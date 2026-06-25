"""训练循环（自动切分 Embedding，Stage 1/2/3）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformer_kit.auto_segment_encoder import flat_segment_importance_weights
from transformer_kit.labels import MarketStateTargets
from transformer_kit.pattern_model import (
    MarketStateOutput,
    future_prediction_loss,
    stage3_prediction_loss,
    cumulative_direction_loss,
)
from transformer_kit.schedulers import WarmupCosineAnnealingWarmRestarts
from transformer_kit.vector_quantizer import VectorQuantizerEMA, soft_code_usage_entropy_loss, z_variance_spread_loss


@dataclass
class TrainStepResult:
    loss: float
    lr: float
    extras: dict[str, float]


def _macro_f1_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    f1s: list[float] = []
    for c in range(num_classes):
        tp = ((pred == c) & (target == c)).sum().item()
        fp = ((pred == c) & (target != c)).sum().item()
        fn = ((pred != c) & (target == c)).sum().item()
        denom = (2 * tp + fp + fn)
        f1s.append(0.0 if denom == 0 else (2.0 * tp) / denom)
    return float(sum(f1s) / max(1, len(f1s)))


def _class_distribution(counts: torch.Tensor) -> dict[str, float]:
    total = float(counts.sum().clamp(min=1.0).item())
    return {f"c{i}": float(counts[i].item() / total) for i in range(counts.numel())}


def _confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> list[list[int]]:
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for p, t in zip(pred.reshape(-1), target.reshape(-1)):
        cm[int(t), int(p)] += 1
    return cm.tolist()


def _focal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    return (((1.0 - pt) ** gamma) * ce).mean()


def market_state_loss(
    output: MarketStateOutput,
    target: MarketStateTargets,
    *,
    return_weight: float = 0.4,
    direction_weight: float = 0.4,
    volatility_weight: float = 0.15,
    risk_weight: float = 0.05,
    cum_direction_weight: float = 0.0,
    cum_return_weight: float = 0.0,
    cum_direction_head_weight: float = 0.0,
    return_consistency_weight: float = 0.0,
    return_horizon_weights: torch.Tensor | None = None,
    direction_class_weight: torch.Tensor | None = None,
    risk_class_weight: torch.Tensor | None = None,
    risk_focal_loss: bool = False,
    focal_gamma: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    ret_loss = F.huber_loss(output.return_pred, target.future_log_ret, delta=0.5, reduction="none")
    if return_horizon_weights is not None:
        w = return_horizon_weights.to(device=ret_loss.device, dtype=ret_loss.dtype)
        if w.numel() != ret_loss.size(-1):
            raise ValueError(
                f"return_horizon_weights length {w.numel()} != pred_horizon {ret_loss.size(-1)}"
            )
        ret_loss = (ret_loss * w.view(1, -1)).mean()
    else:
        ret_loss = ret_loss.mean()
    dir_loss = F.cross_entropy(
        output.direction_logits.reshape(-1, output.direction_logits.size(-1)),
        target.direction_label.reshape(-1),
        weight=direction_class_weight,
    )
    vol_target = torch.log(target.volatility.clamp(min=1e-6))
    vol_pred = torch.log(output.volatility_pred.abs().clamp(min=1e-6))
    vol_loss = F.huber_loss(vol_pred, vol_target, delta=0.5)
    risk_logits = output.risk_logits.reshape(-1, output.risk_logits.size(-1))
    risk_targets = target.risk_label.long().reshape(-1)
    if risk_focal_loss:
        risk_loss = _focal_cross_entropy(
            risk_logits,
            risk_targets,
            weight=risk_class_weight,
            gamma=focal_gamma,
        )
    else:
        risk_loss = F.cross_entropy(risk_logits, risk_targets, weight=risk_class_weight)
    cum_dir_from_return_loss = F.binary_cross_entropy_with_logits(
        output.return_pred.sum(dim=1),
        (target.future_log_ret.sum(dim=1) > 0).float(),
    )
    zero = ret_loss.new_zeros(())
    cum_return_loss = zero
    cum_dir_head_loss = zero
    consistency_loss = zero
    if output.cum_return_pred is not None:
        target_cum_return = target.future_log_ret.sum(dim=1)
        cum_return_loss = F.huber_loss(output.cum_return_pred, target_cum_return, delta=0.5)
        consistency_loss = F.huber_loss(
            output.return_pred.sum(dim=1),
            output.cum_return_pred.detach(),
            delta=0.5,
        )
    if output.cum_direction_logit is not None:
        target_cum_dir = (target.future_log_ret.sum(dim=1) > 0).float()
        cum_dir_head_loss = F.binary_cross_entropy_with_logits(output.cum_direction_logit, target_cum_dir)
    total = (
        return_weight * ret_loss
        + direction_weight * dir_loss
        + volatility_weight * vol_loss
        + risk_weight * risk_loss
        + cum_direction_weight * cum_dir_from_return_loss
        + cum_return_weight * cum_return_loss
        + cum_direction_head_weight * cum_dir_head_loss
        + return_consistency_weight * consistency_loss
    )
    return total, {
        "return_loss": float(ret_loss.detach()),
        "direction_loss": float(dir_loss.detach()),
        "volatility_loss": float(vol_loss.detach()),
        "risk_loss": float(risk_loss.detach()),
        "cum_direction_from_return_loss": float(cum_dir_from_return_loss.detach()),
        "cum_return_loss": float(cum_return_loss.detach()),
        "cum_direction_head_loss": float(cum_dir_head_loss.detach()),
        "return_consistency_loss": float(consistency_loss.detach()),
    }


def _stage3_raw_kwargs(batch: dict, device: torch.device) -> dict:
    if "future_raw_log_ret" not in batch:
        return {}
    out = {
        "target_raw_log_ret": batch["future_raw_log_ret"].to(device),
    }
    if "future_log_ret_mean" in batch:
        out["log_ret_mean"] = batch["future_log_ret_mean"].to(device)
        out["log_ret_std"] = batch["future_log_ret_std"].to(device)
    else:
        out["log_ret_mean"] = batch["log_ret_mean"].to(device)
        out["log_ret_std"] = batch["log_ret_std"].to(device)
    return out


def _direction_seq_targets(future: torch.Tensor) -> torch.Tensor:
    """未来 H 步涨跌标签（1=涨，0=跌/平）。"""
    y = future[..., 0] if future.dim() == 3 else future
    return (y > 0).float()


def _direction_seq_loss(
    pred: torch.Tensor,
    future: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = pred[..., 0] if pred.dim() == 3 else pred
    labels = _direction_seq_targets(future)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if sample_weight is not None:
        bce = bce * sample_weight.unsqueeze(1)
    loss = bce.mean()
    with torch.no_grad():
        pred_bin = (logits > 0).float()
        step_acc = float((pred_bin == labels).float().mean().item())
        all5_acc = float((pred_bin == labels).all(dim=1).float().mean().item())
    return loss, {"step_direction_acc": step_acc, "all5_direction_acc": all5_acc}


def _direction_seq_emissions(pred: torch.Tensor) -> torch.Tensor:
    logits_up = pred[..., 0] if pred.dim() == 3 else pred
    logits_down = torch.zeros_like(logits_up)
    return torch.stack([logits_down, logits_up], dim=-1)


def _crf_path_score(
    emissions: torch.Tensor,
    labels: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
    trans: torch.Tensor,
) -> torch.Tensor:
    b, t, _ = emissions.shape
    rows = torch.arange(b, device=emissions.device)
    score = start[labels[:, 0]] + emissions[rows, 0, labels[:, 0]]
    for i in range(1, t):
        prev = labels[:, i - 1]
        cur = labels[:, i]
        score = score + trans[prev, cur] + emissions[rows, i, cur]
    score = score + end[labels[:, -1]]
    return score


def _crf_log_partition(
    emissions: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
    trans: torch.Tensor,
) -> torch.Tensor:
    alpha = start.unsqueeze(0) + emissions[:, 0, :]
    t = emissions.size(1)
    for i in range(1, t):
        score = alpha.unsqueeze(2) + trans.unsqueeze(0)
        alpha = torch.logsumexp(score, dim=1) + emissions[:, i, :]
    return torch.logsumexp(alpha + end.unsqueeze(0), dim=1)


def _crf_neg_log_likelihood(
    emissions: torch.Tensor,
    labels: torch.Tensor,
    *,
    start: torch.Tensor,
    end: torch.Tensor,
    trans: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    gold = _crf_path_score(emissions, labels, start, end, trans)
    part = _crf_log_partition(emissions, start, end, trans)
    nll = part - gold
    if sample_weight is not None:
        nll = nll * sample_weight
        return nll.sum() / sample_weight.sum().clamp(min=1e-8)
    return nll.mean()


def _crf_viterbi_decode(
    emissions: torch.Tensor,
    *,
    start: torch.Tensor,
    end: torch.Tensor,
    trans: torch.Tensor,
) -> torch.Tensor:
    b, t, c = emissions.shape
    delta = start.unsqueeze(0) + emissions[:, 0, :]
    backpointers: list[torch.Tensor] = []
    for i in range(1, t):
        score = delta.unsqueeze(2) + trans.unsqueeze(0)
        best_score, best_prev = score.max(dim=1)
        delta = best_score + emissions[:, i, :]
        backpointers.append(best_prev)
    delta = delta + end.unsqueeze(0)
    last = delta.argmax(dim=1)
    paths = [last]
    for bp in reversed(backpointers):
        last = bp[torch.arange(b, device=emissions.device), last]
        paths.append(last)
    return torch.stack(list(reversed(paths)), dim=1)


def _direction_seq_crf_loss(
    model: nn.Module,
    pred: torch.Tensor,
    future: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    emissions = _direction_seq_emissions(pred)
    labels = _direction_seq_targets(future).long()
    start = model.direction_seq_start
    end = model.direction_seq_end
    trans = model.direction_seq_trans
    loss = _crf_neg_log_likelihood(
        emissions,
        labels,
        start=start,
        end=end,
        trans=trans,
        sample_weight=sample_weight,
    )
    with torch.no_grad():
        path = _crf_viterbi_decode(emissions, start=start, end=end, trans=trans).float()
        step_acc = float((path == labels.float()).float().mean().item())
        all5_acc = float((path == labels.float()).all(dim=1).float().mean().item())
    return loss, {"step_direction_acc": step_acc, "all5_direction_acc": all5_acc}


def _maybe_clip(model: nn.Module, grad_clip: float) -> None:
    if grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)


@torch.no_grad()
def evaluate_auto_vqvae(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total = recon_sum = vq_sum = br_sum = ppl_sum = 0.0
    n = 0
    for batch in loader:
        bars = batch["ctx_bars"].to(device)
        lengths = batch["ctx_lengths"].to(device)
        out = model(bars, lengths)
        bs = bars.size(0)
        total += out.total_loss.item() * bs
        recon_sum += out.recon_loss.item() * bs
        vq_sum += out.auto_out.vq_loss.item() * bs
        br_sum += out.auto_out.break_reg_loss.item() * bs
        ppl_sum += out.auto_out.perplexity.item() * bs
        n += bs
    return {
        "loss": total / max(1, n),
        "recon": recon_sum / max(1, n),
        "vq": vq_sum / max(1, n),
        "break_reg": br_sum / max(1, n),
        "perplexity": ppl_sum / max(1, n),
    }


def _vq_module(model: nn.Module) -> VectorQuantizerEMA | None:
    enc = getattr(model, "auto_encoder", None) or getattr(model, "pattern_encoder", None)
    vq = getattr(enc, "vq", None) if enc is not None else None
    return vq if isinstance(vq, VectorQuantizerEMA) else None


def _collect_active_z(model: nn.Module, auto_out) -> torch.Tensor | None:
    enc = getattr(model, "auto_encoder", None)
    if enc is None:
        return None
    b, s, max_len, f = auto_out.seg_bars.shape
    flat = auto_out.seg_bars.reshape(b * s, max_len, f)
    flat_lengths = auto_out.seg_lengths.reshape(b * s)
    active = (
        torch.arange(s, device=flat.device).unsqueeze(0) < auto_out.num_segments.unsqueeze(1)
    ).reshape(-1)
    if not active.any():
        return None
    return enc.segment_encoder(flat[active], flat_lengths[active])


def _collect_segment_z(model: nn.Module, auto_out, bars: torch.Tensor) -> torch.Tensor | None:
    return _collect_active_z(model, auto_out)


def _active_vq_codes(auto_out) -> torch.Tensor:
    b, s = auto_out.codes.shape
    active = (
        torch.arange(s, device=auto_out.codes.device).unsqueeze(0)
        < auto_out.num_segments.unsqueeze(1)
    ).reshape(-1)
    return auto_out.codes.reshape(-1)[active]


def train_auto_vqvae_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineAnnealingWarmRestarts | None,
    device: torch.device,
    *,
    grad_clip: float = 1.0,
    diversity_weight: float = 0.05,
    usage_balance_weight: float = 0.0,
    z_spread_weight: float = 0.0,
    break_aware_vq_balance: bool = False,
    break_seg_vq_weight: float = 2.0,
    background_seg_vq_weight: float = 0.35,
    vq_dead_threshold: float = 0.1,
    vq_max_code_frac: float = 0.15,
    vq_kmeans_frac: float = 0.45,
) -> TrainStepResult:
    model.train()
    total = ppl_sum = nseg_sum = 0.0
    n = 0
    epoch_z: list[torch.Tensor] = []
    num_codes = _vq_num_codes(model)
    for batch in loader:
        bars = batch["ctx_bars"].to(device)
        lengths = batch["ctx_lengths"].to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(bars, lengths)
        loss = out.total_loss
        if diversity_weight > 0:
            loss = loss - diversity_weight * out.auto_out.perplexity
        if usage_balance_weight > 0:
            vq = _vq_module(model)
            z_active = out.auto_out.z_vec
            if vq is not None and z_active is not None and z_active.numel() > 0:
                seg_w = None
                if break_aware_vq_balance:
                    ao = out.auto_out
                    seg_w = flat_segment_importance_weights(
                        ao.num_segments,
                        max_segments=ao.codes.size(1),
                        break_seg_weight=break_seg_vq_weight,
                        background_seg_weight=background_seg_vq_weight,
                    )
                loss = loss + usage_balance_weight * soft_code_usage_entropy_loss(
                    z_active, vq.weight, cosine=vq.cosine_distance, sample_weight=seg_w,
                )
        if z_spread_weight > 0:
            z_active = out.auto_out.z_vec
            if z_active is not None and z_active.numel() > 1:
                loss = loss + z_spread_weight * z_variance_spread_loss(z_active)
        loss.backward()
        _maybe_clip(model, grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        with torch.no_grad():
            z = out.auto_out.z_vec
            if z is not None and z.numel() > 0:
                epoch_z.append(z.detach())
        bs = bars.size(0)
        total += out.total_loss.item() * bs
        ppl_sum += out.auto_out.perplexity.item() * bs
        nseg_sum += out.auto_out.num_segments.float().mean().item() * bs
        n += bs
    z_pool = torch.cat(epoch_z, dim=0) if epoch_z else None
    vq = _vq_module(model)
    max_epoch_frac = vq.max_epoch_usage_frac() if vq is not None else 0.0
    max_ema_frac = vq.max_ema_usage_frac() if vq is not None else 0.0
    n_dead, n_dom = rebalance_vq_codes(
        model,
        z_samples=z_pool,
        dead_threshold=vq_dead_threshold,
        max_usage_frac=vq_max_code_frac,
    )
    kmeans_refresh = 0.0
    if (
        vq is not None
        and z_pool is not None
        and z_pool.size(0) >= num_codes
        and max(max_epoch_frac, max_ema_frac) > vq_kmeans_frac
    ):
        vq.init_from_encoder_outputs(z_pool)
        rd2, dm2 = rebalance_vq_codes(
            model,
            z_samples=z_pool,
            dead_threshold=vq_dead_threshold,
            max_usage_frac=vq_max_code_frac,
        )
        n_dead += rd2
        n_dom += dm2
        kmeans_refresh = 1.0
    extras: dict[str, float] = {
        "perplexity": ppl_sum / max(1, n),
        "avg_segments": nseg_sum / max(1, n),
        "vq_reset_dead": float(n_dead),
        "vq_reset_dominant": float(n_dom),
        "vq_kmeans_refresh": kmeans_refresh,
        "vq_max_ema_frac": max_ema_frac,
        "vq_max_epoch_frac": max_epoch_frac,
    }
    return TrainStepResult(
        loss=total / max(1, n),
        lr=optimizer.param_groups[0]["lr"],
        extras=extras,
    )


def _vq_num_codes(model: nn.Module) -> int:
    enc = getattr(model, "auto_encoder", None) or getattr(model, "pattern_encoder", None)
    vq = getattr(enc, "vq", None) if enc is not None else None
    return int(getattr(vq, "num_codes", 0) or 0)


def rebalance_vq_codes(
    model: nn.Module,
    *,
    dead_threshold: float = 0.1,
    max_usage_frac: float = 0.22,
    z_samples: torch.Tensor | None = None,
) -> tuple[int, int]:
    enc = getattr(model, "auto_encoder", None) or getattr(model, "pattern_encoder", None)
    vq = getattr(enc, "vq", None) if enc is not None else None
    if isinstance(vq, VectorQuantizerEMA):
        return vq.rebalance_codes(
            dead_threshold=dead_threshold,
            max_usage_frac=max_usage_frac,
            z_samples=z_samples,
        )
    return 0, 0


def reset_vq_dead_codes(
    model: nn.Module,
    *,
    threshold: float = 0.1,
    z_samples: torch.Tensor | None = None,
) -> int:
    n_dead, _ = rebalance_vq_codes(
        model, dead_threshold=threshold, max_usage_frac=1.0, z_samples=z_samples
    )
    return n_dead


@torch.no_grad()
def init_vq_codebook_from_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> None:
    enc = getattr(model, "auto_encoder", None)
    if enc is None:
        return
    vq = enc.vq
    if not isinstance(vq, VectorQuantizerEMA):
        return
    was_training = model.training
    model.eval()
    zs: list[torch.Tensor] = []
    for batch in loader:
        bars = batch["ctx_bars"].to(device)
        lengths = batch["ctx_lengths"].to(device)
        out = model(bars, lengths)
        z = out.auto_out.z_vec
        if z is not None and z.size(0) > 0:
            zs.append(z)
        if sum(t.size(0) for t in zs) >= max(vq.num_codes * 8, 128):
            break
    if was_training:
        model.train()
    if not zs:
        return
    z_all = torch.cat(zs, dim=0)
    if z_all.size(0) >= vq.num_codes:
        vq.init_from_encoder_outputs(z_all)
        print(f"  VQ codebook k-means++ init from {z_all.size(0)} segment vectors")


# 兼容旧名
evaluate_stage1 = evaluate_auto_vqvae
train_stage1_epoch = train_auto_vqvae_epoch
evaluate_stage2 = evaluate_auto_vqvae
train_stage2_epoch = train_auto_vqvae_epoch


@torch.no_grad()
def evaluate_stage3(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    aux_vq_weight: float = 0.1,
    aux_break_weight: float = 0.05,
    mse_weight: float = 0.7,
    step_corr_weight: float = 0.25,
    cum_corr_weight: float = 0.35,
    sign_weight: float = 0.15,
    rank_weight: float = 0.0,
    direction_weight: float = 0.0,
    shape_weight: float = 0.0,
    path_shape_weight: float = 0.0,
    cum_magnitude_weight: float = 0.0,
    relative_magnitude_weight: float = 0.0,
    raw_mse_weight: float = 0.0,
    magnitude_tolerance: float = 0.2,
    magnitude_min_move: float = 1e-4,
    vol_focus_weight: float = 0.0,
    vol_focus_top_frac: float = 0.3,
    move_focus_weight: float = 0.0,
    move_focus_scale: float = 3.0,
    break_focus_weight: float = 0.0,
    break_focus_tail: int = 16,
    code_supervision_weight: float = 0.0,
    anti_lag_weight: float = 0.0,
    anti_lag_margin: float = 0.05,
    corr_weight: float | None = None,
    use_ic_loss: bool = True,
    direction_seq_only: bool = False,
    direction_seq_weight: float = 1.0,
    direction_seq_crf: bool = False,
) -> dict[str, float]:
    model.eval()
    total = 0.0
    n = 0
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    preds_raw: list[np.ndarray] = []
    targets_raw: list[np.ndarray] = []
    direction_logits: list[torch.Tensor] = []
    code_supervision_logits: list[torch.Tensor] = []
    for batch in loader:
        ctx = batch["ctx_bars"].to(device)
        ctx_len = batch["ctx_lengths"].to(device)
        future = batch["future_bars"].to(device)
        pred, aux = model(ctx, ctx_len, return_aux=True)
        if direction_seq_only:
            if direction_seq_crf:
                cls_loss, cls_extras = _direction_seq_crf_loss(model, pred, future)
            else:
                cls_loss, cls_extras = _direction_seq_loss(pred, future)
            pred_loss = direction_seq_weight * cls_loss
            if direction_weight > 0 and "cum_direction_logit" in aux:
                pred_loss = pred_loss + direction_weight * cumulative_direction_loss(
                    aux["cum_direction_logit"], future, None
                )
            if code_supervision_weight > 0 and "code_supervision_logit" in aux:
                pred_loss = pred_loss + code_supervision_weight * cumulative_direction_loss(
                    aux["code_supervision_logit"], future, None
                )
        elif use_ic_loss:
            raw_kw = _stage3_raw_kwargs(batch, device)
            pred_loss, _ = stage3_prediction_loss(
                pred, future,
                ctx_bars=ctx,
                ctx_lengths=ctx_len,
                break_logits=aux.get("break_logits"),
                direction_logit=aux.get("cum_direction_logit"),
                code_supervision_logit=aux.get("code_supervision_logit"),
                mse_weight=mse_weight, step_corr_weight=step_corr_weight,
                cum_corr_weight=cum_corr_weight, sign_weight=sign_weight,
                rank_weight=rank_weight, direction_weight=direction_weight,
                shape_weight=shape_weight,
                path_shape_weight=path_shape_weight,
                cum_magnitude_weight=cum_magnitude_weight,
                relative_magnitude_weight=relative_magnitude_weight,
                raw_mse_weight=raw_mse_weight,
                vol_focus_weight=vol_focus_weight,
                vol_focus_top_frac=vol_focus_top_frac,
                move_focus_weight=move_focus_weight,
                move_focus_scale=move_focus_scale,
                break_focus_weight=break_focus_weight,
                break_focus_tail=break_focus_tail,
                code_supervision_weight=code_supervision_weight,
                anti_lag_weight=anti_lag_weight,
                anti_lag_margin=anti_lag_margin,
                corr_weight=corr_weight,
                **raw_kw,
            )
        else:
            pred_loss = future_prediction_loss(pred, future)
        loss = pred_loss + aux_vq_weight * aux["vq_loss"] + aux_break_weight * aux["break_reg_loss"]
        bs = ctx.size(0)
        total += loss.item() * bs
        n += bs
        preds.append(pred.detach().cpu())
        targets.append(future[..., :1].detach().cpu())
        if (not direction_seq_only) and "future_raw_log_ret" in batch:
            from transformer_kit.magnitude_metrics import denorm_zscore_log_ret

            pz = pred.detach().cpu().numpy()
            if pz.ndim == 3:
                pz = pz[..., 0]
            if "future_log_ret_mean" in batch:
                preds_raw.append(
                    denorm_zscore_log_ret(
                        pz,
                        batch["future_log_ret_mean"].numpy(),
                        batch["future_log_ret_std"].numpy(),
                    )
                )
            else:
                preds_raw.append(
                    denorm_zscore_log_ret(pz, batch["log_ret_mean"].numpy(), batch["log_ret_std"].numpy())
                )
            targets_raw.append(batch["future_raw_log_ret"].numpy())
        if "cum_direction_logit" in aux:
            direction_logits.append(aux["cum_direction_logit"].detach().cpu())
        if "code_supervision_logit" in aux:
            code_supervision_logits.append(aux["code_supervision_logit"].detach().cpu())
    out = {"loss": total / max(1, n)}
    if preds:
        p = torch.cat(preds, dim=0).numpy()
        y = torch.cat(targets, dim=0).numpy()
        if p.ndim == 3:
            p = p[..., 0]
            y = y[..., 0]
        if direction_seq_only:
            logits = p
            labels = (y > 0).astype(np.float32)
            if direction_seq_crf:
                emissions = _direction_seq_emissions(torch.from_numpy(logits).to(device)).detach()
                path = _crf_viterbi_decode(
                    emissions,
                    start=model.direction_seq_start,
                    end=model.direction_seq_end,
                    trans=model.direction_seq_trans,
                )
                pred_bin = path.detach().cpu().numpy().astype(np.float32)
            else:
                pred_bin = (logits > 0).astype(np.float32)
            out["step_direction_acc"] = float((pred_bin == labels).mean())
            out["all5_direction_acc"] = float((pred_bin == labels).all(axis=1).mean())
            out["cum_direction_acc"] = float(
                ((pred_bin.sum(axis=1) > (pred_bin.shape[1] / 2)).astype(np.float32) == (y.sum(axis=1) > 0).astype(np.float32)).mean()
            )
            out["direction_seq_crf"] = float(direction_seq_crf)
            if direction_logits:
                dlog = torch.cat(direction_logits, dim=0).numpy()
                c_labels = (y.sum(axis=1) > 0).astype(np.float32)
                out["direction_head_acc"] = float(((dlog > 0).astype(np.float32) == c_labels).mean())
            if code_supervision_logits:
                clog = torch.cat(code_supervision_logits, dim=0).numpy()
                c_labels = (y.sum(axis=1) > 0).astype(np.float32)
                out["code_supervision_head_acc"] = float(((clog > 0).astype(np.float32) == c_labels).mean())
            return out
        pr, yr = p.ravel(), y.ravel()
        if pr.std() > 1e-8 and yr.std() > 1e-8:
            out["ic"] = float(np.corrcoef(pr, yr)[0, 1])
        if p.ndim >= 2 and p.shape[0] > 2:
            out["cum_ic"] = float(np.corrcoef(p.sum(axis=1), y.sum(axis=1))[0, 1])
        out["direction_acc"] = float((np.sign(pr) == np.sign(yr)).mean())
        out["cum_direction_acc"] = float((np.sign(p.sum(axis=1)) == np.sign(y.sum(axis=1))).mean())
        if direction_logits:
            logits = torch.cat(direction_logits, dim=0).numpy()
            labels = (y.sum(axis=1) > 0).astype(np.float32)
            out["direction_head_acc"] = float(((logits > 0).astype(np.float32) == labels).mean())
        if code_supervision_logits:
            clogits = torch.cat(code_supervision_logits, dim=0).numpy()
            labels = (y.sum(axis=1) > 0).astype(np.float32)
            out["code_supervision_head_acc"] = float(((clogits > 0).astype(np.float32) == labels).mean())
        from transformer_kit.magnitude_metrics import magnitude_accuracy_metrics

        if preds_raw:
            p_mag = np.concatenate(preds_raw, axis=0)
            y_mag = np.concatenate(targets_raw, axis=0)
        else:
            p_mag, y_mag = p, y
        mag = magnitude_accuracy_metrics(
            p_mag, y_mag, tolerance=magnitude_tolerance, min_move=magnitude_min_move,
        )
        out.update(mag)
    return out


def train_stage3_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineAnnealingWarmRestarts | None,
    device: torch.device,
    *,
    aux_vq_weight: float = 0.1,
    aux_break_weight: float = 0.05,
    grad_clip: float = 1.0,
    mse_weight: float = 0.7,
    step_corr_weight: float = 0.25,
    cum_corr_weight: float = 0.35,
    sign_weight: float = 0.15,
    rank_weight: float = 0.0,
    direction_weight: float = 0.0,
    shape_weight: float = 0.0,
    path_shape_weight: float = 0.0,
    cum_magnitude_weight: float = 0.0,
    relative_magnitude_weight: float = 0.0,
    raw_mse_weight: float = 0.0,
    magnitude_tolerance: float = 0.2,
    magnitude_min_move: float = 1e-4,
    vol_focus_weight: float = 0.0,
    vol_focus_top_frac: float = 0.3,
    move_focus_weight: float = 0.0,
    move_focus_scale: float = 3.0,
    break_focus_weight: float = 0.0,
    break_focus_tail: int = 16,
    code_supervision_weight: float = 0.0,
    anti_lag_weight: float = 0.0,
    anti_lag_margin: float = 0.05,
    corr_weight: float | None = None,
    use_ic_loss: bool = True,
    direction_seq_only: bool = False,
    direction_seq_weight: float = 1.0,
    direction_seq_crf: bool = False,
) -> TrainStepResult:
    model.train()
    total = corr_sum = 0.0
    n = 0
    for batch in loader:
        ctx = batch["ctx_bars"].to(device)
        ctx_len = batch["ctx_lengths"].to(device)
        future = batch["future_bars"].to(device)
        optimizer.zero_grad(set_to_none=True)
        pred, aux = model(ctx, ctx_len, return_aux=True)
        if direction_seq_only:
            if direction_seq_crf:
                cls_loss, cls_extras = _direction_seq_crf_loss(model, pred, future)
            else:
                cls_loss, cls_extras = _direction_seq_loss(pred, future)
            pred_loss = direction_seq_weight * cls_loss
            if direction_weight > 0 and "cum_direction_logit" in aux:
                pred_loss = pred_loss + direction_weight * cumulative_direction_loss(
                    aux["cum_direction_logit"], future, None
                )
            if code_supervision_weight > 0 and "code_supervision_logit" in aux:
                pred_loss = pred_loss + code_supervision_weight * cumulative_direction_loss(
                    aux["code_supervision_logit"], future, None
                )
            corr_sum += cls_extras["step_direction_acc"] * ctx.size(0)
        elif use_ic_loss:
            raw_kw = _stage3_raw_kwargs(batch, device)
            pred_loss, extras = stage3_prediction_loss(
                pred, future,
                ctx_bars=ctx,
                ctx_lengths=ctx_len,
                break_logits=aux.get("break_logits"),
                direction_logit=aux.get("cum_direction_logit"),
                code_supervision_logit=aux.get("code_supervision_logit"),
                mse_weight=mse_weight, step_corr_weight=step_corr_weight,
                cum_corr_weight=cum_corr_weight, sign_weight=sign_weight,
                rank_weight=rank_weight, direction_weight=direction_weight,
                shape_weight=shape_weight,
                path_shape_weight=path_shape_weight,
                cum_magnitude_weight=cum_magnitude_weight,
                relative_magnitude_weight=relative_magnitude_weight,
                raw_mse_weight=raw_mse_weight,
                vol_focus_weight=vol_focus_weight,
                vol_focus_top_frac=vol_focus_top_frac,
                move_focus_weight=move_focus_weight,
                move_focus_scale=move_focus_scale,
                break_focus_weight=break_focus_weight,
                break_focus_tail=break_focus_tail,
                code_supervision_weight=code_supervision_weight,
                anti_lag_weight=anti_lag_weight,
                anti_lag_margin=anti_lag_margin,
                corr_weight=corr_weight,
                **raw_kw,
            )
            corr_sum += extras["step_corr"] * ctx.size(0)
        else:
            pred_loss = future_prediction_loss(pred, future)
        loss = pred_loss + aux_vq_weight * aux["vq_loss"] + aux_break_weight * aux["break_reg_loss"]
        loss.backward()
        _maybe_clip(model, grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        bs = ctx.size(0)
        total += loss.item() * bs
        n += bs
    extras: dict[str, float] = {}
    if use_ic_loss and n > 0:
        extras["batch_corr"] = corr_sum / n
    return TrainStepResult(loss=total / max(1, n), lr=optimizer.param_groups[0]["lr"], extras=extras)


@torch.no_grad()
def evaluate_market_state(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    return_weight: float = 0.4,
    direction_weight: float = 0.4,
    volatility_weight: float = 0.15,
    risk_weight: float = 0.05,
    direction_class_weight: torch.Tensor | None = None,
    risk_class_weight: torch.Tensor | None = None,
    cum_direction_weight: float = 0.0,
    cum_return_weight: float = 0.0,
    cum_direction_head_weight: float = 0.0,
    return_consistency_weight: float = 0.0,
    return_horizon_weights: torch.Tensor | None = None,
    risk_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    with_diagnostics: bool = False,
) -> dict[str, float]:
    model.eval()
    total = 0.0
    n = 0
    ret_preds: list[torch.Tensor] = []
    ret_tgts: list[torch.Tensor] = []
    dir_preds: list[torch.Tensor] = []
    dir_tgts: list[torch.Tensor] = []
    vol_preds: list[torch.Tensor] = []
    vol_tgts: list[torch.Tensor] = []
    risk_preds: list[torch.Tensor] = []
    risk_tgts: list[torch.Tensor] = []
    cum_return_preds: list[torch.Tensor] = []
    cum_direction_logits: list[torch.Tensor] = []
    for batch in loader:
        ctx = batch["ctx_bars"].to(device)
        ctx_len = batch["ctx_lengths"].to(device)
        out = model(ctx, ctx_len)
        if not isinstance(out, MarketStateOutput):
            raise RuntimeError("model must return MarketStateOutput in market-state mode")
        tgt = MarketStateTargets(
            future_log_ret=batch["target_return"].to(device),
            direction_label=batch["target_direction"].to(device),
            volatility=batch["target_volatility"].to(device),
            risk_label=batch["target_risk"].to(device),
            move_label=batch.get("target_move", None).to(device) if "target_move" in batch else None,
        )
        loss, _ = market_state_loss(
            out,
            tgt,
            return_weight=return_weight,
            direction_weight=direction_weight,
            volatility_weight=volatility_weight,
            risk_weight=risk_weight,
            cum_direction_weight=cum_direction_weight,
            cum_return_weight=cum_return_weight,
            cum_direction_head_weight=cum_direction_head_weight,
            return_consistency_weight=return_consistency_weight,
            return_horizon_weights=return_horizon_weights,
            direction_class_weight=direction_class_weight,
            risk_class_weight=risk_class_weight,
            risk_focal_loss=risk_focal_loss,
            focal_gamma=focal_gamma,
        )
        bs = ctx.size(0)
        total += float(loss.item()) * bs
        n += bs
        ret_preds.append(out.return_pred.detach().cpu())
        ret_tgts.append(tgt.future_log_ret.detach().cpu())
        dir_preds.append(out.direction_logits.argmax(dim=-1).detach().cpu())
        dir_tgts.append(tgt.direction_label.detach().cpu())
        vol_preds.append(out.volatility_pred.detach().cpu())
        vol_tgts.append(tgt.volatility.detach().cpu())
        risk_preds.append(out.risk_logits.argmax(dim=-1).detach().cpu())
        risk_tgts.append(tgt.risk_label.detach().cpu())
        if out.cum_return_pred is not None:
            cum_return_preds.append(out.cum_return_pred.detach().cpu())
        if out.cum_direction_logit is not None:
            cum_direction_logits.append(out.cum_direction_logit.detach().cpu())
    ret_p = torch.cat(ret_preds, dim=0)
    ret_y = torch.cat(ret_tgts, dim=0)
    dir_p = torch.cat(dir_preds, dim=0)
    dir_y = torch.cat(dir_tgts, dim=0)
    vol_p = torch.cat(vol_preds, dim=0)
    vol_y = torch.cat(vol_tgts, dim=0)
    risk_p = torch.cat(risk_preds, dim=0)
    risk_y = torch.cat(risk_tgts, dim=0)
    true_cum_return = ret_y.sum(dim=1)
    pred_cum_from_return = ret_p.sum(dim=1)
    if cum_return_preds:
        pred_cum_return = torch.cat(cum_return_preds, dim=0)
    else:
        pred_cum_return = pred_cum_from_return
    if cum_direction_logits:
        pred_cum_dir_head = torch.cat(cum_direction_logits, dim=0)
    else:
        pred_cum_dir_head = pred_cum_from_return
    pr, yr = ret_p.reshape(-1).numpy(), ret_y.reshape(-1).numpy()
    return_ic = float(np.corrcoef(pr, yr)[0, 1]) if pr.std() > 1e-8 and yr.std() > 1e-8 else 0.0
    cum_direction_acc = float(((pred_cum_from_return > 0) == (true_cum_return > 0)).float().mean().item())
    cpr, cyr = pred_cum_return.numpy(), true_cum_return.numpy()
    cum_return_ic = float(np.corrcoef(cpr, cyr)[0, 1]) if cpr.std() > 1e-8 and cyr.std() > 1e-8 else 0.0
    cum_direction_head_acc = float(((pred_cum_dir_head > 0) == (true_cum_return > 0)).float().mean().item())
    cum_direction_from_return_acc = float(((pred_cum_from_return > 0) == (true_cum_return > 0)).float().mean().item())
    step_cum_return_gap_mae = float(torch.mean(torch.abs(pred_cum_from_return - pred_cum_return)).item())
    out_metrics: dict[str, float] = {
        "loss": total / max(1, n),
        "direction_acc": float((dir_p == dir_y).float().mean().item()),
        "direction_macro_f1": _macro_f1_score(dir_p.reshape(-1), dir_y.reshape(-1), num_classes=3),
        "cum_direction_acc": cum_direction_acc,
        "return_ic": return_ic,
        "cum_return_ic": cum_return_ic,
        "return_mae": float(torch.mean(torch.abs(ret_p - ret_y)).item()),
        "cum_return_mae": float(torch.mean(torch.abs(pred_cum_return - true_cum_return)).item()),
        "cum_direction_head_acc": cum_direction_head_acc,
        "cum_direction_from_return_acc": cum_direction_from_return_acc,
        "step_cum_return_gap_mae": step_cum_return_gap_mae,
        "volatility_mae": float(torch.mean(torch.abs(vol_p - vol_y)).item()),
        "risk_f1": _macro_f1_score(risk_p.reshape(-1), risk_y.reshape(-1), num_classes=2),
        "pred_return_std": float(ret_p.reshape(-1).std().item()),
        "pred_volatility_std": float(vol_p.reshape(-1).std().item()),
    }
    if with_diagnostics:
        dir_true_counts = torch.bincount(dir_y.reshape(-1), minlength=3).float()
        dir_pred_counts = torch.bincount(dir_p.reshape(-1), minlength=3).float()
        risk_true = risk_y.reshape(-1).float()
        risk_pred = risk_p.reshape(-1).float()
        out_metrics["risk_positive_rate_true"] = float(risk_true.mean().item())
        out_metrics["risk_positive_rate_pred"] = float(risk_pred.mean().item())
        risk_tp = ((risk_pred == 1) & (risk_true == 1)).sum().item()
        risk_fp = ((risk_pred == 1) & (risk_true == 0)).sum().item()
        risk_fn = ((risk_pred == 0) & (risk_true == 1)).sum().item()
        out_metrics["risk_precision"] = float(risk_tp / max(1, risk_tp + risk_fp))
        out_metrics["risk_recall"] = float(risk_tp / max(1, risk_tp + risk_fn))
        for i in range(3):
            out_metrics[f"direction_true_c{i}"] = float(dir_true_counts[i] / dir_true_counts.sum().clamp(min=1))
            out_metrics[f"direction_pred_c{i}"] = float(dir_pred_counts[i] / dir_pred_counts.sum().clamp(min=1))
            true_i = int(dir_true_counts[i].item())
            pred_i = int(((dir_p.reshape(-1) == i) & (dir_y.reshape(-1) == i)).sum().item())
            out_metrics[f"direction_recall_c{i}"] = float(pred_i / max(1, true_i))
        for h in range(ret_p.size(1)):
            ph = ret_p[:, h].numpy()
            yh = ret_y[:, h].numpy()
            out_metrics[f"direction_acc_h{h + 1}"] = float((dir_p[:, h] == dir_y[:, h]).float().mean().item())
            if ph.std() > 1e-8 and yh.std() > 1e-8:
                out_metrics[f"return_ic_h{h + 1}"] = float(np.corrcoef(ph, yh)[0, 1])
            else:
                out_metrics[f"return_ic_h{h + 1}"] = 0.0
        out_metrics["_direction_confusion_matrix"] = _confusion_matrix(dir_p, dir_y, 3)
        out_metrics["_risk_confusion_matrix"] = _confusion_matrix(risk_p, risk_y, 2)
    return out_metrics


def train_market_state_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineAnnealingWarmRestarts | None,
    device: torch.device,
    *,
    grad_clip: float = 1.0,
    return_weight: float = 0.4,
    direction_weight: float = 0.4,
    volatility_weight: float = 0.15,
    risk_weight: float = 0.05,
    direction_class_weight: torch.Tensor | None = None,
    risk_class_weight: torch.Tensor | None = None,
    cum_direction_weight: float = 0.0,
    cum_return_weight: float = 0.0,
    cum_direction_head_weight: float = 0.0,
    return_consistency_weight: float = 0.0,
    return_horizon_weights: torch.Tensor | None = None,
    risk_focal_loss: bool = False,
    focal_gamma: float = 2.0,
) -> TrainStepResult:
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        ctx = batch["ctx_bars"].to(device)
        ctx_len = batch["ctx_lengths"].to(device)
        tgt = MarketStateTargets(
            future_log_ret=batch["target_return"].to(device),
            direction_label=batch["target_direction"].to(device),
            volatility=batch["target_volatility"].to(device),
            risk_label=batch["target_risk"].to(device),
            move_label=batch.get("target_move", None).to(device) if "target_move" in batch else None,
        )
        optimizer.zero_grad(set_to_none=True)
        out = model(ctx, ctx_len)
        if not isinstance(out, MarketStateOutput):
            raise RuntimeError("model must return MarketStateOutput in market-state mode")
        loss, _ = market_state_loss(
            out,
            tgt,
            return_weight=return_weight,
            direction_weight=direction_weight,
            volatility_weight=volatility_weight,
            risk_weight=risk_weight,
            cum_direction_weight=cum_direction_weight,
            cum_return_weight=cum_return_weight,
            cum_direction_head_weight=cum_direction_head_weight,
            return_consistency_weight=return_consistency_weight,
            return_horizon_weights=return_horizon_weights,
            direction_class_weight=direction_class_weight,
            risk_class_weight=risk_class_weight,
            risk_focal_loss=risk_focal_loss,
            focal_gamma=focal_gamma,
        )
        aux = out.aux
        loss = loss + 0.08 * aux["vq_loss"] + 0.04 * aux["break_reg_loss"]
        loss.backward()
        _maybe_clip(model, grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        bs = ctx.size(0)
        total += float(loss.item()) * bs
        n += bs
    return TrainStepResult(loss=total / max(1, n), lr=optimizer.param_groups[0]["lr"], extras={})


def freeze_legacy_market_state_heads(model: nn.Module) -> int:
    """Stop gradient updates on 0062e legacy heads (return/direction/risk/vol)."""
    msh = getattr(model, "market_state_head", None)
    if msh is None:
        return 0
    frozen = 0
    for attr in (
        "return_head",
        "horizon_return_head",
        "direction_state_head",
        "volatility_head",
        "risk_head",
    ):
        sub = getattr(msh, attr, None)
        if sub is None:
            continue
        for p in sub.parameters():
            if p.requires_grad:
                p.requires_grad = False
                frozen += 1
    return frozen


def collect_leg_align_head_params(model: nn.Module) -> list[nn.Parameter]:
    """Trainable params for participation + hz heads only."""
    msh = getattr(model, "market_state_head", None)
    if msh is None:
        return []
    params: list[nn.Parameter] = []
    for attr in ("participation_logit_long", "participation_logit_short", "hz_return_heads"):
        sub = getattr(msh, attr, None)
        if sub is None:
            continue
        params.extend([p for p in sub.parameters() if p.requires_grad])
    return params


def market_state_teacher_drift_loss(
    student: MarketStateOutput,
    teacher: MarketStateOutput,
    *,
    direction_weight: float = 1.0,
    cum_return_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    zero = student.direction_logits.new_zeros(())
    parts = {"drift_direction_kl": 0.0, "drift_cum_return_mse": 0.0}
    total = zero
    if direction_weight > 0:
        s_log = F.log_softmax(student.direction_logits[:, 0, :], dim=-1)
        t_prob = F.softmax(teacher.direction_logits[:, 0, :].detach(), dim=-1)
        dir_kl = F.kl_div(s_log, t_prob, reduction="batchmean")
        parts["drift_direction_kl"] = float(dir_kl.detach())
        total = total + direction_weight * dir_kl
    if cum_return_weight > 0 and student.cum_return_pred is not None and teacher.cum_return_pred is not None:
        cum_mse = F.mse_loss(student.cum_return_pred, teacher.cum_return_pred.detach())
        parts["drift_cum_return_mse"] = float(cum_mse.detach())
        total = total + cum_return_weight * cum_mse
    return total, parts


def _masked_weighted_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if mask.sum() < 1:
        return logits.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    w = weights * mask.float()
    return (loss * w).sum() / w.sum().clamp(min=1e-6)


def leg_align_aux_loss(
    output: MarketStateOutput,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    *,
    participation_weight: float = 0.0,
    hz_12_weight: float = 0.0,
    hz_24_weight: float = 0.0,
    hz_48_weight: float = 0.0,
    leg_dir_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    anchor = (
        output.participation_logit_long
        if output.participation_logit_long is not None
        else output.return_pred[:, 0, 0]
    )
    zero = anchor.sum() * 0.0
    parts: dict[str, float] = {
        "participation_loss": 0.0,
        "hz_12_loss": 0.0,
        "hz_24_loss": 0.0,
        "hz_48_loss": 0.0,
        "leg_dir_loss": 0.0,
    }
    total = zero
    weights = batch["sample_weight"].to(device)
    confirmed = batch["is_leg_confirmed"].to(device) >= 0.5
    align_up = batch["align_direction_up"].to(device) >= 0.5
    align_down = batch["align_direction_down"].to(device) >= 0.5
    fast_down = batch["leg_type_fast_down"].to(device) >= 0.5

    if participation_weight > 0 and output.participation_logit_long is not None:
        long_mask = confirmed & align_up
        short_mask = confirmed & align_down & fast_down
        long_loss = _masked_weighted_bce(
            output.participation_logit_long,
            batch["ideal_participate_long"].to(device),
            long_mask,
            weights,
        )
        short_loss = _masked_weighted_bce(
            output.participation_logit_short,
            batch["ideal_participate_short"].to(device),
            short_mask,
            weights,
        )
        part_loss = 0.5 * (long_loss + short_loss)
        parts["participation_loss"] = float(part_loss.detach())
        total = total + participation_weight * part_loss

    hz_weights = {12: hz_12_weight, 24: hz_24_weight, 48: hz_48_weight}
    if output.hz_return_pred is not None:
        for h, w in hz_weights.items():
            if w <= 0 or h not in output.hz_return_pred:
                continue
            key = f"target_hz_return_{h}"
            if key not in batch:
                continue
            pred = output.hz_return_pred[h]
            tgt = batch[key].to(device)
            mask = confirmed & (align_up | align_down)
            if mask.sum() < 1:
                continue
            huber = F.huber_loss(pred, tgt, delta=0.5, reduction="none")
            ww = weights * mask.float()
            hz_loss = (huber * ww).sum() / ww.sum().clamp(min=1e-6)
            parts[f"hz_{h}_loss"] = float(hz_loss.detach())
            total = total + w * hz_loss

    if leg_dir_weight > 0:
        dir_logits = output.direction_logits[:, 0, :]
        target = torch.full((dir_logits.size(0),), 1, dtype=torch.long, device=device)
        target[confirmed & align_up] = 2
        target[confirmed & align_down & fast_down] = 0
        leg_mask = confirmed & ((align_up) | (align_down & fast_down))
        if leg_mask.sum() >= 1:
            dir_loss = F.cross_entropy(dir_logits[leg_mask], target[leg_mask])
            parts["leg_dir_loss"] = float(dir_loss.detach())
            total = total + leg_dir_weight * dir_loss

    return total, parts


def _roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        order = np.argsort(y_score)
        y_sorted = y_true[order]
        n_pos = y_sorted.sum()
        n_neg = len(y_sorted) - n_pos
        if n_pos < 1 or n_neg < 1:
            return 0.5
        ranks = np.arange(1, len(y_sorted) + 1)
        sum_ranks_pos = ranks[y_sorted == 1].sum()
        return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def evaluate_leg_align_market_state(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    return_weight: float = 0.35,
    direction_weight: float = 0.30,
    volatility_weight: float = 0.10,
    risk_weight: float = 0.09,
    cum_return_weight: float = 0.18,
    cum_direction_head_weight: float = 0.03,
    return_consistency_weight: float = 0.01,
    participation_weight: float = 0.0,
    hz_12_weight: float = 0.0,
    hz_24_weight: float = 0.0,
    hz_48_weight: float = 0.0,
    leg_dir_weight: float = 0.0,
) -> dict[str, float]:
    base = evaluate_market_state(
        model,
        loader,
        device,
        return_weight=return_weight,
        direction_weight=direction_weight,
        volatility_weight=volatility_weight,
        risk_weight=risk_weight,
        cum_return_weight=cum_return_weight,
        cum_direction_head_weight=cum_direction_head_weight,
        return_consistency_weight=return_consistency_weight,
    )
    model.eval()
    long_scores: list[float] = []
    long_labels: list[float] = []
    short_scores: list[float] = []
    short_labels: list[float] = []
    hz24_pred: list[float] = []
    hz24_true: list[float] = []
    flat_edge_long: list[float] = []
    flat_edge_short: list[float] = []

    for batch in loader:
        ctx = batch["ctx_bars"].to(device)
        ctx_len = batch["ctx_lengths"].to(device)
        out = model(ctx, ctx_len)
        if not isinstance(out, MarketStateOutput):
            raise RuntimeError("model must return MarketStateOutput")
        if out.participation_logit_long is None:
            continue
        pl = torch.sigmoid(out.participation_logit_long).detach().cpu().numpy()
        ps = torch.sigmoid(out.participation_logit_short).detach().cpu().numpy()
        confirmed = batch["is_leg_confirmed"].numpy() >= 0.5
        up = batch["align_direction_up"].numpy() >= 0.5
        down = batch["align_direction_down"].numpy() >= 0.5
        fast_down = batch["leg_type_fast_down"].numpy() >= 0.5
        ideal_l = batch["ideal_participate_long"].numpy()
        ideal_s = batch["ideal_participate_short"].numpy()
        long_mask = confirmed & up
        short_mask = confirmed & down & fast_down
        if long_mask.any():
            long_scores.extend(pl[long_mask].tolist())
            long_labels.extend(ideal_l[long_mask].tolist())
            flat_edge_long.extend((2.0 * pl[long_mask] - 1.0).tolist())
        if short_mask.any():
            short_scores.extend(ps[short_mask].tolist())
            short_labels.extend(ideal_s[short_mask].tolist())
            flat_edge_short.extend((2.0 * ps[short_mask] - 1.0).tolist())
        if out.hz_return_pred is not None and 24 in out.hz_return_pred and "target_hz_return_24" in batch:
            pred24 = out.hz_return_pred[24].detach().cpu().numpy()
            true24 = batch["target_hz_return_24"].numpy()
            leg_mask = confirmed & (up | down)
            if leg_mask.any():
                hz24_pred.extend(pred24[leg_mask].tolist())
                hz24_true.extend(true24[leg_mask].tolist())

    auc_l = _roc_auc_score(np.asarray(long_labels), np.asarray(long_scores)) if long_labels else 0.5
    auc_s = _roc_auc_score(np.asarray(short_labels), np.asarray(short_scores)) if short_labels else 0.5
    participation_auc = 0.5 * (auc_l + auc_s)
    hz_acc = 0.0
    if hz24_pred:
        hp = np.sign(np.asarray(hz24_pred))
        ht = np.sign(np.asarray(hz24_true))
        hz_acc = float((hp == ht).mean())
    base.update(
        {
            "participation_auc": participation_auc,
            "participation_auc_long": auc_l,
            "participation_auc_short": auc_s,
            "confirmed_leg_flat_edge_p50_long": float(np.median(flat_edge_long)) if flat_edge_long else 0.0,
            "confirmed_leg_flat_edge_p50_short": float(np.median(flat_edge_short)) if flat_edge_short else 0.0,
            "hz_direction_acc_24": hz_acc,
        }
    )
    return base


def train_leg_align_market_state_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    *,
    grad_clip: float = 1.0,
    return_weight: float = 0.35,
    direction_weight: float = 0.30,
    volatility_weight: float = 0.10,
    risk_weight: float = 0.09,
    cum_return_weight: float = 0.18,
    cum_direction_head_weight: float = 0.03,
    return_consistency_weight: float = 0.01,
    participation_weight: float = 0.25,
    hz_12_weight: float = 0.0,
    hz_24_weight: float = 0.0,
    hz_48_weight: float = 0.0,
    leg_dir_weight: float = 0.0,
    base_loss_scale: float = 1.0,
    drift_weight: float = 0.0,
    drift_direction_weight: float = 1.0,
    drift_cum_return_weight: float = 1.0,
    teacher: nn.Module | None = None,
    encoder_aux_loss: bool = True,
) -> TrainStepResult:
    model.train()
    total = 0.0
    n = 0
    drift_parts_sum = {"drift_direction_kl": 0.0, "drift_cum_return_mse": 0.0}
    for batch in loader:
        ctx = batch["ctx_bars"].to(device)
        ctx_len = batch["ctx_lengths"].to(device)
        tgt = MarketStateTargets(
            future_log_ret=batch["target_return"].to(device),
            direction_label=batch["target_direction"].to(device),
            volatility=batch["target_volatility"].to(device),
            risk_label=batch["target_risk"].to(device),
        )
        optimizer.zero_grad(set_to_none=True)
        out = model(ctx, ctx_len)
        if not isinstance(out, MarketStateOutput):
            raise RuntimeError("model must return MarketStateOutput")
        aux_loss, _ = leg_align_aux_loss(
            out,
            batch,
            device,
            participation_weight=participation_weight,
            hz_12_weight=hz_12_weight,
            hz_24_weight=hz_24_weight,
            hz_48_weight=hz_48_weight,
            leg_dir_weight=leg_dir_weight,
        )
        loss = aux_loss
        if base_loss_scale > 0:
            base_loss, _ = market_state_loss(
                out,
                tgt,
                return_weight=return_weight,
                direction_weight=direction_weight,
                volatility_weight=volatility_weight,
                risk_weight=risk_weight,
                cum_return_weight=cum_return_weight,
                cum_direction_head_weight=cum_direction_head_weight,
                return_consistency_weight=return_consistency_weight,
            )
            loss = loss + base_loss_scale * base_loss
        if drift_weight > 0 and teacher is not None:
            with torch.no_grad():
                t_out = teacher(ctx, ctx_len)
            if isinstance(t_out, MarketStateOutput):
                drift_loss, drift_parts = market_state_teacher_drift_loss(
                    out,
                    t_out,
                    direction_weight=drift_direction_weight,
                    cum_return_weight=drift_cum_return_weight,
                )
                loss = loss + drift_weight * drift_loss
                for k, v in drift_parts.items():
                    drift_parts_sum[k] += v
        aux = out.aux
        if encoder_aux_loss:
            loss = loss + 0.08 * aux["vq_loss"] + 0.04 * aux["break_reg_loss"]
        if not loss.requires_grad:
            continue
        loss.backward()
        _maybe_clip(model, grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        bs = ctx.size(0)
        total += float(loss.item()) * bs
        n += bs
    extras = {k: v / max(1, len(loader)) for k, v in drift_parts_sum.items()}
    return TrainStepResult(loss=total / max(1, n), lr=optimizer.param_groups[0]["lr"], extras=extras)
