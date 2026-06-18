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


def market_state_loss(
    output: MarketStateOutput,
    target: MarketStateTargets,
    *,
    return_weight: float = 0.4,
    direction_weight: float = 0.4,
    volatility_weight: float = 0.15,
    risk_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    ret_loss = F.huber_loss(output.return_pred, target.future_log_ret, delta=0.5)
    dir_loss = F.cross_entropy(
        output.direction_logits.reshape(-1, output.direction_logits.size(-1)),
        target.direction_label.reshape(-1),
    )
    vol_target = torch.log(target.volatility.clamp(min=1e-6))
    vol_pred = torch.log(output.volatility_pred.abs().clamp(min=1e-6))
    vol_loss = F.huber_loss(vol_pred, vol_target, delta=0.5)
    risk_loss = F.cross_entropy(
        output.risk_logits.reshape(-1, output.risk_logits.size(-1)),
        target.risk_label.long().reshape(-1),
    )
    total = (
        return_weight * ret_loss
        + direction_weight * dir_loss
        + volatility_weight * vol_loss
        + risk_weight * risk_loss
    )
    return total, {
        "return_loss": float(ret_loss.detach()),
        "direction_loss": float(dir_loss.detach()),
        "volatility_loss": float(vol_loss.detach()),
        "risk_loss": float(risk_loss.detach()),
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
    ret_p = torch.cat(ret_preds, dim=0)
    ret_y = torch.cat(ret_tgts, dim=0)
    dir_p = torch.cat(dir_preds, dim=0)
    dir_y = torch.cat(dir_tgts, dim=0)
    vol_p = torch.cat(vol_preds, dim=0)
    vol_y = torch.cat(vol_tgts, dim=0)
    risk_p = torch.cat(risk_preds, dim=0)
    risk_y = torch.cat(risk_tgts, dim=0)
    pr, yr = ret_p.reshape(-1).numpy(), ret_y.reshape(-1).numpy()
    return_ic = float(np.corrcoef(pr, yr)[0, 1]) if pr.std() > 1e-8 and yr.std() > 1e-8 else 0.0
    cum_direction_acc = float(((ret_p.sum(dim=1) > 0) == (ret_y.sum(dim=1) > 0)).float().mean().item())
    return {
        "loss": total / max(1, n),
        "direction_acc": float((dir_p == dir_y).float().mean().item()),
        "direction_macro_f1": _macro_f1_score(dir_p.reshape(-1), dir_y.reshape(-1), num_classes=3),
        "cum_direction_acc": cum_direction_acc,
        "return_ic": return_ic,
        "return_mae": float(torch.mean(torch.abs(ret_p - ret_y)).item()),
        "volatility_mae": float(torch.mean(torch.abs(vol_p - vol_y)).item()),
        "risk_f1": _macro_f1_score(risk_p.reshape(-1), risk_y.reshape(-1), num_classes=2),
    }


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
