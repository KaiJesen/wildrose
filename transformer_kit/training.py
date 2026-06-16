"""训练循环（自动切分 Embedding，Stage 1/2/3）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer_kit.pattern_model import future_prediction_loss, stage3_prediction_loss
from transformer_kit.schedulers import WarmupCosineAnnealingWarmRestarts
from transformer_kit.vector_quantizer import VectorQuantizerEMA, soft_code_usage_entropy_loss, z_variance_spread_loss


@dataclass
class TrainStepResult:
    loss: float
    lr: float
    extras: dict[str, float]


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
    vq_dead_threshold: float = 0.1,
    vq_max_code_frac: float = 0.22,
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
                loss = loss + usage_balance_weight * soft_code_usage_entropy_loss(
                    z_active, vq.weight, cosine=vq.cosine_distance,
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
    corr_weight: float | None = None,
    use_ic_loss: bool = True,
) -> dict[str, float]:
    model.eval()
    total = 0.0
    n = 0
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    direction_logits: list[torch.Tensor] = []
    for batch in loader:
        ctx = batch["ctx_bars"].to(device)
        ctx_len = batch["ctx_lengths"].to(device)
        future = batch["future_bars"].to(device)
        pred, aux = model(ctx, ctx_len, return_aux=True)
        if use_ic_loss:
            pred_loss, _ = stage3_prediction_loss(
                pred, future,
                direction_logit=aux.get("cum_direction_logit"),
                mse_weight=mse_weight, step_corr_weight=step_corr_weight,
                cum_corr_weight=cum_corr_weight, sign_weight=sign_weight,
                rank_weight=rank_weight, direction_weight=direction_weight,
                corr_weight=corr_weight,
            )
        else:
            pred_loss = future_prediction_loss(pred, future)
        loss = pred_loss + aux_vq_weight * aux["vq_loss"] + aux_break_weight * aux["break_reg_loss"]
        bs = ctx.size(0)
        total += loss.item() * bs
        n += bs
        preds.append(pred.detach().cpu())
        targets.append(future[..., :1].detach().cpu())
        if "cum_direction_logit" in aux:
            direction_logits.append(aux["cum_direction_logit"].detach().cpu())
    out = {"loss": total / max(1, n)}
    if preds:
        p = torch.cat(preds, dim=0).numpy()
        y = torch.cat(targets, dim=0).numpy()
        if p.ndim == 3:
            p = p[..., 0]
            y = y[..., 0]
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
    corr_weight: float | None = None,
    use_ic_loss: bool = True,
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
        if use_ic_loss:
            pred_loss, extras = stage3_prediction_loss(
                pred, future,
                direction_logit=aux.get("cum_direction_logit"),
                mse_weight=mse_weight, step_corr_weight=step_corr_weight,
                cum_corr_weight=cum_corr_weight, sign_weight=sign_weight,
                rank_weight=rank_weight, direction_weight=direction_weight,
                corr_weight=corr_weight,
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
