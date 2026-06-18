#!/usr/bin/env python3
"""自动切分 MHA + VQ 模型：训练、验证与绘图报告。

  python examples/plot_auto_segment_report.py --synthetic
  python examples/plot_auto_segment_report.py --skip-train --checkpoint-dir checkpoints/auto_seg
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from torch.utils.data import DataLoader

_EX = Path(__file__).resolve().parent
_ROOT = _EX.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))

from _train_common import add_break_vol_args, add_data_args, add_feature_args, add_segment_args, add_stage3_loss_args, add_train_args, add_vq_args, apply_real_data_defaults, fetch_ohlcv_df, prepare_bar_series_from_args
from transformer_kit.auto_segment_encoder import AutoSegmentConfig, AutoSegmentVQVAE
from transformer_kit.causal_transformer import CausalTransformerConfig
from transformer_kit.pattern_encoder import pattern_config_from_args
from transformer_kit.pattern_model import KlinePatternPredictor, PatternPredictorConfig
from transformer_kit.schedulers import build_adamw_with_warmup_cosine_restarts
from transformer_kit.segment_dataset import (
    BarWindowDataset,
    PatternSequenceDataset,
    build_sequence_sample_indices,
)
from transformer_kit.train_utils import load_checkpoint, save_checkpoint
from transformer_kit.magnitude_metrics import (
    denorm_zscore_log_ret,
    fit_cumulative_magnitude_scale,
    magnitude_accuracy_metrics,
)
from transformer_kit.vector_quantizer import code_usage_stats
from transformer_kit.training import (
    evaluate_auto_vqvae,
    evaluate_stage3,
    init_vq_codebook_from_loader,
    train_auto_vqvae_epoch,
    train_stage3_epoch,
)


@dataclass
class TrainHistory:
    stage1: dict[str, list[float]] = field(
        default_factory=lambda: {"train_loss": [], "valid_loss": [], "perplexity": [], "avg_segments": []}
    )
    stage2: dict[str, list[float]] = field(
        default_factory=lambda: {"train_loss": [], "valid_loss": [], "perplexity": [], "recon": []}
    )
    stage3: dict[str, list[float]] = field(
        default_factory=lambda: {"train_loss": [], "valid_loss": [], "ic": []}
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-segment model report")
    add_data_args(p)
    add_feature_args(p)
    add_train_args(p)
    add_segment_args(p)
    add_vq_args(p)
    add_stage3_loss_args(p)
    add_break_vol_args(p)
    p.add_argument("--epochs1", type=int, default=20)
    p.add_argument("--epochs2", type=int, default=15)
    p.add_argument("--epochs3", type=int, default=20)
    p.add_argument("--pred-horizon", type=int, default=5)
    p.add_argument("--pred-feat-dim", type=int, default=1, help="预测未来特征维度；4=log_ret+实体/影线")
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--trunk-layers", type=int, default=2)
    p.add_argument("--aux-vq-weight", type=float, default=0.1)
    p.add_argument("--aux-break-weight", type=float, default=0.05)
    p.add_argument("--encoder-lr-scale", type=float, default=0.1)
    p.add_argument("--output-dir", default="reports/auto_segment")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-stage12", action="store_true", help="跳过 Stage1/2，从 checkpoint 加载后只训 Stage3")
    p.add_argument("--init-checkpoint-dir", default="", help="Stage1/2 初始权重目录（默认=checkpoint-dir）")
    p.add_argument("--dpi", type=int, default=140)
    return p.parse_args()


def cumulative_path(log_ret: np.ndarray) -> np.ndarray:
    return np.exp(np.cumsum(log_ret, axis=-1))


def _vq_train_kwargs(args: argparse.Namespace) -> dict:
    return {
        "diversity_weight": getattr(args, "diversity_weight", 0.25),
        "usage_balance_weight": getattr(args, "usage_balance_weight", 0.35),
        "z_spread_weight": getattr(args, "z_spread_weight", 0.15),
        "break_aware_vq_balance": getattr(args, "break_aware_vq_balance", True),
        "break_seg_vq_weight": getattr(args, "break_seg_vq_weight", 2.0),
        "background_seg_vq_weight": getattr(args, "background_seg_vq_weight", 0.35),
        "vq_dead_threshold": getattr(args, "vq_dead_threshold", 0.1),
        "vq_max_code_frac": getattr(args, "vq_max_code_frac", 0.15),
        "vq_kmeans_frac": getattr(args, "vq_kmeans_frac", 0.45),
    }


def _stage3_train_kwargs(args: argparse.Namespace) -> dict:
    corr_kw = {} if args.corr_weight <= 0 else {"corr_weight": args.corr_weight}
    return dict(
        aux_vq_weight=args.aux_vq_weight,
        aux_break_weight=args.aux_break_weight,
        mse_weight=args.mse_weight,
        step_corr_weight=args.step_corr_weight,
        cum_corr_weight=args.cum_corr_weight,
        sign_weight=args.sign_weight,
        rank_weight=args.rank_weight,
        direction_weight=args.direction_weight,
        shape_weight=args.shape_weight,
        path_shape_weight=args.path_shape_weight,
        cum_magnitude_weight=args.cum_magnitude_weight,
        relative_magnitude_weight=args.relative_magnitude_weight,
        raw_mse_weight=args.raw_mse_weight,
        vol_focus_weight=args.vol_focus_weight,
        vol_focus_top_frac=args.vol_focus_top_frac,
        move_focus_weight=getattr(args, "move_focus_weight", 0.0),
        move_focus_scale=getattr(args, "move_focus_scale", 3.0),
        break_focus_weight=getattr(args, "break_focus_weight", 0.0),
        break_focus_tail=getattr(args, "break_focus_tail", 16),
        code_supervision_weight=getattr(args, "code_supervision_weight", 0.0),
        anti_lag_weight=args.anti_lag_weight,
        anti_lag_margin=args.anti_lag_margin,
        use_ic_loss=not args.no_ic_loss,
        **corr_kw,
    )


def run_stage1(args, bundle, device, history, ckpt_dir) -> AutoSegmentVQVAE:
    cfg = pattern_config_from_args(args)
    model = AutoSegmentVQVAE(cfg).to(device)
    train_loader = DataLoader(
        BarWindowDataset(bundle.bars, bundle.train_idx, window=args.context_bars,
                         samples_per_epoch=args.samples_per_epoch, seed=args.seed),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    valid_loader = DataLoader(
        BarWindowDataset(bundle.bars, bundle.valid_idx, window=args.context_bars,
                         samples_per_epoch=300, seed=args.seed + 1),
        batch_size=args.batch_size, shuffle=False,
    )
    opt, sched = build_adamw_with_warmup_cosine_restarts(
        list(model.parameters()), lr=args.lr, weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps, t0=args.cosine_t0, t_mult=args.cosine_t_mult, eta_min=args.eta_min,
    )
    init_vq_codebook_from_loader(model, train_loader, device)
    best = float("inf")
    vq_kw = _vq_train_kwargs(args)
    for ep in range(args.epochs1):
        tr = train_auto_vqvae_epoch(
            model, train_loader, opt, sched, device,
            grad_clip=args.grad_clip, **vq_kw,
        )
        va = evaluate_auto_vqvae(model, valid_loader, device)
        history.stage1["train_loss"].append(tr.loss)
        history.stage1["valid_loss"].append(va["loss"])
        history.stage1["perplexity"].append(va["perplexity"])
        history.stage1["avg_segments"].append(tr.extras.get("avg_segments", 0))
        if (ep + 1) % max(1, args.epochs1 // 5) == 0 or ep == 0:
            print(
                f"  s1 ep {ep + 1}: valid={va['loss']:.4f} ppl={va['perplexity']:.2f} "
                f"vq_reset dead={tr.extras.get('vq_reset_dead', 0):.0f} "
                f"dom={tr.extras.get('vq_reset_dominant', 0):.0f} "
                f"max_frac={tr.extras.get('vq_max_epoch_frac', 0):.1%}"
            )
        if va["loss"] < best:
            best = va["loss"]
            save_checkpoint(ckpt_dir / "stage1_auto_segment_vqvae.pt", {"model": model.state_dict(), "config": cfg.__dict__})
    ckpt = load_checkpoint(ckpt_dir / "stage1_auto_segment_vqvae.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    return model


def run_stage2(args, bundle, device, history, ckpt_dir, model: AutoSegmentVQVAE) -> AutoSegmentVQVAE:
    train_loader = DataLoader(
        BarWindowDataset(bundle.bars, bundle.train_idx, window=args.context_bars,
                         samples_per_epoch=args.samples_per_epoch, seed=args.seed),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    valid_loader = DataLoader(
        BarWindowDataset(bundle.bars, bundle.valid_idx, window=args.context_bars,
                         samples_per_epoch=300, seed=args.seed + 1),
        batch_size=args.batch_size, shuffle=False,
    )
    opt, sched = build_adamw_with_warmup_cosine_restarts(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr * 0.5, weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps, t0=args.cosine_t0, t_mult=args.cosine_t_mult, eta_min=args.eta_min,
    )
    best = float("inf")
    vq_kw = _vq_train_kwargs(args)
    for ep in range(args.epochs2):
        tr = train_auto_vqvae_epoch(
            model, train_loader, opt, sched, device,
            grad_clip=args.grad_clip, **vq_kw,
        )
        va = evaluate_auto_vqvae(model, valid_loader, device)
        history.stage2["train_loss"].append(tr.loss)
        history.stage2["valid_loss"].append(va["loss"])
        history.stage2["perplexity"].append(va["perplexity"])
        history.stage2["recon"].append(va["recon"])
        if (ep + 1) % max(1, args.epochs2 // 4) == 0 or ep == 0:
            print(
                f"  s2 ep {ep + 1}: valid={va['loss']:.4f} ppl={va['perplexity']:.2f} "
                f"vq_reset dead={tr.extras.get('vq_reset_dead', 0):.0f} "
                f"dom={tr.extras.get('vq_reset_dominant', 0):.0f} "
                f"max_frac={tr.extras.get('vq_max_epoch_frac', 0):.1%}"
            )
        if va["loss"] < best:
            best = va["loss"]
            save_checkpoint(ckpt_dir / "stage2_vqvae.pt", {"model": model.state_dict()})
    ckpt = load_checkpoint(ckpt_dir / "stage2_vqvae.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    return model


def run_stage3(args, bundle, device, history, ckpt_dir, vqvae: AutoSegmentVQVAE) -> KlinePatternPredictor:
    def split(idx):
        return build_sequence_sample_indices(
            bundle.bars.shape[0], context_bars=args.context_bars, pred_horizon=args.pred_horizon,
            stride=args.stride, index_min=int(idx.min()), index_max=int(idx.max()),
        )

    train_loader = DataLoader(PatternSequenceDataset(bundle.bars, split(bundle.train_idx), bundle.raw_log_ret, zscore_window=bundle.zscore_window),
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(PatternSequenceDataset(bundle.bars, split(bundle.valid_idx), bundle.raw_log_ret, zscore_window=bundle.zscore_window),
                              batch_size=args.batch_size, shuffle=False)

    auto_cfg = pattern_config_from_args(args)
    model = KlinePatternPredictor(PatternPredictorConfig(
        auto_segment=auto_cfg,
        trunk=CausalTransformerConfig(d_model=args.d_model, n_heads=args.n_heads, n_layers=args.trunk_layers),
        pred_horizon=args.pred_horizon, pred_feat_dim=args.pred_feat_dim,
        pool_mode=args.pool_mode,
        learnable_scale=not args.no_learnable_scale,
        use_horizon_head=args.horizon_head,
    )).to(device)
    model.auto_encoder.load_state_dict(vqvae.auto_encoder.state_dict())

    enc_p = list(model.auto_encoder.parameters())
    enc_ids = {id(p) for p in enc_p}
    opt, sched = build_adamw_with_warmup_cosine_restarts(
        [{"params": enc_p, "lr": args.lr * args.encoder_lr_scale},
         {"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": args.lr}],
        lr=args.lr, weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps, t0=args.cosine_t0, t_mult=args.cosine_t_mult, eta_min=args.eta_min,
    )
    best = float("inf")
    best_ic = float("-inf")
    best_cum_ic = float("-inf")
    best_combo = float("-inf")
    best_dir = float("-inf")
    best_mag = float("-inf")
    best_balanced = float("-inf")
    magnitude_focus = args.cum_magnitude_weight > 0 or args.relative_magnitude_weight > 0
    s3kw = _stage3_train_kwargs(args)
    for _ in range(args.epochs3):
        tr = train_stage3_epoch(model, train_loader, opt, sched, device, grad_clip=args.grad_clip, **s3kw)
        va = evaluate_stage3(model, valid_loader, device, **s3kw)
        history.stage3["train_loss"].append(tr.loss)
        history.stage3["valid_loss"].append(va["loss"])
        history.stage3["ic"].append(va.get("ic", 0.0))
        if va["loss"] < best:
            best = va["loss"]
            save_checkpoint(ckpt_dir / "stage3_predictor.pt", {"model": model.state_dict()})
        ic = va.get("ic", float("-inf"))
        if ic > best_ic:
            best_ic = ic
            save_checkpoint(ckpt_dir / "stage3_predictor_best_ic.pt", {"model": model.state_dict()})
        cum_ic = va.get("cum_ic", float("-inf"))
        if cum_ic > best_cum_ic:
            best_cum_ic = cum_ic
            save_checkpoint(ckpt_dir / "stage3_predictor_best_cum_ic.pt", {"model": model.state_dict()})
        combo = va.get("ic", 0.0) + va.get("cum_ic", 0.0)
        if combo > best_combo:
            best_combo = combo
            save_checkpoint(ckpt_dir / "stage3_predictor_best_combo.pt", {"model": model.state_dict()})
        direction_acc = va.get("direction_head_acc", float("-inf"))
        if direction_acc > best_dir:
            best_dir = direction_acc
            save_checkpoint(ckpt_dir / "stage3_predictor_best_direction.pt", {"model": model.state_dict()})
        mag_rate = va.get("magnitude_within_tol_rate", float("-inf"))
        if mag_rate > best_mag:
            best_mag = mag_rate
            save_checkpoint(ckpt_dir / "stage3_predictor_best_magnitude.pt", {"model": model.state_dict()})
        balanced = (
            0.40 * va.get("cum_direction_acc", 0.0)
            + 0.30 * va.get("cum_ic", 0.0)
            + 0.20 * va.get("direction_head_acc", 0.0)
            + 0.10 * va.get("magnitude_within_tol_rate", 0.0)
        )
        if balanced > best_balanced:
            best_balanced = balanced
            save_checkpoint(ckpt_dir / "stage3_predictor_best_balanced.pt", {"model": model.state_dict()})
    if magnitude_focus:
        ic_ckpt = ckpt_dir / "stage3_predictor_best_balanced.pt"
    else:
        ic_ckpt = ckpt_dir / "stage3_predictor_best_combo.pt"
    if not ic_ckpt.is_file():
        ic_ckpt = ckpt_dir / "stage3_predictor_best_cum_ic.pt"
    ckpt_path = ic_ckpt if ic_ckpt.is_file() else ckpt_dir / "stage3_predictor.pt"
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return model


def plot_training_curves(history: TrainHistory, out: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax = axes[0]
    h = history.stage1
    ep = range(1, len(h["train_loss"]) + 1)
    ax.plot(ep, h["train_loss"], "o-", label="Train", ms=3)
    ax.plot(ep, h["valid_loss"], "s-", label="Valid", ms=3)
    ax2 = ax.twinx()
    ax2.plot(ep, h["avg_segments"], "^--", color="green", alpha=0.7, label="Avg segments", ms=3)
    ax.set_title("Stage 1: Auto-segment + VQ-VAE")
    ax.set_xlabel("Epoch")
    ax.legend(loc="upper right", fontsize=8)
    ax2.legend(loc="center right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    h = history.stage2
    ep = range(1, len(h["train_loss"]) + 1)
    ax.plot(ep, h["recon"], "o-", label="Recon", ms=3)
    ax2 = ax.twinx()
    ax2.plot(ep, h["perplexity"], "^--", color="green", label="Perplexity", ms=3)
    ax.set_title("Stage 2: VQ fine-tune")
    ax.set_xlabel("Epoch")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    h = history.stage3
    ep = range(1, len(h["train_loss"]) + 1)
    ax.plot(ep, h["train_loss"], "o-", label="Train", ms=3)
    ax.plot(ep, h["valid_loss"], "s-", label="Valid", ms=3)
    if h["ic"]:
        ax2 = ax.twinx()
        ax2.plot(ep, h["ic"], "^--", color="purple", label="IC", ms=3)
        ax2.set_ylabel("IC")
        ax2.legend(loc="upper right", fontsize=8)
    ax.set_title("Stage 3: Future log_ret prediction")
    ax.set_xlabel("Epoch")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Auto-segment MHA training curves", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "01_training_curves.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def plot_auto_segmentation(
    model: AutoSegmentVQVAE,
    bundle,
    device: torch.device,
    out: Path,
    dpi: int,
    n_samples: int = 3,
) -> dict[str, float]:
    """可视化：MHA 自动切分位置 + 各段 VQ code。"""
    model.eval()
    ds = BarWindowDataset(bundle.bars, bundle.test_idx, window=128, samples_per_epoch=n_samples, seed=99)
    fig, axes = plt.subplots(n_samples, 2, figsize=(14, 3.2 * n_samples))
    if n_samples == 1:
        axes = np.array([axes])

    avg_segs: list[float] = []
    for i in range(n_samples):
        batch = ds[i]
        bars = batch["ctx_bars"].unsqueeze(0).to(device)
        lengths = batch["ctx_lengths"].unsqueeze(0).to(device)
        vq_out = model(bars, lengths)
        ao = vq_out.auto_out
        ln = int(lengths[0].item())
        log_ret = bars[0, :ln, 0].cpu().numpy()
        path = cumulative_path(log_ret)
        breaks = ao.break_hard[0, :ln].cpu().numpy()
        codes = ao.codes[0].cpu().numpy()
        n_seg = int(ao.num_segments[0].item())
        avg_segs.append(n_seg)

        ax = axes[i, 0]
        cuts = np.where(breaks)[0]
        bounds = [0] + [c + 1 for c in cuts] + [ln]
        cmap = plt.cm.tab10
        for j in range(len(bounds) - 1):
            a, b = bounds[j], bounds[j + 1]
            ax.plot(range(a, b), path[a:b], color=cmap(j % 10), linewidth=2, label=f"seg{j} code={int(codes[j])}")
            if b < ln:
                ax.axvline(b - 0.5, color="red", linestyle="--", alpha=0.8, linewidth=1.5)
        ax.set_title(f"Sample {i+1}: auto segmentation ({n_seg} segments, red dashed = break)")
        ax.set_xlabel("Bar index")
        ax.set_ylabel("Norm. price path")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

        ax = axes[i, 1]
        probs = torch.sigmoid(ao.break_logits[0, :ln]).cpu().numpy()
        ax.bar(range(ln), probs, color="steelblue", alpha=0.7, width=1.0)
        for c in cuts:
            ax.axvline(c + 0.5, color="red", linestyle="--", linewidth=1.5)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Sample {i+1}: break probability P(cut after bar t)")
        ax.set_xlabel("Bar index")
        ax.set_ylabel("P(break)")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Stage 1 validation: MHA auto-segmentation + VQ codes", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out / "02_auto_segmentation.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {"avg_test_segments": float(np.mean(avg_segs))}


@torch.no_grad()
def plot_vq_usage(model: AutoSegmentVQVAE, bundle, device, out: Path, dpi: int) -> dict[str, float]:
    model.eval()
    loader = DataLoader(
        BarWindowDataset(bundle.bars, bundle.test_idx, window=128, samples_per_epoch=400, seed=7),
        batch_size=32, shuffle=False,
    )
    all_codes: list[int] = []
    for batch in loader:
        bars = batch["ctx_bars"].to(device)
        lengths = batch["ctx_lengths"].to(device)
        ao = model(bars, lengths).auto_out
        b, s = ao.codes.shape
        for i in range(b):
            n = int(ao.num_segments[i].item())
            all_codes.extend(ao.codes[i, :n].tolist())
    codes = np.array(all_codes)
    unique, counts = np.unique(codes, return_counts=True)
    nc = model.cfg.num_codes
    stats = code_usage_stats(torch.from_numpy(codes), nc)
    usage = stats["active_ratio"]
    norm_ent = stats["norm_entropy"]
    max_frac = stats["max_code_frac"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(unique, counts, color="steelblue", edgecolor="white")
    ax.set_title(
        f"VQ code usage ({len(unique)}/{nc} active, norm_entropy={norm_ent:.3f}, "
        f"max_frac={max_frac:.1%})"
    )
    ax.set_xlabel("Pattern code ID")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "03_vq_usage.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {
        "code_usage_ratio": float(usage),
        "code_entropy_norm": float(norm_ent),
        "max_code_frac": float(max_frac),
    }


@torch.no_grad()
def calibrate_cum_direction_threshold(
    model: KlinePatternPredictor,
    bundle,
    args,
    device: torch.device,
    *,
    pred_scale: np.ndarray | None = None,
    pred_bias: np.ndarray | None = None,
) -> tuple[float, float]:
    """在训练+验证历史窗口上选择累计收益方向阈值。

    如果启用 ``vol_focus_weight``，阈值只在高波动样本子集上校准。
    """
    model.eval()
    idx = np.concatenate([bundle.train_idx, bundle.valid_idx])
    samples = build_sequence_sample_indices(
        bundle.bars.shape[0], context_bars=args.context_bars, pred_horizon=args.pred_horizon,
        stride=args.stride, index_min=int(idx.min()), index_max=int(idx.max()),
    )
    loader = DataLoader(PatternSequenceDataset(bundle.bars, samples, bundle.raw_log_ret, zscore_window=bundle.zscore_window), batch_size=64, shuffle=False)
    scores, labels, vol_scores = [], [], []
    for batch in loader:
        ctx = batch["ctx_bars"].to(device)
        ctx_len = batch["ctx_lengths"].to(device)
        future = batch["future_bars"].to(device)
        pred = model(ctx, ctx_len)
        if pred.dim() == 3:
            pred = pred[..., 0]
        pred_np = pred.cpu().numpy()
        if pred_scale is not None and pred_bias is not None:
            pred_np = pred_np * pred_scale[None, :] + pred_bias[None, :]
        future_ret = future[..., 0]
        scores.append(pred_np.sum(axis=1))
        labels.append((future_ret.sum(dim=1).cpu().numpy() > 0).astype(np.float32))
        vol_scores.append(future_ret.abs().mean(dim=1).cpu().numpy())
    score = np.concatenate(scores)
    label = np.concatenate(labels)
    vol_score = np.concatenate(vol_scores)
    mask = np.ones_like(label, dtype=bool)
    if getattr(args, "vol_focus_weight", 0.0) > 0:
        frac = float(getattr(args, "vol_focus_top_frac", 0.3))
        threshold = np.quantile(vol_score, max(0.0, min(1.0, 1.0 - frac)))
        mask = vol_score >= threshold
    best_threshold = 0.0
    best_acc = float(((score[mask] > 0.0).astype(np.float32) == label[mask]).mean())
    for threshold in np.unique(np.quantile(score[mask], np.linspace(0.0, 1.0, 501))):
        acc = float(((score[mask] > threshold).astype(np.float32) == label[mask]).mean())
        if acc > best_acc:
            best_acc = acc
            best_threshold = float(threshold)
    return best_threshold, best_acc


@torch.no_grad()
def calibrate_prediction_affine(
    model: KlinePatternPredictor,
    bundle,
    args,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """用训练+验证窗口拟合逐 horizon 的 affine 幅度校准。"""
    model.eval()
    idx = np.concatenate([bundle.train_idx, bundle.valid_idx])
    samples = build_sequence_sample_indices(
        bundle.bars.shape[0], context_bars=args.context_bars, pred_horizon=args.pred_horizon,
        stride=args.stride, index_min=int(idx.min()), index_max=int(idx.max()),
    )
    loader = DataLoader(PatternSequenceDataset(bundle.bars, samples, bundle.raw_log_ret, zscore_window=bundle.zscore_window), batch_size=64, shuffle=False)
    preds, targets = [], []
    pred_means, pred_stds = [], []
    for batch in loader:
        ctx = batch["ctx_bars"].to(device)
        ctx_len = batch["ctx_lengths"].to(device)
        future = batch["future_bars"].to(device)
        pred = model(ctx, ctx_len)
        if pred.dim() == 3:
            pred = pred[..., 0]
        p_np = pred.cpu().numpy()
        if "future_raw_log_ret" in batch:
            targets.append(batch["future_raw_log_ret"].numpy())
            pred_means.append(batch["future_log_ret_mean"].numpy())
            pred_stds.append(batch["future_log_ret_std"].numpy())
        else:
            targets.append(future[..., 0].cpu().numpy())
        preds.append(p_np)
    p = np.concatenate(preds, axis=0)
    y = np.concatenate(targets, axis=0)
    if pred_means:
        p = denorm_zscore_log_ret(p, np.concatenate(pred_means, axis=0), np.concatenate(pred_stds, axis=0))
    scale = np.ones(p.shape[1], dtype=np.float32)
    bias = np.zeros(p.shape[1], dtype=np.float32)
    magnitude_focus = getattr(args, "cum_magnitude_weight", 0.0) > 0 or getattr(args, "relative_magnitude_weight", 0.0) > 0
    if not magnitude_focus:
        for t in range(p.shape[1]):
            pt = p[:, t]
            yt = y[:, t]
            var = float(pt.var())
            if var > 1e-10:
                cov = float(((pt - pt.mean()) * (yt - yt.mean())).mean())
                scale[t] = float(np.clip(cov / (var + 1e-10), -5.0, 5.0))
                bias[t] = float(yt.mean() - scale[t] * pt.mean())
    raw_mse = float(((p - y) ** 2).mean())
    cal_mse = float(((p * scale[None, :] + bias[None, :] - y) ** 2).mean())
    if magnitude_focus:
        from transformer_kit.magnitude_metrics import fit_relative_magnitude_scale

        cum_scale = fit_relative_magnitude_scale(p, y)
    else:
        cum_scale = fit_cumulative_magnitude_scale(p, y)
    return scale, bias, raw_mse, cal_mse, cum_scale


@torch.no_grad()
def evaluate_and_plot_prediction(
    model: KlinePatternPredictor,
    bundle,
    args,
    device,
    out: Path,
    dpi: int,
    *,
    cum_direction_threshold: float = 0.0,
    calibration_acc: float = 0.0,
    pred_scale: np.ndarray | None = None,
    pred_bias: np.ndarray | None = None,
    cum_magnitude_scale: float = 1.0,
    magnitude_calibration_mse: float = 0.0,
    magnitude_raw_calibration_mse: float = 0.0,
) -> dict:
    model.eval()
    samples = build_sequence_sample_indices(
        bundle.bars.shape[0], context_bars=args.context_bars, pred_horizon=args.pred_horizon,
        stride=args.stride, index_min=int(bundle.test_idx.min()), index_max=int(bundle.test_idx.max()),
    )
    loader = DataLoader(PatternSequenceDataset(bundle.bars, samples, bundle.raw_log_ret, zscore_window=bundle.zscore_window), batch_size=32, shuffle=False)
    preds, targets, dir_logits, code_sup_logits = [], [], [], []
    preds_z, pred_means, pred_stds = [], [], []
    for batch in loader:
        ctx = batch["ctx_bars"].to(device)
        ctx_len = batch["ctx_lengths"].to(device)
        future = batch["future_bars"].to(device)
        pred, aux = model(ctx, ctx_len, return_aux=True)
        p_np = pred.cpu().numpy()
        if p_np.ndim == 3:
            p_np = p_np[..., 0]
        preds_z.append(p_np)
        if "future_raw_log_ret" in batch:
            targets.append(batch["future_raw_log_ret"].numpy())
            pred_means.append(batch["future_log_ret_mean"].numpy())
            pred_stds.append(batch["future_log_ret_std"].numpy())
        else:
            targets.append(future[..., 0].cpu().numpy())
        if "cum_direction_logit" in aux:
            dir_logits.append(aux["cum_direction_logit"].cpu().numpy())
        if "code_supervision_logit" in aux:
            code_sup_logits.append(aux["code_supervision_logit"].cpu().numpy())
    p_raw = np.concatenate(preds_z, axis=0)
    y = np.concatenate(targets, axis=0)
    if pred_means:
        p_raw = denorm_zscore_log_ret(
            p_raw, np.concatenate(pred_means, axis=0), np.concatenate(pred_stds, axis=0),
        )
    p = p_raw.copy()
    if pred_scale is not None and pred_bias is not None:
        p = p * pred_scale[None, :] + pred_bias[None, :]
    if cum_magnitude_scale != 1.0:
        p = p * cum_magnitude_scale
    raw_mse = float(((p_raw - y) ** 2).mean())
    raw_mae = float(np.abs(p_raw - y).mean())
    mse = float(((p - y) ** 2).mean())
    mae = float(np.abs(p - y).mean())
    ic = float(np.corrcoef(p.ravel(), y.ravel())[0, 1]) if p.std() > 1e-8 and y.std() > 1e-8 else 0.0
    cum_ic = float(np.corrcoef(p.sum(axis=1), y.sum(axis=1))[0, 1]) if p.shape[0] > 2 else 0.0
    dir_acc = float((np.sign(p.ravel()) == np.sign(y.ravel())).mean())
    cum_dir_acc = float((np.sign(p.sum(axis=1)) == np.sign(y.sum(axis=1))).mean())
    calibrated_cum_dir_acc = float(((p.sum(axis=1) > cum_direction_threshold) == (y.sum(axis=1) > 0)).mean())
    vol_score = np.abs(y).mean(axis=1)
    high_vol_threshold = np.quantile(
        vol_score,
        max(0.0, min(1.0, 1.0 - float(getattr(args, "vol_focus_top_frac", 0.3)))),
    )
    high_vol = vol_score >= high_vol_threshold
    high_vol_cum_dir_acc = float((np.sign(p.sum(axis=1)[high_vol]) == np.sign(y.sum(axis=1)[high_vol])).mean())
    high_vol_calibrated_cum_dir_acc = float(
        ((p.sum(axis=1)[high_vol] > cum_direction_threshold) == (y.sum(axis=1)[high_vol] > 0)).mean()
    )
    raw_high_vol_mse = float(((p_raw[high_vol] - y[high_vol]) ** 2).mean())
    high_vol_mse = float(((p[high_vol] - y[high_vol]) ** 2).mean())
    raw_high_vol_mae = float(np.abs(p_raw[high_vol] - y[high_vol]).mean())
    high_vol_mae = float(np.abs(p[high_vol] - y[high_vol]).mean())
    head_dir_acc = 0.0
    high_vol_head_dir_acc = 0.0
    code_sup_head_acc = 0.0
    high_vol_code_sup_head_acc = 0.0
    if dir_logits:
        dlog = np.concatenate(dir_logits, axis=0)
        labels = (y.sum(axis=1) > 0).astype(np.float32)
        head_dir_acc = float(((dlog > 0).astype(np.float32) == labels).mean())
        high_vol_head_dir_acc = float(((dlog[high_vol] > 0).astype(np.float32) == labels[high_vol]).mean())
    if code_sup_logits:
        slog = np.concatenate(code_sup_logits, axis=0)
        labels = (y.sum(axis=1) > 0).astype(np.float32)
        code_sup_head_acc = float(((slog > 0).astype(np.float32) == labels).mean())
        high_vol_code_sup_head_acc = float(((slog[high_vol] > 0).astype(np.float32) == labels[high_vol]).mean())

    tol = float(getattr(args, "magnitude_tolerance", 0.2))
    min_move = float(getattr(args, "magnitude_min_move", 1e-4))
    mag_raw = magnitude_accuracy_metrics(p_raw, y, tolerance=tol, min_move=min_move)
    mag_cal = magnitude_accuracy_metrics(p, y, tolerance=tol, min_move=min_move)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    h = p.shape[1]
    rng = np.random.default_rng(0)
    idx = rng.choice(p.shape[0], size=min(4, p.shape[0]), replace=False)

    ax = axes[0, 0]
    for si in idx:
        ax.plot(range(h), cumulative_path(y[si]), alpha=0.6, linewidth=1.5)
        ax.plot(range(h), cumulative_path(p[si]), "--", alpha=0.6, linewidth=1.5)
    ax.plot([], [], "b-", label="Actual")
    ax.plot([], [], "r--", label="Predicted")
    ax.set_title(f"Future {h}-bar path ({len(idx)} samples)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(y.ravel(), p.ravel(), alpha=0.3, s=10, c="steelblue")
    lim = max(abs(y).max(), abs(p).max(), 1e-3) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title(f"log_ret scatter | IC={ic:.3f} step_dir={dir_acc:.1%}")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    step_mse = [float(((p[:, t] - y[:, t]) ** 2).mean()) for t in range(h)]
    ax.bar(range(h), step_mse, color="coral")
    ax.set_title("Per-step test MSE")
    ax.set_xlabel("Future bar step")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 1]
    si = idx[0]
    w = 0.35
    x = np.arange(h)
    ax.bar(x - w / 2, y[si], width=w, label="Actual", alpha=0.85)
    ax.bar(x + w / 2, p[si], width=w, label="Predicted", alpha=0.85)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Single sample step-wise log_ret")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Stage 3: prediction | MSE={mse:.5f} mag@{tol:.0%}={mag_cal['magnitude_within_tol_rate']:.1%} "
        f"cum_dir={cum_dir_acc:.1%}",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out / "04_prediction.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {
        "test_mse": mse,
        "test_raw_mse": raw_mse,
        "test_mae": mae,
        "test_raw_mae": raw_mae,
        "test_mse_improvement": raw_mse - mse,
        "test_mae_improvement": raw_mae - mae,
        "test_high_vol_mse": high_vol_mse,
        "test_high_vol_raw_mse": raw_high_vol_mse,
        "test_high_vol_mae": high_vol_mae,
        "test_high_vol_raw_mae": raw_high_vol_mae,
        "test_high_vol_mse_improvement": raw_high_vol_mse - high_vol_mse,
        "test_high_vol_mae_improvement": raw_high_vol_mae - high_vol_mae,
        "magnitude_calibration_mse": magnitude_calibration_mse,
        "magnitude_raw_calibration_mse": magnitude_raw_calibration_mse,
        "magnitude_scale_mean": float(np.mean(pred_scale)) if pred_scale is not None else 1.0,
        "magnitude_bias_mean": float(np.mean(pred_bias)) if pred_bias is not None else 0.0,
        "cum_magnitude_scale": float(cum_magnitude_scale),
        "test_ic": ic,
        "test_cum_ic": cum_ic,
        "test_direction_acc": dir_acc,
        "test_cum_direction_acc": cum_dir_acc,
        "test_calibrated_cum_direction_acc": calibrated_cum_dir_acc,
        "test_high_vol_cum_direction_acc": high_vol_cum_dir_acc,
        "test_high_vol_calibrated_cum_direction_acc": high_vol_calibrated_cum_dir_acc,
        "test_high_vol_direction_head_acc": high_vol_head_dir_acc,
        "test_high_vol_code_supervision_head_acc": high_vol_code_sup_head_acc,
        "test_high_vol_frac": float(high_vol.mean()),
        "test_high_vol_threshold": float(high_vol_threshold),
        "calibration_cum_direction_threshold": float(cum_direction_threshold),
        "calibration_cum_direction_acc": float(calibration_acc),
        "test_direction_head_acc": head_dir_acc,
        "test_code_supervision_head_acc": code_sup_head_acc,
        **{f"test_raw_{k}": v for k, v in mag_raw.items()},
        **{f"test_{k}": v for k, v in mag_cal.items()},
    }


def plot_summary(metrics: dict, history: TrainHistory, out: Path, dpi: int) -> None:
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    lines = [
        "=== Auto-Segment Model Summary ===",
        "",
        f"Avg test segments:     {metrics.get('avg_test_segments', 0):.1f}",
        f"VQ code usage:         {metrics.get('code_usage_ratio', 0):.1%}",
        f"VQ norm entropy:       {metrics.get('code_entropy_norm', 0):.3f}",
        f"VQ max code share:     {metrics.get('max_code_frac', 0):.1%}",
        "",
        f"Test MSE (cal/raw):    {metrics.get('test_mse', float('nan')):.5f} / {metrics.get('test_raw_mse', float('nan')):.5f}",
        f"Test MAE (cal/raw):    {metrics.get('test_mae', float('nan')):.5f} / {metrics.get('test_raw_mae', float('nan')):.5f}",
        f"High-vol MSE cal/raw:  {metrics.get('test_high_vol_mse', float('nan')):.5f} / {metrics.get('test_high_vol_raw_mse', float('nan')):.5f}",
        f"Test IC (step):      {metrics.get('test_ic', 0):.3f}",
        f"Test IC (cum):       {metrics.get('test_cum_ic', 0):.3f}",
        f"Test direction acc:    {metrics.get('test_direction_acc', 0):.1%}",
        f"Test cum dir acc:      {metrics.get('test_cum_direction_acc', 0):.1%}",
        f"Mag within {metrics.get('magnitude_tolerance', 0.2):.0%}:   {metrics.get('test_magnitude_within_tol_rate', 0):.1%}",
        f"Mag mean rel err:      {metrics.get('test_magnitude_mean_rel_err', 0):.1%}",
        f"Mag median rel err:    {metrics.get('test_magnitude_median_rel_err', 0):.1%}",
        f"Test cal cum dir:      {metrics.get('test_calibrated_cum_direction_acc', 0):.1%}",
        f"High-vol cal dir:      {metrics.get('test_high_vol_calibrated_cum_direction_acc', 0):.1%}",
        f"Test dir head acc:     {metrics.get('test_direction_head_acc', 0):.1%}",
        f"Code-sup head acc:     {metrics.get('test_code_supervision_head_acc', 0):.1%}",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), va="top", fontsize=11, family="monospace")

    ax = fig.add_subplot(gs[0, 1])
    labels = ["S1 valid", "S2 valid", "S3 valid"]
    vals = [
        history.stage1["valid_loss"][-1] if history.stage1["valid_loss"] else 0,
        history.stage2["valid_loss"][-1] if history.stage2["valid_loss"] else 0,
        history.stage3["valid_loss"][-1] if history.stage3["valid_loss"] else 0,
    ]
    ax.bar(labels, vals, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_title("Final validation loss")
    ax.grid(True, axis="y", alpha=0.3)

    ax = fig.add_subplot(gs[1, :])
    if history.stage1["avg_segments"]:
        ax.plot(history.stage1["avg_segments"], label="Avg segments (S1)", linewidth=2)
    if history.stage2["perplexity"]:
        ax.plot(history.stage2["perplexity"], label="VQ perplexity (S2)", linewidth=2)
    if history.stage3["ic"]:
        ax.plot(history.stage3["ic"], label="Valid IC (S3)", linewidth=2)
    ax.set_title("Segment count / VQ diversity / prediction IC over epochs")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Auto-segment MHA embedding — validation report", fontsize=14)
    fig.savefig(out / "05_summary.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    apply_real_data_defaults(args)

    out_dir = Path(args.output_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print("[1/6] load data")
    bundle = prepare_bar_series_from_args(fetch_ohlcv_df(args), args)
    print(f"  bars={bundle.bars.shape[0]} feat_dim={bundle.bars.shape[1]} trend={args.trend_features}")

    history = TrainHistory()
    cfg = pattern_config_from_args(args)

    if args.skip_train:
        vqvae = AutoSegmentVQVAE(cfg).to(device)
        pred = KlinePatternPredictor(PatternPredictorConfig(
            auto_segment=cfg,
            trunk=CausalTransformerConfig(d_model=args.d_model, n_heads=args.n_heads, n_layers=args.trunk_layers),
            pred_horizon=args.pred_horizon, pred_feat_dim=args.pred_feat_dim,
            pool_mode=args.pool_mode,
            learnable_scale=not args.no_learnable_scale,
            use_horizon_head=args.horizon_head,
        )).to(device)
        for name, mod in [("stage1_auto_segment_vqvae.pt", vqvae), ("stage2_vqvae.pt", vqvae), ("stage3_predictor.pt", pred)]:
            p = ckpt_dir / name
            if not p.is_file():
                raise FileNotFoundError(p)
        vqvae.load_state_dict(
            load_checkpoint(ckpt_dir / "stage2_vqvae.pt", map_location=device)["model"],
            strict=False,
        )
        pred.load_state_dict(
            load_checkpoint(ckpt_dir / "stage3_predictor.pt", map_location=device)["model"],
            strict=False,
        )
    else:
        init_dir = Path(args.init_checkpoint_dir) if args.init_checkpoint_dir else ckpt_dir
        if args.skip_stage12:
            print("[2/6] skip Stage1/2, load checkpoints")
            vqvae = AutoSegmentVQVAE(cfg).to(device)
            s2 = init_dir / "stage2_vqvae.pt"
            if not s2.is_file():
                s2 = init_dir / "stage1_auto_segment_vqvae.pt"
            if not s2.is_file():
                raise FileNotFoundError(f"need stage2 in {init_dir}")
            missing, unexpected = vqvae.load_state_dict(
                load_checkpoint(s2, map_location=device)["model"], strict=False
            )
            if missing or unexpected:
                print(f"  checkpoint compatibility: missing={len(missing)} unexpected={len(unexpected)}")
            print(f"  loaded {s2}")
            print("[3/6] skip Stage 2")
            print("[4/6] Stage 3 train (IC-oriented loss)")
            pred = run_stage3(args, bundle, device, history, ckpt_dir, vqvae)
        else:
            print("[2/6] Stage 1 train")
            vqvae = run_stage1(args, bundle, device, history, ckpt_dir)
            print("[3/6] Stage 2 train")
            vqvae = run_stage2(args, bundle, device, history, ckpt_dir, vqvae)
            print("[4/6] Stage 3 train")
            pred = run_stage3(args, bundle, device, history, ckpt_dir, vqvae)

    print("[5/6] evaluate & plot")
    metrics: dict = {}
    if history.stage1["train_loss"]:
        plot_training_curves(history, out_dir, args.dpi)
    metrics.update(plot_auto_segmentation(vqvae, bundle, device, out_dir, args.dpi))
    metrics.update(plot_vq_usage(vqvae, bundle, device, out_dir, args.dpi))
    pred_scale, pred_bias, raw_cal_mse, cal_mse, cum_scale = calibrate_prediction_affine(pred, bundle, args, device)
    print(
        f"  calibrated prediction magnitude: train+valid mse {raw_cal_mse:.5f} -> {cal_mse:.5f}, "
        f"scale_mean={pred_scale.mean():.3f} cum_scale={cum_scale:.3f}"
    )
    direction_threshold, direction_cal_acc = calibrate_cum_direction_threshold(
        pred,
        bundle,
        args,
        device,
        pred_scale=pred_scale,
        pred_bias=pred_bias,
    )
    print(
        f"  calibrated cumulative direction threshold={direction_threshold:.5f} "
        f"cal_acc={direction_cal_acc:.1%}"
    )
    metrics.update(
        evaluate_and_plot_prediction(
            pred,
            bundle,
            args,
            device,
            out_dir,
            args.dpi,
            cum_direction_threshold=direction_threshold,
            calibration_acc=direction_cal_acc,
            pred_scale=pred_scale,
            pred_bias=pred_bias,
            cum_magnitude_scale=cum_scale,
            magnitude_raw_calibration_mse=raw_cal_mse,
            magnitude_calibration_mse=cal_mse,
        )
    )
    plot_summary(metrics, history, out_dir, args.dpi)

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "history": asdict(history)}, f, indent=2, ensure_ascii=False)

    print("[6/6] done")
    print(f"  report: {out_dir.resolve()}")
    for p in sorted(out_dir.glob("*.png")):
        print(f"    - {p.name}")
    print("\nMetrics:")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
