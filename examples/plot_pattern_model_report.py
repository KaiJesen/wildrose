#!/usr/bin/env python3
"""三阶段形态模型：训练曲线 + 有效性验证 + 绘图报告。

用法:
  python examples/plot_pattern_model_report.py --synthetic
  python examples/plot_pattern_model_report.py --skip-train   # 仅评估已有检查点
  python examples/plot_pattern_model_report.py --output-dir reports/pattern_model
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

_EX = Path(__file__).resolve().parent
_ROOT = _EX.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))

from _train_common import add_data_args, add_segment_args, add_train_args, add_vq_args, fetch_ohlcv_df
from transformer_kit.causal_transformer import CausalTransformerConfig
from transformer_kit.pattern_encoder import PatternEncoderConfig, PatternVQVAE
from transformer_kit.pattern_model import KlinePatternPredictor, PatternPredictorConfig
from transformer_kit.schedulers import build_adamw_with_warmup_cosine_restarts
from transformer_kit.segment_dataset import (
    PatternSequenceDataset,
    VariableSegmentDataset,
    build_sequence_sample_indices,
    prepare_bar_series,
)
from transformer_kit.segment_encoder import SegmentDecoder, SegmentEncoderConfig, SegmentMHAEncoder
from transformer_kit.train_utils import load_pattern_encoder, load_segment_encoder, save_checkpoint
from transformer_kit.training import (
    evaluate_stage1,
    evaluate_stage2,
    evaluate_stage3,
    init_vq_codebook_from_loader,
    reset_vq_dead_codes,
    train_stage1_epoch,
    train_stage2_epoch,
    train_stage3_epoch,
)

FEAT_NAMES = ["log_ret", "body_ratio", "upper_wick", "lower_wick", "log_vol"]
FEAT_LABELS = ["Log return", "Body ratio", "Upper wick", "Lower wick", "Log volume"]


@dataclass
class TrainHistory:
    stage1: dict[str, list[float]] = field(
        default_factory=lambda: {"train_loss": [], "valid_loss": []}
    )
    stage2: dict[str, list[float]] = field(
        default_factory=lambda: {
            "train_loss": [],
            "valid_loss": [],
            "valid_recon": [],
            "valid_vq": [],
            "valid_perplexity": [],
        }
    )
    stage3: dict[str, list[float]] = field(
        default_factory=lambda: {"train_loss": [], "valid_loss": []}
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pattern model training report with plots")
    add_data_args(p)
    add_train_args(p)
    add_segment_args(p)
    add_vq_args(p)
    p.add_argument("--epochs1", type=int, default=30)
    p.add_argument("--epochs2", type=int, default=40)
    p.add_argument("--epochs3", type=int, default=40)
    p.add_argument("--n-segments", type=int, default=8)
    p.add_argument("--context-bars", type=int, default=128)
    p.add_argument("--pred-horizon", type=int, default=5)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--trunk-layers", type=int, default=2)
    p.add_argument("--aux-vq-weight", type=float, default=0.1)
    p.add_argument("--encoder-lr-scale", type=float, default=0.1)
    p.add_argument("--output-dir", default="reports/pattern_model")
    p.add_argument("--skip-train", action="store_true", help="跳过训练，仅加载检查点评估")
    p.add_argument("--dpi", type=int, default=140)
    return p.parse_args()


def cumulative_path(log_ret: np.ndarray) -> np.ndarray:
    """由 log_ret 构造归一化价格路径（起点=1）。"""
    return np.exp(np.cumsum(log_ret, axis=-1))


def run_stage1(
    args: argparse.Namespace,
    bundle,
    device: torch.device,
    history: TrainHistory,
    ckpt_dir: Path,
) -> tuple[SegmentMHAEncoder, SegmentDecoder]:
    enc_cfg = SegmentEncoderConfig(
        feat_dim=5,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.encoder_layers,
        dim_feedforward=args.d_model * 2,
        max_len=args.max_seg_len,
    )
    encoder = SegmentMHAEncoder(enc_cfg).to(device)
    decoder = SegmentDecoder(enc_cfg).to(device)

    train_ds = VariableSegmentDataset(
        bundle.bars, bundle.train_idx,
        min_seg_len=args.min_seg_len, max_seg_len=args.max_seg_len,
        samples_per_epoch=args.samples_per_epoch, seed=args.seed,
    )
    valid_ds = VariableSegmentDataset(
        bundle.bars, bundle.valid_idx,
        min_seg_len=args.min_seg_len, max_seg_len=args.max_seg_len,
        samples_per_epoch=max(300, args.samples_per_epoch // 5),
        seed=args.seed + 1,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

    opt, sched = build_adamw_with_warmup_cosine_restarts(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps, t0=args.cosine_t0,
        t_mult=args.cosine_t_mult, eta_min=args.eta_min,
    )

    best = float("inf")
    for _ in range(args.epochs1):
        tr = train_stage1_epoch(
            encoder, decoder, train_loader, opt, sched, device,
            grad_clip=args.grad_clip,
            contrastive_weight=args.contrastive_weight,
            augment_noise=args.augment_noise,
        )
        va = evaluate_stage1(encoder, decoder, valid_loader, device)
        history.stage1["train_loss"].append(tr.loss)
        history.stage1["valid_loss"].append(va["loss"])
        if va["loss"] < best:
            best = va["loss"]
            save_checkpoint(
                ckpt_dir / "stage1_segment_encoder.pt",
                {"stage": 1, "segment_encoder": encoder.state_dict(), "decoder": decoder.state_dict()},
            )
    load_segment_encoder(encoder, ckpt_dir / "stage1_segment_encoder.pt")
    return encoder, decoder


def run_stage2(
    args: argparse.Namespace,
    bundle,
    device: torch.device,
    history: TrainHistory,
    ckpt_dir: Path,
) -> PatternVQVAE:
    cfg = PatternEncoderConfig(
        feat_dim=5, d_model=args.d_model, n_heads=args.n_heads,
        encoder_layers=args.encoder_layers, max_seg_len=args.max_seg_len,
        num_codes=args.num_codes, vq_beta=args.vq_beta, vq_use_ema=True,
    )
    model = PatternVQVAE(cfg).to(device)
    s1 = ckpt_dir / "stage1_segment_encoder.pt"
    if s1.is_file():
        load_segment_encoder(model.pattern_encoder.segment_encoder, s1)

    train_ds = VariableSegmentDataset(
        bundle.bars, bundle.train_idx,
        min_seg_len=args.min_seg_len, max_seg_len=args.max_seg_len,
        samples_per_epoch=args.samples_per_epoch, seed=args.seed,
    )
    valid_ds = VariableSegmentDataset(
        bundle.bars, bundle.valid_idx,
        min_seg_len=args.min_seg_len, max_seg_len=args.max_seg_len,
        samples_per_epoch=max(300, args.samples_per_epoch // 5),
        seed=args.seed + 1,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

    opt, sched = build_adamw_with_warmup_cosine_restarts(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps, t0=args.cosine_t0,
        t_mult=args.cosine_t_mult, eta_min=args.eta_min,
    )

    best = float("inf")
    init_vq_codebook_from_loader(model, train_loader, device)
    for _ in range(args.epochs2):
        tr = train_stage2_epoch(model, train_loader, opt, sched, device, grad_clip=args.grad_clip)
        reset_vq_dead_codes(model, threshold=args.vq_dead_threshold)
        va = evaluate_stage2(model, valid_loader, device)
        history.stage2["train_loss"].append(tr.loss)
        history.stage2["valid_loss"].append(va["loss"])
        history.stage2["valid_recon"].append(va["recon"])
        history.stage2["valid_vq"].append(va["vq"])
        history.stage2["valid_perplexity"].append(va["perplexity"])
        if va["loss"] < best:
            best = va["loss"]
            save_checkpoint(ckpt_dir / "stage2_vqvae.pt", {"stage": 2, "model": model.state_dict()})

    ckpt = torch.load(ckpt_dir / "stage2_vqvae.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return model


def run_stage3(
    args: argparse.Namespace,
    bundle,
    device: torch.device,
    history: TrainHistory,
    ckpt_dir: Path,
) -> KlinePatternPredictor:
    pat_cfg = PatternEncoderConfig(
        feat_dim=5, d_model=args.d_model, n_heads=args.n_heads,
        encoder_layers=args.encoder_layers, max_seg_len=args.max_seg_len,
        num_codes=args.num_codes, vq_beta=args.vq_beta, vq_use_ema=True,
    )
    model_cfg = PatternPredictorConfig(
        pattern=pat_cfg,
        trunk=CausalTransformerConfig(d_model=args.d_model, n_heads=args.n_heads, n_layers=args.trunk_layers),
        pred_horizon=args.pred_horizon,
        pred_feat_dim=1,
    )
    model = KlinePatternPredictor(model_cfg).to(device)
    s2 = ckpt_dir / "stage2_vqvae.pt"
    if s2.is_file():
        load_pattern_encoder(model.pattern_encoder, s2)

    def split_samples(idx: np.ndarray):
        return build_sequence_sample_indices(
            bundle.bars.shape[0],
            context_bars=args.context_bars,
            pred_horizon=args.pred_horizon,
            stride=args.stride,
            index_min=int(idx.min()),
            index_max=int(idx.max()),
        )

    ds_kw = dict(n_segments=args.n_segments, min_seg_len=args.min_seg_len, max_seg_len=args.max_seg_len)
    train_loader = DataLoader(
        PatternSequenceDataset(bundle.bars, split_samples(bundle.train_idx), **ds_kw),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    valid_loader = DataLoader(
        PatternSequenceDataset(bundle.bars, split_samples(bundle.valid_idx), **ds_kw),
        batch_size=args.batch_size, shuffle=False,
    )

    enc_params = list(model.pattern_encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    trunk_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt, sched = build_adamw_with_warmup_cosine_restarts(
        [
            {"params": enc_params, "lr": args.lr * args.encoder_lr_scale},
            {"params": trunk_params, "lr": args.lr},
        ],
        lr=args.lr, weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps, t0=args.cosine_t0,
        t_mult=args.cosine_t_mult, eta_min=args.eta_min,
    )

    best = float("inf")
    for _ in range(args.epochs3):
        tr = train_stage3_epoch(
            model, train_loader, opt, sched, device,
            aux_vq_weight=args.aux_vq_weight, grad_clip=args.grad_clip,
        )
        va = evaluate_stage3(model, valid_loader, device, aux_vq_weight=args.aux_vq_weight)
        history.stage3["train_loss"].append(tr.loss)
        history.stage3["valid_loss"].append(va["loss"])
        if va["loss"] < best:
            best = va["loss"]
            save_checkpoint(ckpt_dir / "stage3_predictor.pt", {"stage": 3, "model": model.state_dict()})

    ckpt = torch.load(ckpt_dir / "stage3_predictor.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return model


def plot_training_curves(history: TrainHistory, out: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ep = range(1, len(history.stage1["train_loss"]) + 1)
    ax.plot(ep, history.stage1["train_loss"], "o-", label="Train", linewidth=1.5, markersize=4)
    ax.plot(ep, history.stage1["valid_loss"], "s-", label="Valid", linewidth=1.5, markersize=4)
    ax.set_title("Stage 1: Segment encoder reconstruction")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ep = range(1, len(history.stage2["train_loss"]) + 1)
    ax.plot(ep, history.stage2["valid_recon"], "o-", label="Valid recon", linewidth=1.5, markersize=4)
    ax.plot(ep, history.stage2["valid_vq"], "s-", label="Valid VQ", linewidth=1.5, markersize=4)
    ax2 = ax.twinx()
    ax2.plot(ep, history.stage2["valid_perplexity"], "^--", color="tab:green", label="Code perplexity", markersize=4)
    ax.set_title("Stage 2: VQ-VAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax2.set_ylabel("Perplexity")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ep = range(1, len(history.stage3["train_loss"]) + 1)
    ax.plot(ep, history.stage3["train_loss"], "o-", label="Train", linewidth=1.5, markersize=4)
    ax.plot(ep, history.stage3["valid_loss"], "s-", label="Valid", linewidth=1.5, markersize=4)
    ax.set_title("Stage 3: Future bar prediction")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE + aux VQ)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Three-stage training curves", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "01_training_curves.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def plot_stage1_reconstruction(
    encoder: SegmentMHAEncoder,
    decoder: SegmentDecoder,
    bundle,
    device: torch.device,
    out: Path,
    dpi: int,
    n_samples: int = 3,
) -> dict[str, float]:
    encoder.eval()
    decoder.eval()
    ds = VariableSegmentDataset(
        bundle.bars, bundle.test_idx,
        min_seg_len=8, max_seg_len=24,
        samples_per_epoch=n_samples, seed=99,
    )
    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = np.array([axes])

    mse_list: list[float] = []
    for i in range(n_samples):
        batch = ds[i]
        bars = batch["seg_bars"].unsqueeze(0).to(device)
        lengths = batch["seg_lengths"].unsqueeze(0).to(device)
        z = encoder(bars, lengths)
        pred = decoder(z)
        ln = int(lengths[0].item())
        orig = bars[0, :ln].cpu().numpy()
        rec = pred[0, :ln].cpu().numpy()
        mse_list.append(float(((orig - rec) ** 2).mean()))

        ax_path = axes[i, 0]
        t = np.arange(ln)
        ax_path.plot(t, cumulative_path(orig[:, 0]), "b-", label="Original", linewidth=2)
        ax_path.plot(t, cumulative_path(rec[:, 0]), "r--", label="Reconstructed", linewidth=2)
        ax_path.set_title(f"Sample {i + 1}: normalized price path (cum log_ret)")
        ax_path.set_xlabel("Bar index")
        ax_path.legend(fontsize=8)
        ax_path.grid(True, alpha=0.3)

        ax_feat = axes[i, 1]
        for j in range(5):
            ax_feat.plot(t, orig[:, j], alpha=0.8, linewidth=1.2, label=f"orig-{FEAT_LABELS[j]}")
            ax_feat.plot(t, rec[:, j], "--", alpha=0.7, linewidth=1.0)
        ax_feat.set_title(f"Sample {i + 1}: shape features (MSE={mse_list[-1]:.4f})")
        ax_feat.set_xlabel("Bar index")
        ax_feat.legend(fontsize=6, ncol=2, loc="upper right")
        ax_feat.grid(True, alpha=0.3)

    fig.suptitle("Stage 1 validation: variable-length segment reconstruction", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out / "02_stage1_reconstruction.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {"stage1_test_recon_mse": float(np.mean(mse_list))}


@torch.no_grad()
def plot_stage2_vqvae(
    model: PatternVQVAE,
    bundle,
    device: torch.device,
    out: Path,
    dpi: int,
) -> dict[str, float]:
    model.eval()
    loader = DataLoader(
        VariableSegmentDataset(
            bundle.bars, bundle.test_idx,
            min_seg_len=4, max_seg_len=32,
            samples_per_epoch=500, seed=7,
        ),
        batch_size=64, shuffle=False,
    )
    all_codes: list[np.ndarray] = []
    for batch in loader:
        bars = batch["seg_bars"].to(device)
        lengths = batch["seg_lengths"].to(device)
        z = model.pattern_encoder.segment_encoder(bars, lengths)
        codes = model.pattern_encoder.vq(z).codes.cpu().numpy()
        all_codes.append(codes)
    codes = np.concatenate(all_codes)
    unique, counts = np.unique(codes, return_counts=True)
    usage = len(unique) / model.cfg.num_codes
    entropy = -np.sum((counts / counts.sum()) * np.log(counts / counts.sum() + 1e-12))
    max_entropy = np.log(model.cfg.num_codes)
    norm_entropy = entropy / max_entropy

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.bar(unique, counts, width=0.8, color="steelblue", edgecolor="white")
    ax.set_title(f"VQ code usage ({len(unique)}/{model.cfg.num_codes} active)")
    ax.set_xlabel("Pattern code ID")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.3)

    # 重建示例
    ds = VariableSegmentDataset(
        bundle.bars, bundle.test_idx, min_seg_len=10, max_seg_len=model.cfg.max_seg_len,
        samples_per_epoch=1, seed=3,
    )
    batch = ds[0]
    bars = batch["seg_bars"].unsqueeze(0).to(device)
    lengths = batch["seg_lengths"].unsqueeze(0).to(device)
    out_vqvae = model(bars, lengths)
    ln = int(lengths[0].item())
    orig = bars[0, :ln, 0].cpu().numpy()
    rec = out_vqvae.recon[0, :ln, 0].cpu().numpy()
    code = int(out_vqvae.vq_out.codes[0].item())

    ax = axes[1]
    t = np.arange(ln)
    ax.plot(t, cumulative_path(orig), "b-", label="Original path", linewidth=2)
    ax.plot(t, cumulative_path(rec), "r--", label=f"VQ recon (code={code})", linewidth=2)
    ax.set_title("Stage 2: single-segment VQ reconstruction")
    ax.set_xlabel("Bar index")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Stage 2 validation: discrete pattern codes", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "03_stage2_vq_usage.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {
        "stage2_code_usage_ratio": float(usage),
        "stage2_code_entropy_norm": float(norm_entropy),
        "stage2_perplexity_est": float(np.exp(entropy)),
    }


@torch.no_grad()
def evaluate_stage3_metrics(
    model: KlinePatternPredictor,
    bundle,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    samples = build_sequence_sample_indices(
        bundle.bars.shape[0],
        context_bars=args.context_bars,
        pred_horizon=args.pred_horizon,
        stride=args.stride,
        index_min=int(bundle.test_idx.min()),
        index_max=int(bundle.test_idx.max()),
    )
    loader = DataLoader(
        PatternSequenceDataset(
            bundle.bars, samples,
            n_segments=args.n_segments,
            min_seg_len=args.min_seg_len,
            max_seg_len=args.max_seg_len,
        ),
        batch_size=32, shuffle=False,
    )
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch in loader:
        seg_bars = batch["seg_bars"].to(device)
        seg_lengths = batch["seg_lengths"].to(device)
        future = batch["future_bars"].to(device)
        pred = model(seg_bars, seg_lengths)
        preds.append(pred.cpu().numpy())
        targets.append(future.cpu().numpy())
    p = np.concatenate(preds, axis=0)
    y = np.concatenate(targets, axis=0)
    if p.ndim == 3 and p.shape[-1] == 1:
        p = p[..., 0]
    y_ret_full = y[..., 0]

    mse_ret = float(((p - y_ret_full) ** 2).mean())
    p_ret = p.ravel()
    y_ret = y_ret_full.ravel()
    ic = float(np.corrcoef(p_ret, y_ret)[0, 1]) if p_ret.std() > 1e-8 and y_ret.std() > 1e-8 else 0.0
    sign_acc = float((np.sign(p_ret) == np.sign(y_ret)).mean())
    return {
        "stage3_test_mse_all": mse_ret,
        "stage3_test_mse_log_ret": mse_ret,
        "stage3_test_ic_log_ret": ic,
        "stage3_test_direction_acc": sign_acc,
        "pred": p,
        "target": y_ret_full,
    }


def plot_stage3_prediction(metrics: dict, out: Path, dpi: int) -> None:
    p = metrics["pred"]
    y = metrics["target"]
    if p.ndim == 3 and p.shape[-1] == 1:
        p = p[..., 0]
    if y.ndim == 3:
        y = y[..., 0]
    n_show = min(4, p.shape[0])
    rng = np.random.default_rng(0)
    idx = rng.choice(p.shape[0], size=n_show, replace=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 未来路径对比
    ax = axes[0, 0]
    h = p.shape[1]
    for k, si in enumerate(idx):
        ax.plot(range(h), cumulative_path(y[si]), alpha=0.5, linewidth=1.5)
        ax.plot(range(h), cumulative_path(p[si]), "--", alpha=0.5, linewidth=1.5)
    ax.plot([], [], "b-", label="Actual")
    ax.plot([], [], "r--", label="Predicted")
    ax.set_title(f"Next {h} bars: normalized path ({n_show} test samples)")
    ax.set_xlabel("Future bar step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # log_ret 散点
    ax = axes[0, 1]
    pr = p.ravel()
    yr = y.ravel()
    ax.scatter(yr, pr, alpha=0.35, s=12, c="steelblue")
    lim = max(abs(yr).max(), abs(pr).max(), 1e-3) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Actual log_ret")
    ax.set_ylabel("Predicted log_ret")
    ax.set_title(
        f"log_ret scatter | IC={metrics['stage3_test_ic_log_ret']:.3f} "
        f"dir_acc={metrics['stage3_test_direction_acc']:.1%}"
    )
    ax.grid(True, alpha=0.3)

    # 逐步 MSE
    ax = axes[1, 0]
    step_mse = [float(((p[:, t] - y[:, t]) ** 2).mean()) for t in range(h)]
    bars = ax.bar(range(h), step_mse, color="coral", edgecolor="white")
    ax.set_xticks(range(h))
    ax.set_xlabel("Future bar step")
    ax.set_title("Per-step log_ret test MSE")
    ax.set_ylabel("MSE")
    ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, step_mse):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1, 1]
    si = idx[0]
    x = np.arange(h)
    w = 0.35
    ax.bar(x - w / 2, y[si], width=w, label="Actual log_ret", alpha=0.85)
    ax.bar(x + w / 2, p[si], width=w, label="Predicted log_ret", alpha=0.85)
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_title("Single sample: step-wise log_ret")
    ax.set_xlabel("Future bar step")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"Stage 3 validation: future prediction | MSE={metrics['stage3_test_mse_all']:.4f}",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out / "04_stage3_prediction.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_summary_dashboard(
    history: TrainHistory,
    all_metrics: dict[str, float],
    out: Path,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    lines = [
        "=== Model Validation Summary ===",
        "",
        f"Stage 1 test recon MSE: {all_metrics.get('stage1_test_recon_mse', float('nan')):.4f}",
        f"Stage 2 code usage:      {all_metrics.get('stage2_code_usage_ratio', 0):.1%}",
        f"Stage 2 norm entropy:    {all_metrics.get('stage2_code_entropy_norm', 0):.3f}",
        "",
        f"Stage 3 test MSE:        {all_metrics.get('stage3_test_mse_all', float('nan')):.4f}",
        f"Stage 3 log_ret IC:      {all_metrics.get('stage3_test_ic_log_ret', 0):.3f}",
        f"Stage 3 direction acc:   {all_metrics.get('stage3_test_direction_acc', 0):.1%}",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), va="top", fontsize=11, family="monospace")

    ax = fig.add_subplot(gs[0, 1])
    labels = ["S1 valid", "S2 valid", "S3 valid"]
    finals = [
        history.stage1["valid_loss"][-1] if history.stage1["valid_loss"] else 0,
        history.stage2["valid_loss"][-1] if history.stage2["valid_loss"] else 0,
        history.stage3["valid_loss"][-1] if history.stage3["valid_loss"] else 0,
    ]
    ax.bar(labels, finals, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_title("Final validation loss per stage")
    ax.set_ylabel("Loss")
    ax.grid(True, axis="y", alpha=0.3)

    ax = fig.add_subplot(gs[1, :])
    ax.plot(history.stage1["valid_loss"], label="Stage 1", linewidth=2)
    ax.plot(history.stage2["valid_loss"], label="Stage 2", linewidth=2)
    ax.plot(history.stage3["valid_loss"], label="Stage 3", linewidth=2)
    ax.set_title("Validation loss curves (different scales; trend only)")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Pattern encoding model — validation report", fontsize=14)
    fig.savefig(out / "05_summary_dashboard.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_models_from_checkpoint(
    args: argparse.Namespace, device: torch.device, ckpt_dir: Path
) -> tuple[SegmentMHAEncoder, SegmentDecoder, PatternVQVAE, KlinePatternPredictor]:
    enc_cfg = SegmentEncoderConfig(
        feat_dim=5, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.encoder_layers, max_len=args.max_seg_len,
    )
    encoder = SegmentMHAEncoder(enc_cfg).to(device)
    decoder = SegmentDecoder(enc_cfg).to(device)
    load_segment_encoder(encoder, ckpt_dir / "stage1_segment_encoder.pt")

    pat_cfg = PatternEncoderConfig(
        feat_dim=5, d_model=args.d_model, n_heads=args.n_heads,
        encoder_layers=args.encoder_layers, max_seg_len=args.max_seg_len,
        num_codes=args.num_codes, vq_beta=args.vq_beta,
    )
    vqvae = PatternVQVAE(pat_cfg).to(device)
    vqvae.load_state_dict(torch.load(ckpt_dir / "stage2_vqvae.pt", map_location=device, weights_only=False)["model"])

    model_cfg = PatternPredictorConfig(
        pattern=pat_cfg,
        trunk=CausalTransformerConfig(d_model=args.d_model, n_heads=args.n_heads, n_layers=args.trunk_layers),
        pred_horizon=args.pred_horizon,
    )
    predictor = KlinePatternPredictor(model_cfg).to(device)
    predictor.load_state_dict(torch.load(ckpt_dir / "stage3_predictor.pt", map_location=device, weights_only=False)["model"])
    return encoder, decoder, vqvae, predictor


def main() -> int:
    args = parse_args()
    if len(sys.argv) == 1:
        args.synthetic = True

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei", "Arial Unicode MS", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    out_dir = Path(args.output_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print("[1/6] 加载数据")
    df = fetch_ohlcv_df(args)
    bundle = prepare_bar_series(df)
    print(f"  bars={bundle.bars.shape[0]}")

    history = TrainHistory()

    if args.skip_train:
        print("[2/6] 跳过训练，加载检查点")
        for name in ("stage1_segment_encoder.pt", "stage2_vqvae.pt", "stage3_predictor.pt"):
            if not (ckpt_dir / name).is_file():
                raise FileNotFoundError(f"缺少检查点 {ckpt_dir / name}，请先训练或去掉 --skip-train")
        encoder, decoder, vqvae, predictor = load_models_from_checkpoint(args, device, ckpt_dir)
    else:
        print("[2/6] Stage 1 训练")
        encoder, decoder = run_stage1(args, bundle, device, history, ckpt_dir)
        print("[3/6] Stage 2 训练")
        vqvae = run_stage2(args, bundle, device, history, ckpt_dir)
        print("[4/6] Stage 3 训练")
        predictor = run_stage3(args, bundle, device, history, ckpt_dir)

    print("[5/6] 评估与绘图")
    all_metrics: dict[str, float] = {}
    all_metrics.update(plot_stage1_reconstruction(encoder, decoder, bundle, device, out_dir, args.dpi))
    all_metrics.update(plot_stage2_vqvae(vqvae, bundle, device, out_dir, args.dpi))
    stage3 = evaluate_stage3_metrics(predictor, bundle, args, device)
    all_metrics.update({k: v for k, v in stage3.items() if k not in ("pred", "target")})
    plot_stage3_prediction(stage3, out_dir, args.dpi)

    if history.stage1["train_loss"]:
        plot_training_curves(history, out_dir, args.dpi)
        plot_summary_dashboard(history, all_metrics, out_dir, args.dpi)
    else:
        # skip-train 模式：仅画验证图 + 摘要文本
        plot_summary_dashboard(TrainHistory(), all_metrics, out_dir, args.dpi)

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"metrics": all_metrics, "history": asdict(history)}, f, indent=2, ensure_ascii=False)

    print("[6/6] 完成")
    print(f"  报告目录: {out_dir.resolve()}")
    for p in sorted(out_dir.glob("*.png")):
        print(f"    - {p.name}")
    print(f"  指标: {metrics_path.name}")
    print("\n关键指标:")
    for k in sorted(all_metrics):
        print(f"  {k}: {all_metrics[k]:.4f}" if isinstance(all_metrics[k], float) else f"  {k}: {all_metrics[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
