#!/usr/bin/env python3
"""流水编号实验：训练 + 预测效果清晰展示。

  python examples/run_numbered_prediction_showcase.py
  python examples/run_numbered_prediction_showcase.py --source akshare_em --symbol 600519 --interval 60m
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
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

from _train_common import (
    add_break_vol_args,
    add_data_args,
    add_feature_args,
    add_segment_args,
    add_stage3_loss_args,
    add_train_args,
    add_vq_args,
    apply_real_data_defaults,
    fetch_ohlcv_df,
    prepare_bar_series_from_args,
)
from plot_auto_segment_report import (
    TrainHistory,
    calibrate_cum_direction_threshold,
    calibrate_prediction_affine,
    evaluate_and_plot_prediction,
    plot_auto_segmentation,
    plot_summary,
    plot_training_curves,
    plot_vq_usage,
    run_stage1,
    run_stage2,
    run_stage3,
)
from transformer_kit.magnitude_metrics import (
    cumulative_price_change,
    denorm_zscore_log_ret,
    magnitude_accuracy_metrics,
    magnitude_relative_errors,
)
from transformer_kit.auto_segment_encoder import AutoSegmentVQVAE
from transformer_kit.causal_transformer import CausalTransformerConfig
from transformer_kit.pattern_encoder import pattern_config_from_args
from transformer_kit.pattern_model import KlinePatternPredictor, PatternPredictorConfig
from transformer_kit.train_utils import load_checkpoint
from transformer_kit.segment_dataset import PatternSequenceDataset, build_sequence_sample_indices

EXP_ID = "0041"
EXP_SEED = 41
EXP_SLUG = "rebalanced_real_btc"
REPORT_DIR = _ROOT / "reports" / f"{EXP_ID}_{EXP_SLUG}"
CKPT_DIR = _ROOT / "checkpoints" / f"{EXP_ID}_{EXP_SLUG}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Experiment {EXP_ID}: prediction showcase")
    add_data_args(p)
    add_feature_args(p)
    add_train_args(p)
    add_segment_args(p)
    add_vq_args(p)
    add_stage3_loss_args(p)
    add_break_vol_args(p)
    p.add_argument("--epochs1", type=int, default=10)
    p.add_argument("--epochs2", type=int, default=8)
    p.add_argument("--epochs3", type=int, default=24)
    p.add_argument("--pred-horizon", type=int, default=5)
    p.add_argument("--pred-feat-dim", type=int, default=1)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--trunk-layers", type=int, default=2)
    p.add_argument("--aux-vq-weight", type=float, default=0.08)
    p.add_argument("--aux-break-weight", type=float, default=0.04)
    p.add_argument("--encoder-lr-scale", type=float, default=0.0)
    p.add_argument("--skip-train", action="store_true", help="跳过训练，从 checkpoint 加载并仅生成报告")
    p.add_argument("--dpi", type=int, default=150)
    p.set_defaults(
        checkpoint_dir=str(CKPT_DIR),
        output_dir=str(REPORT_DIR),
        trend_features=True,
        mse_weight=0.50,
        raw_mse_weight=1.2,
        step_corr_weight=0.10,
        cum_corr_weight=0.12,
        sign_weight=0.1,
        rank_weight=0.08,
        direction_weight=0.1,
        cum_magnitude_weight=0.75,
        relative_magnitude_weight=0.55,
        magnitude_tolerance=0.2,
        magnitude_min_move=1e-5,
        vol_focus_weight=0.4,
        vol_focus_top_frac=0.25,
        move_focus_weight=0.6,
        move_focus_scale=3.0,
        break_focus_weight=0.2,
        break_focus_tail=16,
        code_supervision_weight=0.1,
        usage_balance_weight=0.55,
        break_aware_vq_balance=True,
        break_seg_vq_weight=2.0,
        background_seg_vq_weight=0.35,
        vq_max_code_frac=0.15,
        vq_inverse_freq_ema=True,
        samples_per_epoch=1500,
        batch_size=32,
    )
    return p.parse_args()


@torch.no_grad()
def collect_test_predictions(
    model: KlinePatternPredictor,
    bundle,
    args,
    device: torch.device,
    *,
    pred_scale: np.ndarray,
    pred_bias: np.ndarray,
    cum_magnitude_scale: float,
) -> dict[str, np.ndarray]:
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
            bundle.bars, samples, bundle.raw_log_ret, zscore_window=bundle.zscore_window,
        ),
        batch_size=64,
        shuffle=False,
    )
    preds_z, targets, means, stds = [], [], [], []
    for batch in loader:
        pred = model(batch["ctx_bars"].to(device), batch["ctx_lengths"].to(device))
        pz = pred[..., 0].cpu().numpy() if pred.dim() == 3 else pred.cpu().numpy()
        preds_z.append(pz)
        targets.append(batch["future_raw_log_ret"].numpy())
        means.append(batch["future_log_ret_mean"].numpy())
        stds.append(batch["future_log_ret_std"].numpy())
    p_raw = denorm_zscore_log_ret(
        np.concatenate(preds_z),
        np.concatenate(means, axis=0),
        np.concatenate(stds, axis=0),
    )
    y = np.concatenate(targets)
    p_cal = p_raw * pred_scale[None, :] + pred_bias[None, :]
    p_cal = p_cal * cum_magnitude_scale
    return {"pred_raw": p_raw, "pred_cal": p_cal, "target": y}


def plot_prediction_showcase(
    data: dict[str, np.ndarray],
    out: Path,
    dpi: int,
    *,
    tolerance: float = 0.2,
    min_move: float = 1e-5,
) -> dict:
    p = data["pred_cal"]
    y = data["target"]
    pred_pct = cumulative_price_change(p) * 100.0
    tgt_pct = cumulative_price_change(y) * 100.0
    price_rel, _ = magnitude_relative_errors(p, y)
    mag = magnitude_accuracy_metrics(p, y, tolerance=tolerance, min_move=min_move)
    dir_hit = (np.sign(pred_pct) == np.sign(tgt_pct))

    # --- 06 价格变动散点 ---
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(tgt_pct, pred_pct, alpha=0.75, s=36, c=np.where(dir_hit, "#2ca02c", "#d62728"), edgecolors="white")
    lim = max(np.abs(tgt_pct).max(), np.abs(pred_pct).max(), 0.05) * 1.15
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Actual cumulative price change (%)")
    ax.set_ylabel("Predicted cumulative price change (%)")
    ax.set_title(
        f"Pred vs Actual price move | Dir acc {dir_hit.mean():.1%} | "
        f"Within +/-{tolerance:.0%} {mag['magnitude_within_tol_rate']:.1%}"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "06_price_change_scatter.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # --- 07 多样本逐步对比 ---
    n_show = min(6, p.shape[0])
    rng = np.random.default_rng(EXP_SEED)
    show_idx = rng.choice(p.shape[0], size=n_show, replace=False)
    fig, axes = plt.subplots(n_show, 1, figsize=(11, 2.2 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]
    h = p.shape[1]
    x = np.arange(1, h + 1)
    for ax, si in zip(axes, show_idx):
        ax.plot(x, cumulative_price_change(y[si]) * 100, "o-", color="#1f77b4", label="Actual", lw=2)
        ax.plot(x, cumulative_price_change(p[si]) * 100, "s--", color="#ff7f0e", label="Predicted", lw=2)
        ax.axhline(0, color="k", lw=0.4)
        rel = price_rel[si]
        ok = rel <= tolerance
        ax.set_ylabel("Cum. change %")
        ax.set_title(
            f"Sample #{si} | true={tgt_pct[si]:+.2f}% pred={pred_pct[si]:+.2f}% | "
            f"dir={'OK' if dir_hit[si] else 'MISS'} rel_err={rel:.1%}{' (OK)' if ok else ''}"
        )
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel(f"Future bar step (horizon={h})")
    fig.suptitle("Test samples: actual vs predicted price paths", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out / "07_sample_paths.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # --- 08 相对误差分布 ---
    mask = np.abs(tgt_pct / 100.0) > min_move
    rel_masked = price_rel[mask] if mask.any() else price_rel
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(rel_masked, bins=20, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(tolerance, color="red", ls="--", lw=2, label=f"Tolerance {tolerance:.0%}")
    if rel_masked.size:
        ax.axvline(float(np.median(rel_masked)), color="orange", ls="-", lw=1.5, label=f"Median {np.median(rel_masked):.1%}")
    ax.set_xlabel("Relative price error |pred-true|/|true|")
    ax.set_ylabel("Count")
    ax.set_title("Magnitude error distribution (test set)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "08_magnitude_error_hist.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # --- 09 逐步 log_ret 对比（最佳+最差样本）---
    order = np.argsort(price_rel)
    pairs = [("best", order[0]), ("worst", order[-1])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    w = 0.35
    for ax, (label, si) in zip(axes, pairs):
        ax.bar(x - w / 2, y[si] * 100, width=w, label="Actual log_ret x100", color="#4C72B0")
        ax.bar(x + w / 2, p[si] * 100, width=w, label="Pred log_ret x100", color="#DD8452")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{label} sample #{si} | true={tgt_pct[si]:+.2f}% pred={pred_pct[si]:+.2f}%")
        ax.set_xlabel("Future bar")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Step-wise log_ret: best vs worst magnitude match", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "09_step_logret_best_worst.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "showcase_direction_acc": float(dir_hit.mean()),
        "showcase_mean_price_chg_pct_pred": float(pred_pct.mean()),
        "showcase_mean_price_chg_pct_true": float(tgt_pct.mean()),
        **mag,
    }


def write_metrics_txt(out: Path, metrics: dict, args, *, showcase: dict) -> None:
    lines = [
        f"=== 实验 {EXP_ID}_{EXP_SLUG} ===",
        "",
        "【数据】",
        f"  synthetic={getattr(args, 'synthetic', False)} source={getattr(args, 'source', '')} "
        f"symbol={getattr(args, 'symbol', '')} horizon={args.pred_horizon}",
        f"  trend_features={args.trend_features}",
        f"  vol/move/break_focus={getattr(args, 'vol_focus_weight', 0)}/"
        f"{getattr(args, 'move_focus_weight', 0)}/{getattr(args, 'break_focus_weight', 0)}",
        f"  code_supervision_weight={getattr(args, 'code_supervision_weight', 0)}",
        f"  break_aware_vq={getattr(args, 'break_aware_vq_balance', False)} "
        f"vq_max_frac={getattr(args, 'vq_max_code_frac', 0.15)}",
        f"  code_supervision_head_acc={metrics.get('test_code_supervision_head_acc', 0):.1%}",
        f"  VQ max_code_frac={metrics.get('max_code_frac', 0):.1%} "
        f"norm_entropy={metrics.get('code_entropy_norm', 0):.3f}",
        "",
        "【测试集预测效果】",
        f"  方向准确率(累计价格):     {showcase.get('showcase_direction_acc', 0):.1%}",
        f"  IC (逐步):                {metrics.get('test_ic', 0):.3f}",
        f"  IC (累计):                {metrics.get('test_cum_ic', 0):.3f}",
        f"  累计方向准确率:           {metrics.get('test_cum_direction_acc', 0):.1%}",
        "",
        "【幅度准确性】",
        f"  容差:                     ±{args.magnitude_tolerance:.0%}",
        f"  命中率(校准后):           {metrics.get('test_magnitude_within_tol_rate', 0):.1%}",
        f"  命中率(校准前):           {metrics.get('test_raw_magnitude_within_tol_rate', 0):.1%}",
        f"  中位相对误差:             {metrics.get('test_magnitude_median_rel_err', 0):.1%}",
        f"  平均相对误差:             {metrics.get('test_magnitude_mean_rel_err', 0):.1%}",
        f"  有效评估样本数:           {int(metrics.get('test_magnitude_eval_samples', 0))}",
        "",
        "【误差指标】",
        f"  MSE (校准/原始):          {metrics.get('test_mse', 0):.6f} / {metrics.get('test_raw_mse', 0):.6f}",
        f"  MAE (校准/原始):          {metrics.get('test_mae', 0):.6f} / {metrics.get('test_raw_mae', 0):.6f}",
        f"  累计幅度缩放系数:         {metrics.get('cum_magnitude_scale', 1):.4f}",
        "",
        "【图表说明】",
        "  01_training_curves.png   - 三阶段训练曲线",
        "  02_auto_segmentation.png - 自动切分可视化",
        "  03_vq_usage.png          - VQ 码本使用",
        "  04_prediction.png        - 标准预测评估",
        "  05_summary.png           - 指标总览",
        "  06_price_change_scatter  - 预测 vs 真实价格变动散点",
        "  07_sample_paths.png      - 6 个样本价格路径对比",
        "  08_magnitude_error_hist  - 幅度相对误差分布",
        "  09_step_logret_best_worst - 最准/最不准样本逐步收益",
        "  predictions_detail.csv   - 逐样本预测明细",
    ]
    out.joinpath("metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_predictions_csv(out: Path, data: dict[str, np.ndarray], tolerance: float) -> None:
    p = data["pred_cal"]
    y = data["target"]
    pred_pct = cumulative_price_change(p) * 100.0
    tgt_pct = cumulative_price_change(y) * 100.0
    price_rel, _ = magnitude_relative_errors(p, y)
    rows = ["sample_id,true_pct,pred_pct,direction_ok,rel_err,within_tol"]
    for i in range(p.shape[0]):
        rows.append(
            f"{i},{tgt_pct[i]:.6f},{pred_pct[i]:.6f},"
            f"{int(np.sign(pred_pct[i]) == np.sign(tgt_pct[i]))},{price_rel[i]:.6f},"
            f"{int(price_rel[i] <= tolerance)}"
        )
    out.joinpath("predictions_detail.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    apply_real_data_defaults(args)

    out_dir = Path(args.output_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[{EXP_ID}] load data -> {out_dir}")
    bundle = prepare_bar_series_from_args(fetch_ohlcv_df(args), args)
    print(f"  bars={bundle.bars.shape[0]} feat_dim={bundle.bars.shape[1]}")

    history = TrainHistory()
    cfg = pattern_config_from_args(args)
    if args.skip_train:
        print(f"[{EXP_ID}] skip train, load checkpoints")
        vqvae = AutoSegmentVQVAE(cfg).to(device)
        pred = KlinePatternPredictor(PatternPredictorConfig(
            auto_segment=cfg,
            trunk=CausalTransformerConfig(d_model=args.d_model, n_heads=args.n_heads, n_layers=args.trunk_layers),
            pred_horizon=args.pred_horizon,
            pred_feat_dim=args.pred_feat_dim,
            pool_mode=args.pool_mode,
            learnable_scale=not args.no_learnable_scale,
            use_horizon_head=args.horizon_head,
        )).to(device)
        vqvae.load_state_dict(load_checkpoint(ckpt_dir / "stage2_vqvae.pt", map_location=device)["model"], strict=False)
        ckpt_path = ckpt_dir / "stage3_predictor_best_balanced.pt"
        if not ckpt_path.is_file():
            ckpt_path = ckpt_dir / "stage3_predictor_best_combo.pt"
        if not ckpt_path.is_file():
            ckpt_path = ckpt_dir / "stage3_predictor.pt"
        pred.load_state_dict(load_checkpoint(ckpt_path, map_location=device)["model"], strict=False)
    else:
        print(f"[{EXP_ID}] Stage 1/2/3 train")
        vqvae = run_stage1(args, bundle, device, history, ckpt_dir)
        vqvae = run_stage2(args, bundle, device, history, ckpt_dir, vqvae)
        pred = run_stage3(args, bundle, device, history, ckpt_dir, vqvae)

    print(f"[{EXP_ID}] evaluate & plot")
    if history.stage1["train_loss"]:
        plot_training_curves(history, out_dir, args.dpi)
    if not args.skip_train:
        seg_metrics = plot_auto_segmentation(vqvae, bundle, device, out_dir, args.dpi)
        vq_metrics = plot_vq_usage(vqvae, bundle, device, out_dir, args.dpi)
    else:
        seg_metrics, vq_metrics = {}, {}

    pred_scale, pred_bias, raw_cal_mse, cal_mse, cum_scale = calibrate_prediction_affine(
        pred, bundle, args, device,
    )
    dir_thr, dir_acc = calibrate_cum_direction_threshold(
        pred, bundle, args, device, pred_scale=pred_scale, pred_bias=pred_bias,
    )
    metrics = evaluate_and_plot_prediction(
        pred, bundle, args, device, out_dir, args.dpi,
        cum_direction_threshold=dir_thr,
        calibration_acc=dir_acc,
        pred_scale=pred_scale,
        pred_bias=pred_bias,
        cum_magnitude_scale=cum_scale,
        magnitude_raw_calibration_mse=raw_cal_mse,
        magnitude_calibration_mse=cal_mse,
    )
    metrics.update(seg_metrics)
    metrics.update(vq_metrics)

    pred_data = collect_test_predictions(
        pred, bundle, args, device,
        pred_scale=pred_scale, pred_bias=pred_bias, cum_magnitude_scale=cum_scale,
    )
    showcase = plot_prediction_showcase(
        pred_data, out_dir, args.dpi,
        tolerance=args.magnitude_tolerance,
        min_move=args.magnitude_min_move,
    )
    metrics.update(showcase)
    write_predictions_csv(out_dir, pred_data, args.magnitude_tolerance)
    write_metrics_txt(out_dir, metrics, args, showcase=showcase)
    plot_summary(metrics, history, out_dir, args.dpi)

    payload = {"experiment_id": EXP_ID, "metrics": metrics, "history": asdict(history), "args": vars(args)}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[{EXP_ID}] done -> {out_dir.resolve()}")
    print(f"  方向准确率: {showcase.get('showcase_direction_acc', 0):.1%}")
    print(f"  幅度@{args.magnitude_tolerance:.0%}内: {metrics.get('test_magnitude_within_tol_rate', 0):.1%}")
    print(f"  中位相对误差: {metrics.get('test_magnitude_median_rel_err', 0):.1%}")
    for name in sorted(out_dir.glob("*.png")):
        print(f"    - {name.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
