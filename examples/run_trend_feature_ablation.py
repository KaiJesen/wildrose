#!/usr/bin/env python3
"""趋势特征消融实验：baseline（5D 形态）vs 多尺度趋势特征（17D）。

  python examples/run_trend_feature_ablation.py --synthetic
  python examples/run_trend_feature_ablation.py --source binance_vision --symbol BTCUSDT
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

_EX = Path(__file__).resolve().parent
_ROOT = _EX.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))

from _train_common import add_data_args, add_feature_args, add_segment_args, add_stage3_loss_args, add_train_args, add_vq_args, apply_real_data_defaults, fetch_ohlcv_df, prepare_bar_series_from_args
from plot_auto_segment_report import (
    TrainHistory,
    calibrate_cum_direction_threshold,
    calibrate_prediction_affine,
    evaluate_and_plot_prediction,
    run_stage1,
    run_stage2,
    run_stage3,
)
from transformer_kit.segment_features import BAR_SHAPE_DIM, feat_dim
from transformer_kit.trend_features import causal_log_price_trend_features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trend feature ablation experiment")
    add_data_args(p)
    add_feature_args(p)
    add_train_args(p)
    add_segment_args(p)
    add_vq_args(p)
    add_stage3_loss_args(p)
    p.add_argument("--epochs1", type=int, default=10)
    p.add_argument("--epochs2", type=int, default=8)
    p.add_argument("--epochs3", type=int, default=15)
    p.add_argument("--pred-horizon", type=int, default=5)
    p.add_argument("--pred-feat-dim", type=int, default=1)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--trunk-layers", type=int, default=2)
    p.add_argument("--aux-vq-weight", type=float, default=0.1)
    p.add_argument("--aux-break-weight", type=float, default=0.05)
    p.add_argument("--encoder-lr-scale", type=float, default=0.1)
    p.add_argument("--output-dir", default="reports/trend_feature_ablation")
    p.add_argument("--skip-sanity", action="store_true")
    return p.parse_args()


def verify_trend_causality() -> None:
    """修改未来 close 不应影响过去的趋势特征。"""
    rng = np.random.default_rng(0)
    n = 200
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, size=n))
    t = 80

    base = causal_log_price_trend_features(close)
    mutated = close.copy()
    mutated[t + 1 :] += 10.0
    alt = causal_log_price_trend_features(mutated)

    if not np.allclose(base[: t + 1], alt[: t + 1], atol=1e-5):
        raise AssertionError("trend features are not causal")
    print("  sanity: trend features are causal ✓")


def run_one_variant(
    label: str,
    args: argparse.Namespace,
    *,
    use_trend: bool,
    out_root: Path,
) -> dict:
    variant_args = deepcopy(args)
    variant_args.trend_features = use_trend
    variant_args.checkpoint_dir = str(out_root / label / "checkpoints")
    variant_args.output_dir = str(out_root / label / "report")

    ckpt_dir = Path(variant_args.checkpoint_dir)
    out_dir = Path(variant_args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(variant_args.device)
    torch.manual_seed(variant_args.seed)
    np.random.seed(variant_args.seed)

    print(f"\n=== {label} (feat_dim={feat_dim(use_trend_features=use_trend)}) ===")
    bundle = prepare_bar_series_from_args(fetch_ohlcv_df(variant_args), variant_args)
    print(f"  bars={bundle.bars.shape[0]} feat_dim={bundle.bars.shape[1]}")

    history = TrainHistory()
    vqvae = run_stage1(variant_args, bundle, device, history, ckpt_dir)
    vqvae = run_stage2(variant_args, bundle, device, history, ckpt_dir, vqvae)
    pred = run_stage3(variant_args, bundle, device, history, ckpt_dir, vqvae)

    pred_scale, pred_bias, raw_cal_mse, cal_mse, _ = calibrate_prediction_affine(pred, bundle, variant_args, device)
    direction_threshold, direction_cal_acc = calibrate_cum_direction_threshold(
        pred,
        bundle,
        variant_args,
        device,
        pred_scale=pred_scale,
        pred_bias=pred_bias,
    )
    metrics = evaluate_and_plot_prediction(
        pred,
        bundle,
        variant_args,
        device,
        out_dir,
        dpi=120,
        cum_direction_threshold=direction_threshold,
        calibration_acc=direction_cal_acc,
        pred_scale=pred_scale,
        pred_bias=pred_bias,
        magnitude_raw_calibration_mse=raw_cal_mse,
        magnitude_calibration_mse=cal_mse,
    )
    metrics["feat_dim"] = int(bundle.bars.shape[1])
    metrics["use_trend_features"] = use_trend
    metrics["valid_ic_final"] = float(history.stage3["ic"][-1]) if history.stage3["ic"] else 0.0

    summary = {
        "label": label,
        "feat_dim": metrics["feat_dim"],
        "test_ic": metrics.get("test_ic", 0.0),
        "test_cum_ic": metrics.get("test_cum_ic", 0.0),
        "test_direction_acc": metrics.get("test_direction_acc", 0.0),
        "test_cum_direction_acc": metrics.get("test_cum_direction_acc", 0.0),
        "test_mse": metrics.get("test_mse", 0.0),
        "valid_ic_final": metrics["valid_ic_final"],
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "metrics": metrics, "history": history.__dict__}, f, indent=2)
    print(
        f"  test_ic={summary['test_ic']:.4f} test_cum_ic={summary['test_cum_ic']:.4f} "
        f"dir_acc={summary['test_cum_direction_acc']:.1%}"
    )
    return summary


def main() -> int:
    args = parse_args()
    apply_real_data_defaults(args)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print("[0/3] sanity checks")
    if not args.skip_sanity:
        verify_trend_causality()
        assert BAR_SHAPE_DIM == 5

    print("[1/3] baseline (no trend features)")
    baseline = run_one_variant("baseline_no_trend", args, use_trend=False, out_root=out_root)

    print("[2/3] with multi-scale trend features")
    with_trend = run_one_variant("with_trend_features", args, use_trend=True, out_root=out_root)

    delta = {
        "test_ic": with_trend["test_ic"] - baseline["test_ic"],
        "test_cum_ic": with_trend["test_cum_ic"] - baseline["test_cum_ic"],
        "test_cum_direction_acc": with_trend["test_cum_direction_acc"] - baseline["test_cum_direction_acc"],
        "test_mse": with_trend["test_mse"] - baseline["test_mse"],
    }
    winner = "with_trend" if delta["test_cum_ic"] > 0 else "baseline"
    if delta["test_cum_ic"] == 0:
        winner = "tie"

    report = {
        "baseline": baseline,
        "with_trend_features": with_trend,
        "delta_with_trend_minus_baseline": delta,
        "winner_by_test_cum_ic": winner,
    }
    report_path = out_root / "comparison.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n[3/3] comparison")
    print(f"  baseline:    cum_ic={baseline['test_cum_ic']:.4f} dir={baseline['test_cum_direction_acc']:.1%}")
    print(f"  with trend:  cum_ic={with_trend['test_cum_ic']:.4f} dir={with_trend['test_cum_direction_acc']:.1%}")
    print(f"  delta:       cum_ic={delta['test_cum_ic']:+.4f} dir={delta['test_cum_direction_acc']:+.1%}")
    print(f"  winner:      {winner}")
    print(f"  report:      {report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
