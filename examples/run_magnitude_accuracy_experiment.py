#!/usr/bin/env python3
"""幅度准确性实验：调参使价格变动预测相对误差 <= 20% 的命中率最大化。

  python examples/run_magnitude_accuracy_experiment.py --synthetic
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
    run_stage1,
    run_stage2,
    run_stage3,
)

MAGNITUDE_PRESETS: list[dict] = [
    {
        "name": "mag_v1_balanced",
        "mse_weight": 1.0,
        "step_corr_weight": 0.08,
        "cum_corr_weight": 0.08,
        "sign_weight": 0.08,
        "rank_weight": 0.0,
        "cum_magnitude_weight": 0.6,
        "relative_magnitude_weight": 0.35,
        "anti_lag_weight": 0.05,
        "no_learnable_scale": False,
    },
    {
        "name": "mag_v2_mse_heavy",
        "mse_weight": 1.4,
        "step_corr_weight": 0.05,
        "cum_corr_weight": 0.05,
        "sign_weight": 0.05,
        "rank_weight": 0.0,
        "cum_magnitude_weight": 1.0,
        "relative_magnitude_weight": 0.5,
        "anti_lag_weight": 0.0,
        "no_learnable_scale": False,
    },
    {
        "name": "mag_v3_rel_focus",
        "mse_weight": 0.8,
        "step_corr_weight": 0.03,
        "cum_corr_weight": 0.03,
        "sign_weight": 0.05,
        "rank_weight": 0.0,
        "cum_magnitude_weight": 0.5,
        "relative_magnitude_weight": 0.8,
        "anti_lag_weight": 0.0,
        "no_learnable_scale": False,
    },
    {
        "name": "mag_v5_raw_scale",
        "mse_weight": 0.15,
        "raw_mse_weight": 1.5,
        "step_corr_weight": 0.03,
        "cum_corr_weight": 0.03,
        "sign_weight": 0.05,
        "rank_weight": 0.0,
        "cum_magnitude_weight": 1.0,
        "relative_magnitude_weight": 0.9,
        "anti_lag_weight": 0.0,
        "no_learnable_scale": False,
        "pred_horizon": 1,
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Magnitude accuracy experiment")
    add_data_args(p)
    add_feature_args(p)
    add_train_args(p)
    add_segment_args(p)
    add_vq_args(p)
    add_stage3_loss_args(p)
    add_break_vol_args(p)
    p.add_argument("--epochs1", type=int, default=8)
    p.add_argument("--epochs2", type=int, default=6)
    p.add_argument("--epochs3", type=int, default=18)
    p.add_argument("--pred-horizon", type=int, default=5)
    p.add_argument("--pred-feat-dim", type=int, default=1)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--trunk-layers", type=int, default=2)
    p.add_argument("--aux-vq-weight", type=float, default=0.08)
    p.add_argument("--aux-break-weight", type=float, default=0.04)
    p.add_argument("--encoder-lr-scale", type=float, default=0.05)
    p.add_argument("--target-magnitude-rate", type=float, default=0.7, help="目标：>=70% 样本幅度相对误差在容差内")
    p.add_argument("--output-dir", default="reports/magnitude_accuracy")
    p.add_argument("--preset", default="", help="只跑指定 preset 名称")
    return p.parse_args()


def apply_preset(args: argparse.Namespace, preset: dict) -> argparse.Namespace:
    out = deepcopy(args)
    out.no_learnable_scale = preset.get("no_learnable_scale", False)
    if "pred_horizon" in preset:
        out.pred_horizon = preset["pred_horizon"]
    for key in (
        "mse_weight",
        "step_corr_weight",
        "cum_corr_weight",
        "sign_weight",
        "rank_weight",
        "cum_magnitude_weight",
        "relative_magnitude_weight",
        "raw_mse_weight",
        "anti_lag_weight",
    ):
        if key in preset:
            setattr(out, key, preset[key])
    return out


def run_preset(
    base_args: argparse.Namespace,
    preset: dict,
    out_root: Path,
    bundle,
    device: torch.device,
) -> dict:
    args = apply_preset(base_args, preset)
    name = preset["name"]
    ckpt_dir = out_root / name / "checkpoints"
    report_dir = out_root / name / "report"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir = str(ckpt_dir)
    args.output_dir = str(report_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n=== preset {name} ===")
    print(
        f"  loss: mse={args.mse_weight} cum_mag={args.cum_magnitude_weight} "
        f"rel_mag={args.relative_magnitude_weight}"
    )

    history = TrainHistory()
    vqvae = run_stage1(args, bundle, device, history, ckpt_dir)
    vqvae = run_stage2(args, bundle, device, history, ckpt_dir, vqvae)
    pred = run_stage3(args, bundle, device, history, ckpt_dir, vqvae)

    pred_scale, pred_bias, raw_cal_mse, cal_mse, cum_scale = calibrate_prediction_affine(
        pred, bundle, args, device,
    )
    direction_threshold, direction_cal_acc = calibrate_cum_direction_threshold(
        pred, bundle, args, device, pred_scale=pred_scale, pred_bias=pred_bias,
    )
    metrics = evaluate_and_plot_prediction(
        pred,
        bundle,
        args,
        device,
        report_dir,
        dpi=120,
        cum_direction_threshold=direction_threshold,
        calibration_acc=direction_cal_acc,
        pred_scale=pred_scale,
        pred_bias=pred_bias,
        cum_magnitude_scale=cum_scale,
        magnitude_raw_calibration_mse=raw_cal_mse,
        magnitude_calibration_mse=cal_mse,
    )

    summary = {
        "preset": name,
        "magnitude_within_tol_rate": metrics.get("test_magnitude_within_tol_rate", 0.0),
        "magnitude_within_tol_rate_raw": metrics.get("test_raw_magnitude_within_tol_rate", 0.0),
        "magnitude_mean_rel_err": metrics.get("test_magnitude_mean_rel_err", 0.0),
        "magnitude_median_rel_err": metrics.get("test_magnitude_median_rel_err", 0.0),
        "test_cum_direction_acc": metrics.get("test_cum_direction_acc", 0.0),
        "test_cum_ic": metrics.get("test_cum_ic", 0.0),
        "test_mse": metrics.get("test_mse", 0.0),
        "cum_magnitude_scale": float(cum_scale),
        "meets_target": bool(
            metrics.get("test_magnitude_median_rel_err", 1.0) <= base_args.magnitude_tolerance
            or metrics.get("test_magnitude_within_tol_rate", 0.0) >= base_args.target_magnitude_rate
        ),
    }
    with (report_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "metrics": metrics}, f, indent=2)

    print(
        f"  mag@{args.magnitude_tolerance:.0%}={summary['magnitude_within_tol_rate']:.1%} "
        f"(raw={summary['magnitude_within_tol_rate_raw']:.1%}) "
        f"median_err={summary['magnitude_median_rel_err']:.1%} "
        f"dir={summary['test_cum_direction_acc']:.1%}"
    )
    return summary


def main() -> int:
    args = parse_args()
    apply_real_data_defaults(args)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    bundle = prepare_bar_series_from_args(fetch_ohlcv_df(args), args)
    print(f"data: bars={bundle.bars.shape[0]} feat_dim={bundle.bars.shape[1]}")

    presets = MAGNITUDE_PRESETS
    if args.preset:
        presets = [p for p in MAGNITUDE_PRESETS if p["name"] == args.preset]
        if not presets:
            raise SystemExit(f"unknown preset: {args.preset}")

    results: list[dict] = []
    for preset in presets:
        results.append(run_preset(args, preset, out_root, bundle, device))

    best = max(results, key=lambda r: r["magnitude_within_tol_rate"])
    report = {
        "tolerance": args.magnitude_tolerance,
        "target_rate": args.target_magnitude_rate,
        "results": results,
        "best_preset": best["preset"],
        "best_magnitude_within_tol_rate": best["magnitude_within_tol_rate"],
        "target_met": best["meets_target"],
    }
    report_path = out_root / "comparison.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== summary ===")
    for r in results:
        mark = "✓" if r["meets_target"] else " "
        print(
            f"  [{mark}] {r['preset']}: mag@{args.magnitude_tolerance:.0%}="
            f"{r['magnitude_within_tol_rate']:.1%} dir={r['test_cum_direction_acc']:.1%}"
        )
    print(f"  best: {best['preset']} ({best['magnitude_within_tol_rate']:.1%})")
    print(f"  report: {report_path.resolve()}")
    return 0 if best["meets_target"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
