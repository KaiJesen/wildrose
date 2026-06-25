#!/usr/bin/env python3
"""024 Phase 2: fit TEQ edge affine calibration on valid split (no test leakage)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

_EX = Path(__file__).resolve().parent
_ROOT = _EX.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))

from _train_common import (
    add_data_args,
    add_feature_args,
    add_segment_args,
    add_vq_args,
    apply_real_data_defaults,
    fetch_ohlcv_df,
    prepare_bar_series_from_args,
)
from trading_system.adapters.market_state_model import ModelSignalProvider
from trading_system.config import TeqEdgeConfig, load_config
from trading_system.teq_edge import TeqEdgeCalibrator, compute_teq_edge_raw, fit_teq_edge_calibrator

DEFAULT_CKPT = "checkpoints/0065a_leg_align_v1/market_state_best.pt"
DEFAULT_CONFIG = "configs/trading_rule_v023_phase1c_0062e.json"
DEFAULT_LABELS = "data/labels/leg_participation/leg_participation_valid.csv"
DEFAULT_OUT = "backtest/v024_phase2/teq_edge_calibration.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate TEQ edge on valid split")
    add_data_args(p)
    add_feature_args(p)
    add_segment_args(p)
    add_vq_args(p)
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--labels", default=DEFAULT_LABELS)
    p.add_argument("--output", default=DEFAULT_OUT)
    p.add_argument("--device", default="cpu")
    p.add_argument("--trunk-layers", type=int, default=2)
    p.set_defaults(source="binance_vision", symbol="BTCUSDT", interval="1h", days=365)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    apply_real_data_defaults(args)
    labels_path = Path(args.labels)
    if not labels_path.is_file():
        raise FileNotFoundError(f"missing labels: {labels_path} (run examples/build_leg_participation_labels.py)")
    labels = pd.read_csv(labels_path)
    if "bar_idx" not in labels.columns:
        raise ValueError("labels must include bar_idx")

    base_cfg = load_config(args.config)
    teq_cfg = TeqEdgeConfig(
        enabled=True,
        weight_edge_5=base_cfg.teq_edge.weight_edge_5,
        weight_edge_24=base_cfg.teq_edge.weight_edge_24,
        weight_participation=base_cfg.teq_edge.weight_participation,
        use_calibrated=False,
    )
    cfg = replace(base_cfg, teq_edge=teq_cfg)

    df = fetch_ohlcv_df(args)
    bundle = prepare_bar_series_from_args(df, args)
    valid_idx = bundle.valid_idx
    start_idx = max(int(valid_idx.min()), args.context_bars + 1)
    end_idx = min(int(valid_idx.max()), len(df) - 2)

    provider = ModelSignalProvider.from_checkpoint(
        checkpoint=args.checkpoint,
        bars=bundle.bars,
        df=df,
        context_bars=args.context_bars,
        d_model=args.d_model,
        n_heads=args.n_heads,
        trunk_layers=args.trunk_layers,
        trend_features=args.trend_features,
        trend_windows=tuple(args.trend_windows),
        max_seg_len=args.max_seg_len,
        max_segments=args.max_segments,
        min_seg_len=args.min_seg_len,
        num_codes=args.num_codes,
        vq_beta=args.vq_beta,
        vq_inverse_freq_ema=args.vq_inverse_freq_ema,
        cfg=cfg,
        device=args.device,
    )

    label_by_idx = labels.set_index("bar_idx")
    raw_long: list[float] = []
    ideal_long: list[float] = []
    raw_short: list[float] = []
    ideal_short: list[float] = []
    part_long_scores: list[float] = []
    part_short_scores: list[float] = []
    legacy_edges: list[float] = []

    for idx in range(start_idx, end_idx + 1):
        if idx not in label_by_idx.index:
            continue
        row = label_by_idx.loc[idx]
        if int(row.get("is_leg_confirmed", 0)) != 1:
            continue
        sig = provider.signal_at(idx)
        part_long = sig.participate_score_long
        part_short = sig.participate_score_short
        edge_5 = float(sig.pred_cum_ret_5 or 0.0)
        edge_24 = sig.edge_long_hz if sig.edge_long_hz else edge_5
        teq_long_raw, teq_short_raw = compute_teq_edge_raw(
            edge_5=edge_5,
            edge_24=edge_24,
            participate_score_long=part_long,
            participate_score_short=part_short,
            cfg=teq_cfg,
        )
        raw_long.append(teq_long_raw)
        ideal_long.append(float(row.get("ideal_participate_long", 0.0)))
        raw_short.append(teq_short_raw)
        ideal_short.append(float(row.get("ideal_participate_short", 0.0)))
        part_long_scores.append(part_long)
        part_short_scores.append(part_short)
        legacy_edges.append(float(sig.edge))

    if not raw_long:
        raise RuntimeError("no confirmed-leg valid bars for calibration")

    calibrator = fit_teq_edge_calibrator(
        np.asarray(raw_long, dtype=np.float64),
        np.asarray(ideal_long, dtype=np.float64),
        np.asarray(raw_short, dtype=np.float64),
        np.asarray(ideal_short, dtype=np.float64),
        part_long=np.asarray(part_long_scores, dtype=np.float64),
        part_short=np.asarray(part_short_scores, dtype=np.float64),
        legacy_edge_long=np.asarray(legacy_edges, dtype=np.float64),
        legacy_edge_short=np.asarray(legacy_edges, dtype=np.float64),
    )
    out_path = Path(args.output)
    calibrator.save(out_path)
    summary = {
        "checkpoint": args.checkpoint,
        "labels": str(labels_path),
        "confirmed_bars": len(raw_long),
        "ideal_long_rate": float(np.mean(ideal_long)),
        "ideal_short_rate": float(np.mean(ideal_short)),
        "calibrator": calibrator.to_dict(),
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved calibration: {out_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
