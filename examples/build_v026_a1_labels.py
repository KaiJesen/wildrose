#!/usr/bin/env python3
"""026 Phase 2 A1: build three-tier ordered participation labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

_EX = Path(__file__).resolve().parent
_ROOT = _EX.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))

from _train_common import add_data_args, add_feature_args, add_segment_args, apply_real_data_defaults, fetch_ohlcv_df, prepare_bar_series_from_args
from market_data.schema import COL_CLOSE, COL_HIGH, COL_LOW
from trading_system.adapters.market_state_model import compute_atr
from trading_system.config import load_config
from trading_system.leg_participation_labels import (
    LegParticipationLabelConfig,
    LegParticipationLabelMetadata,
    compute_ideal_participation_labels,
    label_summary,
    replay_segment_bars,
)

DEFAULT_CONFIG = "configs/trading_rule_v022_trend_quality_0062e.json"
DEFAULT_OUT = "data/labels/leg_participation_a1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_idx(bundle, split: str):
    if split == "train":
        return bundle.train_idx
    if split == "valid":
        return bundle.valid_idx
    return bundle.test_idx


def build_labels_for_split(
    *,
    split: str,
    df,
    bundle,
    cfg,
    label_cfg: LegParticipationLabelConfig,
    out_dir: Path,
    context_bars: int,
) -> dict:
    idx = _split_idx(bundle, split)
    start_idx = max(int(idx.min()), context_bars + 1)
    end_idx = min(int(idx.max()), len(df) - 2)
    eval_idx = idx[(idx >= start_idx) & (idx <= end_idx)]

    bars = replay_segment_bars(df, cfg, idx=eval_idx, start_bar=start_idx)
    high = df[COL_HIGH].to_numpy(dtype=np.float64)
    low = df[COL_LOW].to_numpy(dtype=np.float64)
    close = df[COL_CLOSE].to_numpy(dtype=np.float64)
    atr_arr = compute_atr(high, low, close, cfg.execution.atr_period)

    labels = compute_ideal_participation_labels(
        bars,
        close,
        high,
        low,
        atr_arr,
        label_cfg=label_cfg,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"leg_participation_{split}.csv"
    labels.to_csv(csv_path, index=False)

    summary = label_summary(labels)
    summary["split"] = split
    summary["csv_path"] = str(csv_path.relative_to(_ROOT))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build 026 A1 three-tier leg participation labels")
    add_data_args(p)
    add_feature_args(p)
    add_segment_args(p)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--split", action="append", default=[], choices=["train", "valid", "test"])
    p.add_argument("--output-dir", default=DEFAULT_OUT)
    p.add_argument("--fee-bps", type=float, default=4.0)
    p.add_argument("--slippage-bps", type=float, default=2.0)
    p.set_defaults(source="binance_vision", symbol="BTCUSDT", interval="1h", days=365)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    apply_real_data_defaults(args)
    splits = args.split or ["train", "valid", "test"]

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (_ROOT / cfg_path).resolve()
    cfg = load_config(cfg_path)
    label_cfg = LegParticipationLabelConfig(
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        enable_a1_tiers=True,
    )

    df = fetch_ohlcv_df(args)
    bundle = prepare_bar_series_from_args(df, args)
    out_dir = Path(args.output_dir).resolve()

    split_summaries: dict[str, dict] = {}
    for split in splits:
        print(f"building A1 labels: split={split}")
        split_summaries[split] = build_labels_for_split(
            split=split,
            df=df,
            bundle=bundle,
            cfg=cfg,
            label_cfg=label_cfg,
            out_dir=out_dir,
            context_bars=args.context_bars,
        )
        s = split_summaries[split]
        print(
            f"  bars={int(s['bar_count'])} tier2_long={s.get('participate_tier2_long_rate', 0):.4f} "
            f"tier1_long={s.get('participate_tier1_long_rate', 0):.4f}"
        )

    metadata = LegParticipationLabelMetadata(
        trend_signal_config_sha256=_sha256(cfg_path),
        rule_config_sha256=_sha256(cfg_path),
        teacher_label_version="leg_participation_v026_a1",
        label_config={
            "fee_bps": label_cfg.fee_bps,
            "slippage_bps": label_cfg.slippage_bps,
            "mae_atr_limit": label_cfg.mae_atr_limit,
            "max_entry_progress": label_cfg.max_entry_progress,
            "tier1_mae_atr_limit": label_cfg.tier1_mae_atr_limit,
            "tier1_max_entry_progress": label_cfg.tier1_max_entry_progress,
            "enable_a1_tiers": True,
            "round_trip_cost": label_cfg.round_trip_cost,
        },
    )
    meta_path = out_dir / "metadata.json"
    meta_doc = {
        **asdict(metadata),
        "config_path": str(cfg_path.relative_to(_ROOT)),
        "splits": split_summaries,
    }
    meta_path.write_text(json.dumps(meta_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
