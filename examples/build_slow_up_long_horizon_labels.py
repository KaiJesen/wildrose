#!/usr/bin/env python3
"""Build v019 slow-up long-horizon labels for offline detector evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_EX = Path(__file__).resolve().parent
_ROOT = _EX.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))

from _train_common import add_data_args, apply_real_data_defaults, fetch_ohlcv_df
from best_point.labeler import LabelerConfig, build_slow_up_long_horizon_labels, save_label_outputs
from market_data.schema import COL_TIME


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build v019 slow-up long-horizon labels")
    add_data_args(p)
    p.add_argument("--config", default="configs/best_point_signal_v019_slow_up_long_horizon.json")
    p.add_argument("--out-dir", default="data/labels/best_point_v019_slow_up_long_horizon")
    p.add_argument("--prefix", default="BTCUSDT_1h")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    apply_real_data_defaults(args)
    cfg_payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    label_cfg = cfg_payload.get("label", {})
    df = fetch_ohlcv_df(args).reset_index(drop=True)
    labels, trades, summary = build_slow_up_long_horizon_labels(
        df,
        fee_rate=float(label_cfg.get("fee_rate", 0.0004)),
        leverage=float(label_cfg.get("leverage", 20.0)),
        min_net_roi=float(label_cfg.get("min_net_roi", 0.10)),
        min_holding_bars=int(label_cfg.get("min_holding_bars", 8)),
        max_holding_bars=int(label_cfg.get("max_holding_bars", 72)),
        max_adverse_excursion_ratio=float(label_cfg.get("max_adverse_excursion_ratio", 0.5)),
        cooldown_after_trade=int(label_cfg.get("cooldown_after_trade", 3)),
        cfg=LabelerConfig(
            pre_entry_bars=int(label_cfg.get("pre_entry_bars", 2)),
            post_entry_bars=int(label_cfg.get("post_entry_bars", 2)),
            pre_exit_bars=int(label_cfg.get("pre_exit_bars", 2)),
            post_exit_bars=int(label_cfg.get("post_exit_bars", 1)),
        ),
    )
    labels.insert(0, COL_TIME, df[COL_TIME].to_numpy())
    paths = save_label_outputs(labels=labels, trades=trades, summary=summary, out_dir=args.out_dir, prefix=args.prefix)
    print(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
