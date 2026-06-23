#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EX = Path(__file__).resolve().parent
_ROOT = _EX.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))

from _train_common import add_data_args, apply_real_data_defaults, fetch_ohlcv_df
from best_point.labeler import LabelerConfig, build_best_point_labels, save_label_outputs
from market_data.schema import COL_TIME


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build hindsight best-point labels (v017)")
    add_data_args(p)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--leverage", type=float, default=20.0)
    p.add_argument("--min-net-roi", type=float, default=0.03)
    p.add_argument("--max-holding-bars", type=int, default=72)
    p.add_argument("--min-holding-bars", type=int, default=1)
    p.add_argument("--cooldown-after-trade", type=int, default=0)
    p.add_argument("--allow-long", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow-short", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mode", choices=("major_legs", "dp"), default="major_legs")
    p.add_argument("--zigzag-min-move-atr", type=float, default=1.8)
    p.add_argument("--zigzag-atr-period", type=int, default=14)
    p.add_argument("--merge-pullback-atr", type=float, default=2.0)
    p.add_argument("--min-leg-bars", type=int, default=2)
    p.add_argument("--out-dir", default="data/labels/best_point_v017_major_legs")
    p.add_argument("--prefix", default="BTCUSDT_1h")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    apply_real_data_defaults(args)
    df = fetch_ohlcv_df(args).reset_index(drop=True)
    labels, trades, summary = build_best_point_labels(
        df,
        fee_rate=args.fee_rate,
        leverage=args.leverage,
        min_net_roi=args.min_net_roi,
        max_holding_bars=args.max_holding_bars,
        allow_long=args.allow_long,
        allow_short=args.allow_short,
        price_field="close",
        cfg=LabelerConfig(
            mode=args.mode,
            zigzag_min_move_atr=args.zigzag_min_move_atr,
            zigzag_atr_period=args.zigzag_atr_period,
            merge_pullback_atr=args.merge_pullback_atr,
            min_leg_bars=args.min_leg_bars,
        ),
        min_holding_bars=args.min_holding_bars,
        cooldown_after_trade=args.cooldown_after_trade,
    )
    labels.insert(0, COL_TIME, df[COL_TIME].to_numpy())
    out = save_label_outputs(labels=labels, trades=trades, summary=summary, out_dir=args.out_dir, prefix=args.prefix)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

