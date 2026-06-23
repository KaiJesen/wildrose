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
    p.add_argument("--allow-long", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow-short", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out-dir", default="data/labels/best_point_v017")
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
        cfg=LabelerConfig(),
    )
    labels.insert(0, COL_TIME, df[COL_TIME].to_numpy())
    out = save_label_outputs(labels=labels, trades=trades, summary=summary, out_dir=args.out_dir, prefix=args.prefix)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

