#!/usr/bin/env python3
"""Build v020 teacher trend-leg labels from OHLCV."""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    cmd = [
        sys.executable,
        "tools/trend_label_tool.py",
        "--input",
        "data/cache/kline/binance_vision_BTCUSDT_1h_365d_end20260623.csv",
        "--config",
        "configs/trading_rule_v020_trend_segment_0062e.json",
        "--out-dir",
        "data/labels/trend_leg_v020_teacher",
    ]
    cmd.extend(sys.argv[1:])
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
