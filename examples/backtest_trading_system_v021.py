#!/usr/bin/env python3
"""Backtest entry for v021 trend-bias phased rollout."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_VARIANT_CONFIG = {
    "observe": "configs/trading_rule_v021_observe_0062e.json",
    "open_bias": "configs/trading_rule_v021_open_bias_0062e.json",
    "open_size_bias": "configs/trading_rule_v021_open_size_bias_0062e.json",
    "full_bias": "configs/trading_rule_v021_full_bias_0062e.json",
    "legacy": "configs/trading_rule_v021_trend_bias_0062e.json",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest trading system v021 trend bias")
    p.add_argument(
        "--variant",
        choices=sorted(_VARIANT_CONFIG),
        default="observe",
        help="021 rollout phase config preset",
    )
    p.add_argument("--config", default="", help="override config path")
    p.add_argument("--output-dir", default="", help="override output directory")
    return p.parse_known_args()


def main() -> int:
    args, rest = parse_args()
    config = args.config or _VARIANT_CONFIG[args.variant]
    out_dir = args.output_dir or f"backtest/backtest_v021_{args.variant}_test"
    cmd = [
        sys.executable,
        "examples/backtest_trading_system_v014.py",
        "--config",
        config,
        "--output-dir",
        out_dir,
    ]
    cmd.extend(rest)
    print(f"v021 variant={args.variant} config={config} out={out_dir}")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
