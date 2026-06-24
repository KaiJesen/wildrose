#!/usr/bin/env python3
"""Backtest entry for v022 trend-quality rollout."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_DEFAULT_CONFIG = "configs/trading_rule_v022_trend_quality_0062e.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest trading system v022 trend quality")
    p.add_argument("--config", default=_DEFAULT_CONFIG)
    p.add_argument("--output-dir", default="backtest/022_trend_quality_test")
    return p.parse_known_args()


def main() -> int:
    args, rest = parse_args()
    cmd = [
        sys.executable,
        "examples/backtest_trading_system_v014.py",
        "--config",
        args.config,
        "--output-dir",
        args.output_dir,
    ]
    cmd.extend(rest)
    print(f"v022 config={args.config} out={args.output_dir}")
    return subprocess.call(cmd, cwd=Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    raise SystemExit(main())
