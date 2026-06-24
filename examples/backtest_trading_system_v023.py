#!/usr/bin/env python3
"""Backtest entry for v023 baseline (frozen v022 recipe)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_DEFAULT_CONFIG = "configs/trading_rule_v023_baseline_0062e.json"
_DEFAULT_CKPT = "prod/v0.0.0/checkpoint/market_state_best.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest trading system v023 baseline")
    p.add_argument("--config", default=_DEFAULT_CONFIG)
    p.add_argument("--checkpoint", default=_DEFAULT_CKPT)
    p.add_argument("--output-dir", default="backtest/v023_baseline/test")
    return p.parse_known_args()


def main() -> int:
    args, rest = parse_args()
    cmd = [
        sys.executable,
        "examples/backtest_trading_system_v014.py",
        "--config",
        args.config,
        "--checkpoint",
        args.checkpoint,
        "--output-dir",
        args.output_dir,
    ]
    cmd.extend(rest)
    print(f"v023 config={args.config} ckpt={args.checkpoint} out={args.output_dir}")
    return subprocess.call(cmd, cwd=Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    raise SystemExit(main())
