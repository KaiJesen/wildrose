#!/usr/bin/env python3
"""Thin wrapper for v018 lifecycle backtest."""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    cmd = [
        sys.executable,
        "examples/backtest_trading_system_v014.py",
        "--config",
        "configs/trading_rule_v018_lifecycle_0062e.json",
    ]
    cmd.extend(sys.argv[1:])
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

