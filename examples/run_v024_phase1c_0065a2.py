#!/usr/bin/env python3
"""024 Phase 1c: train 0065a-2 from 0065a-1 best checkpoint."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
INIT = _ROOT / "checkpoints/0065a_leg_align_v1/market_state_best.pt"


def main() -> int:
    if not INIT.is_file():
        raise FileNotFoundError(f"run 0065a-1 first: {INIT}")
    cmd = [
        sys.executable,
        "examples/train_market_state_0065a.py",
        "--variant",
        "2",
        "--init-checkpoint",
        str(INIT.relative_to(_ROOT)),
        "--epochs",
        "12",
        "--early-stop-patience",
        "5",
        "--batch-size",
        "32",
    ]
    subprocess.check_call(cmd, cwd=_ROOT)
    subprocess.check_call([
        sys.executable,
        "examples/eval_model_participation.py",
        "--checkpoint",
        "checkpoints/0065a_leg_align_v2/market_state_best.pt",
        "--output",
        "backtest/v024_phase1/eval_model_participation_v2.json",
    ], cwd=_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
