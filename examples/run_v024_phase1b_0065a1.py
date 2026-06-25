#!/usr/bin/env python3
"""024 Phase 1b: train 0065a-1 from tuned 0065a-0 best checkpoint."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
V0_BEST = _ROOT / "backtest/v024_phase1_tune/s1_os30_pw15/checkpoint/market_state_best.pt"
FALLBACK = _ROOT / "checkpoints/0065a_leg_align_v0/market_state_best.pt"


def main() -> int:
    init = V0_BEST if V0_BEST.is_file() else FALLBACK
    if not init.is_file():
        raise FileNotFoundError(f"missing 0065a-0 checkpoint: {init}")
    cmd = [
        sys.executable,
        "examples/train_market_state_0065a.py",
        "--variant",
        "1",
        "--init-checkpoint",
        str(init.relative_to(_ROOT)),
        "--epochs",
        "12",
        "--early-stop-patience",
        "5",
        "--batch-size",
        "32",
    ]
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=_ROOT)
    subprocess.check_call([
        sys.executable,
        "examples/eval_model_participation.py",
        "--checkpoint",
        "checkpoints/0065a_leg_align_v1/market_state_best.pt",
        "--output",
        "backtest/v024_phase1/eval_model_participation_v1.json",
    ], cwd=_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
