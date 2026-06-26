"""025 frozen artifacts for cross-machine reproducibility."""

from __future__ import annotations

import hashlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PROD_V111 = _ROOT / "prod/v1.1.1"

# 024 B0 backtests used this cache snapshot (see REPORT_024_CONSTRAINED_FINAL).
_REPO_KLINE = _ROOT / "data/cache/kline/binance_vision_BTCUSDT_1h_365d_end20260625.csv"
_PROD_KLINE = _PROD_V111 / "data/kline/binance_vision_BTCUSDT_1h_365d_end20260625.csv"
FROZEN_KLINE_CSV = _REPO_KLINE if _REPO_KLINE.is_file() else _PROD_KLINE

_REPO_CKPT = _ROOT / "checkpoints/0065a_leg_align_c1_pw20/market_state_best.pt"
_PROD_CKPT = _PROD_V111 / "checkpoint/market_state_best.pt"
PW20_CKPT = _REPO_CKPT if _REPO_CKPT.is_file() else _PROD_CKPT

_REPO_CALIB = _ROOT / "backtest/v024_constrained/teq_edge_calibration.json"
_PROD_CALIB = _PROD_V111 / "calibration/teq_edge_calibration.json"
TEQ_CALIBRATION = _REPO_CALIB if _REPO_CALIB.is_file() else _PROD_CALIB

# Observed on 024 original machine; used as a presence/consistency hint only.
EXPECTED_PW20_CKPT_HASH_PREFIX = "82ca51cf637a258c"


def kline_backtest_args() -> list[str]:
    if not FROZEN_KLINE_CSV.is_file():
        raise FileNotFoundError(
            f"missing frozen kline cache for 025 reproduction: {FROZEN_KLINE_CSV}\n"
            "Copy from 024 machine or run once with matching end date, then pin this path."
        )
    return ["--csv", str(FROZEN_KLINE_CSV.relative_to(_ROOT))]


def sha256_prefix(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def verify_pw20_checkpoint() -> str:
    if not PW20_CKPT.is_file():
        raise FileNotFoundError(
            f"missing 024 B0 checkpoint: {PW20_CKPT}\n"
            "Copy from original 024 machine (gitignored) or retrain with documented recipe."
        )
    digest = sha256_prefix(PW20_CKPT)
    if digest != EXPECTED_PW20_CKPT_HASH_PREFIX:
        print(
            f"warning: pw20 checkpoint hash {digest} != expected {EXPECTED_PW20_CKPT_HASH_PREFIX}; "
            "metrics may diverge from 024 B0 report"
        )
    return digest
