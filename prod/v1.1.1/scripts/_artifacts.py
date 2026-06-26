"""Frozen B0 artifacts for prod/v1.1.1 (self-contained paths)."""
from __future__ import annotations

import hashlib
from pathlib import Path

_PROD_ROOT = Path(__file__).resolve().parents[1]

FROZEN_KLINE_CSV = _PROD_ROOT / "data/kline/binance_vision_BTCUSDT_1h_365d_end20260625.csv"
PW20_CKPT = _PROD_ROOT / "checkpoint/market_state_best.pt"
TEQ_CALIBRATION = _PROD_ROOT / "calibration/teq_edge_calibration.json"
B0_CONFIG = _PROD_ROOT / "config/trading_rule.json"
EXPECTED_PW20_CKPT_HASH_PREFIX = "82ca51cf637a258c"


def kline_backtest_args() -> list[str]:
    if not FROZEN_KLINE_CSV.is_file():
        raise FileNotFoundError(f"missing frozen kline: {FROZEN_KLINE_CSV}")
    return ["--csv", str(FROZEN_KLINE_CSV)]


def sha256_prefix(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def verify_pw20_checkpoint() -> str:
    if not PW20_CKPT.is_file():
        raise FileNotFoundError(f"missing checkpoint: {PW20_CKPT}")
    digest = sha256_prefix(PW20_CKPT)
    if digest != EXPECTED_PW20_CKPT_HASH_PREFIX:
        print(
            f"warning: checkpoint hash {digest} != expected {EXPECTED_PW20_CKPT_HASH_PREFIX}"
        )
    return digest
