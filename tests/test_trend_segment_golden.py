from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trading_system.config import TrendSegmentConfig
from trading_system.trend_segment import TrendSegmentEngine

_GOLDEN_PATH = Path(__file__).resolve().parent / "fixtures" / "trend_segment_golden_500.json"


def _synthetic_series(n: int = 500) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    close = 100.0 + np.cumsum(rng.normal(0.05, 0.18, n))
    for i in range(80, 140):
        close[i] = close[i - 1] + 0.25
    for i in range(220, 280):
        close[i] = close[i - 1] - 0.22
    high = close + 0.45
    low = close - 0.35
    atr = np.full(n, 1.0)
    return close, high, low, atr


def _run_snapshot(n: int = 500) -> list[dict]:
    close, high, low, atr = _synthetic_series(n)
    engine = TrendSegmentEngine(TrendSegmentConfig())
    rows: list[dict] = []
    for i in range(n):
        ctx = engine.update(bar_idx=i, high=float(high[i]), low=float(low[i]), close=float(close[i]), atr=float(atr[i]))
        if i % 25 != 0 and i != n - 1:
            continue
        rows.append(
            {
                "bar_idx": i,
                "leg_type": ctx.leg_type.value,
                "sub_phase": ctx.sub_phase.value,
                "leg_state": ctx.leg_state.value,
                "bars_since_leg_start": ctx.bars_since_leg_start,
                "regime": ctx.regime.value,
            }
        )
    return rows


def test_segment_golden_snapshot_matches() -> None:
    if not _GOLDEN_PATH.is_file():
        _GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN_PATH.write_text(json.dumps(_run_snapshot(), indent=2), encoding="utf-8")
    expected = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))
    actual = _run_snapshot()
    assert actual == expected
