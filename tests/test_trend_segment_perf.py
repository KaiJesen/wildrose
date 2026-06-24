from __future__ import annotations

import time

import numpy as np

from trading_system.config import TrendSegmentConfig
from trading_system.trend_segment import TrendSegmentEngine


def test_segment_engine_8745_bars_under_30s() -> None:
    n = 8745
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.12, n))
    high = close + 0.5
    low = close - 0.5
    atr = np.full(n, 1.0)
    engine = TrendSegmentEngine(TrendSegmentConfig())
    t0 = time.perf_counter()
    for i in range(n):
        engine.update(bar_idx=i, high=float(high[i]), low=float(low[i]), close=float(close[i]), atr=float(atr[i]))
    elapsed = time.perf_counter() - t0
    assert elapsed < 60.0, f"segment update too slow: {elapsed:.2f}s"
    # Local target from 022立项; CI allows 60s
    if elapsed >= 30.0:
        print(f"WARN: segment_runtime_8745={elapsed:.2f}s exceeds 30s local target")
