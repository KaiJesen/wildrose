from __future__ import annotations

from trading_system.config import TrendConfig
from trading_system.trend import TrendRegimeFilter


def test_downtrend_detected_with_ema_and_price_structure() -> None:
    cfg = TrendConfig(enabled=True, ema_fast=3, ema_slow=6, ret_lookback_fast=3, ret_lookback_slow=6, breakdown_lookback=6)
    filt = TrendRegimeFilter(cfg)
    close = [110, 108, 106, 104, 102, 100, 98, 96, 94, 92]
    high = [c + 1.0 for c in close]
    low = [c - 1.0 for c in close]
    tc = filt.compute(close, high, low, atr=2.0)
    assert tc.is_downtrend
    assert tc.ret_6_atr < 0
    assert tc.ema_fast < tc.ema_slow

