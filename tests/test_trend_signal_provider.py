from __future__ import annotations

from trading_system.config import TrendSignalConfig
from trading_system.trend_signal import TrendDirection, TrendMemory, TrendSignalProvider


def test_downtrend_confirmed_output() -> None:
    cfg = TrendSignalConfig(enabled=True, ema_fast=6, ema_mid=12, ema_slow=18)
    provider = TrendSignalProvider(cfg)
    memory = TrendMemory()
    close = [120 - i * 0.8 for i in range(90)]
    high = [c + 0.4 for c in close]
    low = [c - 0.6 for c in close]
    atr = [1.0 for _ in close]
    out = provider.compute(close_hist=close, high_hist=high, low_hist=low, atr_hist=atr, memory=memory)
    assert out.direction == TrendDirection.DOWN
    assert out.is_confirmed


def test_choppy_market_outputs_none() -> None:
    cfg = TrendSignalConfig(enabled=True, ema_fast=6, ema_mid=12, ema_slow=18)
    provider = TrendSignalProvider(cfg)
    memory = TrendMemory()
    close = [100 + (0.2 if i % 2 == 0 else -0.2) for i in range(90)]
    high = [c + 0.2 for c in close]
    low = [c - 0.2 for c in close]
    atr = [0.8 for _ in close]
    out = provider.compute(close_hist=close, high_hist=high, low_hist=low, atr_hist=atr, memory=memory)
    assert out.direction == TrendDirection.NONE

