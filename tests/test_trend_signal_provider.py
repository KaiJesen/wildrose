from __future__ import annotations

from trading_system.config import TrendSignalConfig
from trading_system.trend_signal import ConfirmTier, TrendDirection, TrendMemory, TrendSignalProvider


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
    cfg = TrendSignalConfig(enabled=True, ema_fast=6, ema_mid=12, ema_slow=18, chop_ema_slope_bars=24)
    provider = TrendSignalProvider(cfg)
    memory = TrendMemory()
    close = [100 + (0.2 if i % 2 == 0 else -0.2) for i in range(90)]
    high = [c + 0.2 for c in close]
    low = [c - 0.2 for c in close]
    atr = [0.8 for _ in close]
    out = provider.compute(close_hist=close, high_hist=high, low_hist=low, atr_hist=atr, memory=memory)
    assert out.direction == TrendDirection.NONE
    assert "CHOP_HARD" in out.reason_codes or out.confirm_tier == ConfirmTier.NONE


def test_slow_uptrend_proxy_avoids_chop_hard() -> None:
    cfg = TrendSignalConfig(enabled=True, ema_fast=6, ema_mid=12, ema_slow=18, chop_ema_slope_bars=24)
    provider = TrendSignalProvider(cfg)
    memory = TrendMemory()
    close = [100.0 + i * 0.05 for i in range(120)]
    high = [c + 0.15 for c in close]
    low = [c - 0.15 for c in close]
    atr = [0.9 for _ in close]
    out = provider.compute(close_hist=close, high_hist=high, low_hist=low, atr_hist=atr, memory=memory)
    assert "CHOP_HARD" not in out.reason_codes
    assert out.direction in (TrendDirection.UP, TrendDirection.NONE)

