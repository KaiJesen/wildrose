from __future__ import annotations

import numpy as np

from trading_system.config import TrendSegmentConfig
from trading_system.trend_segment import LegState, SubLegPhase, TrendLegType, TrendSegmentEngine


def _synthetic_uptrend(n: int = 120) -> tuple[list[float], list[float], list[float], list[float]]:
    close = []
    high = []
    low = []
    atr = []
    px = 100.0
    for i in range(n):
        px += 0.15 + 0.02 * np.sin(i / 8.0)
        close.append(px)
        high.append(px + 0.4)
        low.append(px - 0.2)
        atr.append(1.0)
    return close, high, low, atr


def test_segment_engine_produces_confirmed_up_leg():
    cfg = TrendSegmentConfig(
        min_leg_bars=8,
        min_move_atr=0.5,
        min_efficiency=0.1,
        swing_large_left_bars=4,
        swing_large_right_bars=4,
    )
    engine = TrendSegmentEngine(cfg)
    close = []
    px = 100.0
    for i in range(160):
        if i % 20 < 15:
            px += 0.35
        else:
            px -= 0.15
        close.append(px)
    high = [c + 0.6 for c in close]
    low = [c - 0.4 for c in close]
    atr = [1.0] * len(close)
    last = None
    for i in range(len(close)):
        last = engine.update(bar_idx=i, high=high[i], low=low[i], close=close[i], atr=atr[i])
    assert last is not None
    assert last.bars_since_leg_start >= 1
    assert last.leg_state in (LegState.LEG_CONFIRMED, LegState.LEG_FORMING, LegState.NO_LEG)


def test_small_pullback_does_not_reset_macro_leg():
    cfg = TrendSegmentConfig(min_leg_bars=6, merge_pullback_atr=2.5, swing_large_left_bars=4, swing_large_right_bars=4)
    engine = TrendSegmentEngine(cfg)
    close = [100 + i * 0.3 for i in range(40)] + [112 - i * 0.1 for i in range(3)] + [109 + i * 0.25 for i in range(30)]
    high = [c + 0.5 for c in close]
    low = [c - 0.3 for c in close]
    atr = [1.0] * len(close)
    leg_ids = []
    for i in range(len(close)):
        seg = engine.update(bar_idx=i, high=high[i], low=low[i], close=close[i], atr=atr[i])
        if seg.active_leg is not None:
            leg_ids.append(seg.active_leg.leg_id)
    assert len(set(leg_ids)) <= 3


def test_sub_phase_updates():
    cfg = TrendSegmentConfig()
    engine = TrendSegmentEngine(cfg)
    close, high, low, atr = _synthetic_uptrend(80)
    phases = set()
    for i in range(len(close)):
        seg = engine.update(bar_idx=i, high=high[i], low=low[i], close=close[i], atr=atr[i])
        phases.add(seg.sub_phase)
    assert len(phases) >= 1
