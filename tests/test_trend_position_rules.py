from __future__ import annotations

from datetime import datetime

from trading_system.config import TradingSystemConfig
from trading_system.enums import ActionType, Side
from trading_system.portfolio import PortfolioState
from trading_system.rules import RuleEngine
from trading_system.signal import TradingSignal
from trading_system.trend_signal import TrendDirection, TrendPhase, TrendSignal, TrendStrength


def _sig(price: float = 100.0) -> TradingSignal:
    return TradingSignal(
        ts=datetime(2026, 1, 1),
        price=price,
        atr=2.0,
        p_up=0.3,
        p_down=0.3,
        p_flat=0.4,
        p_risk=0.2,
        pred_ret_1=0.0,
        pred_ret_2=0.0,
        pred_ret_3=0.0,
        pred_ret_4=0.0,
        pred_ret_5=0.0,
        pred_cum_ret_5=0.0,
    ).finalize(0.45)


def _trend(direction: TrendDirection, phase: TrendPhase, confirmed: bool = True) -> TrendSignal:
    return TrendSignal(
        direction=direction,
        strength=TrendStrength.STRONG if confirmed else TrendStrength.WEAK,
        phase=phase,
        score_up=5.0 if direction == TrendDirection.UP else 1.0,
        score_down=5.0 if direction == TrendDirection.DOWN else 1.0,
        score_abs=5.0 if confirmed else 2.0,
        confidence=0.8,
        trend_age=6,
        invalid_count=0,
        is_confirmed=confirmed,
        is_broken=False,
        is_accelerating=phase == TrendPhase.ACCELERATION,
        is_exhausted=phase == TrendPhase.EXHAUSTION,
        ret_6_atr=2.0 if direction == TrendDirection.UP else -2.0,
        ret_12_atr=3.0 if direction == TrendDirection.UP else -3.0,
        ret_24_atr=4.0 if direction == TrendDirection.UP else -4.0,
        ema_fast=100.0,
        ema_mid=99.0,
        ema_slow=98.0,
        close_to_ema_fast_atr=1.0,
        distance_from_ema_slow_atr=2.0,
        rolling_high_break=direction == TrendDirection.UP,
        rolling_low_break=direction == TrendDirection.DOWN,
        higher_high_low=direction == TrendDirection.UP,
        lower_high_low=direction == TrendDirection.DOWN,
        persistence_ratio=0.8,
        range_expansion=1.6,
        reason_codes=[],
    )


def test_profitable_short_upgrades_to_trend_short() -> None:
    cfg = TradingSystemConfig()
    engine = RuleEngine(cfg)
    pf = PortfolioState()
    pf.position.side = Side.SHORT
    pf.position.entry_price = 100.0
    pf.position.position_ratio = 0.08
    sig = _sig(price=95.0)
    action = engine.decide(sig, pf, trend_signal=_trend(TrendDirection.DOWN, TrendPhase.CONTINUATION))
    assert action.reason_code == "UPGRADE_TO_TREND_SHORT"


def test_trend_exhaustion_triggers_reduce() -> None:
    cfg = TradingSystemConfig()
    engine = RuleEngine(cfg)
    pf = PortfolioState()
    pf.position.side = Side.LONG
    pf.position.hold_mode = "TREND"
    pf.position.position_ratio = 0.1
    pf.position.entry_price = 100.0
    pf.position.trend_peak_score = 6.0
    sig = _sig(price=105.0)
    action = engine.decide(sig, pf, trend_signal=_trend(TrendDirection.UP, TrendPhase.EXHAUSTION))
    assert action.action == ActionType.REDUCE
    assert action.reason_code == "REDUCE_TREND_EXHAUSTION"

