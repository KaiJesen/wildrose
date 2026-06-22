from __future__ import annotations

from datetime import datetime

from trading_system.config import TradingSystemConfig
from trading_system.enums import ActionType, Side
from trading_system.portfolio import PortfolioState
from trading_system.risk import RiskManager
from trading_system.rules import RuleEngine, TradingAction
from trading_system.signal import TradingSignal
from trading_system.sizing import PositionSizer
from trading_system.trend import TrendContext


def _sig(**kwargs) -> TradingSignal:
    sig = TradingSignal(
        ts=datetime(2026, 1, 1),
        price=100.0,
        atr=2.0,
        p_up=0.2,
        p_down=0.5,
        p_flat=0.3,
        p_risk=0.2,
        pred_ret_1=-0.001,
        pred_ret_2=-0.001,
        pred_ret_3=-0.001,
        pred_ret_4=-0.001,
        pred_ret_5=-0.001,
        pred_cum_ret_5=-0.005,
        source="test",
    )
    for k, v in kwargs.items():
        setattr(sig, k, v)
    return sig.finalize(0.45)


def _downtrend(strong: bool = False) -> TrendContext:
    return TrendContext(
        is_downtrend=True,
        is_strong_downtrend=strong,
        is_uptrend=False,
        trend_score=2.0,
        ret_3_atr=-1.0,
        ret_6_atr=-2.0 if strong else -1.4,
        ema_fast=98.0,
        ema_slow=101.0,
        breakdown_low_n=True,
        lower_high_low=True,
        reason_codes=["DOWN_RET6_ATR", "EMA_FAST_LT_EMA_SLOW"],
    )


def test_sentinel_short_open_in_strong_downtrend() -> None:
    cfg = TradingSystemConfig()
    rule_engine = RuleEngine(cfg)
    sig = _sig(p_up=0.32, p_down=0.34, p_flat=0.34, p_risk=0.2, pred_cum_ret_5=-0.001, price=95.0)
    action = rule_engine.decide(sig, PortfolioState(), trend_context=_downtrend(strong=True))
    assert action.action == ActionType.OPEN_SHORT
    assert action.reason_code == "OPEN_SHORT_SENTINEL"


def test_block_long_in_downtrend() -> None:
    cfg = TradingSystemConfig()
    risk = RiskManager(cfg)
    action = TradingAction(ActionType.OPEN_LONG, Side.LONG, "OPEN_LONG_SIGNAL")
    sig = _sig(p_up=0.7, p_down=0.1, p_flat=0.2, pred_cum_ret_5=0.01)
    out = risk.validate_action(action, sig, PortfolioState(), trend_context=_downtrend())
    assert out.action == ActionType.BLOCK
    assert out.reason_code == "BLOCK_LONG_DOWNTREND"


def test_short_exit_hysteresis_requires_two_bars() -> None:
    cfg = TradingSystemConfig()
    rule_engine = RuleEngine(cfg)
    pf = PortfolioState()
    pf.position.side = Side.SHORT
    pf.position.position_ratio = 0.05
    sig = _sig(pred_cum_ret_5=0.01, p_up=0.6, p_down=0.2)
    a1 = rule_engine.decide(sig, pf, trend_context=_downtrend())
    assert a1.action == ActionType.HOLD
    assert a1.reason_code == "HOLD_SHORT_REVERSE_WAIT_CONFIRM"
    a2 = rule_engine.decide(sig, pf, trend_context=_downtrend())
    assert a2.action == ActionType.CLOSE
    assert a2.reason_code == "CLOSE_SHORT_REVERSE_CONFIRMED"


def test_sentinel_short_blocked_by_high_risk() -> None:
    cfg = TradingSystemConfig()
    risk = RiskManager(cfg)
    sig = _sig(p_risk=0.8, p_up=0.36, p_down=0.4)
    action = TradingAction(ActionType.OPEN_SHORT, Side.SHORT, "OPEN_SHORT_SENTINEL")
    out = risk.validate_action(action, sig, PortfolioState(), trend_context=_downtrend(strong=True))
    assert out.action == ActionType.BLOCK
    assert out.reason_code == "BLOCK_OPEN_RISK_HIGH"


def test_sentinel_short_position_cap_applied() -> None:
    cfg = TradingSystemConfig()
    sizer = PositionSizer(cfg)
    sig = _sig(p_up=0.31, p_down=0.33, p_flat=0.3)
    action = TradingAction(ActionType.OPEN_SHORT, Side.SHORT, "OPEN_SHORT_SENTINEL")
    sized = sizer.apply(action, sig, PortfolioState(), trend_context=_downtrend(strong=True))
    assert sized.position_ratio <= cfg.sentinel_short.sentinel_max_position_ratio
    assert sized.position_ratio <= cfg.base.max_position_ratio


def test_hard_stop_still_has_priority() -> None:
    cfg = TradingSystemConfig()
    risk = RiskManager(cfg)
    pf = PortfolioState()
    pf.position.side = Side.SHORT
    pf.position.stop_price = 100.0
    event = risk.check_pre_decision_risk(_sig(), pf, bar_index=100, current_price=101.0)
    assert event.force_close
    assert event.reason == "CLOSE_HARD_STOP"

