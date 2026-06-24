from __future__ import annotations

from dataclasses import dataclass

from trading_system.config import TradingSystemConfig
from trading_system.enums import ActionType, Side
from trading_system.portfolio import PortfolioState
from trading_system.rules import TradingAction
from trading_system.signal import TradingSignal
from trading_system.trend import TrendContext
from trading_system.trend_signal import TrendSignal
from trading_system.slow_trend import SlowTrendContext
from trading_system.trend_bias import CounterTrendLevel, TrendBiasContext


@dataclass
class SizedAction:
    action: ActionType
    target_side: Side
    reason_code: str
    position_ratio: float
    notional_exposure: float
    margin_required: float


class PositionSizer:
    _SCOPE_ORDER = {"observe": 0, "open_only": 1, "open_size": 2, "full": 3}

    def __init__(self, cfg: TradingSystemConfig) -> None:
        self.cfg = cfg

    def _bias_size_active(self) -> bool:
        tb = self.cfg.trend_bias
        if not tb.enabled:
            return False
        return self._SCOPE_ORDER.get(tb.decision_scope, 0) >= self._SCOPE_ORDER.get("open_size", 0)

    def _size_bias_multiplier(self, trend_bias: TrendBiasContext, *, side: Side) -> float:
        if side == Side.LONG:
            mult = trend_bias.size_bias_long
            counter = trend_bias.counter_level_long
            align = trend_bias.alignment_score_long
        elif side == Side.SHORT:
            mult = trend_bias.size_bias_short
            counter = trend_bias.counter_level_short
            align = trend_bias.alignment_score_short
        else:
            return 1.0
        if counter != CounterTrendLevel.NONE:
            return mult if mult < 1.0 else 1.0
        if mult > 1.0 and align < 1:
            return 1.0
        return mult

    def _apply_size_bias(
        self,
        ratio: float,
        *,
        side: Side,
        trend_bias: TrendBiasContext | None,
        reason_code: str,
    ) -> float:
        if trend_bias is None or not self._bias_size_active():
            return ratio
        tb = self.cfg.trend_bias
        mult = self._size_bias_multiplier(trend_bias, side=side)
        ratio *= mult
        if side == Side.LONG:
            if (
                tb.allow_hard_counter_probe
                and trend_bias.counter_level_long == CounterTrendLevel.HARD_BLOCK
                and reason_code.startswith("OPEN_")
            ):
                ratio = min(ratio, tb.hard_counter_probe_ratio)
        elif side == Side.SHORT:
            if (
                tb.allow_hard_counter_probe
                and trend_bias.counter_level_short == CounterTrendLevel.HARD_BLOCK
                and reason_code.startswith("OPEN_")
            ):
                ratio = min(ratio, tb.hard_counter_probe_ratio)
        return min(ratio, self.cfg.base.max_position_ratio)

    def _base_ratio_from_conf(self, conf: float, p_risk: float) -> float:
        s = self.cfg.sizing
        if conf < s.weak_conf_min:
            lo, hi = s.weak_range
        elif conf < s.medium_conf_min:
            lo, hi = s.medium_range
        elif conf < s.strong_conf_min:
            lo, hi = s.strong_range
        else:
            lo, hi = s.very_strong_range
        base = lo + (hi - lo) * min(conf / max(s.strong_conf_min, 1e-6), 1.0)
        if p_risk < self.cfg.rule.risk_open_max:
            base *= 1.0
        return min(base, self.cfg.base.max_position_ratio)

    def apply(
        self,
        action: TradingAction,
        signal: TradingSignal,
        portfolio: PortfolioState,
        trend_context: TrendContext | None = None,
        trend_signal: TrendSignal | None = None,
        slow_context: SlowTrendContext | None = None,
        trend_bias: TrendBiasContext | None = None,
    ) -> SizedAction:
        side = action.target_side
        cur = portfolio.position.position_ratio
        if action.action == ActionType.BLOCK:
            return SizedAction(action.action, Side.FLAT, action.reason_code, cur, cur * self.cfg.base.fixed_leverage, cur)
        if action.action in (ActionType.CLOSE, ActionType.FORCE_CLOSE):
            return SizedAction(action.action, Side.FLAT, action.reason_code, 0.0, 0.0, 0.0)
        if action.action == ActionType.REDUCE:
            if action.reason_code == "REDUCE_TREND_PROFIT_LOCK":
                scale = self.cfg.trend_lifecycle.runner_reduce_scale
            elif action.reason_code == "REDUCE_SLOW_UP_PROFIT_LOCK":
                scale = self.cfg.slow_up_position.runner_reduce_scale
            elif action.reason_code == "REDUCE_TREND_EXHAUSTION":
                scale = self.cfg.trend_position.exhaustion_reduce_scale
            else:
                scale = self.cfg.rule.reduce_scale
            new_ratio = max(0.0, cur * scale)
        elif action.action == ActionType.ADD:
            base = self._base_ratio_from_conf(signal.conf, signal.p_risk)
            if action.reason_code == "UPGRADE_SENTINEL_TO_MODEL_SHORT":
                new_ratio = min(self.cfg.base.max_position_ratio, max(cur, base))
            elif action.reason_code == "UPGRADE_CRASH_TO_MODEL_SHORT":
                new_ratio = min(self.cfg.base.max_position_ratio, max(cur, base))
            elif action.reason_code in ("ADD_TREND_CONTINUATION", "UPGRADE_TO_TREND_LONG", "UPGRADE_TO_TREND_SHORT"):
                new_ratio = min(self.cfg.base.max_position_ratio, cur * 1.3)
            else:
                new_ratio = min(self.cfg.base.max_position_ratio, cur + base * 0.5)
        else:
            if action.reason_code == "OPEN_SHORT_CRASH":
                c = self.cfg.crash_short
                crash_ratio = c.strong_position_ratio if (trend_context and getattr(trend_context, "is_strong_downtrend", False)) else c.position_ratio
                new_ratio = min(self.cfg.base.max_position_ratio, min(crash_ratio, c.max_position_ratio))
            elif action.reason_code == "OPEN_LONG_SLOW_TREND":
                sp = self.cfg.slow_up_position
                slow_ratio = sp.stable_position_ratio
                if slow_context is not None and not slow_context.is_stable_slow_uptrend:
                    slow_ratio = sp.position_ratio
                new_ratio = min(self.cfg.base.max_position_ratio, min(slow_ratio, sp.max_position_ratio))
            elif action.reason_code in ("OPEN_LONG_TREND_QUALIFIED", "OPEN_SHORT_TREND_QUALIFIED"):
                tq = self.cfg.trend_entry_qualifier
                new_ratio = min(self.cfg.base.max_position_ratio, min(tq.position_ratio, tq.max_position_ratio))
            elif action.reason_code == "OPEN_SHORT_SENTINEL":
                s = self.cfg.sentinel_short
                new_ratio = min(self.cfg.base.max_position_ratio, min(s.sentinel_position_ratio, s.sentinel_max_position_ratio))
            else:
                new_ratio = self._base_ratio_from_conf(signal.conf, signal.p_risk) if side != Side.FLAT else 0.0
            if portfolio.weekly_defensive_mode:
                new_ratio *= self.cfg.risk.defensive_size_scale
        new_ratio = self._apply_size_bias(
            new_ratio,
            side=side if side != Side.FLAT else action.target_side,
            trend_bias=trend_bias,
            reason_code=action.reason_code,
        )
        notional = new_ratio * self.cfg.base.fixed_leverage
        margin = new_ratio
        return SizedAction(action.action, side, action.reason_code, new_ratio, notional, margin)

