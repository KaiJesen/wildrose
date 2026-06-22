from __future__ import annotations

from dataclasses import dataclass

from trading_system.config import TradingSystemConfig
from trading_system.enums import ActionType, Side
from trading_system.portfolio import PortfolioState
from trading_system.rules import TradingAction
from trading_system.signal import TradingSignal
from trading_system.trend import TrendContext


@dataclass
class SizedAction:
    action: ActionType
    target_side: Side
    reason_code: str
    position_ratio: float
    notional_exposure: float
    margin_required: float


class PositionSizer:
    def __init__(self, cfg: TradingSystemConfig) -> None:
        self.cfg = cfg

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
    ) -> SizedAction:
        side = action.target_side
        cur = portfolio.position.position_ratio
        if action.action == ActionType.BLOCK:
            return SizedAction(action.action, Side.FLAT, action.reason_code, cur, cur * self.cfg.base.fixed_leverage, cur)
        if action.action in (ActionType.CLOSE, ActionType.FORCE_CLOSE):
            return SizedAction(action.action, Side.FLAT, action.reason_code, 0.0, 0.0, 0.0)
        if action.action == ActionType.REDUCE:
            new_ratio = max(0.0, cur * self.cfg.rule.reduce_scale)
        elif action.action == ActionType.ADD:
            base = self._base_ratio_from_conf(signal.conf, signal.p_risk)
            if action.reason_code == "UPGRADE_SENTINEL_TO_MODEL_SHORT":
                new_ratio = min(self.cfg.base.max_position_ratio, max(cur, base))
            else:
                new_ratio = min(self.cfg.base.max_position_ratio, cur + base * 0.5)
        else:
            if action.reason_code == "OPEN_SHORT_SENTINEL":
                s = self.cfg.sentinel_short
                new_ratio = min(self.cfg.base.max_position_ratio, min(s.sentinel_position_ratio, s.sentinel_max_position_ratio))
            else:
                new_ratio = self._base_ratio_from_conf(signal.conf, signal.p_risk) if side != Side.FLAT else 0.0
            if portfolio.weekly_defensive_mode:
                new_ratio *= self.cfg.risk.defensive_size_scale
        notional = new_ratio * self.cfg.base.fixed_leverage
        margin = new_ratio
        return SizedAction(action.action, side, action.reason_code, new_ratio, notional, margin)

