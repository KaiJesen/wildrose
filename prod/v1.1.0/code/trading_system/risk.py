from __future__ import annotations

from dataclasses import dataclass

from trading_system.config import TradingSystemConfig
from trading_system.enums import ActionType, Side
from trading_system.portfolio import PortfolioState
from trading_system.rules import TradingAction
from trading_system.signal import TradingSignal
from trading_system.trend import TrendContext
from trading_system.trend_signal import TrendStrength, TrendSignal
from trading_system.slow_trend import SlowTrendContext
from trading_system.trend_bias import RiskBudget


@dataclass
class RiskEvent:
    force_close: bool = False
    block_open: bool = False
    reason: str = ""


class RiskManager:
    def __init__(self, cfg: TradingSystemConfig) -> None:
        self.cfg = cfg

    def check_pre_decision_risk(
        self,
        signal: TradingSignal,
        portfolio: PortfolioState,
        bar_index: int,
        current_price: float,
        trend_context: TrendContext | None = None,
        trend_signal: TrendSignal | None = None,
        slow_context: SlowTrendContext | None = None,
    ) -> RiskEvent:
        if not signal.is_valid:
            return RiskEvent(block_open=True, reason=signal.reason_code or "INVALID_SIGNAL")
        if portfolio.account_circuit_breaker:
            return RiskEvent(block_open=True, reason="BLOCK_ACCOUNT_CIRCUIT")
        if portfolio.daily_open_block:
            return RiskEvent(block_open=True, reason="BLOCK_DAILY_DRAWDOWN")
        if bar_index <= portfolio.cooldown_until:
            return RiskEvent(block_open=True, reason="BLOCK_LOSS_STREAK_COOLDOWN")
        pos = portfolio.position
        if not pos.is_flat:
            if pos.side == Side.LONG and current_price <= pos.stop_price:
                return RiskEvent(force_close=True, reason="CLOSE_HARD_STOP")
            if pos.side == Side.SHORT and current_price >= pos.stop_price:
                return RiskEvent(force_close=True, reason="CLOSE_HARD_STOP")
            # Risk-prob exit.
            if signal.p_risk >= self.cfg.rule.risk_exit_threshold:
                if self.cfg.rule.risk_exit_mode == "reduce_first":
                    return RiskEvent(force_close=False, block_open=False, reason="REDUCE_RISK_PROB_HIGH")
                return RiskEvent(force_close=True, reason="CLOSE_RISK_PROB_HIGH")
            # Catastrophic margin guard.
            if pos.margin_used > 0:
                margin_loss_ratio = max(0.0, -portfolio.unrealized_pnl / max(1e-12, pos.margin_used))
                if margin_loss_ratio >= self.cfg.base.catastrophe_margin_loss_buffer:
                    return RiskEvent(force_close=True, reason="CLOSE_CATASTROPHE_MARGIN_LOSS")
            # Time exit: NORMAL and TREND use different hold limits.
            if pos.hold_mode == "CRASH" and pos.side == Side.SHORT and self.cfg.crash_short.enabled:
                crash_max = self.cfg.crash_short.max_hold_bars
                if trend_context and getattr(trend_context, "is_strong_downtrend", False):
                    crash_max = self.cfg.crash_short.strong_max_hold_bars
                if pos.bars_held >= crash_max:
                    return RiskEvent(force_close=True, reason="CLOSE_CRASH_MAX_HOLD_BARS")
            elif pos.entry_was_slow_up or pos.hold_mode == "SLOW_UP":
                slow_max = self.cfg.slow_up_position.max_hold_bars
                if pos.bars_held >= slow_max:
                    return RiskEvent(force_close=True, reason="CLOSE_SLOW_UP_MAX_HOLD_BARS")
            elif pos.lifecycle in ("TREND", "PROTECT_PROFIT", "RUNNER") and self.cfg.trend_signal.enabled:
                trend_max_hold = self.cfg.trend_position.max_trend_hold_bars
                if trend_signal and trend_signal.strength in (TrendStrength.STRONG, TrendStrength.EXTREME):
                    trend_max_hold = self.cfg.trend_position.strong_trend_hold_bars
                if pos.bars_held >= trend_max_hold:
                    return RiskEvent(force_close=True, reason="CLOSE_TREND_MAX_HOLD_BARS")
            else:
                normal_max = self.cfg.trend_hold.normal_max_hold_bars if self.cfg.trend_hold.enabled else self.cfg.rule.max_hold_bars
                if pos.bars_held >= normal_max:
                    return RiskEvent(force_close=True, reason="CLOSE_MAX_HOLD_BARS")
        return RiskEvent()

    def compute_risk_budget(self, portfolio: PortfolioState, *, bar_index: int) -> RiskBudget:
        max_ratio = self.cfg.base.max_position_ratio
        cur = portfolio.position.position_ratio
        margin = portfolio.position.margin_used
        worst_case = 0.0
        if margin > 0:
            worst_case = max(0.0, -portfolio.unrealized_pnl / max(1e-12, margin))
        remaining_pos = max(0.0, max_ratio - cur)
        loss_budget = max(0.0, self.cfg.base.catastrophe_margin_loss_buffer - worst_case)
        allow_open = (
            not portfolio.account_circuit_breaker
            and not portfolio.daily_open_block
            and bar_index > portfolio.cooldown_until
        )
        allow_add = allow_open and remaining_pos > 1e-6 and loss_budget > 0.05
        reject_open = ""
        reject_add = ""
        if portfolio.account_circuit_breaker:
            reject_open = "BLOCK_ACCOUNT_CIRCUIT"
        elif portfolio.daily_open_block:
            reject_open = "BLOCK_DAILY_DRAWDOWN"
        elif bar_index <= portfolio.cooldown_until:
            reject_open = "BLOCK_LOSS_STREAK_COOLDOWN"
        if remaining_pos <= 1e-6:
            reject_add = "RISK_NO_REMAINING_POSITION"
        elif loss_budget <= 0.05:
            reject_add = "RISK_LOSS_BUDGET_LOW"
        return RiskBudget(
            current_position_ratio=cur,
            max_position_ratio=max_ratio,
            current_margin_ratio=margin,
            remaining_position_ratio=remaining_pos,
            worst_case_loss_ratio=worst_case,
            remaining_loss_budget_ratio=loss_budget,
            allow_open_long=allow_open,
            allow_open_short=allow_open,
            allow_add_long=allow_add,
            allow_add_short=allow_add,
            allow_reverse=allow_open and loss_budget > 0.1,
            open_reject_reason_long=reject_open,
            open_reject_reason_short=reject_open,
            add_reject_reason_long=reject_add,
            add_reject_reason_short=reject_add,
        )

    def validate_action(
        self,
        action: TradingAction,
        signal: TradingSignal,
        portfolio: PortfolioState,
        trend_context: TrendContext | None = None,
    ) -> TradingAction:
        tb = self.cfg.trend_bias
        legacy_long_block = (
            action.action == ActionType.OPEN_LONG
            and trend_context
            and trend_context.is_downtrend
            and self.cfg.protection.block_long_in_downtrend
            and not (tb.enabled and tb.disable_legacy_trend_rules and tb.decision_scope != "observe")
        )
        if legacy_long_block:
            return TradingAction(ActionType.BLOCK, Side.FLAT, "BLOCK_LONG_DOWNTREND", blocked_by="trend")
        if action.action in (ActionType.OPEN_LONG, ActionType.OPEN_SHORT, ActionType.ADD):
            risk_open_max = self.cfg.rule.risk_open_max
            flat_max = self.cfg.rule.open_flat_max
            if action.reason_code == "OPEN_SHORT_SENTINEL":
                risk_open_max = self.cfg.sentinel_short.sentinel_risk_max
                flat_max = self.cfg.sentinel_short.sentinel_flat_max
            elif action.reason_code == "OPEN_SHORT_CRASH":
                risk_open_max = self.cfg.crash_short.risk_max
                flat_max = self.cfg.crash_short.flat_max
            elif action.reason_code == "OPEN_LONG_SLOW_TREND":
                risk_open_max = self.cfg.slow_up_position.risk_max
                flat_max = 1.0
            if signal.p_risk > risk_open_max:
                return TradingAction(ActionType.BLOCK, Side.FLAT, "BLOCK_OPEN_RISK_HIGH", blocked_by="risk")
            if signal.p_flat > flat_max:
                return TradingAction(ActionType.BLOCK, Side.FLAT, "BLOCK_OPEN_FLAT_HIGH", blocked_by="risk")
        return action

