from __future__ import annotations

from dataclasses import dataclass

from trading_system.config import TradingSystemConfig
from trading_system.enums import ActionType, Side
from trading_system.execution import BacktestExecutionEngine
from trading_system.logger import TradeLogger
from trading_system.portfolio import PortfolioState
from trading_system.risk import RiskManager
from trading_system.rules import RuleEngine, TradingAction
from trading_system.signal import TradingSignal
from trading_system.sizing import PositionSizer
from trading_system.trend import TrendContext, TrendRegimeFilter


@dataclass
class Bar:
    idx: int
    ts: object
    open: float
    high: float
    low: float
    close: float
    atr: float


class TradingEngine:
    def __init__(self, cfg: TradingSystemConfig, logger: TradeLogger) -> None:
        self.cfg = cfg
        self.logger = logger
        self.portfolio = PortfolioState()
        self.rule_engine = RuleEngine(cfg)
        self.risk_manager = RiskManager(cfg)
        self.position_sizer = PositionSizer(cfg)
        self.execution_engine = BacktestExecutionEngine(cfg)
        self.trend_filter = TrendRegimeFilter(cfg.trend)
        self.position_limit_violations = 0
        self.risk_rule_violations = 0
        self.max_margin_loss_ratio_observed = 0.0
        self.close_hist: list[float] = []
        self.high_hist: list[float] = []
        self.low_hist: list[float] = []
        self._last_trend_context: TrendContext | None = None
        self._downtrend_regime_active: bool = False
        self._sentinel_used_in_regime: bool = False

    def _mark_to_market(self, current_bar: Bar) -> None:
        pos = self.portfolio.position
        if pos.is_flat or current_bar.idx <= 0:
            self.portfolio.unrealized_pnl = 0.0
            return
        # open-to-close approximation for current bar mark.
        pnl = 0.0
        if pos.side == Side.LONG:
            pnl = pos.notional_exposure * (current_bar.close - pos.avg_price) / max(1e-12, pos.avg_price)
        elif pos.side == Side.SHORT:
            pnl = pos.notional_exposure * (pos.avg_price - current_bar.close) / max(1e-12, pos.avg_price)
            profit_atr = (pos.entry_price - current_bar.close) / max(current_bar.atr, 1e-12)
            pos.peak_profit_atr = max(pos.peak_profit_atr, float(profit_atr))
        self.portfolio.unrealized_pnl = pnl
        if pos.margin_used > 0:
            ratio = max(0.0, -pnl / max(1e-12, pos.margin_used))
            self.max_margin_loss_ratio_observed = max(self.max_margin_loss_ratio_observed, ratio)

    def _apply_fill(self, fill, current_bar: Bar) -> None:
        p = self.portfolio
        pos = p.position
        p.equity = max(1e-9, p.equity - fill.fee)
        p.cash = p.equity
        # Handle close record before state reset.
        if fill.action in (ActionType.CLOSE, ActionType.FORCE_CLOSE) and not pos.is_flat:
            if pos.side == Side.LONG:
                trade_ret = (fill.price - pos.entry_price) / max(1e-12, pos.entry_price)
            else:
                trade_ret = (pos.entry_price - fill.price) / max(1e-12, pos.entry_price)
            pnl_eq = trade_ret * pos.notional_exposure
            p.realized_pnl += pnl_eq
            p.equity += pnl_eq
            self.logger.record_trade(
                {
                    "entry_ts": pos.entry_ts,
                    "exit_ts": fill.ts,
                    "side": pos.side.value,
                    "entry_price": pos.entry_price,
                    "exit_price": fill.price,
                    "max_position_ratio": pos.position_ratio,
                    "avg_position_ratio": pos.position_ratio,
                    "add_count": pos.add_count,
                    "bars_held": pos.bars_held,
                    "entry_reason": pos.entry_signal_snapshot.get("reason_code", ""),
                    "exit_reason": fill.reason_code,
                    "gross_pnl": pnl_eq,
                    "fee": fill.fee,
                    "slippage_cost": 0.0,
                    "net_pnl": pnl_eq - fill.fee,
                    "return_on_equity": pnl_eq,
                    "return_on_margin": pnl_eq / max(1e-12, pos.margin_used),
                    "max_adverse_excursion": 0.0,
                    "max_favorable_excursion": pos.peak_unrealized_pnl,
                    "entry_trend_context": pos.entry_signal_snapshot.get("entry_trend_context", ""),
                    "exit_trend_context": "|".join(self._last_trend_context.reason_codes) if self._last_trend_context else "",
                    "entry_was_probe": int(pos.entry_was_probe),
                    "entry_was_sentinel": int(pos.entry_was_sentinel),
                    "hold_mode": pos.hold_mode,
                }
            )
            if pnl_eq < 0:
                p.loss_streak += 1
                if p.loss_streak >= self.cfg.risk.loss_streak_limit:
                    p.cooldown_until = current_bar.idx + self.cfg.risk.cooldown_bars
                    p.loss_streak = 0
            else:
                p.loss_streak = 0
            if pos.entry_was_sentinel and fill.reason_code == "CLOSE_SENTINEL_NOT_CONFIRMED":
                p.cooldown_until = max(p.cooldown_until, current_bar.idx + self.cfg.sentinel_short.sentinel_cooldown_bars)
        # Position state update.
        if fill.action in (ActionType.CLOSE, ActionType.FORCE_CLOSE):
            p.position = type(pos)()
        elif fill.action in (ActionType.OPEN_LONG, ActionType.OPEN_SHORT, ActionType.REVERSE):
            side = Side.LONG if fill.side == Side.LONG else Side.SHORT
            pos.side = side
            pos.entry_ts = fill.ts
            pos.entry_price = fill.price
            pos.avg_price = fill.price
            pos.position_ratio = fill.filled_position_ratio
            pos.notional_exposure = fill.notional
            pos.margin_used = fill.margin_required
            pos.leverage = self.cfg.base.fixed_leverage
            pos.bars_held = 0
            pos.add_count = 0
            pos.continue_fail_count = 0
            pos.short_reverse_confirm_count = 0
            pos.entry_was_probe = False
            pos.entry_was_sentinel = bool(fill.reason_code == "OPEN_SHORT_SENTINEL")
            pos.hold_mode = "NORMAL"
            pos.trend_hold_bars = 0
            pos.trend_break_count = 0
            pos.sentinel_bars = 0
            pos.peak_profit_atr = 0.0
            pos.stop_price = (
                fill.price - self.cfg.risk.stop_atr_mult * current_bar.atr
                if side == Side.LONG
                else fill.price + self.cfg.risk.stop_atr_mult * current_bar.atr
            )
            pos.take_profit_1 = (
                fill.price + self.cfg.risk.tp1_atr_mult * current_bar.atr
                if side == Side.LONG
                else fill.price - self.cfg.risk.tp1_atr_mult * current_bar.atr
            )
            pos.take_profit_2 = (
                fill.price + self.cfg.risk.tp2_atr_mult * current_bar.atr
                if side == Side.LONG
                else fill.price - self.cfg.risk.tp2_atr_mult * current_bar.atr
            )
            pos.entry_signal_snapshot = {
                "reason_code": fill.reason_code,
                "entry_trend_context": "|".join(self._last_trend_context.reason_codes) if self._last_trend_context else "",
            }
        elif fill.action in (ActionType.REDUCE, ActionType.ADD, ActionType.HOLD, ActionType.BLOCK):
            if fill.action == ActionType.REDUCE:
                pos.position_ratio = fill.filled_position_ratio
                pos.notional_exposure = fill.notional
                pos.margin_used = fill.margin_required
            elif fill.action == ActionType.ADD:
                pos.position_ratio = fill.filled_position_ratio
                pos.notional_exposure = fill.notional
                pos.margin_used = fill.margin_required
                pos.add_count += 1
                if fill.reason_code == "UPGRADE_SENTINEL_TO_MODEL_SHORT":
                    pos.entry_was_sentinel = False
                    pos.hold_mode = "TREND"
                    pos.entry_signal_snapshot["reason_code"] = "UPGRADE_SENTINEL_TO_MODEL_SHORT"

    def on_bar_close(self, signal: TradingSignal, current_bar: Bar, next_bar: Bar) -> None:
        self.close_hist.append(current_bar.close)
        self.high_hist.append(current_bar.high)
        self.low_hist.append(current_bar.low)
        trend_context = self.trend_filter.compute(self.close_hist, self.high_hist, self.low_hist, current_bar.atr)
        self._last_trend_context = trend_context
        if trend_context.is_downtrend and not self._downtrend_regime_active:
            self._downtrend_regime_active = True
            self._sentinel_used_in_regime = False
        elif not trend_context.is_downtrend:
            self._downtrend_regime_active = False
        self.portfolio.update_time_gates(
            current_bar.ts,
            day_drawdown_stop=self.cfg.risk.day_drawdown_stop,
            week_drawdown_defensive=self.cfg.risk.week_drawdown_defensive,
        )
        self._mark_to_market(current_bar)
        if not self.portfolio.position.is_flat:
            self.portfolio.position.bars_held += 1
            if self.portfolio.position.hold_mode == "TREND":
                self.portfolio.position.trend_hold_bars += 1
            if self.portfolio.position.entry_was_sentinel:
                self.portfolio.position.sentinel_bars += 1
        if not signal.is_valid:
            action = TradingAction(ActionType.BLOCK, Side.FLAT, signal.reason_code or "INVALID_SIGNAL", blocked_by="signal")
            self.logger.record_decision(
                signal,
                action,
                self.portfolio,
                trend_context=trend_context,
                blocked_reason=action.reason_code,
            )
            self.logger.record_equity(current_bar.ts, self.portfolio.equity)
            return
        risk_event = self.risk_manager.check_pre_decision_risk(
            signal,
            self.portfolio,
            current_bar.idx,
            current_bar.close,
            trend_context=trend_context,
        )
        if risk_event.force_close:
            action = TradingAction(ActionType.CLOSE, Side.FLAT, risk_event.reason, blocked_by="risk")
        elif risk_event.reason == "REDUCE_RISK_PROB_HIGH" and not self.portfolio.position.is_flat:
            action = TradingAction(ActionType.REDUCE, self.portfolio.position.side, risk_event.reason, blocked_by="risk")
        elif risk_event.block_open and self.portfolio.position.is_flat:
            action = TradingAction(ActionType.BLOCK, Side.FLAT, risk_event.reason, blocked_by="risk")
        else:
            action = self.rule_engine.decide(signal, self.portfolio, trend_context=trend_context)
        if action.reason_code == "OPEN_SHORT_SENTINEL" and self._sentinel_used_in_regime:
            action = TradingAction(ActionType.BLOCK, Side.FLAT, "BLOCK_SENTINEL_ONCE_PER_REGIME", blocked_by="trend")
        action = self.risk_manager.validate_action(action, signal, self.portfolio, trend_context=trend_context)
        sized = self.position_sizer.apply(action, signal, self.portfolio, trend_context=trend_context)
        if sized.position_ratio > self.cfg.base.max_position_ratio + 1e-12:
            self.position_limit_violations += 1
        fill = self.execution_engine.execute(
            sized,
            ts=next_bar.ts,
            next_open=next_bar.open,
            current_position_ratio=self.portfolio.position.position_ratio,
        )
        self.logger.record_decision(
            signal,
            action,
            self.portfolio,
            trend_context=trend_context,
            blocked_reason=action.reason_code if action.action == ActionType.BLOCK else "",
        )
        self.logger.record_order(
            {
                "ts": next_bar.ts,
                "action": sized.action.value,
                "side": sized.target_side.value,
                "position_ratio": sized.position_ratio,
                "notional_exposure": sized.notional_exposure,
                "margin_required": sized.margin_required,
                "reason_code": sized.reason_code,
            }
        )
        self.logger.record_fill(fill)
        self._apply_fill(fill, current_bar)
        if fill.reason_code == "OPEN_SHORT_SENTINEL":
            self._sentinel_used_in_regime = True
        self.logger.record_equity(next_bar.ts, self.portfolio.equity)

