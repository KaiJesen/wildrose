from __future__ import annotations

from dataclasses import dataclass, field

from trading_system.config import TradingSystemConfig
from trading_system.crash import CrashContext
from trading_system.enums import ActionType, Side
from trading_system.portfolio import PortfolioState
from trading_system.signal import TradingSignal
from trading_system.trend import TrendContext
from trading_system.trend_signal import TrendDirection, TrendPhase, TrendSignal


@dataclass
class TradingAction:
    action: ActionType
    target_side: Side
    reason_code: str
    blocked_by: str = ""
    diagnostics: dict = field(default_factory=dict)


class RuleEngine:
    def __init__(self, cfg: TradingSystemConfig) -> None:
        self.cfg = cfg

    def _is_standard_short_entry(self, signal: TradingSignal) -> bool:
        r = self.cfg.rule
        return (
            signal.edge <= -r.open_edge_threshold
            and signal.p_down >= r.open_prob_threshold
            and signal.p_flat <= r.open_flat_max
            and signal.pred_cum_ret_5 < 0
            and signal.risk_ok
        )

    def _is_sentinel_short_entry(self, signal: TradingSignal, tc: TrendContext | None) -> bool:
        if tc is None or not tc.is_strong_downtrend:
            return False
        s = self.cfg.sentinel_short
        if tc.ret_6_atr > s.sentinel_ret6_atr_threshold:
            return False
        if not (signal.price < tc.ema_fast < tc.ema_slow):
            return False
        return (
            signal.p_risk <= s.sentinel_risk_max
            and signal.p_flat <= s.sentinel_flat_max
            and signal.edge <= 0.0
            and signal.pred_cum_ret_5 <= 0.0
        )

    def decide(
        self,
        signal: TradingSignal,
        portfolio: PortfolioState,
        trend_context: TrendContext | None = None,
        crash_context: CrashContext | None = None,
        trend_signal: TrendSignal | None = None,
    ) -> TradingAction:
        pos = portfolio.position
        r = self.cfg.rule
        protection = self.cfg.protection
        tc = trend_context
        if pos.is_flat:
            if (
                signal.edge >= r.open_edge_threshold
                and signal.p_up >= r.open_prob_threshold
                and signal.p_flat <= r.open_flat_max
                and signal.pred_cum_ret_5 > 0
                and signal.risk_ok
            ):
                return TradingAction(ActionType.OPEN_LONG, Side.LONG, "OPEN_LONG_SIGNAL")
            if self._is_standard_short_entry(signal):
                return TradingAction(ActionType.OPEN_SHORT, Side.SHORT, "OPEN_SHORT_SIGNAL")
            cc = crash_context
            if (
                cc
                and self.cfg.crash_short.enabled
                and cc.is_model_blind_crash
                and signal.p_risk <= self.cfg.crash_short.risk_max
                and signal.p_flat <= self.cfg.crash_short.flat_max
            ):
                return TradingAction(
                    ActionType.OPEN_SHORT,
                    Side.SHORT,
                    "OPEN_SHORT_CRASH",
                    diagnostics={"is_crash_short": True, "strong_crash": cc.strong_crash},
                )
            if protection.allow_sentinel_short and self.cfg.sentinel_short.enabled and self._is_sentinel_short_entry(signal, tc):
                return TradingAction(
                    ActionType.OPEN_SHORT,
                    Side.SHORT,
                    "OPEN_SHORT_SENTINEL",
                    diagnostics={"is_sentinel_short": True},
                )
            return TradingAction(ActionType.HOLD, Side.FLAT, "HOLD_NO_ENTRY")

        side = pos.side
        ts = trend_signal
        if not pos.is_flat and ts is not None:
            # Upgrade profitable position into trend-hold mode.
            profit_atr = (
                (signal.price - pos.entry_price) / max(signal.atr, 1e-12)
                if side == Side.LONG
                else (pos.entry_price - signal.price) / max(signal.atr, 1e-12)
            )
            upgrade_profit_atr = (
                self.cfg.trend_position.crash_upgrade_profit_atr
                if pos.hold_mode == "CRASH" and self.cfg.trend_position.allow_crash_trend_upgrade
                else self.cfg.trend_position.upgrade_profit_atr
            )
            can_upgrade = pos.hold_mode != "TREND"
            if pos.hold_mode == "CRASH" and not self.cfg.trend_position.allow_crash_trend_upgrade:
                can_upgrade = False
            if (
                can_upgrade
                and profit_atr >= upgrade_profit_atr
                and ts.is_confirmed
                and ts.trend_age >= self.cfg.trend_position.min_trend_age_for_upgrade
                and ts.phase in (TrendPhase.CONTINUATION, TrendPhase.ACCELERATION)
                and signal.p_risk < self.cfg.rule.risk_exit_threshold
            ):
                if side == Side.LONG and ts.direction == TrendDirection.UP:
                    return TradingAction(ActionType.HOLD, Side.LONG, "UPGRADE_TO_TREND_LONG")
                if side == Side.SHORT and ts.direction == TrendDirection.DOWN:
                    return TradingAction(ActionType.HOLD, Side.SHORT, "UPGRADE_TO_TREND_SHORT")
            if pos.hold_mode == "TREND":
                if ts.is_broken or ts.phase == TrendPhase.REVERSAL_RISK:
                    return TradingAction(ActionType.CLOSE, Side.FLAT, "CLOSE_TREND_BROKEN")
                score_drop = max(0.0, pos.trend_peak_score - ts.score_abs)
                if ts.phase == TrendPhase.EXHAUSTION or score_drop >= 2.0:
                    return TradingAction(ActionType.REDUCE, side, "REDUCE_TREND_EXHAUSTION")
                if (
                    ts.phase == TrendPhase.ACCELERATION
                    and pos.add_count < self.cfg.base.max_add_count
                    and pos.position_ratio < self.cfg.base.max_position_ratio
                    and signal.p_risk < self.cfg.rule.risk_open_max
                    and profit_atr >= self.cfg.trend_position.add_profit_atr
                ):
                    return TradingAction(ActionType.ADD, side, "ADD_TREND_CONTINUATION")
                return TradingAction(ActionType.HOLD, side, "HOLD_TREND_CONTINUATION")
        if side == Side.LONG and tc and tc.is_downtrend:
            if signal.edge <= 0.0 or signal.pred_cum_ret_5 <= 0.0:
                return TradingAction(ActionType.CLOSE, Side.FLAT, "CLOSE_LONG_DOWNTREND_CONFIRMED")
            return TradingAction(ActionType.REDUCE, Side.LONG, "REDUCE_LONG_DOWNTREND_RISK")

        # Hard reverse signal.
        if side == Side.LONG and (signal.edge <= -r.reverse_edge_threshold or signal.pred_cum_ret_5 < 0):
            if r.allow_reverse and signal.edge <= -r.open_edge_threshold and signal.p_down >= r.open_prob_threshold:
                return TradingAction(ActionType.REVERSE, Side.SHORT, "REVERSE_SIGNAL")
            return TradingAction(ActionType.CLOSE, Side.FLAT, "CLOSE_REVERSE_SIGNAL")
        if side == Side.SHORT:
            if pos.entry_was_crash and self._is_standard_short_entry(signal):
                return TradingAction(
                    ActionType.ADD,
                    Side.SHORT,
                    "UPGRADE_CRASH_TO_MODEL_SHORT",
                    diagnostics={"upgrade_crash": True},
                )
            if pos.entry_was_crash and crash_context is not None:
                fail_by_price = signal.price > pos.entry_price + self.cfg.crash_short.fail_stop_atr * max(signal.atr, 1e-12)
                fail_by_votes = crash_context.crash_votes < self.cfg.crash.min_crash_votes
                if fail_by_price or (fail_by_votes and pos.crash_bars >= 2):
                    return TradingAction(ActionType.CLOSE, Side.FLAT, "CLOSE_CRASH_FAILED")
                if pos.peak_profit_atr >= self.cfg.crash_short.trail_start_atr:
                    cur_profit_atr = (pos.entry_price - signal.price) / max(1e-12, signal.atr)
                    if (pos.peak_profit_atr - cur_profit_atr) >= self.cfg.crash_short.trail_back_atr:
                        return TradingAction(ActionType.CLOSE, Side.FLAT, "CLOSE_CRASH_TRAIL")

            # Sentinel must be upgraded quickly, otherwise exit to avoid probe-like drag.
            if pos.entry_was_sentinel and self._is_standard_short_entry(signal):
                return TradingAction(
                    ActionType.ADD,
                    Side.SHORT,
                    "UPGRADE_SENTINEL_TO_MODEL_SHORT",
                    diagnostics={"upgrade_sentinel": True},
                )
            if pos.entry_was_sentinel and pos.sentinel_bars >= self.cfg.sentinel_short.sentinel_max_hold_bars:
                return TradingAction(ActionType.CLOSE, Side.FLAT, "CLOSE_SENTINEL_NOT_CONFIRMED")

            in_trend_hold = pos.hold_mode == "TREND"
            if in_trend_hold and tc:
                trend_break = (
                    signal.price > tc.ema_fast
                    or signal.edge >= r.reverse_edge_threshold
                    or signal.pred_cum_ret_5 > 0
                    or tc.ret_3_atr > 0.8
                )
                trend_restore = signal.price < tc.ema_fast and signal.pred_cum_ret_5 <= 0 and signal.edge < r.reverse_edge_threshold
                if trend_restore:
                    pos.trend_break_count = 0
                elif trend_break:
                    pos.trend_break_count += 1
                if pos.trend_break_count >= self.cfg.trend_hold.trend_break_confirm_bars:
                    return TradingAction(ActionType.CLOSE, Side.FLAT, "CLOSE_SHORT_TREND_BROKEN")
                if pos.peak_profit_atr >= 2.0:
                    cur_profit_atr = (pos.entry_price - signal.price) / max(1e-12, signal.atr)
                    if (pos.peak_profit_atr - cur_profit_atr) >= 1.0:
                        return TradingAction(ActionType.CLOSE, Side.FLAT, "CLOSE_SHORT_TREND_TRAIL")

            if signal.edge >= r.reverse_edge_threshold or signal.pred_cum_ret_5 > 0:
                if tc and tc.is_downtrend:
                    if protection.short_exit_require_edge_and_cum:
                        reverse_confirm = signal.edge >= r.reverse_edge_threshold and signal.pred_cum_ret_5 > 0
                    else:
                        reverse_confirm = signal.edge >= r.reverse_edge_threshold or signal.pred_cum_ret_5 > 0
                    if reverse_confirm:
                        pos.short_reverse_confirm_count += 1
                    else:
                        pos.short_reverse_confirm_count = 0
                    if pos.short_reverse_confirm_count < protection.short_exit_confirm_bars:
                        return TradingAction(
                            ActionType.HOLD,
                            Side.SHORT,
                            "HOLD_SHORT_REVERSE_WAIT_CONFIRM",
                            diagnostics={"reverse_confirm_count": pos.short_reverse_confirm_count},
                        )
                    pos.short_reverse_confirm_count = 0
                    return TradingAction(ActionType.CLOSE, Side.FLAT, "CLOSE_SHORT_REVERSE_CONFIRMED")
                if r.allow_reverse and signal.edge >= r.open_edge_threshold and signal.p_up >= r.open_prob_threshold:
                    return TradingAction(ActionType.REVERSE, Side.LONG, "REVERSE_SIGNAL")
                return TradingAction(ActionType.CLOSE, Side.FLAT, "CLOSE_REVERSE_SIGNAL")

        # Continue condition.
        if side == Side.LONG:
            keep = signal.edge > r.long_continue_edge_min and signal.pred_cum_ret_5 >= 0
        else:
            keep = signal.edge < r.short_continue_edge_max and signal.pred_cum_ret_5 <= 0
        if keep:
            pos.continue_fail_count = 0
            pos.short_reverse_confirm_count = 0
            if (
                side == Side.SHORT
                and tc
                and self.cfg.trend_hold.enabled
                and tc.is_downtrend
                and signal.pred_cum_ret_5 < 0
                and signal.p_risk <= self.cfg.rule.risk_exit_threshold
                and (
                    not self.cfg.trend_hold.allow_extend_only_for_model_short
                    or (
                        pos.entry_signal_snapshot.get("reason_code")
                        in ("OPEN_SHORT_SIGNAL", "UPGRADE_SENTINEL_TO_MODEL_SHORT", "UPGRADE_CRASH_TO_MODEL_SHORT")
                    )
                )
            ):
                pos.hold_mode = "TREND"
            if (
                pos.add_count < self.cfg.base.max_add_count
                and signal.risk_ok
                and ((side == Side.LONG and signal.pred_cum_ret_5 > 0) or (side == Side.SHORT and signal.pred_cum_ret_5 < 0))
            ):
                if pos.entry_was_sentinel and side == Side.SHORT and not self._is_standard_short_entry(signal):
                    return TradingAction(ActionType.HOLD, side, "HOLD_SENTINEL_WAIT_STANDARD_CONFIRM")
                return TradingAction(ActionType.ADD, side, "ADD_SIGNAL_CONFIRMED")
            return TradingAction(ActionType.HOLD, side, "HOLD_SIGNAL_VALID")

        pos.continue_fail_count += 1
        if side == Side.SHORT and (signal.edge < r.reverse_edge_threshold or signal.pred_cum_ret_5 <= 0.0):
            pos.short_reverse_confirm_count = 0
        if pos.continue_fail_count >= r.continue_fail_limit:
            return TradingAction(ActionType.CLOSE, Side.FLAT, "CLOSE_CONTINUE_SIGNAL_FAILED")
        return TradingAction(ActionType.REDUCE, side, "REDUCE_WEAK_CONTINUE_SIGNAL")

