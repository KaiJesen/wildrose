"""027 Satellite slot: best-point tactical rules (independent of TEQ)."""

from __future__ import annotations

from dataclasses import dataclass

from trading_system.adapters.best_point_model import BestPointSignal
from trading_system.enums import ActionType, Side
from trading_system.portfolio_slots import AccountEquity
from trading_system.rules import TradingAction
from trading_system.signal import TradingSignal
from trading_system.state import PositionState


@dataclass(frozen=True)
class SatelliteSlotConfig:
    enabled: bool = True
    max_position_ratio: float = 0.08
    max_hold_bars: int = 48
    max_daily_opens: int = 5
    long_entry_threshold: float = 0.50
    short_entry_threshold: float = 0.50
    exit_prob_threshold: float = 0.70
    hold_min_prob: float = 0.30
    min_opportunity_roi: float = 0.0
    min_pred_cum_ret_5_long: float = 0.0
    risk_open_max: float = 0.50
    stop_atr_mult: float = 1.4
    require_core_flat: bool = False
    require_core_watch_slow_uptrend: bool = False


class SatelliteRuleEngine:
    def __init__(self, cfg: SatelliteSlotConfig) -> None:
        self.cfg = cfg

    def _bp_exit(self, bp: BestPointSignal, *, side: Side) -> bool:
        if side == Side.LONG:
            return bp.p_exit_long >= self.cfg.exit_prob_threshold or bp.p_hold_long <= self.cfg.hold_min_prob
        if side == Side.SHORT:
            return bp.p_exit_short >= self.cfg.exit_prob_threshold or bp.p_hold_short <= self.cfg.hold_min_prob
        return False

    def decide(
        self,
        signal: TradingSignal,
        bp: BestPointSignal | None,
        position: PositionState,
        account: AccountEquity,
        *,
        bar_index: int,
        current_price: float,
        daily_open_count: int,
        core_reason_code: str = "",
    ) -> TradingAction:
        if not self.cfg.enabled or bp is None or not signal.is_valid:
            return TradingAction(ActionType.HOLD, Side.FLAT, "SAT_HOLD_NO_SIGNAL", blocked_by="satellite")

        pos = position
        if not pos.is_flat:
            if pos.side == Side.LONG and current_price <= pos.stop_price:
                return TradingAction(ActionType.CLOSE, Side.FLAT, "SAT_CLOSE_HARD_STOP", blocked_by="satellite")
            if pos.side == Side.SHORT and current_price >= pos.stop_price:
                return TradingAction(ActionType.CLOSE, Side.FLAT, "SAT_CLOSE_HARD_STOP", blocked_by="satellite")
            if pos.bars_held >= self.cfg.max_hold_bars:
                return TradingAction(ActionType.CLOSE, Side.FLAT, "SAT_CLOSE_MAX_HOLD", blocked_by="satellite")
            if self._bp_exit(bp, side=pos.side):
                return TradingAction(ActionType.CLOSE, Side.FLAT, "SAT_CLOSE_BP_EXIT", blocked_by="satellite")
            return TradingAction(ActionType.HOLD, pos.side, "SAT_HOLD_POSITION", blocked_by="satellite")

        if daily_open_count >= self.cfg.max_daily_opens:
            return TradingAction(ActionType.HOLD, Side.FLAT, "SAT_HOLD_DAILY_CAP", blocked_by="satellite")
        if self.cfg.require_core_flat and not account.core_position.is_flat:
            return TradingAction(ActionType.HOLD, Side.FLAT, "SAT_HOLD_CORE_ACTIVE", blocked_by="satellite")
        if self.cfg.require_core_watch_slow_uptrend:
            if core_reason_code != "WATCH_SLOW_UPTREND":
                return TradingAction(ActionType.HOLD, Side.FLAT, "SAT_HOLD_NOT_WATCH_SLOW", blocked_by="satellite")
            if not account.core_position.is_flat:
                return TradingAction(ActionType.HOLD, Side.FLAT, "SAT_HOLD_CORE_ACTIVE", blocked_by="satellite")
        if signal.p_risk > self.cfg.risk_open_max:
            return TradingAction(ActionType.HOLD, Side.FLAT, "SAT_BLOCK_RISK", blocked_by="satellite")
        if bp.expected_opportunity_roi < self.cfg.min_opportunity_roi:
            return TradingAction(ActionType.HOLD, Side.FLAT, "SAT_BLOCK_LOW_OPP", blocked_by="satellite")

        long_ok = (
            bp.p_long_entry_zone >= self.cfg.long_entry_threshold
            and float(signal.pred_cum_ret_5 or 0.0) >= self.cfg.min_pred_cum_ret_5_long
        )
        short_ok = (
            bp.p_short_entry_zone >= self.cfg.short_entry_threshold
            and not self.cfg.require_core_watch_slow_uptrend
        )

        if long_ok:
            if not account.allow_satellite_long():
                return TradingAction(ActionType.BLOCK, Side.FLAT, account.dad_block_reason(side=Side.LONG), blocked_by="dad")
            return TradingAction(ActionType.OPEN_LONG, Side.LONG, "SAT_OPEN_LONG_BP", blocked_by="satellite")
        if short_ok:
            if not account.allow_satellite_short():
                return TradingAction(ActionType.BLOCK, Side.FLAT, account.dad_block_reason(side=Side.SHORT), blocked_by="dad")
            return TradingAction(ActionType.OPEN_SHORT, Side.SHORT, "SAT_OPEN_SHORT_BP", blocked_by="satellite")
        return TradingAction(ActionType.HOLD, Side.FLAT, "SAT_HOLD_NO_ENTRY", blocked_by="satellite")
