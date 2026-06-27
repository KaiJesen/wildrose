"""027 Satellite slot execution (best-point tactical layer)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from trading_system.adapters.best_point_model import BestPointSignal
from trading_system.config import TradingSystemConfig
from trading_system.engine import Bar
from trading_system.enums import ActionType, Side
from trading_system.execution import BacktestExecutionEngine, FillEvent
from trading_system.portfolio_slots import AccountEquity
from trading_system.rules import TradingAction
from trading_system.satellite_rules import SatelliteRuleEngine, SatelliteSlotConfig
from trading_system.signal import TradingSignal
from trading_system.state import PositionState


@dataclass
class SatellitePortfolio:
    equity: float = 1.0
    cash: float = 1.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    position: PositionState = field(default_factory=PositionState)
    peak_equity: float = 1.0
    _daily_opens: int = 0
    _last_date: datetime | None = None

    def reset_daily_opens(self, ts: datetime) -> None:
        if self._last_date is None or ts.date() != self._last_date.date():
            self._daily_opens = 0
            self._last_date = ts

    def bump_daily_open(self) -> None:
        self._daily_opens += 1

    @property
    def daily_open_count(self) -> int:
        return self._daily_opens


class SatelliteEngine:
    def __init__(self, base_cfg: TradingSystemConfig, sat_cfg: SatelliteSlotConfig) -> None:
        self.base_cfg = base_cfg
        self.sat_cfg = sat_cfg
        self.rules = SatelliteRuleEngine(sat_cfg)
        self.execution = BacktestExecutionEngine(base_cfg)
        self.portfolio = SatellitePortfolio()
        self.trades: list[dict] = []
        self.decisions: list[dict] = []
        self._entry_core_reason: str = ""

    def _mark_to_market(self, price: float) -> None:
        pos = self.portfolio.position
        if pos.is_flat:
            self.portfolio.unrealized_pnl = 0.0
            return
        if pos.side == Side.LONG:
            pnl = pos.notional_exposure * (price - pos.avg_price) / max(1e-12, pos.avg_price)
        else:
            pnl = pos.notional_exposure * (pos.avg_price - price) / max(1e-12, pos.avg_price)
        self.portfolio.unrealized_pnl = pnl

    def _apply_fill(self, fill: FillEvent, *, current_bar: Bar, atr: float) -> None:
        p = self.portfolio
        pos = p.position
        p.equity = max(1e-9, p.equity - fill.fee)
        p.cash = p.equity

        if fill.action in (ActionType.CLOSE, ActionType.FORCE_CLOSE) and not pos.is_flat:
            if pos.side == Side.LONG:
                pnl = pos.notional_exposure * (fill.price - pos.avg_price) / max(1e-12, pos.avg_price)
            else:
                pnl = pos.notional_exposure * (pos.avg_price - fill.price) / max(1e-12, pos.avg_price)
            pnl -= fill.fee
            p.realized_pnl += pnl
            p.equity += pnl
            p.cash = p.equity
            self.trades.append(
                {
                    "slot_id": "satellite",
                    "entry_ts": pos.entry_ts,
                    "exit_ts": fill.ts,
                    "side": pos.side.value,
                    "entry_price": pos.entry_price,
                    "exit_price": fill.price,
                    "bars_held": pos.bars_held,
                    "net_pnl": pnl,
                    "fee": fill.fee,
                    "reason_code": fill.reason_code,
                    "core_reason_at_entry": self._entry_core_reason,
                }
            )
            p.position = PositionState()
            return

        if fill.action in (ActionType.OPEN_LONG, ActionType.OPEN_SHORT):
            side = Side.LONG if fill.side == Side.LONG else Side.SHORT
            pos.side = side
            pos.entry_ts = fill.ts
            pos.entry_price = fill.price
            pos.avg_price = fill.price
            pos.position_ratio = fill.filled_position_ratio
            pos.notional_exposure = fill.notional
            pos.margin_used = fill.margin_required
            pos.leverage = self.base_cfg.base.fixed_leverage
            pos.bars_held = 0
            stop_mult = self.sat_cfg.stop_atr_mult
            pos.stop_price = (
                fill.price - stop_mult * atr if side == Side.LONG else fill.price + stop_mult * atr
            )
            p.bump_daily_open()

    def on_bar_close(
        self,
        signal: TradingSignal,
        bp: BestPointSignal | None,
        current_bar: Bar,
        next_bar: Bar,
        account: AccountEquity,
        *,
        core_reason_code: str = "",
    ) -> TradingAction:
        self.portfolio.reset_daily_opens(current_bar.ts)
        self._mark_to_market(current_bar.close)
        if not self.portfolio.position.is_flat:
            self.portfolio.position.bars_held += 1

        action = self.rules.decide(
            signal,
            bp,
            self.portfolio.position,
            account,
            bar_index=current_bar.idx,
            current_price=current_bar.close,
            daily_open_count=self.portfolio.daily_open_count,
            core_reason_code=core_reason_code,
        )

        self.decisions.append(
            {
                "slot_id": "satellite",
                "ts": signal.ts,
                "action": action.action.value,
                "reason_code": action.reason_code,
                "bp_p_long": bp.p_long_entry_zone if bp else 0.0,
                "bp_p_short": bp.p_short_entry_zone if bp else 0.0,
                "core_reason": core_reason_code,
                "state": self.portfolio.position.side.value,
            }
        )

        if action.action in (ActionType.OPEN_LONG, ActionType.OPEN_SHORT, ActionType.CLOSE):
            if action.action in (ActionType.OPEN_LONG, ActionType.OPEN_SHORT):
                self._entry_core_reason = core_reason_code
            elif action.action == ActionType.CLOSE:
                self._entry_core_reason = ""
            ratio = self.sat_cfg.max_position_ratio if action.action != ActionType.CLOSE else 0.0
            if action.action in (ActionType.OPEN_LONG, ActionType.OPEN_SHORT):
                ratio = min(ratio, self.sat_cfg.max_position_ratio)
            from trading_system.sizing import SizedAction

            sized = SizedAction(
                action.action,
                action.target_side,
                action.reason_code,
                ratio,
                ratio * self.base_cfg.base.fixed_leverage,
                ratio,
            )
            fill = self.execution.execute(
                sized,
                ts=next_bar.ts,
                next_open=next_bar.open,
                current_position_ratio=self.portfolio.position.position_ratio,
            )
            self._apply_fill(fill, current_bar=current_bar, atr=signal.atr)

        self.portfolio.peak_equity = max(self.portfolio.peak_equity, self.portfolio.equity)
        account.satellite_position = self.portfolio.position
        account._refresh_aggregate()
        return action

    def incremental_pnl(self) -> float:
        return self.portfolio.equity - 1.0
