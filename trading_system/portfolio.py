from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from trading_system.state import PositionState


@dataclass
class PortfolioState:
    equity: float = 1.0
    cash: float = 1.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    margin_used: float = 0.0
    equity_curve: list[float] = field(default_factory=list)
    day_start_equity: float = 1.0
    week_start_equity: float = 1.0
    loss_streak: int = 0
    cooldown_until: int = -1
    daily_open_block: bool = False
    weekly_defensive_mode: bool = False
    account_circuit_breaker: bool = False
    position: PositionState = field(default_factory=PositionState)
    crash_regime_active: bool = False
    crash_regime_id: int = 0
    crash_short_used_in_regime: bool = False
    crash_short_cooldown_until: int = -1
    crash_release_count: int = 0

    _last_ts: datetime | None = None

    def update_time_gates(
        self,
        ts: datetime,
        *,
        day_drawdown_stop: float,
        week_drawdown_defensive: float,
    ) -> None:
        if self._last_ts is None:
            self.day_start_equity = self.equity
            self.week_start_equity = self.equity
            self._last_ts = ts
        else:
            if ts.date() != self._last_ts.date():
                self.day_start_equity = self.equity
                self.daily_open_block = False
            if ts.isocalendar().week != self._last_ts.isocalendar().week or ts.year != self._last_ts.year:
                self.week_start_equity = self.equity
                self.weekly_defensive_mode = False
            self._last_ts = ts
        day_dd = (self.equity - self.day_start_equity) / max(1e-12, self.day_start_equity)
        week_dd = (self.equity - self.week_start_equity) / max(1e-12, self.week_start_equity)
        if day_dd <= -day_drawdown_stop:
            self.daily_open_block = True
        if week_dd <= -week_drawdown_defensive:
            self.weekly_defensive_mode = True

