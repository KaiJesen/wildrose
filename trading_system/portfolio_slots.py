"""027 dual-slot account model: Core + Satellite with DAD conflict gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from trading_system.enums import Side
from trading_system.portfolio import PortfolioState
from trading_system.state import PositionState


class SlotId(str, Enum):
    CORE = "core"
    SATELLITE = "satellite"


@dataclass
class AccountEquity:
    """Combined account over Core and Satellite slots (027)."""

    equity: float = 1.0
    cash: float = 1.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    margin_used: float = 0.0
    peak_equity: float = 1.0
    core_position: PositionState = field(default_factory=PositionState)
    satellite_position: PositionState = field(default_factory=PositionState)
    # Mirrors legacy portfolio time gates when Core drives the account.
    day_start_equity: float = 1.0
    week_start_equity: float = 1.0
    daily_open_block: bool = False
    weekly_defensive_mode: bool = False
    account_circuit_breaker: bool = False
    loss_streak: int = 0
    cooldown_until: int = -1

    def sync_from_core_portfolio(self, portfolio: PortfolioState) -> None:
        """Phase 0: Core-only path — account mirrors legacy single-slot portfolio."""
        self.equity = portfolio.equity
        self.cash = portfolio.cash
        self.realized_pnl = portfolio.realized_pnl
        self.unrealized_pnl = portfolio.unrealized_pnl
        self.margin_used = portfolio.position.margin_used
        self.core_position = portfolio.position
        self.day_start_equity = portfolio.day_start_equity
        self.week_start_equity = portfolio.week_start_equity
        self.daily_open_block = portfolio.daily_open_block
        self.weekly_defensive_mode = portfolio.weekly_defensive_mode
        self.account_circuit_breaker = portfolio.account_circuit_breaker
        self.loss_streak = portfolio.loss_streak
        self.cooldown_until = portfolio.cooldown_until
        self._refresh_aggregate()

    def _refresh_aggregate(self) -> None:
        if self.satellite_position.is_flat:
            self.margin_used = self.core_position.margin_used
            return
        self.margin_used = self.core_position.margin_used + self.satellite_position.margin_used

    def update_peak_equity(self) -> None:
        self.peak_equity = max(self.peak_equity, self.equity)

    @property
    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity

    @property
    def net_side(self) -> Side:
        if not self.core_position.is_flat:
            return self.core_position.side
        if not self.satellite_position.is_flat:
            return self.satellite_position.side
        return Side.FLAT

    def allow_satellite_long(self) -> bool:
        """DAD: Core SHORT blocks Satellite LONG."""
        if self.core_position.side == Side.SHORT:
            return False
        return True

    def allow_satellite_short(self) -> bool:
        """DAD: Core LONG blocks Satellite SHORT."""
        if self.core_position.side == Side.LONG:
            return False
        return True

    def dad_block_reason(self, *, side: Side) -> str:
        if side == Side.LONG and not self.allow_satellite_long():
            return "BLOCK_DAD_CORE_SHORT"
        if side == Side.SHORT and not self.allow_satellite_short():
            return "BLOCK_DAD_CORE_LONG"
        return ""
