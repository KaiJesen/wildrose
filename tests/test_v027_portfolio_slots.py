"""027 portfolio_slots and DAD gates."""

from __future__ import annotations

from trading_system.enums import Side
from trading_system.portfolio import PortfolioState
from trading_system.portfolio_slots import AccountEquity
from trading_system.state import PositionState


def test_sync_from_core_portfolio() -> None:
    p = PortfolioState(equity=1.05, realized_pnl=0.05)
    p.position = PositionState(side=Side.LONG, position_ratio=0.1, margin_used=0.05)
    acct = AccountEquity()
    acct.sync_from_core_portfolio(p)
    assert acct.equity == 1.05
    assert acct.core_position.side == Side.LONG
    assert acct.satellite_position.is_flat


def test_dad_blocks_cross_slot_reverse() -> None:
    acct = AccountEquity()
    acct.core_position = PositionState(side=Side.LONG)
    assert acct.allow_satellite_short() is False
    assert acct.allow_satellite_long() is True
    assert acct.dad_block_reason(side=Side.SHORT) == "BLOCK_DAD_CORE_LONG"

    acct.core_position = PositionState(side=Side.SHORT)
    assert acct.allow_satellite_long() is False
    assert acct.dad_block_reason(side=Side.LONG) == "BLOCK_DAD_CORE_SHORT"

    acct.core_position = PositionState(side=Side.FLAT)
    assert acct.allow_satellite_long() is True
    assert acct.allow_satellite_short() is True


def test_drawdown_on_peak() -> None:
    acct = AccountEquity(equity=0.9, peak_equity=1.0)
    assert abs(acct.drawdown - 0.1) < 1e-9
