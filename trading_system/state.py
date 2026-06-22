from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from trading_system.enums import Side


@dataclass
class PositionState:
    side: Side = Side.FLAT
    entry_ts: datetime | None = None
    entry_price: float = 0.0
    avg_price: float = 0.0
    position_ratio: float = 0.0
    notional_exposure: float = 0.0
    margin_used: float = 0.0
    leverage: float = 0.0
    add_count: int = 0
    bars_held: int = 0
    continue_fail_count: int = 0
    stop_price: float = 0.0
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0
    trailing_stop_price: float = 0.0
    peak_unrealized_pnl: float = 0.0
    tp1_done: bool = False
    tp2_done: bool = False
    short_reverse_confirm_count: int = 0
    entry_was_probe: bool = False
    hold_mode: str = "NORMAL"
    trend_hold_bars: int = 0
    trend_break_count: int = 0
    entry_was_sentinel: bool = False
    sentinel_bars: int = 0
    peak_profit_atr: float = 0.0
    entry_was_crash: bool = False
    crash_bars: int = 0
    crash_regime_id: int = 0
    entry_trend_direction: str = "NONE"
    entry_trend_strength: str = "NONE"
    trend_upgrade_done: bool = False
    trend_position_type: str = "NONE"
    trend_entry_score: float = 0.0
    trend_peak_score: float = 0.0
    trend_invalid_count: int = 0
    entry_signal_snapshot: dict = field(default_factory=dict)

    @property
    def is_flat(self) -> bool:
        return self.side == Side.FLAT

