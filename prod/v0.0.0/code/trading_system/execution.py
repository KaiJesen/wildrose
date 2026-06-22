from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from trading_system.config import TradingSystemConfig
from trading_system.enums import ActionType, Side
from trading_system.sizing import SizedAction


@dataclass
class FillEvent:
    ts: datetime
    action: ActionType
    side: Side
    requested_position_ratio: float
    filled_position_ratio: float
    price: float
    fee: float
    slippage: float
    notional: float
    margin_required: float
    realized_pnl: float
    reason_code: str


class BacktestExecutionEngine:
    def __init__(self, cfg: TradingSystemConfig) -> None:
        self.cfg = cfg

    def execute(
        self,
        action: SizedAction,
        *,
        ts: datetime,
        next_open: float,
        current_position_ratio: float,
    ) -> FillEvent:
        if action.target_side == Side.LONG:
            price = next_open * (1.0 + self.cfg.execution.slippage_bps / 10000.0)
            slip = self.cfg.execution.slippage_bps
        elif action.target_side == Side.SHORT:
            price = next_open * (1.0 - self.cfg.execution.slippage_bps / 10000.0)
            slip = self.cfg.execution.slippage_bps
        else:
            price = next_open
            slip = 0.0
        delta = abs(action.position_ratio - current_position_ratio) * self.cfg.base.fixed_leverage
        fee = delta * self.cfg.execution.fee_bps / 10000.0
        return FillEvent(
            ts=ts,
            action=action.action,
            side=action.target_side,
            requested_position_ratio=action.position_ratio,
            filled_position_ratio=action.position_ratio,
            price=price,
            fee=fee,
            slippage=slip,
            notional=action.notional_exposure,
            margin_required=action.margin_required,
            realized_pnl=0.0,
            reason_code=action.reason_code,
        )

