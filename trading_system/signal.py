from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import math

from trading_system.enums import SignalStatus


@dataclass
class TradingSignal:
    ts: datetime
    price: float
    atr: float
    p_up: float
    p_down: float
    p_flat: float
    p_risk: float
    pred_ret_1: float
    pred_ret_2: float
    pred_ret_3: float
    pred_ret_4: float
    pred_ret_5: float
    pred_cum_ret_5: float | None = None
    source: str = "model"
    raw: dict = field(default_factory=dict)
    status: SignalStatus = SignalStatus.VALID
    reason_code: str = ""
    fallback_reason: str = ""

    edge: float = 0.0
    conf: float = 0.0
    cum_edge: float = 0.0
    risk_ok: bool = False
    participate_score_long: float = 0.0
    participate_score_short: float = 0.0
    teq_edge_long: float = 0.0
    teq_edge_short: float = 0.0
    edge_long_hz: float = 0.0
    edge_short_hz: float = 0.0

    def finalize(self, risk_open_max: float) -> "TradingSignal":
        if self.pred_cum_ret_5 is None:
            self.pred_cum_ret_5 = self.pred_ret_1 + self.pred_ret_2 + self.pred_ret_3 + self.pred_ret_4 + self.pred_ret_5
            self.fallback_reason = "cum_return_fallback_from_step_sum"
        self.edge = self.p_up - self.p_down
        self.conf = abs(self.edge)
        self.cum_edge = float(self.pred_cum_ret_5)
        self.risk_ok = self.p_risk <= risk_open_max
        self._validate()
        return self

    @property
    def is_valid(self) -> bool:
        return self.status == SignalStatus.VALID

    def _validate(self) -> None:
        if self.price <= 0:
            self.status = SignalStatus.INVALID
            self.reason_code = "INVALID_SIGNAL_PRICE"
            return
        if self.atr <= 0:
            self.status = SignalStatus.INVALID
            self.reason_code = "INVALID_SIGNAL_ATR"
            return
        vals = [self.p_up, self.p_down, self.p_flat, self.p_risk]
        for v in vals:
            if not math.isfinite(v):
                self.status = SignalStatus.INVALID
                self.reason_code = "INVALID_SIGNAL_NAN_PROB"
                return
        if any(v < 0 or v > 1 for v in vals):
            self.status = SignalStatus.INVALID
            self.reason_code = "INVALID_SIGNAL_PROB_RANGE"
            return
        self.status = SignalStatus.VALID
        self.reason_code = ""

