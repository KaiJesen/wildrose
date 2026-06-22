from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from trading_system.config import TrendConfig


@dataclass
class TrendContext:
    is_downtrend: bool
    is_strong_downtrend: bool
    is_uptrend: bool
    trend_score: float
    ret_3_atr: float
    ret_6_atr: float
    ema_fast: float
    ema_slow: float
    breakdown_low_n: bool
    lower_high_low: bool
    reason_codes: list[str] = field(default_factory=list)


class TrendRegimeFilter:
    def __init__(self, cfg: TrendConfig) -> None:
        self.cfg = cfg

    @staticmethod
    def _ema_from_history(values: list[float], period: int) -> float:
        if not values:
            return 0.0
        alpha = 2.0 / (period + 1.0)
        ema = float(values[0])
        for v in values[1:]:
            ema = alpha * float(v) + (1.0 - alpha) * ema
        return ema

    @staticmethod
    def _lower_high_low(high_hist: list[float], low_hist: list[float], lookback: int) -> bool:
        if len(high_hist) < lookback or len(low_hist) < lookback:
            return False
        hs = np.asarray(high_hist[-lookback:], dtype=np.float64)
        ls = np.asarray(low_hist[-lookback:], dtype=np.float64)
        return bool(np.all(np.diff(hs) <= 0.0) and np.all(np.diff(ls) <= 0.0))

    def compute(self, close_hist: list[float], high_hist: list[float], low_hist: list[float], atr: float) -> TrendContext:
        if (not self.cfg.enabled) or len(close_hist) < max(self.cfg.ema_slow, self.cfg.ret_lookback_slow + 1):
            return TrendContext(
                is_downtrend=False,
                is_strong_downtrend=False,
                is_uptrend=False,
                trend_score=0.0,
                ret_3_atr=0.0,
                ret_6_atr=0.0,
                ema_fast=close_hist[-1] if close_hist else 0.0,
                ema_slow=close_hist[-1] if close_hist else 0.0,
                breakdown_low_n=False,
                lower_high_low=False,
                reason_codes=[],
            )
        c = float(close_hist[-1])
        atr_safe = max(atr, 1e-12)
        ret_3_atr = (c - float(close_hist[-1 - self.cfg.ret_lookback_fast])) / atr_safe
        ret_6_atr = (c - float(close_hist[-1 - self.cfg.ret_lookback_slow])) / atr_safe
        ema_fast = self._ema_from_history(close_hist[-self.cfg.ema_fast :], self.cfg.ema_fast)
        ema_slow = self._ema_from_history(close_hist[-self.cfg.ema_slow :], self.cfg.ema_slow)
        rolling_low = float(np.min(np.asarray(low_hist[-self.cfg.breakdown_lookback :], dtype=np.float64)))
        breakdown = c <= rolling_low
        lower_hl = self._lower_high_low(high_hist, low_hist, lookback=self.cfg.ret_lookback_slow + 1)

        votes = 0
        reason_codes: list[str] = []
        if ret_6_atr <= self.cfg.down_ret_atr_threshold:
            votes += 1
            reason_codes.append("DOWN_RET6_ATR")
        if c < ema_fast:
            votes += 1
            reason_codes.append("CLOSE_LT_EMA_FAST")
        if ema_fast < ema_slow:
            votes += 1
            reason_codes.append("EMA_FAST_LT_EMA_SLOW")
        if breakdown:
            votes += 1
            reason_codes.append("BREAKDOWN_LOW_N")
        if lower_hl:
            votes += 1
            reason_codes.append("LOWER_HIGH_LOW")

        strong_votes = votes
        is_downtrend = votes >= self.cfg.min_downtrend_votes
        is_strong_downtrend = strong_votes >= self.cfg.min_strong_downtrend_votes or (
            ret_6_atr <= self.cfg.strong_down_ret_atr_threshold and c < ema_fast
        )
        is_uptrend = c > ema_fast and ema_fast > ema_slow and ret_6_atr > 0.8
        trend_score = float((-ret_6_atr) + max(0.0, ema_slow - ema_fast) / max(1e-12, atr_safe))
        return TrendContext(
            is_downtrend=is_downtrend,
            is_strong_downtrend=is_strong_downtrend,
            is_uptrend=is_uptrend,
            trend_score=trend_score,
            ret_3_atr=float(ret_3_atr),
            ret_6_atr=float(ret_6_atr),
            ema_fast=float(ema_fast),
            ema_slow=float(ema_slow),
            breakdown_low_n=bool(breakdown),
            lower_high_low=bool(lower_hl),
            reason_codes=reason_codes,
        )

