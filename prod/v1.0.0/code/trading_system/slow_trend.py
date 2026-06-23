from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from trading_system.config import SlowUptrendConfig


@dataclass
class SlowTrendContext:
    is_slow_uptrend: bool
    is_stable_slow_uptrend: bool
    slow_up_score: float
    slope_24_atr: float
    slope_48_atr: float
    persistence_above_ema_fast: float
    persistence_above_ema_mid: float
    pullback_depth_atr: float
    max_drawdown_24_atr: float
    volatility_compression: float
    ema_fast: float
    ema_mid: float
    ema_slow: float
    ret_6_atr: float
    is_range: bool
    reason_codes: list[str] = field(default_factory=list)


class SlowUptrendDetector:
    def __init__(self, cfg: SlowUptrendConfig) -> None:
        self.cfg = cfg

    @staticmethod
    def _ema_series(values: np.ndarray, period: int) -> np.ndarray:
        alpha = 2.0 / (period + 1.0)
        out = np.empty_like(values, dtype=np.float64)
        out[0] = float(values[0])
        for i in range(1, len(values)):
            out[i] = alpha * float(values[i]) + (1.0 - alpha) * out[i - 1]
        return out

    def compute(
        self,
        close_hist: list[float],
        high_hist: list[float],
        low_hist: list[float],
        atr: float,
        *,
        p_risk: float = 0.0,
        p_flat: float = 0.0,
    ) -> SlowTrendContext:
        c = self.cfg
        empty = SlowTrendContext(
            is_slow_uptrend=False,
            is_stable_slow_uptrend=False,
            slow_up_score=0.0,
            slope_24_atr=0.0,
            slope_48_atr=0.0,
            persistence_above_ema_fast=0.0,
            persistence_above_ema_mid=0.0,
            pullback_depth_atr=0.0,
            max_drawdown_24_atr=0.0,
            volatility_compression=0.0,
            ema_fast=0.0,
            ema_mid=0.0,
            ema_slow=0.0,
            ret_6_atr=0.0,
            is_range=True,
            reason_codes=["DISABLED"] if not c.enabled else ["INSUFFICIENT_HISTORY"],
        )
        min_len = max(c.ema_slow, 48, 24) + 1
        if (not c.enabled) or len(close_hist) < min_len:
            return empty

        close = np.asarray(close_hist, dtype=np.float64)
        high = np.asarray(high_hist, dtype=np.float64)
        low = np.asarray(low_hist, dtype=np.float64)
        atr_v = max(float(atr), 1e-12)

        ema_fast_s = self._ema_series(close, c.ema_fast)
        ema_mid_s = self._ema_series(close, c.ema_mid)
        ema_slow_s = self._ema_series(close, c.ema_slow)
        ema_fast = float(ema_fast_s[-1])
        ema_mid = float(ema_mid_s[-1])
        ema_slow = float(ema_slow_s[-1])
        price = float(close[-1])

        slope_24_atr = (price - float(close[-25])) / atr_v if len(close) >= 25 else 0.0
        slope_48_atr = (price - float(close[-49])) / atr_v if len(close) >= 49 else 0.0
        ret_6_atr = (price - float(close[-7])) / atr_v if len(close) >= 7 else 0.0

        win = close[-24:]
        ema_fast_win = ema_fast_s[-24:]
        ema_mid_win = ema_mid_s[-24:]
        persistence_fast = float(np.mean(win > ema_fast_win))
        persistence_mid = float(np.mean(win > ema_mid_win))

        pullback_depth_atr = max(0.0, (ema_fast - price) / atr_v)
        rolling_high_24 = float(np.max(high[-24:]))
        max_drawdown_24_atr = max(0.0, (rolling_high_24 - price) / atr_v)

        if len(high) >= 48 and len(low) >= 48:
            rh48 = float(np.max(high[-48:]))
            rl48 = float(np.min(low[-48:]))
            volatility_compression = (rh48 - rl48) / max(atr_v, 1e-12)
        else:
            volatility_compression = 0.0

        ema_mid_rising = len(ema_mid_s) >= 3 and ema_mid_s[-1] > ema_mid_s[-3]
        ema_structure_ok = ema_fast > ema_mid and (ema_mid >= ema_slow or ema_mid_rising)

        is_range = False
        reason_codes: list[str] = []
        if slope_48_atr < 1.0 and volatility_compression < 4.0:
            is_range = True
            reason_codes.append("RANGE_LOW_SLOPE")
        if p_flat >= 0.50 and slope_48_atr < c.slope_48_atr_min:
            is_range = True
            reason_codes.append("RANGE_HIGH_FLAT")

        score = 0.0
        if price > ema_fast:
            score += 1.0
            reason_codes.append("ABOVE_EMA_FAST")
        if ema_fast > ema_mid:
            score += 1.0
            reason_codes.append("EMA_FAST_ABOVE_MID")
        if ema_mid >= ema_slow or ema_mid_rising:
            score += 1.0
            reason_codes.append("EMA_MID_STRUCTURE")
        if slope_24_atr >= c.slope_24_atr_min or slope_48_atr >= c.slope_48_atr_min:
            score += 1.0
            reason_codes.append("SLOPE_OK")
        if slope_48_atr >= c.slope_48_atr_min:
            score += 1.0
            reason_codes.append("SLOPE_48_OK")
        if persistence_fast >= c.persistence_fast_min:
            score += 1.0
            reason_codes.append("PERSIST_FAST")
        if persistence_mid >= c.persistence_mid_min:
            score += 1.0
            reason_codes.append("PERSIST_MID")
        if max_drawdown_24_atr <= c.max_drawdown_24_atr:
            score += 1.0
            reason_codes.append("DRAWDOWN_OK")
        if p_risk <= c.risk_score_max:
            score += 1.0
            reason_codes.append("RISK_OK")

        is_slow = (not is_range) and score >= c.min_score and ema_structure_ok
        is_stable = (
            is_slow
            and score >= c.stable_score
            and slope_24_atr >= c.stable_slope_24_atr_min
            and slope_48_atr >= c.stable_slope_48_atr_min
            and persistence_fast >= c.stable_persistence_fast_min
            and persistence_mid >= c.stable_persistence_mid_min
        )
        if is_slow:
            reason_codes.append("SLOW_UPTREND")
        if is_stable:
            reason_codes.append("STABLE_SLOW_UPTREND")
        if pullback_depth_atr > c.pullback_depth_atr_max:
            is_slow = False
            is_stable = False
            reason_codes.append("PULLBACK_TOO_DEEP")

        return SlowTrendContext(
            is_slow_uptrend=is_slow,
            is_stable_slow_uptrend=is_stable,
            slow_up_score=score,
            slope_24_atr=slope_24_atr,
            slope_48_atr=slope_48_atr,
            persistence_above_ema_fast=persistence_fast,
            persistence_above_ema_mid=persistence_mid,
            pullback_depth_atr=pullback_depth_atr,
            max_drawdown_24_atr=max_drawdown_24_atr,
            volatility_compression=volatility_compression,
            ema_fast=ema_fast,
            ema_mid=ema_mid,
            ema_slow=ema_slow,
            ret_6_atr=ret_6_atr,
            is_range=is_range,
            reason_codes=reason_codes,
        )
