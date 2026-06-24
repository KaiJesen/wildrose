from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from trading_system.config import TrendSignalConfig


class TrendDirection(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    NONE = "NONE"


class TrendStrength(str, Enum):
    NONE = "NONE"
    WEAK = "WEAK"
    NORMAL = "NORMAL"
    STRONG = "STRONG"
    EXTREME = "EXTREME"


class TrendPhase(str, Enum):
    NONE = "NONE"
    EARLY = "EARLY"
    CONTINUATION = "CONTINUATION"
    ACCELERATION = "ACCELERATION"
    EXHAUSTION = "EXHAUSTION"
    REVERSAL_RISK = "REVERSAL_RISK"


class ConfirmTier(str, Enum):
    NONE = "NONE"
    CANDIDATE = "CANDIDATE"
    CONFIRMED = "CONFIRMED"
    SUSTAINED = "SUSTAINED"


@dataclass
class TrendMemory:
    current_direction: TrendDirection = TrendDirection.NONE
    trend_age: int = 0
    invalid_count: int = 0
    phase: TrendPhase = TrendPhase.NONE


@dataclass
class TrendSignal:
    direction: TrendDirection
    strength: TrendStrength
    phase: TrendPhase
    score_up: float
    score_down: float
    score_abs: float
    confidence: float
    trend_age: int
    invalid_count: int
    is_confirmed: bool
    is_broken: bool
    is_accelerating: bool
    is_exhausted: bool
    ret_6_atr: float
    ret_12_atr: float
    ret_24_atr: float
    ema_fast: float
    ema_mid: float
    ema_slow: float
    close_to_ema_fast_atr: float
    distance_from_ema_slow_atr: float
    rolling_high_break: bool
    rolling_low_break: bool
    higher_high_low: bool
    lower_high_low: bool
    persistence_ratio: float
    range_expansion: float
    is_sustained: bool = False
    confirm_tier: ConfirmTier = ConfirmTier.NONE
    reason_codes: list[str] = field(default_factory=list)


class TrendSignalProvider:
    def __init__(self, cfg: TrendSignalConfig) -> None:
        self.cfg = cfg

    @staticmethod
    def _ema(values: np.ndarray, period: int) -> float:
        alpha = 2.0 / (period + 1.0)
        ema = float(values[0])
        for v in values[1:]:
            ema = alpha * float(v) + (1.0 - alpha) * ema
        return ema

    @staticmethod
    def _trend_structure(high: np.ndarray, low: np.ndarray) -> tuple[bool, bool]:
        if len(high) < 4 or len(low) < 4:
            return False, False
        higher = bool(np.all(np.diff(high[-4:]) >= 0.0) and np.all(np.diff(low[-4:]) >= 0.0))
        lower = bool(np.all(np.diff(high[-4:]) <= 0.0) and np.all(np.diff(low[-4:]) <= 0.0))
        return higher, lower

    @staticmethod
    def _path_efficiency(close: np.ndarray, lookback: int) -> float:
        if len(close) < lookback + 1:
            return 1.0
        window = close[-(lookback + 1) :]
        net = abs(float(window[-1] - window[0]))
        gross = float(np.sum(np.abs(np.diff(window))))
        return net / max(gross, 1e-12)

    def _ema_flip_count(self, close: np.ndarray, ema_period: int, lookback: int) -> int:
        if len(close) < lookback + ema_period:
            return 0
        flips = 0
        prev_sign: int | None = None
        n = len(close)
        for offset in range(-lookback, 0):
            idx = n + offset
            ema = self._ema(close[idx - ema_period + 1 : idx + 1], ema_period)
            diff = float(close[idx]) - ema
            sign = 1 if diff > 0 else (-1 if diff < 0 else 0)
            if prev_sign is not None and sign != 0 and prev_sign != 0 and sign != prev_sign:
                flips += 1
            if sign != 0:
                prev_sign = sign
        return flips

    def _slow_trend_exception(
        self,
        close: np.ndarray,
        *,
        ema_mid: float,
        ema_slow: float,
        atr: float,
        persistence_above_ema_mid: float,
        efficiency_48: float,
    ) -> tuple[bool, list[str]]:
        c = self.cfg
        if not c.chop_slow_trend_exception_enabled:
            return False, []
        reasons: list[str] = []
        slope_bars = c.chop_ema_slope_bars
        if len(close) > slope_bars:
            ema_now = self._ema(close[-c.ema_slow :], c.ema_slow)
            ema_prev = self._ema(close[-(c.ema_slow + slope_bars) : -slope_bars], c.ema_slow)
            slope_atr = (ema_now - ema_prev) / max(atr, 1e-12)
            if slope_atr >= c.chop_ema_slope_atr_min and ema_now > ema_mid > ema_slow:
                reasons.append("SLOW_TREND_EMA_SLOPE")
        if persistence_above_ema_mid >= c.chop_persistence_ema_mid_min:
            reasons.append("SLOW_TREND_PERSISTENCE")
        if efficiency_48 >= c.chop_efficiency_48_min:
            reasons.append("SLOW_TREND_EFFICIENCY_48")
        return bool(reasons), reasons

    def _evaluate_chop(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        *,
        atr: float,
        ret_24_atr: float,
        ema_mid: float,
        ema_slow: float,
    ) -> tuple[str, list[str]]:
        c = self.cfg
        if not c.chop_guard_enabled:
            return "NONE", []
        chop_range_atr = (float(np.max(high[-c.structure_lookback:])) - float(np.min(low[-c.structure_lookback:]))) / max(
            atr, 1e-12
        )
        flip_count = self._ema_flip_count(close, c.ema_fast, c.chop_flip_lookback)
        efficiency_12 = self._path_efficiency(close, c.chop_efficiency_lookback)
        efficiency_48 = self._path_efficiency(close, 48) if len(close) >= 49 else 1.0
        close_window = close[-c.persistence_lookback :]
        persistence_above_ema_mid = float(np.mean(close_window > ema_mid))

        candidates: list[str] = []
        if chop_range_atr < c.chop_range_atr_max and abs(ret_24_atr) < c.chop_ret24_atr_max:
            candidates.append("CHOP_RANGE")
        if flip_count >= c.chop_flip_max:
            candidates.append("CHOP_FLIP")
        if efficiency_12 < c.chop_efficiency_min:
            candidates.append("CHOP_INEFFICIENT")

        if not candidates:
            return "NONE", []

        slow_ok, slow_reasons = self._slow_trend_exception(
            close,
            ema_mid=ema_mid,
            ema_slow=ema_slow,
            atr=atr,
            persistence_above_ema_mid=persistence_above_ema_mid,
            efficiency_48=efficiency_48,
        )
        if slow_ok and len(candidates) >= 2:
            return "NONE", slow_reasons

        if len(candidates) >= 2:
            return "CHOP_HARD", ["CHOP_HARD", *candidates]
        return "CHOP_SOFT", ["CHOP_SOFT", *candidates]

    def _confirm_tier_for(
        self,
        *,
        direction: TrendDirection,
        is_confirmed: bool,
        is_sustained: bool,
        chop_level: str,
    ) -> ConfirmTier:
        if chop_level == "CHOP_HARD" or direction == TrendDirection.NONE:
            return ConfirmTier.NONE
        if is_confirmed:
            return ConfirmTier.CONFIRMED
        if is_sustained:
            return ConfirmTier.SUSTAINED
        if direction != TrendDirection.NONE:
            return ConfirmTier.CANDIDATE
        return ConfirmTier.NONE

    def _empty_signal(self, memory: TrendMemory, *, close_hist: list[float], reasons: list[str] | None = None) -> TrendSignal:
        c = self.cfg
        broken_at = max(c.invalid_confirm_bars, c.invalid_reset_bars)
        return TrendSignal(
            direction=TrendDirection.NONE,
            strength=TrendStrength.NONE,
            phase=TrendPhase.NONE,
            score_up=0.0,
            score_down=0.0,
            score_abs=0.0,
            confidence=0.0,
            trend_age=memory.trend_age,
            invalid_count=memory.invalid_count,
            is_confirmed=False,
            is_broken=memory.invalid_count >= broken_at,
            is_accelerating=False,
            is_exhausted=False,
            is_sustained=False,
            confirm_tier=ConfirmTier.NONE,
            ret_6_atr=0.0,
            ret_12_atr=0.0,
            ret_24_atr=0.0,
            ema_fast=close_hist[-1] if close_hist else 0.0,
            ema_mid=close_hist[-1] if close_hist else 0.0,
            ema_slow=close_hist[-1] if close_hist else 0.0,
            close_to_ema_fast_atr=0.0,
            distance_from_ema_slow_atr=0.0,
            rolling_high_break=False,
            rolling_low_break=False,
            higher_high_low=False,
            lower_high_low=False,
            persistence_ratio=0.0,
            range_expansion=1.0,
            reason_codes=list(reasons or []),
        )

    def compute(
        self,
        *,
        close_hist: list[float],
        high_hist: list[float],
        low_hist: list[float],
        atr_hist: list[float],
        memory: TrendMemory,
    ) -> TrendSignal:
        c = self.cfg
        min_len = max(
            c.ema_slow,
            c.ret_slow + 1,
            c.structure_lookback + 1,
            c.persistence_lookback + 1,
            c.chop_ema_slope_bars + c.ema_slow + 1,
            49,
        )
        if (not c.enabled) or len(close_hist) < min_len:
            memory.current_direction = TrendDirection.NONE
            memory.trend_age = 0
            memory.invalid_count += 1
            return self._empty_signal(memory, close_hist=close_hist)

        close = np.asarray(close_hist, dtype=np.float64)
        high = np.asarray(high_hist, dtype=np.float64)
        low = np.asarray(low_hist, dtype=np.float64)
        atr = max(float(atr_hist[-1]), 1e-12)
        cur = float(close[-1])
        ema_fast = self._ema(close[-c.ema_fast :], c.ema_fast)
        ema_mid = self._ema(close[-c.ema_mid :], c.ema_mid)
        ema_slow = self._ema(close[-c.ema_slow :], c.ema_slow)

        ret_6_atr = (cur - float(close[-1 - c.ret_fast])) / atr
        ret_12_atr = (cur - float(close[-1 - c.ret_mid])) / atr
        ret_24_atr = (cur - float(close[-1 - c.ret_slow])) / atr

        chop_level, chop_reasons = self._evaluate_chop(
            close,
            high,
            low,
            atr=atr,
            ret_24_atr=ret_24_atr,
            ema_mid=ema_mid,
            ema_slow=ema_slow,
        )
        if chop_level == "CHOP_HARD":
            memory.invalid_count += 1
            if memory.invalid_count >= c.invalid_reset_bars:
                memory.current_direction = TrendDirection.NONE
                memory.trend_age = 0
            return self._empty_signal(memory, close_hist=close_hist, reasons=chop_reasons)

        rolling_high = float(np.max(high[-c.structure_lookback:]))
        rolling_low = float(np.min(low[-c.structure_lookback:]))
        rolling_high_break = cur >= rolling_high
        rolling_low_break = cur <= rolling_low
        higher_hl, lower_hl = self._trend_structure(high[-c.structure_lookback :], low[-c.structure_lookback :])

        close_window = close[-c.persistence_lookback :]
        persistence_up = float(np.mean(close_window > ema_fast))
        persistence_down = float(np.mean(close_window < ema_fast))

        tr = np.maximum(high[-c.structure_lookback :] - low[-c.structure_lookback :], 1e-12)
        range_expansion = float(np.mean(tr[-3:]) / max(np.mean(tr), 1e-12))

        score_up = 0
        score_down = 0
        reasons: list[str] = list(chop_reasons)
        if cur > ema_fast:
            score_up += 1
            reasons.append("CLOSE_GT_EMA_FAST")
        if ema_fast > ema_mid > ema_slow:
            score_up += 1
            reasons.append("EMA_BULL_STACK")
        if ret_12_atr >= 1.5:
            score_up += 1
            reasons.append("RET12_UP")
        if ret_24_atr >= 2.5:
            score_up += 1
        if rolling_high_break:
            score_up += 1
        if persistence_up >= 0.67:
            score_up += 1
        if higher_hl:
            score_up += 1

        if cur < ema_fast:
            score_down += 1
            reasons.append("CLOSE_LT_EMA_FAST")
        if ema_fast < ema_mid < ema_slow:
            score_down += 1
            reasons.append("EMA_BEAR_STACK")
        if ret_12_atr <= -1.5:
            score_down += 1
            reasons.append("RET12_DOWN")
        if ret_24_atr <= -2.5:
            score_down += 1
        if rolling_low_break:
            score_down += 1
        if persistence_down >= 0.67:
            score_down += 1
        if lower_hl:
            score_down += 1

        if score_up - score_down >= c.direction_margin:
            direction = TrendDirection.UP
            score_abs = float(score_up)
            persistence_ratio = persistence_up
        elif score_down - score_up >= c.direction_margin:
            direction = TrendDirection.DOWN
            score_abs = float(score_down)
            persistence_ratio = persistence_down
        else:
            direction = TrendDirection.NONE
            score_abs = float(max(score_up, score_down))
            persistence_ratio = max(persistence_up, persistence_down)

        raw_confirmed = score_abs >= c.confirmed_score and direction != TrendDirection.NONE
        hold_ok = (
            memory.trend_age >= c.min_trend_age_for_hold
            and score_abs >= c.hold_confirm_score
            and direction == memory.current_direction
            and direction != TrendDirection.NONE
        )
        is_confirmed = raw_confirmed and chop_level != "CHOP_SOFT"
        is_sustained = (
            direction != TrendDirection.NONE
            and direction == memory.current_direction
            and (is_confirmed or hold_ok)
        )

        prev_direction = memory.current_direction
        if is_confirmed and direction == memory.current_direction:
            memory.trend_age += 1
            memory.invalid_count = 0
        elif is_confirmed:
            memory.current_direction = direction
            memory.trend_age = 1
            memory.invalid_count = 0
        elif hold_ok:
            memory.trend_age += 1
            memory.invalid_count = max(0, memory.invalid_count - 1)
        elif (
            direction != TrendDirection.NONE
            and prev_direction != TrendDirection.NONE
            and direction != prev_direction
            and abs(score_up - score_down) >= c.direction_margin
        ):
            memory.invalid_count += 2
        else:
            memory.invalid_count += 1
            if memory.invalid_count >= c.invalid_reset_bars:
                memory.current_direction = TrendDirection.NONE
                memory.trend_age = 0

        is_broken = memory.invalid_count >= c.invalid_reset_bars
        confirm_tier = self._confirm_tier_for(
            direction=direction,
            is_confirmed=is_confirmed,
            is_sustained=is_sustained,
            chop_level=chop_level,
        )

        if direction == TrendDirection.NONE:
            strength = TrendStrength.NONE if score_abs <= 1 else TrendStrength.WEAK
        elif score_abs >= c.extreme_score and abs(ret_24_atr) >= 4.0:
            strength = TrendStrength.EXTREME
        elif score_abs >= c.strong_score:
            strength = TrendStrength.STRONG
        elif score_abs >= c.confirmed_score:
            strength = TrendStrength.NORMAL
        else:
            strength = TrendStrength.WEAK

        is_accelerating = (
            direction != TrendDirection.NONE
            and range_expansion >= c.acceleration_range_expansion
            and abs(ret_6_atr) >= 2.0
        )
        distance_slow = (cur - ema_slow) / atr
        is_exhausted = abs(distance_slow) >= c.exhaustion_distance_atr and range_expansion >= 2.0

        phase = TrendPhase.NONE
        if direction == TrendDirection.NONE:
            phase = TrendPhase.NONE
        elif memory.trend_age <= 3 and score_abs >= 3:
            phase = TrendPhase.EARLY
        elif is_exhausted:
            phase = TrendPhase.EXHAUSTION
        elif is_accelerating and score_abs >= c.strong_score:
            phase = TrendPhase.ACCELERATION
        elif is_broken:
            phase = TrendPhase.REVERSAL_RISK
        else:
            phase = TrendPhase.CONTINUATION

        confidence = abs(score_up - score_down) / max(score_up, score_down, 1)
        return TrendSignal(
            direction=direction,
            strength=strength,
            phase=phase,
            score_up=float(score_up),
            score_down=float(score_down),
            score_abs=float(score_abs),
            confidence=float(confidence),
            trend_age=memory.trend_age,
            invalid_count=memory.invalid_count,
            is_confirmed=is_confirmed,
            is_broken=is_broken,
            is_accelerating=is_accelerating,
            is_exhausted=is_exhausted,
            is_sustained=is_sustained,
            confirm_tier=confirm_tier,
            ret_6_atr=float(ret_6_atr),
            ret_12_atr=float(ret_12_atr),
            ret_24_atr=float(ret_24_atr),
            ema_fast=float(ema_fast),
            ema_mid=float(ema_mid),
            ema_slow=float(ema_slow),
            close_to_ema_fast_atr=float((cur - ema_fast) / atr),
            distance_from_ema_slow_atr=float(distance_slow),
            rolling_high_break=rolling_high_break,
            rolling_low_break=rolling_low_break,
            higher_high_low=higher_hl,
            lower_high_low=lower_hl,
            persistence_ratio=float(persistence_ratio),
            range_expansion=float(range_expansion),
            reason_codes=reasons,
        )
