from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from trading_system.config import TrendSegmentConfig
from trading_system.crash import CrashContext
from trading_system.slow_trend import SlowTrendContext
from trading_system.trend_signal import TrendDirection, TrendPhase, TrendSignal


class TrendLegType(str, Enum):
    SLOW_UP_LEG = "SLOW_UP_LEG"
    FAST_UP_LEG = "FAST_UP_LEG"
    SLOW_DOWN_LEG = "SLOW_DOWN_LEG"
    FAST_DOWN_LEG = "FAST_DOWN_LEG"
    CRASH_LEG = "CRASH_LEG"
    SURGE_LEG = "SURGE_LEG"
    RANGE_LEG = "RANGE_LEG"
    TRANSITION_LEG = "TRANSITION_LEG"
    NONE = "NONE"


class SubLegPhase(str, Enum):
    IMPULSE = "IMPULSE"
    PULLBACK = "PULLBACK"
    BASE = "BASE"
    BREAKOUT = "BREAKOUT"
    EXHAUSTION = "EXHAUSTION"
    LEG_END = "LEG_END"
    NONE = "NONE"


class MarketRegime(str, Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    NEUTRAL = "NEUTRAL"
    HIGH_VOL = "HIGH_VOL"


class LegState(str, Enum):
    NO_LEG = "NO_LEG"
    LEG_FORMING = "LEG_FORMING"
    LEG_CONFIRMED = "LEG_CONFIRMED"
    LEG_CLOSING = "LEG_CLOSING"


@dataclass
class TrendLeg:
    leg_id: int
    leg_type: TrendLegType
    direction: TrendDirection
    start_bar_idx: int
    end_bar_idx: int | None
    duration_bars: int
    leg_return_atr: float
    leg_slope_atr: float
    leg_efficiency: float
    leg_pullback_ratio: float
    leg_vol_expansion: float
    is_confirmed: bool
    is_active: bool
    sub_phase: SubLegPhase
    sub_phase_age: int
    leg_state: LegState
    reason_codes: list[str] = field(default_factory=list)


@dataclass
class SegmentContext:
    regime: MarketRegime
    active_leg: TrendLeg | None
    previous_leg: TrendLeg | None
    leg_type: TrendLegType
    sub_phase: SubLegPhase
    leg_state: LegState
    bars_since_leg_start: int
    bars_to_estimated_leg_end: int | None
    leg_progress_ratio: float
    aligned_with_regime: bool
    should_hold_trend: bool
    should_avoid_counter: bool
    reason_codes: list[str] = field(default_factory=list)


class _AbsBarBuffer:
    """Bounded OHLC/ATR history with absolute bar index mapping (022 C0/C1)."""

    def __init__(self, maxlen: int) -> None:
        self.maxlen = maxlen
        self._data: deque[float] = deque()
        self.buffer_start_bar_idx = 0

    def append(self, value: float) -> None:
        if len(self._data) == self.maxlen:
            self._data.popleft()
            self.buffer_start_bar_idx += 1
        self._data.append(float(value))

    def __len__(self) -> int:
        return len(self._data)

    def abs_to_local(self, abs_idx: int) -> int:
        return abs_idx - self.buffer_start_bar_idx

    def has_abs(self, abs_idx: int) -> bool:
        local = self.abs_to_local(abs_idx)
        return 0 <= local < len(self._data)

    def get_abs(self, abs_idx: int) -> float:
        local = self.abs_to_local(abs_idx)
        if local < 0 or local >= len(self._data):
            raise IndexError(f"abs_idx {abs_idx} outside buffer [{self.buffer_start_bar_idx}, ...)")
        return self._data[local]

    def tail_array(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.asarray([], dtype=np.float64)
        return np.asarray(list(self._data)[-n:], dtype=np.float64)

    def mean_abs_range(self, start_abs: int, end_abs: int) -> float:
        if not self.has_abs(start_abs) or not self.has_abs(end_abs):
            raise IndexError("range outside buffer")
        ls = self.abs_to_local(start_abs)
        le = self.abs_to_local(end_abs)
        return float(np.mean(list(self._data)[ls : le + 1]))


@dataclass
class _LegMetrics:
    start_close: float
    start_atr: float
    prev_close: float
    min_close: float
    max_close: float
    ret_sum: float = 0.0
    gross_sum: float = 0.0
    atr_run_sum: float = 0.0
    atr_run_count: int = 0
    atr_baseline: float = 1.0
    last_bar_idx: int = -1

    @classmethod
    def start(cls, *, bar_idx: int, close: float, atr: float, atr_baseline: float) -> _LegMetrics:
        return cls(
            start_close=close,
            start_atr=max(atr, 1e-12),
            prev_close=close,
            min_close=close,
            max_close=close,
            atr_baseline=max(atr_baseline, 1e-12),
            last_bar_idx=bar_idx,
        )

    def on_bar(self, bar_idx: int, close: float, atr: float) -> None:
        if bar_idx == self.last_bar_idx:
            self.min_close = min(self.min_close, close)
            self.max_close = max(self.max_close, close)
            return
        if bar_idx > self.last_bar_idx:
            ret = (close - self.prev_close) / max(self.prev_close, 1e-12)
            self.ret_sum += ret
            self.gross_sum += abs(ret)
            self.prev_close = close
            self.min_close = min(self.min_close, close)
            self.max_close = max(self.max_close, close)
            self.atr_run_sum += atr
            self.atr_run_count += 1
            self.last_bar_idx = bar_idx

    def efficiency(self) -> float:
        return float(abs(self.ret_sum) / max(self.gross_sum, 1e-12))

    def return_atr(self, close: float) -> float:
        return float((close - self.start_close) / max(self.start_atr, 1e-12))

    def pullback_ratio(self, close: float, direction: TrendDirection) -> float:
        start_px = self.start_close
        leg_ret = abs(close - start_px) / max(start_px, 1e-12)
        if leg_ret < 1e-9:
            return 0.0
        if direction == TrendDirection.UP:
            adverse = max(0.0, (start_px - self.min_close) / max(start_px, 1e-12))
        elif direction == TrendDirection.DOWN:
            adverse = max(0.0, (self.max_close - start_px) / max(start_px, 1e-12))
        else:
            adverse = 0.0
        return float(adverse / leg_ret)

    def vol_expansion(self) -> float:
        cur = self.atr_run_sum / max(self.atr_run_count, 1)
        return float(cur / max(self.atr_baseline, 1e-12))


class SwingDetector:
    def __init__(self, left_bars: int, right_bars: int) -> None:
        self.left_bars = left_bars
        self.right_bars = right_bars
        self._maxlen = left_bars + right_bars + 16
        self.high_hist: deque[float] = deque(maxlen=self._maxlen)
        self.low_hist: deque[float] = deque(maxlen=self._maxlen)
        self.last_swing_high: tuple[int, float] | None = None
        self.last_swing_low: tuple[int, float] | None = None

    def on_bar(self, bar_idx: int, high: float, low: float) -> tuple[bool, bool]:
        self.high_hist.append(float(high))
        self.low_hist.append(float(low))
        confirmed_high = False
        confirmed_low = False
        current = len(self.high_hist) - 1
        candidate = current - self.right_bars
        if candidate >= self.left_bars:
            if self._is_swing_high_confirmed(candidate, current):
                self.last_swing_high = (bar_idx - (current - candidate), float(self.high_hist[candidate]))
                confirmed_high = True
            if self._is_swing_low_confirmed(candidate, current):
                self.last_swing_low = (bar_idx - (current - candidate), float(self.low_hist[candidate]))
                confirmed_low = True
        return confirmed_high, confirmed_low

    def _is_swing_high_confirmed(self, candidate: int, current: int) -> bool:
        peak = self.high_hist[candidate]
        for i in range(max(0, candidate - self.left_bars), current + 1):
            if self.high_hist[i] > peak + 1e-12:
                return False
        return True

    def _is_swing_low_confirmed(self, candidate: int, current: int) -> bool:
        trough = self.low_hist[candidate]
        for i in range(max(0, candidate - self.left_bars), current + 1):
            if self.low_hist[i] < trough - 1e-12:
                return False
        return True


class TrendSegmentEngine:
    def __init__(self, cfg: TrendSegmentConfig) -> None:
        self.cfg = cfg
        self.swing_small = SwingDetector(cfg.swing_small_left_bars, cfg.swing_small_right_bars)
        self.swing_large = SwingDetector(cfg.swing_large_left_bars, cfg.swing_large_right_bars)
        self._leg_id = 0
        self._active_leg: TrendLeg | None = None
        self._previous_leg: TrendLeg | None = None
        self._leg_state = LegState.NO_LEG
        buf = max(cfg.history_buffer_bars, 168 + 48)
        self._close_hist = _AbsBarBuffer(buf)
        self._high_hist = _AbsBarBuffer(buf)
        self._low_hist = _AbsBarBuffer(buf)
        self._atr_hist = _AbsBarBuffer(buf)
        self._leg_metrics: _LegMetrics | None = None
        self._last_large_swing: str = "NONE"
        self._sub_phase_age = 0
        self._typical_leg_lengths: dict[str, list[int]] = {
            TrendLegType.SLOW_UP_LEG.value: [],
            TrendLegType.FAST_DOWN_LEG.value: [],
            TrendLegType.SLOW_DOWN_LEG.value: [],
        }

    def _atr_baseline_before(self, bar_idx: int) -> float:
        if bar_idx <= 0:
            return 1.0
        start = max(0, bar_idx - 24)
        end = bar_idx - 1
        if self._atr_hist.has_abs(start) and self._atr_hist.has_abs(end):
            return self._atr_hist.mean_abs_range(start, end)
        return float(self._atr_hist.tail_array(1)[-1]) if len(self._atr_hist) else 1.0

    def _classify_leg_type(
        self,
        *,
        leg_return_atr: float,
        leg_slope_atr: float,
        leg_efficiency: float,
        duration_bars: int,
        vol_expansion: float,
        direction: TrendDirection,
        slow_ctx: SlowTrendContext | None,
        crash_ctx: CrashContext | None,
        is_model_blind: bool,
    ) -> TrendLegType:
        if crash_ctx and crash_ctx.is_crash and is_model_blind and leg_return_atr <= -3.0 and vol_expansion >= 1.8:
            return TrendLegType.CRASH_LEG
        if slow_ctx and slow_ctx.is_stable_slow_uptrend and leg_return_atr >= 0.8 and duration_bars >= 12:
            return TrendLegType.SLOW_UP_LEG
        if leg_return_atr >= 2.5 and leg_slope_atr >= 0.08:
            return TrendLegType.FAST_UP_LEG
        if leg_return_atr <= -2.5 and leg_slope_atr <= -0.08:
            return TrendLegType.FAST_DOWN_LEG
        if leg_return_atr <= -1.0 and leg_efficiency >= 0.35 and duration_bars >= 12:
            return TrendLegType.SLOW_DOWN_LEG
        if leg_return_atr >= 0.8 and leg_efficiency >= 0.30 and duration_bars >= 24:
            return TrendLegType.SLOW_UP_LEG
        if leg_return_atr >= 2.0 and vol_expansion >= 1.5 and duration_bars <= 12:
            return TrendLegType.SURGE_LEG
        if abs(leg_return_atr) < 0.8 and leg_efficiency < 0.25:
            return TrendLegType.RANGE_LEG
        if duration_bars < self.cfg.min_leg_bars or abs(leg_return_atr) < self.cfg.min_move_atr:
            return TrendLegType.TRANSITION_LEG
        if leg_efficiency < self.cfg.min_efficiency:
            return TrendLegType.RANGE_LEG
        if direction == TrendDirection.UP:
            return TrendLegType.SLOW_UP_LEG if leg_slope_atr < 0.08 else TrendLegType.FAST_UP_LEG
        if direction == TrendDirection.DOWN:
            return TrendLegType.SLOW_DOWN_LEG if leg_slope_atr > -0.08 else TrendLegType.FAST_DOWN_LEG
        return TrendLegType.TRANSITION_LEG

    def _refresh_active_leg(
        self,
        bar_idx: int,
        close: float,
        atr: float,
        slow_ctx: SlowTrendContext | None,
        crash_ctx: CrashContext | None,
        is_model_blind: bool,
    ) -> None:
        if self._active_leg is None or self._leg_metrics is None:
            return
        leg = self._active_leg
        self._leg_metrics.on_bar(bar_idx, close, atr)
        leg.duration_bars = bar_idx - leg.start_bar_idx + 1
        leg.leg_return_atr = self._leg_metrics.return_atr(close)
        leg.leg_slope_atr = leg.leg_return_atr / max(leg.duration_bars, 1)
        leg.leg_efficiency = self._leg_metrics.efficiency()
        leg.leg_pullback_ratio = self._leg_metrics.pullback_ratio(close, leg.direction)
        leg.leg_vol_expansion = self._leg_metrics.vol_expansion()
        leg.leg_type = self._classify_leg_type(
            leg_return_atr=leg.leg_return_atr,
            leg_slope_atr=leg.leg_slope_atr,
            leg_efficiency=leg.leg_efficiency,
            duration_bars=leg.duration_bars,
            vol_expansion=leg.leg_vol_expansion,
            direction=leg.direction,
            slow_ctx=slow_ctx,
            crash_ctx=crash_ctx,
            is_model_blind=is_model_blind,
        )
        confirmed = (
            leg.duration_bars >= self.cfg.min_leg_bars
            and abs(leg.leg_return_atr) >= self.cfg.min_move_atr
            and leg.leg_efficiency >= self.cfg.min_efficiency
            and leg.leg_type not in (TrendLegType.RANGE_LEG, TrendLegType.TRANSITION_LEG)
        )
        leg.is_confirmed = confirmed
        if confirmed:
            self._leg_state = LegState.LEG_CONFIRMED
            leg.leg_state = LegState.LEG_CONFIRMED
        else:
            self._leg_state = LegState.LEG_FORMING
            leg.leg_state = LegState.LEG_FORMING

    def _start_leg(self, bar_idx: int, direction: TrendDirection, close: float, atr: float) -> None:
        self._leg_id += 1
        self._leg_metrics = _LegMetrics.start(
            bar_idx=bar_idx,
            close=close,
            atr=atr,
            atr_baseline=self._atr_baseline_before(bar_idx),
        )
        self._active_leg = TrendLeg(
            leg_id=self._leg_id,
            leg_type=TrendLegType.TRANSITION_LEG,
            direction=direction,
            start_bar_idx=bar_idx,
            end_bar_idx=None,
            duration_bars=1,
            leg_return_atr=0.0,
            leg_slope_atr=0.0,
            leg_efficiency=0.0,
            leg_pullback_ratio=0.0,
            leg_vol_expansion=1.0,
            is_confirmed=False,
            is_active=True,
            sub_phase=SubLegPhase.IMPULSE,
            sub_phase_age=0,
            leg_state=LegState.LEG_FORMING,
            reason_codes=["LEG_START"],
        )
        self._leg_state = LegState.LEG_FORMING
        self._sub_phase_age = 0

    def _close_active_leg(self, bar_idx: int) -> None:
        if self._active_leg is None:
            return
        leg = self._active_leg
        leg.end_bar_idx = bar_idx
        leg.is_active = False
        leg.leg_state = LegState.LEG_CLOSING
        leg.sub_phase = SubLegPhase.LEG_END
        self._previous_leg = leg
        hist = self._typical_leg_lengths.setdefault(leg.leg_type.value, [])
        hist.append(leg.duration_bars)
        if len(hist) > 50:
            hist.pop(0)
        self._active_leg = None
        self._leg_metrics = None
        self._leg_state = LegState.NO_LEG

    def _update_sub_phase(self, *, confirmed_small_high: bool, confirmed_small_low: bool, trend_signal: TrendSignal | None) -> None:
        if self._active_leg is None:
            return
        leg = self._active_leg
        new_phase = leg.sub_phase
        if self._leg_state == LegState.LEG_CLOSING:
            new_phase = SubLegPhase.LEG_END
        elif leg.direction == TrendDirection.UP and confirmed_small_low:
            new_phase = SubLegPhase.PULLBACK
        elif leg.direction == TrendDirection.DOWN and confirmed_small_high:
            new_phase = SubLegPhase.PULLBACK
        elif trend_signal and trend_signal.phase == TrendPhase.EXHAUSTION:
            new_phase = SubLegPhase.EXHAUSTION
        elif trend_signal and trend_signal.phase == TrendPhase.ACCELERATION:
            new_phase = SubLegPhase.IMPULSE
        else:
            new_phase = SubLegPhase.IMPULSE
        if new_phase == leg.sub_phase:
            self._sub_phase_age += 1
        else:
            self._sub_phase_age = 0
        leg.sub_phase = new_phase
        leg.sub_phase_age = self._sub_phase_age

    def _compute_regime(self) -> MarketRegime:
        if len(self._close_hist) < 168:
            return MarketRegime.NEUTRAL
        close_tail = self._close_hist.tail_array(168)
        ema72 = float(close_tail[-72:].mean())
        ema168 = float(close_tail.mean())
        rets = np.diff(np.log(np.clip(close_tail[-48:], 1e-12, None)))
        vol = float(np.std(rets)) if len(rets) > 2 else 0.0
        vols: list[float] = []
        for i in range(48, len(close_tail)):
            r = np.diff(np.log(np.clip(close_tail[i - 48 : i], 1e-12, None)))
            vols.append(float(np.std(r)) if len(r) > 2 else 0.0)
        vol_pct = float(np.mean(np.asarray(vols) <= vol)) if vols else 0.5
        if vol_pct >= 0.85:
            return MarketRegime.HIGH_VOL
        if ema72 > ema168 * 1.002:
            return MarketRegime.BULL
        if ema72 < ema168 * 0.998:
            return MarketRegime.BEAR
        return MarketRegime.NEUTRAL

    def _build_context(self) -> SegmentContext:
        regime = self._compute_regime() if self.cfg.use_regime_filter else MarketRegime.NEUTRAL
        leg = self._active_leg
        leg_type = leg.leg_type if leg else TrendLegType.NONE
        sub_phase = leg.sub_phase if leg else SubLegPhase.NONE
        bars_since = leg.duration_bars if leg else 0
        progress = 0.0
        est_end: int | None = None
        if leg and leg.leg_type.value in self._typical_leg_lengths and self._typical_leg_lengths[leg.leg_type.value]:
            typical = float(np.median(self._typical_leg_lengths[leg.leg_type.value]))
            progress = min(1.0, bars_since / max(typical, 1.0))
            est_end = max(0, int(typical - bars_since))
        aligned = True
        if leg and regime == MarketRegime.BULL and leg.direction == TrendDirection.DOWN:
            aligned = False
        if leg and regime == MarketRegime.BEAR and leg.direction == TrendDirection.UP:
            aligned = False
        should_hold = bool(
            leg
            and leg.is_confirmed
            and leg.leg_type
            in (
                TrendLegType.SLOW_UP_LEG,
                TrendLegType.FAST_UP_LEG,
                TrendLegType.SLOW_DOWN_LEG,
                TrendLegType.FAST_DOWN_LEG,
                TrendLegType.CRASH_LEG,
            )
            and sub_phase in (SubLegPhase.IMPULSE, SubLegPhase.PULLBACK, SubLegPhase.BREAKOUT)
        )
        should_avoid_counter = bool(
            self.cfg.counter_trend_block
            and leg
            and leg.is_confirmed
            and leg.leg_type
            in (
                TrendLegType.SLOW_UP_LEG,
                TrendLegType.FAST_UP_LEG,
                TrendLegType.SLOW_DOWN_LEG,
                TrendLegType.FAST_DOWN_LEG,
                TrendLegType.CRASH_LEG,
            )
        )
        return SegmentContext(
            regime=regime,
            active_leg=leg,
            previous_leg=self._previous_leg,
            leg_type=leg_type,
            sub_phase=sub_phase,
            leg_state=self._leg_state,
            bars_since_leg_start=bars_since,
            bars_to_estimated_leg_end=est_end,
            leg_progress_ratio=progress,
            aligned_with_regime=aligned,
            should_hold_trend=should_hold,
            should_avoid_counter=should_avoid_counter,
            reason_codes=leg.reason_codes if leg else [],
        )

    def update(
        self,
        *,
        bar_idx: int,
        high: float,
        low: float,
        close: float,
        atr: float,
        trend_signal: TrendSignal | None = None,
        slow_ctx: SlowTrendContext | None = None,
        crash_ctx: CrashContext | None = None,
        is_model_blind: bool = False,
    ) -> SegmentContext:
        if not self.cfg.enabled:
            return SegmentContext(
                regime=MarketRegime.NEUTRAL,
                active_leg=None,
                previous_leg=None,
                leg_type=TrendLegType.NONE,
                sub_phase=SubLegPhase.NONE,
                leg_state=LegState.NO_LEG,
                bars_since_leg_start=0,
                bars_to_estimated_leg_end=None,
                leg_progress_ratio=0.0,
                aligned_with_regime=True,
                should_hold_trend=False,
                should_avoid_counter=False,
                reason_codes=["DISABLED"],
            )

        self._close_hist.append(float(close))
        self._high_hist.append(float(high))
        self._low_hist.append(float(low))
        self._atr_hist.append(max(float(atr), 1e-12))

        sh_s, sl_s = self.swing_small.on_bar(bar_idx, high, low)
        sh_l, sl_l = self.swing_large.on_bar(bar_idx, high, low)

        new_large_dir: TrendDirection | None = None
        if sh_l and self.swing_large.last_swing_high is not None:
            new_large_dir = TrendDirection.DOWN
            self._last_large_swing = "HIGH"
        elif sl_l and self.swing_large.last_swing_low is not None:
            new_large_dir = TrendDirection.UP
            self._last_large_swing = "LOW"

        if new_large_dir is not None:
            if self._active_leg is None:
                self._start_leg(bar_idx, new_large_dir, close, atr)
            else:
                same_dir = (
                    (self._active_leg.direction == TrendDirection.UP and new_large_dir == TrendDirection.UP)
                    or (self._active_leg.direction == TrendDirection.DOWN and new_large_dir == TrendDirection.DOWN)
                )
                pullback_atr = 0.0
                if self._leg_metrics is not None:
                    if self._active_leg.direction == TrendDirection.UP:
                        pullback_atr = (self._leg_metrics.max_close - close) / max(atr, 1e-12)
                    else:
                        pullback_atr = (close - self._leg_metrics.min_close) / max(atr, 1e-12)
                if same_dir and pullback_atr < self.cfg.merge_pullback_atr:
                    pass
                else:
                    self._refresh_active_leg(bar_idx, close, atr, slow_ctx, crash_ctx, is_model_blind)
                    if self._active_leg and (
                        not same_dir
                        or pullback_atr >= self.cfg.leg_end_pullback_atr
                        or self._active_leg.leg_efficiency < self.cfg.min_efficiency * 0.67
                    ):
                        self._close_active_leg(bar_idx)
                        self._start_leg(bar_idx, new_large_dir, close, atr)

        if self._active_leg is not None:
            self._refresh_active_leg(bar_idx, close, atr, slow_ctx, crash_ctx, is_model_blind)
            self._update_sub_phase(confirmed_small_high=sh_s, confirmed_small_low=sl_s, trend_signal=trend_signal)
            if (
                self._active_leg.leg_pullback_ratio > 0
                and self._active_leg.leg_return_atr != 0
                and self._active_leg.leg_pullback_ratio >= 1.0 / max(abs(self._active_leg.leg_return_atr), 0.1)
                and self._active_leg.sub_phase == SubLegPhase.PULLBACK
            ):
                self._active_leg.sub_phase = SubLegPhase.EXHAUSTION

        return self._build_context()
