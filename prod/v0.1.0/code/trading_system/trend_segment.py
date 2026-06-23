from __future__ import annotations

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


class SwingDetector:
    def __init__(self, left_bars: int, right_bars: int) -> None:
        self.left_bars = left_bars
        self.right_bars = right_bars
        self.high_hist: list[float] = []
        self.low_hist: list[float] = []
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
                self.last_swing_high = (candidate, self.high_hist[candidate])
                confirmed_high = True
            if self._is_swing_low_confirmed(candidate, current):
                self.last_swing_low = (candidate, self.low_hist[candidate])
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
        self._close_hist: list[float] = []
        self._high_hist: list[float] = []
        self._low_hist: list[float] = []
        self._atr_hist: list[float] = []
        self._last_large_swing: str = "NONE"
        self._sub_phase_age = 0
        self._typical_leg_lengths: dict[str, list[int]] = {
            TrendLegType.SLOW_UP_LEG.value: [],
            TrendLegType.FAST_DOWN_LEG.value: [],
            TrendLegType.SLOW_DOWN_LEG.value: [],
        }

    def _leg_efficiency(self, start: int, end: int) -> float:
        if end <= start:
            return 0.0
        rets = []
        for i in range(start + 1, end + 1):
            prev = max(self._close_hist[i - 1], 1e-12)
            rets.append((self._close_hist[i] - prev) / prev)
        if not rets:
            return 0.0
        net = abs(sum(rets))
        gross = sum(abs(r) for r in rets)
        return float(net / max(gross, 1e-12))

    def _leg_return_atr(self, start: int, end: int) -> float:
        start_atr = max(self._atr_hist[start], 1e-12)
        return float((self._close_hist[end] - self._close_hist[start]) / start_atr)

    def _max_adverse_ratio(self, start: int, end: int, direction: TrendDirection) -> float:
        start_px = self._close_hist[start]
        seg = self._close_hist[start : end + 1]
        leg_ret = abs(self._close_hist[end] - start_px) / max(start_px, 1e-12)
        if leg_ret < 1e-9:
            return 0.0
        if direction == TrendDirection.UP:
            adverse = max(0.0, (start_px - min(seg)) / max(start_px, 1e-12))
        elif direction == TrendDirection.DOWN:
            adverse = max(0.0, (max(seg) - start_px) / max(start_px, 1e-12))
        else:
            adverse = 0.0
        return float(adverse / leg_ret)

    def _vol_expansion(self, start: int, end: int) -> float:
        if start < 24:
            return 1.0
        prev = np.mean(self._atr_hist[max(0, start - 24) : start])
        cur = np.mean(self._atr_hist[start : end + 1])
        return float(cur / max(prev, 1e-12))

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

    def _infer_direction(self, start: int, end: int) -> TrendDirection:
        ret = self._close_hist[end] - self._close_hist[start]
        if ret > 0:
            return TrendDirection.UP
        if ret < 0:
            return TrendDirection.DOWN
        return TrendDirection.NONE

    def _refresh_active_leg(self, bar_idx: int, slow_ctx: SlowTrendContext | None, crash_ctx: CrashContext | None, is_model_blind: bool) -> None:
        if self._active_leg is None:
            return
        leg = self._active_leg
        start = leg.start_bar_idx
        leg.duration_bars = bar_idx - start + 1
        leg.leg_return_atr = self._leg_return_atr(start, bar_idx)
        leg.leg_slope_atr = leg.leg_return_atr / max(leg.duration_bars, 1)
        leg.leg_efficiency = self._leg_efficiency(start, bar_idx)
        leg.leg_pullback_ratio = self._max_adverse_ratio(start, bar_idx, leg.direction)
        leg.leg_vol_expansion = self._vol_expansion(start, bar_idx)
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

    def _start_leg(self, bar_idx: int, direction: TrendDirection) -> None:
        self._leg_id += 1
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
        close = np.asarray(self._close_hist, dtype=np.float64)
        ema72 = close[-72:].mean()
        ema168 = close[-168:].mean()
        rets = np.diff(np.log(np.clip(close[-48:], 1e-12, None)))
        vol = float(np.std(rets)) if len(rets) > 2 else 0.0
        vol_hist = [float(np.std(np.diff(np.log(np.clip(close[i - 48 : i], 1e-12, None))))) for i in range(48, len(close))]
        vol_pct = 0.5
        if vol_hist:
            vol_pct = float(np.mean(np.asarray(vol_hist) <= vol))
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
        idx = len(self._close_hist) - 1

        sh_s, sl_s = self.swing_small.on_bar(idx, high, low)
        sh_l, sl_l = self.swing_large.on_bar(idx, high, low)

        new_large_dir: TrendDirection | None = None
        if sh_l and self.swing_large.last_swing_high is not None:
            new_large_dir = TrendDirection.DOWN
            self._last_large_swing = "HIGH"
        elif sl_l and self.swing_large.last_swing_low is not None:
            new_large_dir = TrendDirection.UP
            self._last_large_swing = "LOW"

        if new_large_dir is not None:
            if self._active_leg is None:
                self._start_leg(idx, new_large_dir)
            else:
                same_dir = (
                    (self._active_leg.direction == TrendDirection.UP and new_large_dir == TrendDirection.UP)
                    or (self._active_leg.direction == TrendDirection.DOWN and new_large_dir == TrendDirection.DOWN)
                )
                pullback_atr = 0.0
                if self._active_leg.direction == TrendDirection.UP:
                    peak = max(self._close_hist[self._active_leg.start_bar_idx : idx + 1])
                    pullback_atr = (peak - close) / max(atr, 1e-12)
                else:
                    trough = min(self._close_hist[self._active_leg.start_bar_idx : idx + 1])
                    pullback_atr = (close - trough) / max(atr, 1e-12)
                if same_dir and pullback_atr < self.cfg.merge_pullback_atr:
                    pass
                else:
                    self._refresh_active_leg(idx, slow_ctx, crash_ctx, is_model_blind)
                    if self._active_leg and (
                        not same_dir
                        or pullback_atr >= self.cfg.leg_end_pullback_atr
                        or self._active_leg.leg_efficiency < self.cfg.min_efficiency * 0.67
                    ):
                        self._close_active_leg(idx)
                        self._start_leg(idx, new_large_dir)

        if self._active_leg is not None:
            self._refresh_active_leg(idx, slow_ctx, crash_ctx, is_model_blind)
            self._update_sub_phase(confirmed_small_high=sh_s, confirmed_small_low=sl_s, trend_signal=trend_signal)
            if (
                self._active_leg.leg_pullback_ratio > 0
                and self._active_leg.leg_return_atr != 0
                and self._active_leg.leg_pullback_ratio >= 1.0 / max(abs(self._active_leg.leg_return_atr), 0.1)
                and self._active_leg.sub_phase == SubLegPhase.PULLBACK
            ):
                self._active_leg.sub_phase = SubLegPhase.EXHAUSTION

        return self._build_context()
