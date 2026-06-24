from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from trading_system.config import TrendBiasConfig
from trading_system.crash import CrashContext
from trading_system.slow_trend import SlowTrendContext
from trading_system.trend import TrendContext
from trading_system.trend_signal import TrendDirection, TrendPhase, TrendSignal
from trading_system.trend_segment import SegmentContext, SubLegPhase, TrendLegType


class BiasDirection(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    NONE = "NONE"


class CounterTrendLevel(str, Enum):
    NONE = "NONE"
    LIGHT = "LIGHT"
    MEDIUM = "MEDIUM"
    HARD_BLOCK = "HARD_BLOCK"


@dataclass(frozen=True)
class TrendBiasContext:
    macro_direction: BiasDirection = BiasDirection.NONE
    leg_direction: BiasDirection = BiasDirection.NONE
    micro_direction: BiasDirection = BiasDirection.NONE
    alignment_score_long: int = 0
    alignment_score_short: int = 0
    counter_level_long: CounterTrendLevel = CounterTrendLevel.LIGHT
    counter_level_short: CounterTrendLevel = CounterTrendLevel.LIGHT
    open_bias_long: float = 1.0
    open_bias_short: float = 1.0
    size_bias_long: float = 1.0
    size_bias_short: float = 1.0
    hold_bias_long: float = 1.0
    hold_bias_short: float = 1.0
    exit_bias_long: float = 1.0
    exit_bias_short: float = 1.0
    risk_tolerance_bias_long: float = 1.0
    risk_tolerance_bias_short: float = 1.0
    allow_open_long: bool = True
    allow_open_short: bool = True
    allow_add_long: bool = False
    allow_add_short: bool = False
    force_exit_long: bool = False
    force_exit_short: bool = False
    active_leg_id: int | None = None
    active_leg_type: str = ""
    sub_phase: str = ""
    leg_progress_ratio: float = 0.0
    regime_strength: str = "NORMAL"
    regime_phase: str = "CONTINUATION"
    is_confirmed: bool = False
    is_trend_breaking: bool = False
    slow_up_active: bool = False
    crash_short_active: bool = False
    source_confidence: float = 0.5
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RiskBudget:
    current_position_ratio: float = 0.0
    max_position_ratio: float = 0.2
    current_margin_ratio: float = 0.0
    remaining_position_ratio: float = 0.2
    worst_case_loss_ratio: float = 0.0
    remaining_loss_budget_ratio: float = 1.0
    allow_open_long: bool = True
    allow_open_short: bool = True
    allow_add_long: bool = True
    allow_add_short: bool = True
    allow_reverse: bool = True
    open_reject_reason_long: str = ""
    open_reject_reason_short: str = ""
    add_reject_reason_long: str = ""
    add_reject_reason_short: str = ""


def neutral_trend_bias() -> TrendBiasContext:
    return TrendBiasContext()


def _to_bias_direction(direction: TrendDirection | None) -> BiasDirection:
    if direction == TrendDirection.UP:
        return BiasDirection.UP
    if direction == TrendDirection.DOWN:
        return BiasDirection.DOWN
    return BiasDirection.NONE


def _score_layer(direction: BiasDirection, target: BiasDirection, weight: int | float) -> float:
    if direction == BiasDirection.NONE or weight <= 0:
        return 0.0
    if direction == target:
        return float(weight)
    return -float(weight)


def _clamp_alignment(raw: int) -> int:
    if raw >= 3:
        return 2
    if raw >= 1:
        return 1
    if raw == 0:
        return 0
    if raw >= -2:
        return -1
    return -2


def _counter_from_alignment(align: int) -> CounterTrendLevel:
    if align >= 1:
        return CounterTrendLevel.NONE
    if align == 0:
        return CounterTrendLevel.LIGHT
    if align == -1:
        return CounterTrendLevel.MEDIUM
    return CounterTrendLevel.HARD_BLOCK


def _apply_hard_block(
    *,
    allow_open: bool,
    open_bias: float,
    size_bias: float,
    allow_add: bool,
    counter: CounterTrendLevel,
) -> tuple[bool, float, float, bool]:
    if counter != CounterTrendLevel.HARD_BLOCK:
        return allow_open, open_bias, size_bias, allow_add
    return False, 0.0, 0.0, False


class TrendBiasBuilder:
    def __init__(self, cfg: TrendBiasConfig) -> None:
        self.cfg = cfg

    def build(
        self,
        *,
        trend_signal: TrendSignal | None,
        segment_context: SegmentContext | None,
        slow_context: SlowTrendContext | None,
        crash_context: CrashContext | None,
        trend_context: TrendContext | None,
    ) -> TrendBiasContext:
        if not self.cfg.enabled:
            return neutral_trend_bias()

        reasons: list[str] = []
        seg = segment_context
        ts = trend_signal
        slow = slow_context
        crash = crash_context

        leg_direction = BiasDirection.NONE
        if seg and seg.active_leg and seg.active_leg.is_confirmed:
            leg_direction = _to_bias_direction(seg.active_leg.direction)
            reasons.append(f"SEGMENT_LEG_{leg_direction.value}")
        elif ts and ts.is_confirmed:
            leg_direction = _to_bias_direction(ts.direction)
            reasons.append("TREND_SIGNAL_FALLBACK")
        else:
            reasons.append("LEG_NONE")

        micro_direction = _to_bias_direction(ts.direction) if ts else BiasDirection.NONE
        chop_soft_active = bool(ts and "CHOP_SOFT" in ts.reason_codes)
        micro_for_align = BiasDirection.NONE if chop_soft_active else micro_direction
        if chop_soft_active:
            reasons.append("CHOP_SOFT_MICRO_WEAK")
        sub_phase = seg.sub_phase.value if seg else SubLegPhase.NONE.value
        if seg and seg.sub_phase == SubLegPhase.PULLBACK:
            reasons.append("MICRO_PULLBACK")
        elif ts and ts.phase == TrendPhase.ACCELERATION:
            reasons.append("MICRO_ACCELERATION")
        else:
            reasons.append("MICRO_ALIGNED")

        macro_direction = BiasDirection.NONE
        if seg and seg.regime.value in ("BULL", "BEAR"):
            macro_direction = BiasDirection.UP if seg.regime.value == "BULL" else BiasDirection.DOWN
            reasons.append("MACRO_FROM_REGIME")
        elif ts and ts.ema_slow > 0 and ts.ema_fast > ts.ema_slow:
            macro_direction = BiasDirection.UP
            reasons.append("MACRO_DERIVED_WEAK")
        elif ts and ts.ema_slow > 0 and ts.ema_fast < ts.ema_slow:
            macro_direction = BiasDirection.DOWN
            reasons.append("MACRO_DERIVED_WEAK")

        macro_for_align = macro_direction
        if seg and seg.leg_type in (TrendLegType.RANGE_LEG, TrendLegType.TRANSITION_LEG):
            macro_for_align = BiasDirection.NONE
            reasons.append("MACRO_SOFT_RANGE_LEG")

        micro_weight = 1.0 if not chop_soft_active else max(0.0, self.cfg.chop_soft_micro_weight)
        raw_long = (
            _score_layer(macro_for_align, BiasDirection.UP, 1)
            + _score_layer(leg_direction, BiasDirection.UP, 2)
            + _score_layer(micro_for_align, BiasDirection.UP, micro_weight)
        )
        raw_short = (
            _score_layer(macro_for_align, BiasDirection.DOWN, 1)
            + _score_layer(leg_direction, BiasDirection.DOWN, 2)
            + _score_layer(micro_for_align, BiasDirection.DOWN, micro_weight)
        )
        align_long = _clamp_alignment(int(round(raw_long)))
        align_short = _clamp_alignment(int(round(raw_short)))
        counter_long = _counter_from_alignment(align_long)
        counter_short = _counter_from_alignment(align_short)

        open_bias_long = 1.0
        open_bias_short = 1.0
        size_bias_long = 1.0
        size_bias_short = 1.0
        hold_bias_long = 1.0
        hold_bias_short = 1.0
        exit_bias_long = 1.0
        exit_bias_short = 1.0
        allow_open_long = True
        allow_open_short = True
        allow_add_long = False
        allow_add_short = False
        force_exit_long = False
        force_exit_short = False
        slow_up_active = bool(slow and slow.is_stable_slow_uptrend)
        crash_short_active = bool(crash and crash.is_model_blind_crash)

        leg_confirmed = bool(seg and seg.active_leg and seg.active_leg.is_confirmed)
        leg_confirmed_up = leg_confirmed and leg_direction == BiasDirection.UP
        leg_confirmed_down = leg_confirmed and leg_direction == BiasDirection.DOWN

        if align_long >= 1 and leg_confirmed_up:
            open_bias_long = self.cfg.long_open_relax
            size_bias_long = self.cfg.trend_size_boost
            hold_bias_long = 1.1
            exit_bias_long = self.cfg.trend_exit_vote_bonus
        elif align_long == 0:
            open_bias_long = self.cfg.light_counter_tighten
            size_bias_long = self.cfg.light_counter_size_penalty
        elif align_long == -1:
            if leg_confirmed_down:
                open_bias_long = self.cfg.medium_counter_tighten
                size_bias_long = self.cfg.medium_counter_size_penalty
                hold_bias_long = 0.8
                exit_bias_long = self.cfg.counter_exit_vote_penalty
            else:
                open_bias_long = self.cfg.light_counter_tighten
                size_bias_long = self.cfg.light_counter_size_penalty
                counter_long = CounterTrendLevel.LIGHT

        if align_short >= 1 and leg_confirmed_down:
            open_bias_short = self.cfg.short_open_relax
            size_bias_short = self.cfg.trend_size_boost
            hold_bias_short = 1.1
            exit_bias_short = self.cfg.trend_exit_vote_bonus
        elif align_short == 0:
            open_bias_short = self.cfg.light_counter_tighten
            size_bias_short = self.cfg.light_counter_size_penalty
        elif align_short == -1:
            if leg_confirmed_up:
                open_bias_short = self.cfg.medium_counter_tighten
                size_bias_short = self.cfg.medium_counter_size_penalty
                hold_bias_short = 0.8
                exit_bias_short = self.cfg.counter_exit_vote_penalty
            else:
                open_bias_short = self.cfg.light_counter_tighten
                size_bias_short = self.cfg.light_counter_size_penalty
                counter_short = CounterTrendLevel.LIGHT

        if seg and seg.should_avoid_counter and leg_confirmed:
            if leg_direction == BiasDirection.DOWN:
                counter_long = CounterTrendLevel.HARD_BLOCK
                open_bias_long = min(open_bias_long, self.cfg.medium_counter_tighten)
                reasons.append("SEGMENT_CONFIRMED_DOWN_LEG_BLOCK_LONG")
            elif leg_direction == BiasDirection.UP:
                counter_short = CounterTrendLevel.HARD_BLOCK
                open_bias_short = min(open_bias_short, self.cfg.medium_counter_tighten)
                reasons.append("SEGMENT_CONFIRMED_UP_LEG_BLOCK_SHORT")

        if seg and seg.sub_phase in (SubLegPhase.IMPULSE, SubLegPhase.BREAKOUT):
            if leg_direction == BiasDirection.UP:
                allow_add_long = True
            if leg_direction == BiasDirection.DOWN:
                allow_add_short = True
            reasons.append("SEGMENT_ADD_ALLOWED")

        is_trend_breaking = bool(
            (ts and ts.is_broken)
            or (seg and seg.sub_phase in (SubLegPhase.EXHAUSTION, SubLegPhase.LEG_END))
        )
        if is_trend_breaking:
            exit_bias_long = min(exit_bias_long, 0.85)
            exit_bias_short = min(exit_bias_short, 0.85)
            hold_bias_long = min(hold_bias_long, 0.85)
            hold_bias_short = min(hold_bias_short, 0.85)
            reasons.append("TREND_BREAKING_TIGHTEN")

        if (
            seg
            and seg.sub_phase == SubLegPhase.EXHAUSTION
            and counter_long == CounterTrendLevel.HARD_BLOCK
            and not (seg.should_avoid_counter and leg_confirmed and leg_direction == BiasDirection.DOWN)
        ):
            counter_long = CounterTrendLevel.MEDIUM
            reasons.append("EXHAUSTION_DOWNGRADE_HARD_BLOCK")

        if crash and crash.is_crash:
            reasons.append("CRASH_P1_BLOCK_LONG")
            if self.cfg.crash_hard_block_long:
                allow_open_long = False
                open_bias_long = 0.0
            else:
                open_bias_long = min(open_bias_long, self.cfg.medium_counter_tighten)
                size_bias_long = min(size_bias_long, self.cfg.light_counter_size_penalty)
                reasons.append("CRASH_SOFT_LONG")
            if crash.is_model_blind_crash:
                crash_short_active = True
                open_bias_short = self.cfg.crash_short_open_boost
                reasons.append("CRASH_P1_BOOST_SHORT_OPEN")
            if crash_short_active and counter_short == CounterTrendLevel.MEDIUM:
                counter_short = CounterTrendLevel.NONE
                reasons.append("CRASH_UPGRADE_SHORT_COUNTER")

        if slow_up_active and leg_direction != BiasDirection.DOWN:
            size_bias_long = max(size_bias_long, self.cfg.slow_up_size_boost)
            hold_bias_long = max(hold_bias_long, self.cfg.slow_up_hold_boost)
            allow_add_long = allow_add_long or bool(seg and seg.should_hold_trend)
            reasons.append("SLOW_UP_BOOST_LONG_SIZE")

        if trend_context and trend_context.is_downtrend:
            up_leg = seg and seg.leg_type in (TrendLegType.SLOW_UP_LEG, TrendLegType.FAST_UP_LEG)
            safe_long_in_downtrend = bool(leg_confirmed_up and up_leg and align_long >= 1)
            if not safe_long_in_downtrend:
                if self.cfg.legacy_down_hard_block:
                    allow_open_long = False
                    open_bias_long = min(open_bias_long, self.cfg.medium_counter_tighten)
                    reasons.append("LEGACY_DOWNTREND_BLOCK_LONG")
                else:
                    open_bias_long = min(open_bias_long, self.cfg.medium_counter_tighten)
                    size_bias_long = min(size_bias_long, self.cfg.light_counter_size_penalty)
                    reasons.append("LEGACY_DOWNTREND_SOFT_LONG")

        allow_open_long, open_bias_long, size_bias_long, allow_add_long = _apply_hard_block(
            allow_open=allow_open_long,
            open_bias=open_bias_long,
            size_bias=size_bias_long,
            allow_add=allow_add_long,
            counter=counter_long,
        )
        allow_open_short, open_bias_short, size_bias_short, allow_add_short = _apply_hard_block(
            allow_open=allow_open_short,
            open_bias=open_bias_short,
            size_bias=size_bias_short,
            allow_add=allow_add_short,
            counter=counter_short,
        )

        if seg and seg.sub_phase == SubLegPhase.LEG_END:
            if leg_direction == BiasDirection.UP:
                force_exit_long = True
            if leg_direction == BiasDirection.DOWN:
                force_exit_short = True
            reasons.append("LEG_END_FORCE_EXIT")

        is_confirmed = bool((ts and ts.is_confirmed) or (seg and seg.active_leg and seg.active_leg.is_confirmed))
        if seg and seg.active_leg and seg.active_leg.is_confirmed:
            source_confidence = 1.0
        elif ts and ts.is_confirmed:
            source_confidence = 0.7
        elif "MACRO_DERIVED_WEAK" in reasons:
            source_confidence = 0.5
        else:
            source_confidence = 0.3

        regime_strength = ts.strength.value if ts else "NORMAL"
        regime_phase = ts.phase.value if ts else "CONTINUATION"
        active_leg_id = seg.active_leg.leg_id if seg and seg.active_leg else None
        active_leg_type = seg.leg_type.value if seg else ""
        leg_progress = seg.leg_progress_ratio if seg else 0.0

        return TrendBiasContext(
            macro_direction=macro_direction,
            leg_direction=leg_direction,
            micro_direction=micro_direction,
            alignment_score_long=align_long,
            alignment_score_short=align_short,
            counter_level_long=counter_long,
            counter_level_short=counter_short,
            open_bias_long=open_bias_long,
            open_bias_short=open_bias_short,
            size_bias_long=size_bias_long,
            size_bias_short=size_bias_short,
            hold_bias_long=hold_bias_long,
            hold_bias_short=hold_bias_short,
            exit_bias_long=exit_bias_long,
            exit_bias_short=exit_bias_short,
            risk_tolerance_bias_long=1.0,
            risk_tolerance_bias_short=1.0,
            allow_open_long=allow_open_long,
            allow_open_short=allow_open_short,
            allow_add_long=allow_add_long,
            allow_add_short=allow_add_short,
            force_exit_long=force_exit_long,
            force_exit_short=force_exit_short,
            active_leg_id=active_leg_id,
            active_leg_type=active_leg_type,
            sub_phase=sub_phase,
            leg_progress_ratio=leg_progress,
            regime_strength=regime_strength,
            regime_phase=regime_phase,
            is_confirmed=is_confirmed,
            is_trend_breaking=is_trend_breaking,
            slow_up_active=slow_up_active,
            crash_short_active=crash_short_active,
            source_confidence=source_confidence,
            reason_codes=tuple(reasons),
        )
