from __future__ import annotations

from dataclasses import dataclass, field

from trading_system.config import TrendEntryQualifierConfig
from trading_system.crash import CrashContext
from trading_system.slow_trend import SlowTrendContext
from trading_system.trend_signal import TrendDirection, TrendSignal
from trading_system.trend_segment import SegmentContext, TrendLegType


@dataclass
class TrendEntryQualification:
    allow_trend_entry_long: bool = False
    allow_trend_entry_short: bool = False
    entry_tier: str = "NONE"
    relax_edge_mult: float = 1.0
    relax_prob_delta: float = 0.0
    reason_codes: list[str] = field(default_factory=list)


def empty_trend_entry_qualification() -> TrendEntryQualification:
    return TrendEntryQualification()


class TrendEntryQualifier:
    _LONG_LEG_TYPES = frozenset({TrendLegType.FAST_UP_LEG, TrendLegType.SLOW_UP_LEG})
    _SHORT_LEG_TYPES = frozenset({TrendLegType.FAST_DOWN_LEG})

    def __init__(self, cfg: TrendEntryQualifierConfig) -> None:
        self.cfg = cfg

    def _chop_hard(self, trend_signal: TrendSignal | None) -> bool:
        if trend_signal is None:
            return False
        return "CHOP_HARD" in trend_signal.reason_codes

    def _long_leg_ok(self, segment_context: SegmentContext | None) -> bool:
        if segment_context is None or not self.cfg.require_segment:
            return segment_context is not None
        if segment_context.leg_type not in self._LONG_LEG_TYPES:
            return False
        leg = segment_context.active_leg
        if leg is None:
            return False
        if self.cfg.require_confirmed_leg and not leg.is_confirmed:
            return False
        return True

    def _short_leg_ok(self, segment_context: SegmentContext | None) -> bool:
        if segment_context is None or not self.cfg.require_segment:
            return segment_context is not None
        if segment_context.leg_type not in self._SHORT_LEG_TYPES:
            return False
        leg = segment_context.active_leg
        if leg is None:
            return False
        if self.cfg.require_confirmed_leg and not leg.is_confirmed:
            return False
        return True

    def compute(
        self,
        *,
        trend_signal: TrendSignal | None,
        segment_context: SegmentContext | None,
        slow_context: SlowTrendContext | None = None,
        crash_context: CrashContext | None = None,
    ) -> TrendEntryQualification:
        del slow_context  # reserved for Phase 2/3 extensions
        if not self.cfg.enabled:
            return empty_trend_entry_qualification()
        if trend_signal is None:
            return empty_trend_entry_qualification()
        if self._chop_hard(trend_signal):
            return TrendEntryQualification(reason_codes=["TEQ_BLOCK_CHOP_HARD"])

        reasons: list[str] = []
        allow_long = False
        allow_short = False

        if (
            trend_signal.direction == TrendDirection.UP
            and trend_signal.is_confirmed
            and self._long_leg_ok(segment_context)
            and not (crash_context and crash_context.is_crash and self.cfg.block_long_in_crash)
        ):
            allow_long = True
            leg = segment_context.leg_type.value if segment_context else "NONE"
            reasons.append(f"TREND_QUALIFIED_LONG_{leg}")

        if trend_signal.direction == TrendDirection.DOWN and trend_signal.is_confirmed and self._short_leg_ok(segment_context):
            allow_short = True
            leg = segment_context.leg_type.value if segment_context else "NONE"
            reasons.append(f"TREND_QUALIFIED_SHORT_{leg}")

        if not allow_long and not allow_short:
            return TrendEntryQualification(reason_codes=reasons or ["TEQ_NONE"])

        tier = "PROBE"
        if trend_signal.strength.value in ("STRONG", "EXTREME"):
            tier = "FULL"

        return TrendEntryQualification(
            allow_trend_entry_long=allow_long,
            allow_trend_entry_short=allow_short,
            entry_tier=tier,
            relax_edge_mult=self.cfg.relax_edge_mult,
            relax_prob_delta=self.cfg.relax_prob_delta,
            reason_codes=reasons,
        )
