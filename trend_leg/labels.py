from __future__ import annotations

from trading_system.trend_segment import SubLegPhase, TrendLegType

LEG_TYPES = [
    TrendLegType.TRANSITION_LEG.value,
    TrendLegType.SLOW_UP_LEG.value,
    TrendLegType.FAST_UP_LEG.value,
    TrendLegType.SLOW_DOWN_LEG.value,
    TrendLegType.FAST_DOWN_LEG.value,
    TrendLegType.CRASH_LEG.value,
    TrendLegType.SURGE_LEG.value,
    TrendLegType.RANGE_LEG.value,
    TrendLegType.NONE.value,
]

SUB_PHASES = [
    SubLegPhase.BASE.value,
    SubLegPhase.IMPULSE.value,
    SubLegPhase.PULLBACK.value,
    SubLegPhase.BREAKOUT.value,
    SubLegPhase.EXHAUSTION.value,
    SubLegPhase.LEG_END.value,
    SubLegPhase.NONE.value,
]

LEG_TYPE_TO_IDX = {name: i for i, name in enumerate(LEG_TYPES)}
SUB_PHASE_TO_IDX = {name: i for i, name in enumerate(SUB_PHASES)}


def encode_leg_type(value: str) -> int:
    return LEG_TYPE_TO_IDX.get(str(value), LEG_TYPE_TO_IDX[TrendLegType.TRANSITION_LEG.value])


def encode_sub_phase(value: str) -> int:
    return SUB_PHASE_TO_IDX.get(str(value), SUB_PHASE_TO_IDX[SubLegPhase.BASE.value])
