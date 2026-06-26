from __future__ import annotations

import math
from collections import Counter

from trading_system.trend_bias import CounterTrendLevel, TrendBiasContext

_LONG_BLOCK_TAGS: tuple[tuple[str, str], ...] = (
    ("CRASH_P1_BLOCK_LONG", "CRASH"),
    ("CRASH_SOFT_LONG", "CRASH"),
    ("SEGMENT_CONFIRMED_DOWN_LEG_BLOCK_LONG", "SEGMENT"),
    ("LEGACY_DOWNTREND_BLOCK_LONG", "LEGACY"),
    ("LEGACY_DOWNTREND_SOFT_LONG", "LEGACY"),
    ("MACRO_SOFT_RANGE_LEG", "MACRO"),
)

_SHORT_BLOCK_TAGS: tuple[tuple[str, str], ...] = (
    ("SEGMENT_CONFIRMED_UP_LEG_BLOCK_SHORT", "SEGMENT"),
    ("CRASH_P1_BOOST_SHORT_OPEN", "CRASH"),
)


def derive_block_reason(
    bias: TrendBiasContext,
    *,
    side: str,
) -> str | None:
    codes = set(bias.reason_codes)
    if side == "long" and not bias.allow_open_long:
        for tag, bucket in _LONG_BLOCK_TAGS:
            if tag in codes:
                return bucket
        if bias.counter_level_long == CounterTrendLevel.HARD_BLOCK:
            return "COUNTER"
        return "OTHER"
    if side == "short" and not bias.allow_open_short:
        for tag, bucket in _SHORT_BLOCK_TAGS:
            if tag in codes:
                return bucket
        if bias.counter_level_short == CounterTrendLevel.HARD_BLOCK:
            return "COUNTER"
        return "OTHER"
    return None


def aggregate_block_stats(rows: list[dict]) -> dict[str, object]:
    n = len(rows)
    long_block = sum(1 for r in rows if not r["allow_open_long"])
    short_block = sum(1 for r in rows if not r["allow_open_short"])
    long_by_reason: Counter[str] = Counter()
    short_by_reason: Counter[str] = Counter()
    for r in rows:
        if not r["allow_open_long"]:
            long_by_reason[r.get("block_reason_long") or "OTHER"] += 1
        if not r["allow_open_short"]:
            short_by_reason[r.get("block_reason_short") or "OTHER"] += 1

    def _entropy(counter: Counter[str]) -> float:
        total = sum(counter.values())
        if total <= 0:
            return 0.0
        ent = 0.0
        for c in counter.values():
            p = c / total
            ent -= p * math.log(p + 1e-12)
        return float(ent)

    range_rows = [r for r in rows if r.get("teacher_regime") == "RANGE_TRANSITION"]
    range_long = sum(1 for r in range_rows if not r["allow_open_long"])
    range_short = sum(1 for r in range_rows if not r["allow_open_short"])
    range_ratio = (range_long / max(range_short, 1)) if range_rows else 0.0

    return {
        "hard_block_long_ratio": long_block / max(n, 1),
        "hard_block_short_ratio": short_block / max(n, 1),
        "block_long_by_reason": dict(long_by_reason),
        "block_short_by_reason": dict(short_by_reason),
        "block_reason_entropy_long": _entropy(long_by_reason),
        "block_reason_entropy_short": _entropy(short_by_reason),
        "hard_block_symmetry_range_transition_ratio": range_ratio,
    }
