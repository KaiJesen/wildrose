"""023 participation metrics (§5.3 frozen formulas)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

CONFIRMED_LEG_TYPES = frozenset(
    {"SLOW_UP_LEG", "FAST_UP_LEG", "SLOW_DOWN_LEG", "FAST_DOWN_LEG"}
)
EXCLUDED_LEG_TYPES = frozenset({"TRANSITION_LEG", "RANGE_LEG"})


def leg_direction_from_type(leg_type: str) -> str | None:
    if leg_type in ("FAST_UP_LEG", "SLOW_UP_LEG"):
        return "UP"
    if leg_type in ("FAST_DOWN_LEG", "SLOW_DOWN_LEG"):
        return "DOWN"
    return None


def position_side_aligned(state: str, leg_direction: str) -> bool:
    if leg_direction == "UP":
        return state == "LONG"
    if leg_direction == "DOWN":
        return state == "SHORT"
    return False


def position_side_counter(state: str, leg_direction: str) -> bool:
    if leg_direction == "UP":
        return state == "SHORT"
    if leg_direction == "DOWN":
        return state == "LONG"
    return False


@dataclass
class ParticipationConfig:
    min_effective_position_ratio: float = 0.02
    small_move_threshold: float = 0.003
    exclude_chop_hard: bool = True


@dataclass
class LegParticipation:
    leg_id: int
    leg_type: str
    leg_direction: str
    leg_bars: int
    aligned_overlap_bars: int
    counter_overlap_bars: int
    avg_pos_ratio: float
    effective_covered: bool
    ideal_leg_pnl: float
    captured_pnl: float
    small_move: bool


@dataclass
class ParticipationMetrics:
    leg_count: int = 0
    leg_count_covered: int = 0
    leg_count_coverage_ratio: float = 0.0
    leg_bar_coverage_ratio: float = 0.0
    leg_pnl_capture_ratio: float = 0.0
    counter_leg_participation_count: int = 0
    counter_overlap_bar_ratio: float = 0.0
    small_move_leg_count: int = 0
    leg_loss_coverage_count: int = 0
    slow_up_watch_to_open_ratio: float = 0.0
    slow_up_false_entry_count: int = 0
    legs: list[LegParticipation] = field(default_factory=list)

    def to_dict(self) -> dict[str, float]:
        return {
            "leg_count": float(self.leg_count),
            "leg_count_covered": float(self.leg_count_covered),
            "leg_count_coverage_ratio": float(self.leg_count_coverage_ratio),
            "leg_bar_coverage_ratio": float(self.leg_bar_coverage_ratio),
            "leg_pnl_capture_ratio": float(self.leg_pnl_capture_ratio),
            "counter_leg_participation_count": float(self.counter_leg_participation_count),
            "counter_overlap_bar_ratio": float(self.counter_overlap_bar_ratio),
            "small_move_leg_count": float(self.small_move_leg_count),
            "leg_loss_coverage_count": float(self.leg_loss_coverage_count),
            "slow_up_watch_to_open_ratio": float(self.slow_up_watch_to_open_ratio),
            "slow_up_false_entry_count": float(self.slow_up_false_entry_count),
        }


def _is_chop_hard_row(row: pd.Series) -> bool:
    codes = str(row.get("bias_reason_codes", "") or "")
    seg = str(row.get("segment_reason_codes", "") or "")
    return "CHOP_HARD" in codes or "CHOP_HARD" in seg


def _build_confirmed_legs(decisions: pd.DataFrame, cfg: ParticipationConfig) -> dict[int, pd.DataFrame]:
    df = decisions.copy()
    if cfg.exclude_chop_hard and "bias_reason_codes" in df.columns:
        chop_mask = df.apply(_is_chop_hard_row, axis=1)
    else:
        chop_mask = pd.Series(False, index=df.index)

    mask = (
        (df.get("is_leg_confirmed", 0).astype(int) == 1)
        & df["leg_type"].isin(CONFIRMED_LEG_TYPES)
        & ~chop_mask
    )
    leg_df = df.loc[mask].copy()
    if leg_df.empty:
        return {}

    legs: dict[int, pd.DataFrame] = {}
    for leg_id, grp in leg_df.groupby("leg_id"):
        lid = int(leg_id)
        if lid < 0:
            continue
        legs[lid] = grp.sort_values("ts").reset_index(drop=True)
    return legs


def _leg_ideal_pnl(leg_rows: pd.DataFrame) -> tuple[float, bool]:
    prices = leg_rows["price"].astype(float)
    if len(prices) < 2:
        return 0.0, True
    p0 = float(prices.iloc[0])
    p1 = float(prices.iloc[-1])
    if p0 <= 0:
        return 0.0, True
    ideal = abs(p1 - p0) / p0
    return ideal, ideal < 0.003  # default threshold applied by caller


def compute_participation_metrics(
    decisions: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    *,
    cfg: ParticipationConfig | None = None,
    slow_up_min_hold_bars: int = 8,
) -> ParticipationMetrics:
    cfg = cfg or ParticipationConfig()
    legs_map = _build_confirmed_legs(decisions, cfg)
    trades = trades if trades is not None else pd.DataFrame()

    trade_pnl_by_leg: dict[int, float] = {}
    if not trades.empty and "entry_leg_id" in trades.columns:
        for _, tr in trades.iterrows():
            lid = int(tr.get("entry_leg_id", -1))
            if lid >= 0:
                trade_pnl_by_leg[lid] = trade_pnl_by_leg.get(lid, 0.0) + float(tr.get("net_pnl", 0.0))

    leg_records: list[LegParticipation] = []
    total_leg_bars = 0
    total_aligned = 0
    total_counter = 0
    pnl_num = 0.0
    pnl_den = 0.0
    covered_count = 0
    counter_leg_count = 0
    loss_covered = 0
    small_move_count = 0

    for leg_id, leg_rows in legs_map.items():
        leg_type = str(leg_rows["leg_type"].iloc[0])
        leg_dir = leg_direction_from_type(leg_type)
        if leg_dir is None:
            continue

        leg_bars = len(leg_rows)
        total_leg_bars += leg_bars

        states = leg_rows["state"].astype(str)
        pos_ratio = leg_rows.get("position_ratio", pd.Series(0.0, index=leg_rows.index)).astype(float)

        aligned_mask = states.apply(lambda s: position_side_aligned(s, leg_dir))
        counter_mask = states.apply(lambda s: position_side_counter(s, leg_dir))

        aligned_overlap = int(aligned_mask.sum())
        counter_overlap = int(counter_mask.sum())
        total_aligned += aligned_overlap
        total_counter += counter_overlap

        if counter_overlap > 0:
            counter_leg_count += 1

        avg_pos = float(pos_ratio.loc[aligned_mask].mean()) if aligned_overlap > 0 else 0.0
        min_overlap = min(3, max(1, int(leg_bars * 0.15)))
        effective = (
            aligned_overlap >= min_overlap
            and avg_pos >= cfg.min_effective_position_ratio
        )
        if effective:
            covered_count += 1

        ideal, small_move = _leg_ideal_pnl(leg_rows)
        if ideal < cfg.small_move_threshold:
            small_move_count += 1
        else:
            pnl_den += ideal

        captured = trade_pnl_by_leg.get(leg_id, 0.0)
        if not small_move:
            pnl_num += captured

        if effective and captured < 0:
            loss_covered += 1

        leg_records.append(
            LegParticipation(
                leg_id=leg_id,
                leg_type=leg_type,
                leg_direction=leg_dir,
                leg_bars=leg_bars,
                aligned_overlap_bars=aligned_overlap,
                counter_overlap_bars=counter_overlap,
                avg_pos_ratio=avg_pos,
                effective_covered=effective,
                ideal_leg_pnl=ideal,
                captured_pnl=captured,
                small_move=small_move,
            )
        )

    leg_count = len(leg_records)
    watch_count = int((decisions.get("reason_code", pd.Series(dtype=str)) == "WATCH_SLOW_UPTREND").sum())
    slow_open = int((decisions.get("reason_code", pd.Series(dtype=str)) == "OPEN_LONG_SLOW_TREND").sum())

    slow_false = 0
    if not trades.empty and "entry_was_slow_up" in trades.columns:
        for _, tr in trades.iterrows():
            if int(tr.get("entry_was_slow_up", 0)) != 1:
                continue
            if float(tr.get("net_pnl", 0.0)) < 0 and float(tr.get("bars_held", 0.0)) < slow_up_min_hold_bars:
                slow_false += 1

    return ParticipationMetrics(
        leg_count=leg_count,
        leg_count_covered=covered_count,
        leg_count_coverage_ratio=covered_count / max(1, leg_count),
        leg_bar_coverage_ratio=total_aligned / max(1, total_leg_bars),
        leg_pnl_capture_ratio=pnl_num / max(1e-12, pnl_den) if pnl_den > 0 else 0.0,
        counter_leg_participation_count=counter_leg_count,
        counter_overlap_bar_ratio=total_counter / max(1, total_leg_bars),
        small_move_leg_count=small_move_count,
        leg_loss_coverage_count=loss_covered,
        slow_up_watch_to_open_ratio=slow_open / max(1, watch_count),
        slow_up_false_entry_count=slow_false,
        legs=leg_records,
    )


def compare_runner_metrics(
    runner_metrics: dict[str, Any],
    participation: ParticipationMetrics,
) -> dict[str, dict[str, float]]:
    """Legacy v022 leg_coverage_ratio vs 023 aligned bar ratio."""
    legacy = float(runner_metrics.get("leg_coverage_ratio", 0.0))
    return {
        "leg_coverage_ratio_legacy": {
            "runner": legacy,
            "participation_aligned_bar": participation.leg_bar_coverage_ratio,
            "delta": participation.leg_bar_coverage_ratio - legacy,
        }
    }
