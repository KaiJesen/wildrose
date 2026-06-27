"""024 leg-alignment participation labels (hindsight, training-only)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from market_data.schema import COL_CLOSE, COL_HIGH, COL_LOW, COL_TIME
from trading_system.adapters.market_state_model import compute_atr
from trading_system.config import TradingSystemConfig
from trading_system.crash import CrashRegimeDetector
from trading_system.participation import CONFIRMED_LEG_TYPES, leg_direction_from_type
from trading_system.slow_trend import SlowUptrendDetector
from trading_system.trend import TrendRegimeFilter
from trading_system.trend_segment import SegmentContext, TrendLegType, TrendSegmentEngine
from trading_system.trend_signal import TrendMemory, TrendSignalProvider


def _neutral_model_signal(ts, price: float, atr: float):
    from datetime import datetime

    from trading_system.signal import TradingSignal

    return TradingSignal(
        ts=ts if ts is not None else datetime(2026, 1, 1),
        price=price,
        atr=atr,
        p_up=0.33,
        p_down=0.33,
        p_flat=0.34,
        p_risk=0.2,
        pred_ret_1=0.0,
        pred_ret_2=0.0,
        pred_ret_3=0.0,
        pred_ret_4=0.0,
        pred_ret_5=0.0,
        pred_cum_ret_5=0.0,
        edge=0.0,
    )

LONG_LABEL_LEG_TYPES = frozenset({TrendLegType.FAST_UP_LEG.value, TrendLegType.SLOW_UP_LEG.value})
SHORT_LABEL_LEG_TYPES = frozenset({TrendLegType.FAST_DOWN_LEG.value})
EXCLUDED_SUB_PHASES = frozenset({"EXHAUSTION", "LEG_END"})


@dataclass
class LegParticipationLabelConfig:
    fee_bps: float = 4.0
    slippage_bps: float = 2.0
    mae_atr_limit: float = 1.5
    max_entry_progress: float = 0.35
    forward_horizons: tuple[int, ...] = (12, 24, 48)
    # 026 A1 ordered tiers (optional; tier2 matches mae_atr_limit + max_entry_progress).
    enable_a1_tiers: bool = False
    tier1_mae_atr_limit: float = 2.5
    tier1_max_entry_progress: float = 0.50

    @property
    def round_trip_cost(self) -> float:
        return 2.0 * (self.fee_bps + self.slippage_bps) / 10000.0


def _participation_tier(
    roi: float,
    mae: float,
    progress: float,
    *,
    tier2_mae: float,
    tier1_mae: float,
    tier2_progress: float,
    tier1_progress: float,
) -> int:
    """026 A1: 0=none, 1=weak participate, 2=strong (ideal)."""
    if roi <= 0:
        return 0
    if mae <= tier2_mae and progress <= tier2_progress:
        return 2
    if mae <= tier1_mae and progress <= tier1_progress:
        return 1
    return 0


@dataclass
class LegParticipationLabelMetadata:
    segment_module_version: str = "022_trend_quality"
    trend_signal_config_sha256: str = ""
    rule_config_sha256: str = ""
    teacher_label_version: str = "leg_participation_v024_v1"
    aligned_leg_types_long: list[str] = field(
        default_factory=lambda: sorted(LONG_LABEL_LEG_TYPES)
    )
    aligned_leg_types_short: list[str] = field(
        default_factory=lambda: sorted(SHORT_LABEL_LEG_TYPES)
    )
    label_config: dict[str, float] = field(default_factory=dict)


def _chop_hard(trend_reason_codes: list[str], segment_reason_codes: list[str]) -> bool:
    codes = set(trend_reason_codes) | set(segment_reason_codes)
    return "CHOP_HARD" in codes


def replay_segment_bars(
    df: pd.DataFrame,
    cfg: TradingSystemConfig,
    *,
    idx: np.ndarray | None = None,
    start_bar: int = 0,
) -> pd.DataFrame:
    """Replay TrendSegment + TrendSignal on OHLCV; return per-bar segment features."""
    high = df[COL_HIGH].to_numpy(dtype=np.float64)
    low = df[COL_LOW].to_numpy(dtype=np.float64)
    close = df[COL_CLOSE].to_numpy(dtype=np.float64)
    atr_arr = compute_atr(high, low, close, cfg.execution.atr_period)
    times = df[COL_TIME].tolist()

    trend_filter = TrendRegimeFilter(cfg.trend)
    signal_provider = TrendSignalProvider(cfg.trend_signal)
    segment_engine = TrendSegmentEngine(cfg.trend_segment)
    crash_detector = CrashRegimeDetector(cfg.crash)
    slow_detector = SlowUptrendDetector(cfg.slow_uptrend)
    memory = TrendMemory()

    close_hist: list[float] = []
    high_hist: list[float] = []
    low_hist: list[float] = []
    atr_hist: list[float] = []

    idx_set = None if idx is None else {int(i) for i in idx.tolist()}
    end_i = int(idx.max()) if idx is not None else len(df) - 1

    rows: list[dict] = []
    for i in range(start_bar, end_i + 1):
        close_hist.append(float(close[i]))
        high_hist.append(float(high[i]))
        low_hist.append(float(low[i]))
        atr_hist.append(float(atr_arr[i]))

        atr_v = float(atr_arr[i])
        trend_filter.compute(close_hist, high_hist, low_hist, atr_v)
        trend_signal = signal_provider.compute(
            close_hist=close_hist,
            high_hist=high_hist,
            low_hist=low_hist,
            atr_hist=atr_hist,
            memory=memory,
        )
        model_sig = _neutral_model_signal(times[i], float(close[i]), atr_v)
        crash_ctx = crash_detector.compute(
            close_hist,
            high_hist,
            low_hist,
            atr_hist,
            model_sig,
            standard_open_short=False,
            is_flat=True,
        )
        slow_ctx = slow_detector.compute(
            close_hist,
            high_hist,
            low_hist,
            atr_v,
            p_risk=model_sig.p_risk,
            p_flat=model_sig.p_flat,
        )
        segment_ctx: SegmentContext = segment_engine.update(
            bar_idx=i,
            high=float(high[i]),
            low=float(low[i]),
            close=float(close[i]),
            atr=float(atr_arr[i]),
            trend_signal=trend_signal,
            slow_ctx=slow_ctx,
            crash_ctx=crash_ctx,
            is_model_blind=bool(crash_ctx.is_model_blind_crash),
        )

        if idx_set is not None and i not in idx_set:
            continue

        leg = segment_ctx.active_leg
        leg_type = segment_ctx.leg_type.value
        leg_id = int(leg.leg_id) if leg is not None else -1
        is_confirmed = bool(leg.is_confirmed) if leg is not None else False
        sub_phase = segment_ctx.sub_phase.value

        rows.append(
            {
                "bar_idx": i,
                COL_TIME: times[i],
                "close": float(close[i]),
                "high": float(high[i]),
                "low": float(low[i]),
                "atr": float(atr_arr[i]),
                "leg_id": leg_id,
                "leg_type": leg_type,
                "is_leg_confirmed": int(is_confirmed),
                "leg_progress_ratio": float(segment_ctx.leg_progress_ratio),
                "sub_phase": sub_phase,
                "align_direction": leg_direction_from_type(leg_type) or "NONE",
                "chop_hard": int(
                    _chop_hard(list(trend_signal.reason_codes), list(segment_ctx.reason_codes))
                ),
                "trend_signal_reason_codes": "|".join(trend_signal.reason_codes),
                "segment_reason_codes": "|".join(segment_ctx.reason_codes),
            }
        )

    return pd.DataFrame(rows)


def _leg_spans(bars: pd.DataFrame) -> dict[int, tuple[int, int]]:
    """Map leg_id -> (start_bar_idx, end_bar_idx) from replayed segment bars."""
    out: dict[int, tuple[int, int]] = {}
    active = bars[bars["leg_id"] >= 0]
    for leg_id, grp in active.groupby("leg_id"):
        out[int(leg_id)] = (int(grp["bar_idx"].min()), int(grp["bar_idx"].max()))
    return out


def _label_progress_ratio(t: int, leg_id: int, leg_spans: dict[int, tuple[int, int]]) -> float:
    if leg_id not in leg_spans:
        return 1.0
    start, end = leg_spans[leg_id]
    span = max(1, end - start)
    return min(1.0, max(0.0, (t - start) / span))


def _leg_end_index(bars: pd.DataFrame) -> dict[int, int]:
    """Map leg_id -> last bar_idx for confirmed aligned legs."""
    out: dict[int, int] = {}
    confirmed = bars[(bars["is_leg_confirmed"] == 1) & (bars["leg_id"] >= 0)]
    for leg_id, grp in confirmed.groupby("leg_id"):
        out[int(leg_id)] = int(grp["bar_idx"].max())
    return out


def _forward_roi(close: np.ndarray, t: int, end: int, *, long_side: bool, cost: float) -> float:
    entry = float(close[t])
    exit_p = float(close[end])
    if entry <= 0:
        return 0.0
    raw = (exit_p - entry) / entry if long_side else (entry - exit_p) / entry
    return raw - cost


def _forward_mae(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    t: int,
    end: int,
    *,
    long_side: bool,
) -> float:
    entry = float(close[t])
    atr_t = max(float(atr[t]), 1e-8)
    if long_side:
        adverse = max(0.0, (entry - float(low[t : end + 1].min())) / atr_t)
    else:
        adverse = max(0.0, (float(high[t : end + 1].max()) - entry) / atr_t)
    return adverse


def _forward_horizon_roi(
    close: np.ndarray,
    t: int,
    horizon: int,
    *,
    long_side: bool,
) -> float:
    end = min(len(close) - 1, t + horizon)
    entry = float(close[t])
    exit_p = float(close[end])
    if entry <= 0:
        return 0.0
    return (exit_p - entry) / entry if long_side else (entry - exit_p) / entry


def compute_ideal_participation_labels(
    bars: pd.DataFrame,
    full_close: np.ndarray,
    full_high: np.ndarray,
    full_low: np.ndarray,
    full_atr: np.ndarray,
    *,
    label_cfg: LegParticipationLabelConfig | None = None,
) -> pd.DataFrame:
    """Add hindsight ideal_participate_* and forward ROI columns to replayed bars."""
    label_cfg = label_cfg or LegParticipationLabelConfig()
    cost = label_cfg.round_trip_cost
    leg_ends = _leg_end_index(bars)
    leg_spans = _leg_spans(bars)

    ideal_long: list[int] = []
    ideal_short: list[int] = []
    tier_long: list[int] = []
    tier_short: list[int] = []
    eligible_long: list[int] = []
    eligible_short: list[int] = []
    label_progress: list[float] = []
    fwd: dict[str, list[float]] = {f"forward_leg_roi_{h}": [] for h in label_cfg.forward_horizons}
    use_tiers = label_cfg.enable_a1_tiers

    for row in bars.itertuples(index=False):
        t = int(row.bar_idx)
        leg_id = int(row.leg_id)
        leg_type = str(row.leg_type)
        align = str(row.align_direction)
        progress = _label_progress_ratio(t, leg_id, leg_spans)
        label_progress.append(progress)
        confirmed = int(row.is_leg_confirmed) == 1
        chop = int(row.chop_hard) == 1
        sub_phase = str(row.sub_phase)

        base_eligible = (
            confirmed
            and leg_id >= 0
            and align in ("UP", "DOWN")
            and progress <= label_cfg.max_entry_progress
            and not chop
            and sub_phase not in EXCLUDED_SUB_PHASES
            and leg_type in CONFIRMED_LEG_TYPES
        )
        long_elig = base_eligible and leg_type in LONG_LABEL_LEG_TYPES and align == "UP"
        short_elig = base_eligible and leg_type in SHORT_LABEL_LEG_TYPES and align == "DOWN"
        eligible_long.append(int(long_elig))
        eligible_short.append(int(short_elig))

        end = leg_ends.get(leg_id, t)
        tier_l = 0
        tier_s = 0
        if use_tiers and confirmed and not chop and sub_phase not in EXCLUDED_SUB_PHASES and leg_id in leg_ends:
            end = leg_ends[leg_id]
            if leg_type in LONG_LABEL_LEG_TYPES and align == "UP":
                roi = _forward_roi(full_close, t, end, long_side=True, cost=cost)
                mae = _forward_mae(full_high, full_low, full_close, full_atr, t, end, long_side=True)
                tier_l = _participation_tier(
                    roi,
                    mae,
                    progress,
                    tier2_mae=label_cfg.mae_atr_limit,
                    tier1_mae=label_cfg.tier1_mae_atr_limit,
                    tier2_progress=label_cfg.max_entry_progress,
                    tier1_progress=label_cfg.tier1_max_entry_progress,
                )
            if leg_type in SHORT_LABEL_LEG_TYPES and align == "DOWN":
                roi = _forward_roi(full_close, t, end, long_side=False, cost=cost)
                mae = _forward_mae(full_high, full_low, full_close, full_atr, t, end, long_side=False)
                tier_s = _participation_tier(
                    roi,
                    mae,
                    progress,
                    tier2_mae=label_cfg.mae_atr_limit,
                    tier1_mae=label_cfg.tier1_mae_atr_limit,
                    tier2_progress=label_cfg.max_entry_progress,
                    tier1_progress=label_cfg.tier1_max_entry_progress,
                )

        if long_elig and leg_id in leg_ends:
            if use_tiers:
                ideal_long.append(int(tier_l >= 2))
            else:
                roi = _forward_roi(full_close, t, end, long_side=True, cost=cost)
                mae = _forward_mae(full_high, full_low, full_close, full_atr, t, end, long_side=True)
                ideal_long.append(int(roi > 0 and mae <= label_cfg.mae_atr_limit))
        else:
            ideal_long.append(int(tier_l >= 2) if use_tiers else 0)

        if short_elig and leg_id in leg_ends:
            if use_tiers:
                ideal_short.append(int(tier_s >= 2))
            else:
                roi = _forward_roi(full_close, t, end, long_side=False, cost=cost)
                mae = _forward_mae(full_high, full_low, full_close, full_atr, t, end, long_side=False)
                ideal_short.append(int(roi > 0 and mae <= label_cfg.mae_atr_limit))
        else:
            ideal_short.append(int(tier_s >= 2) if use_tiers else 0)

        if use_tiers:
            tier_long.append(tier_l)
            tier_short.append(tier_s)

        for h in label_cfg.forward_horizons:
            fwd[f"forward_leg_roi_{h}"].append(
                _forward_horizon_roi(full_close, t, h, long_side=(align == "UP"))
                if align in ("UP", "DOWN")
                else 0.0
            )

    out = bars.copy()
    out["label_leg_progress_ratio"] = label_progress
    out["eligible_participate_long"] = eligible_long
    out["eligible_participate_short"] = eligible_short
    out["ideal_participate_long"] = ideal_long
    out["ideal_participate_short"] = ideal_short
    if use_tiers:
        out["participate_tier_long"] = tier_long
        out["participate_tier_short"] = tier_short
    for k, v in fwd.items():
        out[k] = v
    return out


def label_summary(labels: pd.DataFrame) -> dict[str, float]:
    n = max(1, len(labels))
    confirmed = labels[labels["is_leg_confirmed"] == 1]
    out: dict[str, float] = {
        "bar_count": float(len(labels)),
        "confirmed_leg_bar_count": float(len(confirmed)),
        "ideal_participate_long_rate": float(labels["ideal_participate_long"].mean()),
        "ideal_participate_short_rate": float(labels["ideal_participate_short"].mean()),
        "eligible_long_rate": float(labels["eligible_participate_long"].mean()),
        "eligible_short_rate": float(labels["eligible_participate_short"].mean()),
        "confirmed_leg_count": float(confirmed["leg_id"].nunique()),
        "ideal_long_per_1000_bars": float(labels["ideal_participate_long"].sum()) / n * 1000.0,
        "ideal_short_per_1000_bars": float(labels["ideal_participate_short"].sum()) / n * 1000.0,
    }
    if "participate_tier_long" in labels.columns:
        tier_l = labels["participate_tier_long"]
        tier_s = labels["participate_tier_short"]
        out.update(
            {
                "participate_tier2_long_rate": float((tier_l >= 2).mean()),
                "participate_tier1_long_rate": float((tier_l >= 1).mean()),
                "participate_tier2_short_rate": float((tier_s >= 2).mean()),
                "participate_tier1_short_rate": float((tier_s >= 1).mean()),
            }
        )
    return out


def count_confirmed_legs_from_bars(
    bars: pd.DataFrame,
    *,
    exclude_chop_hard: bool = True,
) -> int:
    """Count unique confirmed legs aligned with eval_participation §5.3."""
    df = bars.copy()
    if exclude_chop_hard and "chop_hard" in df.columns:
        df = df[df["chop_hard"] == 0]
    mask = (df["is_leg_confirmed"] == 1) & df["leg_type"].isin(CONFIRMED_LEG_TYPES)
    confirmed = df.loc[mask]
    if confirmed.empty:
        return 0
    return int(confirmed["leg_id"].nunique())

