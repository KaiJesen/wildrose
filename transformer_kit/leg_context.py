"""026 C1 leg-context features for participation head (anchor bar)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from market_data.schema import COL_CLOSE, COL_HIGH, COL_LOW, COL_TIME
from trading_system.config import TradingSystemConfig
from trading_system.crash import CrashRegimeDetector
from trading_system.slow_trend import SlowUptrendDetector
from trading_system.trend import TrendRegimeFilter
from trading_system.trend_segment import TrendSegmentEngine
from trading_system.trend_signal import TrendMemory, TrendSignalProvider

# Frozen vocabulary — inference TrendSegmentEngine must map into same ids (026 C2).
LEG_TYPE_TO_ID: dict[str, int] = {
    "NONE": 0,
    "FAST_UP_LEG": 1,
    "FAST_DOWN_LEG": 2,
    "SLOW_UP_LEG": 3,
    "SLOW_DOWN_LEG": 4,
    "RANGE_LEG": 5,
    "TRANSITION_LEG": 6,
    "SURGE_LEG": 7,
    "CHOP": 8,
}
NUM_LEG_TYPES = max(LEG_TYPE_TO_ID.values()) + 1
LEG_CONTEXT_VERSION = "026_c1_v1"
BARS_SINCE_LOG_DENOM = math.log1p(128.0)


def leg_type_to_id(leg_type: str) -> int:
    return LEG_TYPE_TO_ID.get(str(leg_type or "NONE"), 0)


def bars_since_norm(bars_since: int) -> float:
    if bars_since <= 0:
        return 0.0
    return float(math.log1p(bars_since) / BARS_SINCE_LOG_DENOM)


def build_leg_starts(label_df: pd.DataFrame) -> dict[int, int]:
    starts: dict[int, int] = {}
    if "leg_id" not in label_df.columns:
        return starts
    for row in label_df.itertuples(index=False):
        leg_id = int(getattr(row, "leg_id", -1))
        if leg_id < 0:
            continue
        bar_idx = int(row.bar_idx)
        prev = starts.get(leg_id)
        if prev is None or bar_idx < prev:
            starts[leg_id] = bar_idx
    return starts


def anchor_leg_fields(row, *, leg_starts: dict[int, int]) -> dict[str, float]:
    leg_id = int(getattr(row, "leg_id", -1))
    progress = float(getattr(row, "leg_progress_ratio", 0.0) or 0.0)
    leg_type = str(getattr(row, "leg_type", "NONE") or "NONE")
    if leg_id < 0:
        bars_since = 0
        progress = 0.0
        leg_type = "NONE"
    else:
        start = leg_starts.get(leg_id, int(row.bar_idx))
        bars_since = max(1, int(row.bar_idx) - start + 1)
    return {
        "leg_type_id": float(leg_type_to_id(leg_type)),
        "leg_progress_ratio": max(0.0, min(1.0, progress)),
        "bars_since_norm": bars_since_norm(bars_since),
    }


def d1_sample_weight(row: dict[str, float]) -> float:
    """026 D1 tier weights (ideal / hard-negative / other)."""
    ideal = row.get("ideal_participate_long", 0.0) >= 1.0 or row.get("ideal_participate_short", 0.0) >= 1.0
    if ideal:
        return 10.0
    confirmed = row.get("is_leg_confirmed", 0.0) >= 1.0
    if confirmed and not ideal:
        return 2.0
    return 0.5


class LegContextFusion(nn.Module):
    """Learnable leg_type embedding + numeric progress features → d_model bias for query."""

    def __init__(self, d_model: int, *, num_leg_types: int = NUM_LEG_TYPES, embed_dim: int = 4) -> None:
        super().__init__()
        self.leg_type_emb = nn.Embedding(num_leg_types, embed_dim)
        self.numeric_proj = nn.Linear(2, embed_dim)
        self.out_proj = nn.Linear(embed_dim * 2, d_model)

    def forward(self, leg_context: dict[str, torch.Tensor]) -> torch.Tensor:
        type_id = leg_context["leg_type_id"].long().clamp(0, self.leg_type_emb.num_embeddings - 1)
        numeric = torch.stack(
            (
                leg_context["leg_progress_ratio"],
                leg_context["bars_since_norm"],
            ),
            dim=-1,
        )
        type_vec = self.leg_type_emb(type_id)
        num_vec = self.numeric_proj(numeric)
        return self.out_proj(torch.cat([type_vec, num_vec], dim=-1))


def leg_context_from_batch(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor] | None:
    if "leg_type_id" not in batch:
        return None
    return {
        "leg_type_id": batch["leg_type_id"].to(device),
        "leg_progress_ratio": batch["leg_progress_ratio"].to(device),
        "bars_since_norm": batch["bars_since_norm"].to(device),
    }


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


def precompute_inference_leg_context(df: pd.DataFrame, cfg: TradingSystemConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal leg context per bar for 026 C1 inference (026 C2: matches TrendSegmentEngine online)."""
    from trading_system.adapters.market_state_model import compute_atr

    n = len(df)
    leg_type_id = np.zeros(n, dtype=np.float32)
    leg_progress = np.zeros(n, dtype=np.float32)
    bars_since = np.zeros(n, dtype=np.float32)

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

    for i in range(n):
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
        segment_ctx = segment_engine.update(
            bar_idx=i,
            high=float(high[i]),
            low=float(low[i]),
            close=float(close[i]),
            atr=atr_v,
            trend_signal=trend_signal,
            slow_ctx=slow_ctx,
            crash_ctx=crash_ctx,
            is_model_blind=bool(crash_ctx.is_model_blind_crash),
        )
        leg_type_id[i] = float(leg_type_to_id(segment_ctx.leg_type.value))
        leg_progress[i] = float(max(0.0, min(1.0, segment_ctx.leg_progress_ratio)))
        bars_since[i] = bars_since_norm(int(segment_ctx.bars_since_leg_start))

    return leg_type_id, leg_progress, bars_since
