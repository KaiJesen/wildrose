"""026 C1 leg-context features for participation head (anchor bar)."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import torch
import torch.nn as nn

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
