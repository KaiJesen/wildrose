"""024 leg-alignment training dataset (join sequence samples with participation labels)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from transformer_kit.leg_context import anchor_leg_fields, build_leg_starts, d1_sample_weight
from transformer_kit.segment_dataset import PatternSequenceDataset, SequenceSampleIndex


def _default_label_row(*, use_leg_context: bool = False) -> dict[str, float]:
    row = {
        "ideal_participate_long": 0.0,
        "ideal_participate_short": 0.0,
        "is_leg_confirmed": 0.0,
        "chop_hard": 0.0,
        "align_direction_up": 0.0,
        "align_direction_down": 0.0,
        "leg_type_fast_down": 0.0,
        "forward_leg_roi_12": 0.0,
        "forward_leg_roi_24": 0.0,
        "forward_leg_roi_48": 0.0,
        "sample_weight": 1.0,
    }
    if use_leg_context:
        row.update({"leg_type_id": 0.0, "leg_progress_ratio": 0.0, "bars_since_norm": 0.0})
    return row


class LegParticipationSequenceDataset(PatternSequenceDataset):
    """PatternSequenceDataset + per-anchor leg participation supervision."""

    def __init__(
        self,
        bars: np.ndarray,
        sample_indices: list[SequenceSampleIndex],
        raw_log_ret: np.ndarray | None,
        label_df: pd.DataFrame,
        *,
        zscore_window: int = 120,
        direction_threshold: float = 0.0,
        risk_vol_threshold: float = 0.01,
        leg_align_horizons: tuple[int, ...] = (),
        confirmed_leg_weight: float = 3.0,
        chop_hard_weight: float = 0.2,
        use_leg_context: bool = False,
        d1_sampling: bool = False,
    ) -> None:
        super().__init__(
            bars,
            sample_indices,
            raw_log_ret,
            zscore_window=zscore_window,
            return_market_state_targets=True,
            direction_threshold=direction_threshold,
            risk_vol_threshold=risk_vol_threshold,
        )
        self.leg_align_horizons = tuple(int(h) for h in leg_align_horizons)
        self.confirmed_leg_weight = float(confirmed_leg_weight)
        self.chop_hard_weight = float(chop_hard_weight)
        self.use_leg_context = use_leg_context
        self.d1_sampling = d1_sampling
        leg_starts = build_leg_starts(label_df) if use_leg_context or d1_sampling else {}
        self._label_by_bar: dict[int, dict[str, float]] = {}
        for row in label_df.itertuples(index=False):
            bar_idx = int(row.bar_idx)
            align = str(getattr(row, "align_direction", "NONE"))
            confirmed = float(getattr(row, "is_leg_confirmed", 0))
            chop = float(getattr(row, "chop_hard", 0))
            leg_type = str(getattr(row, "leg_type", ""))
            ideal_l = float(getattr(row, "ideal_participate_long", 0))
            ideal_s = float(getattr(row, "ideal_participate_short", 0))
            row_data = {
                "ideal_participate_long": ideal_l,
                "ideal_participate_short": ideal_s,
                "is_leg_confirmed": confirmed,
                "chop_hard": chop,
                "align_direction_up": 1.0 if align == "UP" else 0.0,
                "align_direction_down": 1.0 if align == "DOWN" else 0.0,
                "leg_type_fast_down": 1.0 if leg_type == "FAST_DOWN_LEG" else 0.0,
                "forward_leg_roi_12": float(getattr(row, "forward_leg_roi_12", 0)),
                "forward_leg_roi_24": float(getattr(row, "forward_leg_roi_24", 0)),
                "forward_leg_roi_48": float(getattr(row, "forward_leg_roi_48", 0)),
            }
            if use_leg_context:
                row_data.update(anchor_leg_fields(row, leg_starts=leg_starts))
            else:
                row_data.update(
                    {
                        "leg_type_id": 0.0,
                        "leg_progress_ratio": 0.0,
                        "bars_since_norm": 0.0,
                    }
                )
            if d1_sampling:
                row_data["sample_weight"] = d1_sample_weight(row_data)
            else:
                weight = 1.0
                if confirmed >= 1.0:
                    weight *= self.confirmed_leg_weight
                if chop >= 1.0:
                    weight *= self.chop_hard_weight
                row_data["sample_weight"] = weight
            self._label_by_bar[bar_idx] = row_data

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        out = super().__getitem__(i)
        anchor = self.sample_indices[i].context_end - 1
        labels = self._label_by_bar.get(anchor, _default_label_row(use_leg_context=self.use_leg_context))
        for key, value in labels.items():
            out[key] = torch.tensor(value, dtype=torch.float32)
        out["anchor_bar_idx"] = torch.tensor(anchor, dtype=torch.long)
        if self.raw_log_ret is not None:
            for h in self.leg_align_horizons:
                end = min(len(self.raw_log_ret), anchor + h + 1)
                if end > anchor + 1:
                    hz_ret = float(self.raw_log_ret[anchor + 1 : end].sum())
                else:
                    hz_ret = 0.0
                out[f"target_hz_return_{h}"] = torch.tensor(hz_ret, dtype=torch.float32)
        return out


def load_label_dataframe(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
