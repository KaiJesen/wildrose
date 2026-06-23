from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from trend_leg.labels import encode_leg_type, encode_sub_phase


@dataclass(frozen=True)
class DatasetSplit:
    train_end: int
    valid_end: int


def time_split_indices(n: int, train_ratio: float = 0.7, valid_ratio: float = 0.15) -> DatasetSplit:
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    return DatasetSplit(train_end=train_end, valid_end=valid_end)


class TrendLegDataset(Dataset):
    def __init__(
        self,
        feat_df: pd.DataFrame,
        label_df: pd.DataFrame,
        *,
        context_bars: int,
        start_idx: int,
        end_idx: int,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        min_teacher_conf: float = 0.0,
    ) -> None:
        self.context_bars = int(context_bars)
        self.feat = feat_df.to_numpy(dtype=np.float32)
        self.leg_type = np.array([encode_leg_type(v) for v in label_df["trend_leg_type"]], dtype=np.int64)
        self.sub_phase = np.array([encode_sub_phase(v) for v in label_df["sub_phase"]], dtype=np.int64)
        self.leg_progress = label_df["leg_progress"].to_numpy(dtype=np.float32)
        self.is_confirmed = label_df["is_leg_confirmed"].to_numpy(dtype=np.float32)
        self.teacher_conf = label_df.get("teacher_confidence", pd.Series(np.ones(len(label_df)))).to_numpy(dtype=np.float32)
        self.start = max(start_idx, self.context_bars)
        self.end = min(end_idx, len(self.feat))
        self.min_teacher_conf = float(min_teacher_conf)
        if feature_mean is None or feature_std is None:
            arr = self.feat[self.start : self.end]
            feature_mean = arr.mean(axis=0)
            feature_std = arr.std(axis=0)
        self.mean = feature_mean.astype(np.float32)
        self.std = np.clip(feature_std.astype(np.float32), 1e-6, None)

    def __len__(self) -> int:
        return max(0, self.end - self.start)

    def __getitem__(self, idx: int):
        t = self.start + idx
        x = (self.feat[t - self.context_bars : t] - self.mean) / self.std
        return {
            "x": torch.from_numpy(x),
            "leg_type": torch.tensor(self.leg_type[t], dtype=torch.long),
            "sub_phase": torch.tensor(self.sub_phase[t], dtype=torch.long),
            "leg_progress": torch.tensor([self.leg_progress[t]], dtype=torch.float32),
            "is_confirmed": torch.tensor([self.is_confirmed[t]], dtype=torch.float32),
            "teacher_conf": torch.tensor([self.teacher_conf[t]], dtype=torch.float32),
        }
