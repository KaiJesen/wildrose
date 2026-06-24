from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetSplit:
    train_end: int
    valid_end: int


def time_split_indices(n: int, train_ratio: float = 0.7, valid_ratio: float = 0.15) -> DatasetSplit:
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    return DatasetSplit(train_end=train_end, valid_end=valid_end)


class BestPointDataset(Dataset):
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
    ) -> None:
        self.context_bars = int(context_bars)
        self.feat = feat_df.to_numpy(dtype=np.float32)
        self.entry = label_df["entry_label"].to_numpy(dtype=np.int64)
        self.hold = label_df["hold_label"].to_numpy(dtype=np.int64)
        self.exit = label_df["exit_label"].to_numpy(dtype=np.int64)
        self.opp = label_df["future_best_net_roi"].to_numpy(dtype=np.float32)
        self.start = max(start_idx, self.context_bars)
        self.end = min(end_idx, len(self.feat))
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
            "entry": torch.tensor(self.entry[t], dtype=torch.long),
            "hold": torch.tensor(self.hold[t], dtype=torch.long),
            "exit": torch.tensor(self.exit[t], dtype=torch.long),
            "opp": torch.tensor([self.opp[t]], dtype=torch.float32),
        }

