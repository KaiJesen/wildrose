"""片段级数据集（Stage 1/2/3）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from transformer_kit.segment_features import (
    build_bar_feature_frame,
    normalize_segment,
    pad_segments,
    partition_bars_into_segments,
)
from transformer_kit.labels import build_market_state_targets
from transformer_kit.trend_features import DEFAULT_TREND_WINDOWS


@dataclass(frozen=True)
class BarSeriesBundle:
    """全序列 bar 形状特征与切分索引。"""

    bars: np.ndarray
    raw_log_ret: np.ndarray
    train_idx: np.ndarray
    valid_idx: np.ndarray
    test_idx: np.ndarray
    zscore_window: int = 120


def time_ordered_split(
    n_samples: int,
    *,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_samples < 3:
        raise ValueError("need at least 3 samples")
    t1 = int(n_samples * train_ratio)
    t2 = int(n_samples * (train_ratio + valid_ratio))
    t1 = max(1, min(t1, n_samples - 2))
    t2 = max(t1 + 1, min(t2, n_samples - 1))
    idx = np.arange(n_samples)
    return idx[:t1], idx[t1:t2], idx[t2:]


def global_causal_zscore_bars(bars: np.ndarray, *, window: int = 120) -> np.ndarray:
    """对 bar 特征逐列做 trailing z-score（因果，无泄漏）。

    统计量始终基于原始特征值计算，避免窗口内混入已归一化数据导致无法反标准化。
    """
    raw = bars.astype(np.float64, copy=False)
    out = bars.astype(np.float32, copy=True)
    n, f = out.shape
    for j in range(f):
        for i in range(n):
            start = max(0, i - window + 1)
            seg = raw[start : i + 1, j]
            m = float(seg.mean())
            s = float(seg.std()) + 1e-8
            out[i, j] = (raw[i, j] - m) / s
    return out


def prepare_bar_series(
    df: pd.DataFrame,
    *,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    zscore_window: int = 120,
    use_trend_features: bool = True,
    trend_windows: tuple[int, ...] = DEFAULT_TREND_WINDOWS,
) -> BarSeriesBundle:
    feat_df, _ = build_bar_feature_frame(
        df,
        use_trend_features=use_trend_features,
        trend_windows=trend_windows,
    )
    feat_arr = feat_df.to_numpy(dtype=np.float32)
    raw_log_ret = feat_df["log_ret"].to_numpy(dtype=np.float32)
    bars = global_causal_zscore_bars(feat_arr, window=zscore_window)
    n = bars.shape[0]
    train_idx, valid_idx, test_idx = time_ordered_split(
        n, train_ratio=train_ratio, valid_ratio=valid_ratio
    )
    return BarSeriesBundle(
        bars=bars,
        raw_log_ret=raw_log_ret,
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        zscore_window=zscore_window,
    )


class BarWindowDataset(Dataset):
    """连续 K 线窗口（供自动切分 Embedding 使用）。"""

    def __init__(
        self,
        bars: np.ndarray,
        indices: np.ndarray,
        *,
        window: int = 128,
        samples_per_epoch: int = 2000,
        seed: int = 0,
    ) -> None:
        self.bars = bars
        self.indices = np.asarray(indices, dtype=np.int64)
        self.window = window
        self.samples_per_epoch = samples_per_epoch
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.rng.integers(0, 2**31 - 1))
        valid_starts = self.indices[self.indices + self.window <= self.bars.shape[0]]
        if valid_starts.size == 0:
            raise ValueError("no valid window start; increase data or reduce window")
        start = int(rng.choice(valid_starts))
        ctx = self.bars[start : start + self.window].astype(np.float32)
        return {
            "ctx_bars": torch.from_numpy(ctx),
            "ctx_lengths": torch.tensor(self.window, dtype=torch.long),
        }


class VariableSegmentDataset(Dataset):
    """Stage 1/2：随机起点 + 随机长度的变长片段。"""

    def __init__(
        self,
        bars: np.ndarray,
        indices: np.ndarray,
        *,
        min_seg_len: int = 4,
        max_seg_len: int = 32,
        samples_per_epoch: int = 2000,
        seed: int = 0,
    ) -> None:
        self.bars = bars
        self.indices = np.asarray(indices, dtype=np.int64)
        self.min_seg_len = min_seg_len
        self.max_seg_len = max_seg_len
        self.samples_per_epoch = samples_per_epoch
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.rng.integers(0, 2**31 - 1))
        length = int(rng.integers(self.min_seg_len, self.max_seg_len + 1))
        valid_starts = self.indices[self.indices + length <= self.bars.shape[0]]
        if valid_starts.size == 0:
            raise ValueError("no valid segment start; increase data or reduce max_seg_len")
        start = int(rng.choice(valid_starts))
        seg = self.bars[start : start + length].copy()

        padded = np.zeros((self.max_seg_len, seg.shape[1]), dtype=np.float32)
        ln = min(length, self.max_seg_len)
        padded[:ln] = seg[:ln]
        return {
            "seg_bars": torch.from_numpy(padded),
            "seg_lengths": torch.tensor(ln, dtype=torch.long),
        }


@dataclass(frozen=True)
class SequenceSampleIndex:
    """Stage 3 样本：上下文 bar 区间与未来目标区间。"""

    context_start: int
    context_end: int
    future_end: int


def build_sequence_sample_indices(
    bars_len: int,
    *,
    context_bars: int,
    pred_horizon: int,
    stride: int = 8,
    index_min: int = 0,
    index_max: int | None = None,
) -> list[SequenceSampleIndex]:
    need = context_bars + pred_horizon
    if index_max is None:
        index_max = bars_len - 1
    out: list[SequenceSampleIndex] = []
    for start in range(0, bars_len - need + 1, stride):
        ctx_end = start + context_bars
        fut_end = ctx_end + pred_horizon
        if start < index_min or fut_end - 1 > index_max:
            continue
        out.append(
            SequenceSampleIndex(
                context_start=start,
                context_end=ctx_end,
                future_end=fut_end,
            )
        )
    if not out:
        raise ValueError("no sequence samples; reduce context_bars or pred_horizon")
    return out


def trailing_raw_log_ret_stats(
    raw_log_ret: np.ndarray,
    anchor: int,
    *,
    window: int = 120,
) -> tuple[float, float]:
    """与 ``global_causal_zscore_bars`` 一致的 trailing 均值/标准差。"""
    start = max(0, anchor - window + 1)
    seg = raw_log_ret[start : anchor + 1]
    return float(seg.mean()), float(seg.std() + 1e-8)


class PatternSequenceDataset(Dataset):
    """Stage 3：连续上下文窗口 + 未来 bar（切分由模型自动完成）。"""

    def __init__(
        self,
        bars: np.ndarray,
        sample_indices: list[SequenceSampleIndex],
        raw_log_ret: np.ndarray | None = None,
        *,
        zscore_window: int = 120,
        return_market_state_targets: bool = False,
        direction_threshold: float = 0.0,
        risk_vol_threshold: float = 0.01,
    ) -> None:
        self.bars = bars
        self.sample_indices = sample_indices
        self.raw_log_ret = raw_log_ret
        self.zscore_window = zscore_window
        self.return_market_state_targets = return_market_state_targets
        self.direction_threshold = float(direction_threshold)
        self.risk_vol_threshold = float(risk_vol_threshold)

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        spec = self.sample_indices[i]
        ctx = self.bars[spec.context_start : spec.context_end].astype(np.float32)
        future = self.bars[spec.context_end : spec.future_end].astype(np.float32)
        out = {
            "ctx_bars": torch.from_numpy(ctx),
            "ctx_lengths": torch.tensor(ctx.shape[0], dtype=torch.long),
            "future_bars": torch.from_numpy(future),
        }
        if self.raw_log_ret is not None:
            anchor = spec.context_end - 1
            m, s = trailing_raw_log_ret_stats(
                self.raw_log_ret, anchor, window=self.zscore_window,
            )
            future_raw = self.raw_log_ret[spec.context_end : spec.future_end].astype(np.float32)
            fut_means: list[float] = []
            fut_stds: list[float] = []
            for idx in range(spec.context_end, spec.future_end):
                fm, fs = trailing_raw_log_ret_stats(
                    self.raw_log_ret, idx, window=self.zscore_window,
                )
                fut_means.append(fm)
                fut_stds.append(fs)
            out["future_raw_log_ret"] = torch.from_numpy(future_raw)
            out["future_log_ret_mean"] = torch.tensor(fut_means, dtype=torch.float32)
            out["future_log_ret_std"] = torch.tensor(fut_stds, dtype=torch.float32)
            out["log_ret_mean"] = torch.tensor(m, dtype=torch.float32)
            out["log_ret_std"] = torch.tensor(s, dtype=torch.float32)
            if self.return_market_state_targets:
                tgt = build_market_state_targets(
                    future_raw,
                    direction_threshold=self.direction_threshold,
                    risk_vol_threshold=self.risk_vol_threshold,
                )
                out["target_return"] = tgt.future_log_ret
                out["target_direction"] = tgt.direction_label
                out["target_volatility"] = tgt.volatility
                out["target_risk"] = tgt.risk_label
                if tgt.move_label is not None:
                    out["target_move"] = tgt.move_label
        return out


def make_synthetic_ohlcv(n: int = 1200, seed: int = 0) -> pd.DataFrame:
    from market_data.schema import COL_CLOSE, COL_TIME

    rng = np.random.default_rng(seed)
    t = pd.date_range("2020-01-01 09:30", periods=n, freq="60min")
    close = 100.0 + np.cumsum(rng.normal(0, 0.4, size=n))
    return pd.DataFrame(
        {
            COL_TIME: t,
            "open": close + rng.normal(0, 0.05, size=n),
            "high": close + np.abs(rng.normal(0, 0.2, size=n)),
            "low": close - np.abs(rng.normal(0, 0.2, size=n)),
            COL_CLOSE: close,
            "volume": rng.integers(1000, 50000, size=n).astype(float),
        }
    )
