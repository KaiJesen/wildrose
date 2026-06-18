"""K 线形态片段的特征工程（尺度不变、严格因果）。

每个 bar 转为形状特征；可选附加多尺度线性趋势特征（斜率 / 残差离散 / R² / 强度）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_data.schema import COL_CLOSE, COL_HIGH, COL_LOW, COL_OPEN, COL_VOLUME
from transformer_kit.trend_features import (
    DEFAULT_TREND_WINDOWS,
    causal_log_price_trend_features,
    trend_column_names,
    trend_feature_dim,
)

BAR_SHAPE_COLS: tuple[str, ...] = ("log_ret", "body_ratio", "upper_wick", "lower_wick", "log_vol")
BAR_SHAPE_DIM: int = len(BAR_SHAPE_COLS)
LOG_RET_COL: int = 0


def feat_dim(*, use_trend_features: bool = True, windows: tuple[int, ...] = DEFAULT_TREND_WINDOWS) -> int:
    """单 bar 特征维度。"""
    dim = BAR_SHAPE_DIM
    if use_trend_features:
        dim += trend_feature_dim(windows)
    return dim


def all_feature_cols(
    *,
    use_trend_features: bool = True,
    windows: tuple[int, ...] = DEFAULT_TREND_WINDOWS,
) -> list[str]:
    cols = list(BAR_SHAPE_COLS)
    if use_trend_features:
        cols.extend(trend_column_names(windows))
    return cols


def bars_to_shape_features(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """单根 K 柱 → 形状特征向量 [N, 5]。

    列顺序: ``log_ret, body_ratio, upper_wick, lower_wick, log_vol``。
    """
    o = open_.astype(np.float64)
    h = high.astype(np.float64)
    l = low.astype(np.float64)
    c = close.astype(np.float64)
    v = np.clip(volume.astype(np.float64), 0.0, None)

    log_ret = np.zeros_like(c)
    if c.shape[0] > 1:
        prev = np.clip(c[:-1], eps, None)
        log_ret[1:] = np.log(np.clip(c[1:], eps, None) / prev)

    rng = np.clip(h - l, eps, None)
    body = c - o
    body_ratio = body / rng
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    upper_wick = upper / rng
    lower_wick = lower / rng
    log_vol = np.log(v + 1.0)

    return np.stack([log_ret, body_ratio, upper_wick, lower_wick, log_vol], axis=-1).astype(
        np.float32
    )


def build_bar_shape_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """从 OHLCV DataFrame 构建 bar 级形状特征表（不含趋势特征）。"""
    return build_bar_feature_frame(df, use_trend_features=False)


def build_bar_feature_frame(
    df: pd.DataFrame,
    *,
    use_trend_features: bool = True,
    trend_windows: tuple[int, ...] = DEFAULT_TREND_WINDOWS,
) -> tuple[pd.DataFrame, list[str]]:
    """从 OHLCV DataFrame 构建 bar 级特征表（形状 + 可选多尺度趋势）。"""
    for col in (COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME):
        if col not in df.columns:
            raise KeyError(f"missing column {col!r}")

    shape_arr = bars_to_shape_features(
        df[COL_OPEN].to_numpy(),
        df[COL_HIGH].to_numpy(),
        df[COL_LOW].to_numpy(),
        df[COL_CLOSE].to_numpy(),
        df[COL_VOLUME].to_numpy(),
    )
    cols = list(BAR_SHAPE_COLS)
    parts = [shape_arr]

    if use_trend_features:
        trend_arr = causal_log_price_trend_features(
            df[COL_CLOSE].to_numpy(),
            windows=trend_windows,
        )
        cols.extend(trend_column_names(trend_windows))
        parts.append(trend_arr)

    feat_df = pd.DataFrame(np.concatenate(parts, axis=-1), columns=cols)
    return feat_df, cols


def normalize_segment(segment: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """片段内 z-score（仅 trailing 统计，对每列独立；长度 1 时填 0）。"""
    if segment.ndim != 2:
        raise ValueError(f"segment must be [L, F], got {segment.shape}")
    out = segment.astype(np.float32, copy=True)
    if out.shape[0] == 1:
        return np.zeros_like(out)
    mean = out.mean(axis=0, keepdims=True)
    std = out.std(axis=0, keepdims=True)
    return ((out - mean) / (std + eps)).astype(np.float32)


def augment_segment(segment: np.ndarray, rng: np.random.Generator, *, noise_std: float = 0.05) -> np.ndarray:
    """语义保持增强：小幅噪声（用于对比学习可选扩展）。"""
    out = segment.copy()
    out += rng.normal(0.0, noise_std, size=out.shape).astype(np.float32)
    return out


def partition_bars_into_segments(
    bars: np.ndarray,
    n_segments: int,
    *,
    min_seg_len: int = 2,
) -> list[np.ndarray]:
    """将 ``[T, F]`` 连续 bar 均分为 ``n_segments`` 段（最后一段吸收余数）。"""
    t = bars.shape[0]
    if n_segments < 1:
        raise ValueError("n_segments must be >= 1")
    if t < n_segments * min_seg_len:
        raise ValueError(f"not enough bars ({t}) for {n_segments} segments (min_seg_len={min_seg_len})")

    base = t // n_segments
    segments: list[np.ndarray] = []
    start = 0
    for i in range(n_segments):
        extra = t - start - (n_segments - i) * base
        length = base + (1 if extra > 0 else 0)
        end = start + length
        segments.append(bars[start:end])
        start = end
    return segments


def pad_segments(
    segments: list[np.ndarray],
    *,
    max_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """变长片段列表 → ``[S, max_len, F]`` 与 ``lengths [S]``。"""
    if not segments:
        raise ValueError("empty segments")
    f = segments[0].shape[1]
    s = len(segments)
    out = np.zeros((s, max_len, f), dtype=np.float32)
    lengths = np.zeros(s, dtype=np.int64)
    for i, seg in enumerate(segments):
        ln = min(seg.shape[0], max_len)
        out[i, :ln] = seg[:ln]
        lengths[i] = ln
    return out, lengths
