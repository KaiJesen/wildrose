"""K 线序列的因果特征工程（feed 给 KlineBertEmbedding）。

所有变换严格因果：位置 i 只使用 <= i 的信息。
- ``add_log_returns``    : 计算对数收益
- ``add_calendar_features``: 提取 minute-of-day / day-of-week
- ``causal_zscore``       : 滚动窗口 z-score（仅 trailing）
- ``build_feature_frame`` : 把上面拼成 embedding 所需的 (feature_df, columns)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_data.schema import (
    COL_CLOSE,
    COL_HIGH,
    COL_LOW,
    COL_OPEN,
    COL_TIME,
    COL_VOLUME,
)


def add_log_returns(
    df: pd.DataFrame,
    *,
    col_close: str = COL_CLOSE,
    out_col: str = "log_ret",
) -> pd.DataFrame:
    """新增 ``log_ret`` 列: log(close_t / close_{t-1})。首行填 0。"""
    if col_close not in df.columns:
        raise KeyError(f"missing column {col_close!r} in df")
    out = df.copy()
    closes = out[col_close].to_numpy(dtype=np.float64)
    log_ret = np.zeros_like(closes)
    if closes.shape[0] > 1:
        prev = np.clip(closes[:-1], 1e-12, None)
        log_ret[1:] = np.log(closes[1:] / prev)
    out[out_col] = log_ret
    return out


def add_calendar_features(
    df: pd.DataFrame,
    *,
    col_time: str = COL_TIME,
    minute_col: str = "minute_of_day",
    dow_col: str = "dow",
) -> pd.DataFrame:
    """新增 ``minute_of_day`` (0..1439) 与 ``dow`` (0..6) 整数列。"""
    if col_time not in df.columns:
        raise KeyError(f"missing column {col_time!r} in df")
    out = df.copy()
    ts = pd.to_datetime(out[col_time])
    out[minute_col] = (ts.dt.hour * 60 + ts.dt.minute).astype(np.int64)
    out[dow_col] = ts.dt.dayofweek.astype(np.int64)
    return out


def causal_zscore(
    series: pd.Series,
    *,
    window: int = 60,
    min_periods: int | None = None,
    eps: float = 1e-8,
) -> pd.Series:
    """trailing-window z-score。窗口起点不足时填 0。"""
    if min_periods is None:
        min_periods = max(2, window // 4)
    s = series.astype(float)
    mean = s.rolling(window=window, min_periods=min_periods).mean()
    std = s.rolling(window=window, min_periods=min_periods).std(ddof=0)
    z = (s - mean) / (std + eps)
    return z.fillna(0.0)


DEFAULT_VALUE_COLS: tuple[str, ...] = (
    COL_OPEN,
    COL_HIGH,
    COL_LOW,
    COL_CLOSE,
    COL_VOLUME,
    "log_ret",
)


def build_feature_frame(
    df: pd.DataFrame,
    *,
    value_cols: tuple[str, ...] | list[str] = DEFAULT_VALUE_COLS,
    zscore_window: int = 60,
) -> tuple[pd.DataFrame, list[str]]:
    """从标准化的 K 线 DataFrame 构建因果特征表。

    返回 ``(feature_df, feature_columns)``:
    - ``feature_df`` 包含 time、连续特征列 (前缀 ``z_``) 与 ``minute_of_day`` / ``dow``。
    - ``feature_columns`` 是有序的连续特征列名（喂给 embedding 的 feats）。
    """
    base = add_log_returns(df)
    base = add_calendar_features(base)

    missing = [c for c in value_cols if c not in base.columns]
    if missing:
        raise KeyError(f"missing value columns {missing} in df (available: {list(base.columns)})")

    feat = pd.DataFrame({COL_TIME: base[COL_TIME]})
    feat_cols: list[str] = []
    for col in value_cols:
        new_col = f"z_{col}"
        feat[new_col] = causal_zscore(base[col], window=zscore_window)
        feat_cols.append(new_col)

    feat["minute_of_day"] = base["minute_of_day"]
    feat["dow"] = base["dow"]
    return feat, feat_cols


def make_sliding_windows(
    feat_array: np.ndarray,
    minute_arr: np.ndarray,
    dow_arr: np.ndarray,
    *,
    window: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 [N, F] 的特征数组切成 [B, T, F] 滑动窗口（同时切日历数组）。

    每个窗口包含连续 ``window`` 步，相邻窗口偏移 ``stride``。
    """
    if feat_array.ndim != 2:
        raise ValueError(f"feat_array must be 2D [N, F], got shape {feat_array.shape}")
    n = feat_array.shape[0]
    if n < window:
        raise ValueError(f"not enough rows ({n}) for window={window}")
    if minute_arr.shape[0] != n or dow_arr.shape[0] != n:
        raise ValueError("minute_arr/dow_arr length must match feat_array")

    starts = np.arange(0, n - window + 1, stride)
    if starts.size == 0:
        raise ValueError("no windows can be built; check stride/window/n")

    feats = np.stack([feat_array[i : i + window] for i in starts], axis=0)
    minutes = np.stack([minute_arr[i : i + window] for i in starts], axis=0)
    dows = np.stack([dow_arr[i : i + window] for i in starts], axis=0)
    return feats, minutes, dows
