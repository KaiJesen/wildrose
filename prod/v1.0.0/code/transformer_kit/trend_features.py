"""多尺度因果趋势特征：线性回归斜率 + 残差离散度 + 拟合优度 + 趋势强度。

每个窗口 W 在 bar t 仅使用 ``[t-W+1, t]`` 的 log(close) 做 OLS，严格因果、无泄漏。
"""

from __future__ import annotations

import numpy as np

DEFAULT_TREND_WINDOWS: tuple[int, ...] = (20, 60, 120)
TREND_METRIC_SUFFIXES: tuple[str, ...] = ("slope", "resid_std", "r2", "strength")


def trend_column_names(windows: tuple[int, ...] = DEFAULT_TREND_WINDOWS) -> list[str]:
    """趋势特征列名，例如 ``trend_slope_20``。"""
    cols: list[str] = []
    for w in windows:
        for suffix in TREND_METRIC_SUFFIXES:
            cols.append(f"trend_{suffix}_{w}")
    return cols


def trend_feature_dim(windows: tuple[int, ...] = DEFAULT_TREND_WINDOWS) -> int:
    return len(windows) * len(TREND_METRIC_SUFFIXES)


def causal_log_price_trend_features(
    close: np.ndarray,
    *,
    windows: tuple[int, ...] = DEFAULT_TREND_WINDOWS,
    eps: float = 1e-8,
) -> np.ndarray:
    """对 log(close) 做多窗口因果线性趋势分解。

    返回 ``[N, len(windows)*4]``，每个窗口四列依次为：
    ``slope, resid_std, r2, strength``，其中 ``strength = |slope| / (resid_std + eps)``。
    """
    c = np.clip(close.astype(np.float64), eps, None)
    log_close = np.log(c)
    n = log_close.shape[0]
    out_cols: list[np.ndarray] = []

    for window in windows:
        slope = np.zeros(n, dtype=np.float64)
        resid_std = np.zeros(n, dtype=np.float64)
        r2 = np.zeros(n, dtype=np.float64)

        for i in range(n):
            start = max(0, i - window + 1)
            y = log_close[start : i + 1]
            m = y.shape[0]
            if m < 3:
                continue
            x = np.arange(m, dtype=np.float64)
            x_mean = x.mean()
            y_mean = y.mean()
            dx = x - x_mean
            dy = y - y_mean
            ss_xx = float((dx * dx).sum())
            if ss_xx < eps:
                continue
            ss_xy = float((dx * dy).sum())
            beta = ss_xy / ss_xx
            alpha = y_mean - beta * x_mean
            resid = y - (alpha + beta * x)
            ss_res = float((resid * resid).sum())
            ss_tot = float((dy * dy).sum()) + eps
            slope[i] = beta
            resid_std[i] = np.sqrt(ss_res / m)
            r2[i] = max(0.0, min(1.0, 1.0 - ss_res / ss_tot))

        strength = np.abs(slope) / (resid_std + eps)
        out_cols.extend([slope, resid_std, r2, strength])

    return np.stack(out_cols, axis=-1).astype(np.float32)


def trend_col_index(
    window: int,
    metric: str,
    *,
    windows: tuple[int, ...] = DEFAULT_TREND_WINDOWS,
    bar_shape_dim: int = 5,
) -> int:
    """趋势列在完整特征向量中的下标（含前置 bar 形状特征）。"""
    if metric not in TREND_METRIC_SUFFIXES:
        raise KeyError(f"unknown trend metric {metric!r}")
    if window not in windows:
        raise KeyError(f"window {window} not in {windows}")
    w_idx = windows.index(window)
    m_idx = TREND_METRIC_SUFFIXES.index(metric)
    return bar_shape_dim + w_idx * len(TREND_METRIC_SUFFIXES) + m_idx
