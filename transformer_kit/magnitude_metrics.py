"""预测幅度评估：价格变动相对误差与容差命中率。"""

from __future__ import annotations

import numpy as np


def cumulative_price_change(log_ret: np.ndarray) -> np.ndarray:
    """逐步 log_ret ``[N,H]`` → 累计价格变动比例 ``exp(sum)-1``。"""
    if log_ret.ndim == 1:
        return np.expm1(log_ret)
    return np.expm1(log_ret.sum(axis=-1))


def magnitude_relative_errors(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (价格相对误差, log累计相对误差)，形状 ``[N]``。"""
    if pred.ndim == 2:
        pred_cum = pred.sum(axis=1)
        tgt_cum = target.sum(axis=1)
    else:
        pred_cum = pred
        tgt_cum = target
    pred_chg = np.expm1(pred_cum)
    tgt_chg = np.expm1(tgt_cum)
    denom_price = np.maximum(np.abs(tgt_chg), eps)
    denom_log = np.maximum(np.abs(tgt_cum), eps)
    price_rel = np.abs(pred_chg - tgt_chg) / denom_price
    log_rel = np.abs(pred_cum - tgt_cum) / denom_log
    return price_rel, log_rel


def magnitude_accuracy_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    tolerance: float = 0.2,
    min_move: float = 1e-4,
    eps: float = 1e-5,
) -> dict[str, float]:
    """评估预测幅度是否在容差内（默认 20% 相对误差）。

    仅在 ``|真实价格变动| > min_move`` 的样本上统计命中率，避免极小波动导致除法失真。
    """
    price_rel, log_rel = magnitude_relative_errors(pred, target, eps=eps)
    if pred.ndim == 2:
        tgt_cum = target.sum(axis=1)
    else:
        tgt_cum = target
    tgt_chg = np.expm1(tgt_cum)
    mask = np.abs(tgt_chg) > min_move
    n_eval = int(mask.sum())
    if n_eval == 0:
        return {
            "magnitude_within_tol_rate": 0.0,
            "magnitude_within_tol_rate_log": 0.0,
            "magnitude_mean_rel_err": 0.0,
            "magnitude_median_rel_err": 0.0,
            "magnitude_p90_rel_err": 0.0,
            "magnitude_eval_samples": 0.0,
            "magnitude_tolerance": float(tolerance),
        }
    pr = price_rel[mask]
    lr = log_rel[mask]
    return {
        "magnitude_within_tol_rate": float((pr <= tolerance).mean()),
        "magnitude_within_tol_rate_log": float((lr <= tolerance).mean()),
        "magnitude_mean_rel_err": float(pr.mean()),
        "magnitude_median_rel_err": float(np.median(pr)),
        "magnitude_p90_rel_err": float(np.quantile(pr, 0.9)),
        "magnitude_eval_samples": float(n_eval),
        "magnitude_tolerance": float(tolerance),
    }


def fit_cumulative_magnitude_scale(pred: np.ndarray, target: np.ndarray) -> float:
    """在 train+valid 上拟合累计幅度缩放，使 ``scale * sum(pred) ≈ sum(target)``。"""
    pred_cum = pred.sum(axis=1) if pred.ndim == 2 else pred
    tgt_cum = target.sum(axis=1) if target.ndim == 2 else target
    var = float(pred_cum.var())
    if var < 1e-10:
        return 1.0
    cov = float(((pred_cum - pred_cum.mean()) * (tgt_cum - tgt_cum.mean())).mean())
    return float(np.clip(cov / (var + 1e-10), 0.05, 5.0))


def fit_relative_magnitude_scale(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    eps: float = 1e-5,
    grid_size: int = 240,
) -> float:
    """搜索累计缩放系数，使价格变动相对误差最小。"""
    pred_cum = pred.sum(axis=1) if pred.ndim == 2 else pred
    tgt_cum = target.sum(axis=1) if target.ndim == 2 else target
    pred_chg = np.expm1(pred_cum)
    tgt_chg = np.expm1(tgt_cum)
    mask = np.abs(tgt_chg) > eps
    if not mask.any():
        return fit_cumulative_magnitude_scale(pred, target)
    pred_c = pred_chg[mask]
    tgt_c = tgt_chg[mask]
    lo = max(0.05, float(np.percentile(pred_c / np.maximum(np.abs(tgt_c), eps), 5)) * 0.5)
    hi = min(5.0, float(np.percentile(pred_c / np.maximum(np.abs(tgt_c), eps), 95)) * 1.5 + 1e-3)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = 0.05, 5.0
    best_s, best_err = 1.0, float("inf")
    for s in np.linspace(lo, hi, grid_size):
        err = np.abs(s * pred_c - tgt_c) / np.maximum(np.abs(tgt_c), eps)
        score = float(err.mean())
        if score < best_err:
            best_err = score
            best_s = float(s)
    return best_s


def denorm_zscore_log_ret(
    z: np.ndarray,
    mean: float | np.ndarray,
    std: float | np.ndarray,
) -> np.ndarray:
    """将 z-score 的 log_ret 还原到原始尺度（支持逐步 mean/std）。"""
    mean_a = np.asarray(mean, dtype=np.float64)
    std_a = np.asarray(std, dtype=np.float64)
    z_a = np.asarray(z, dtype=np.float64)
    if z_a.ndim == 2 and mean_a.ndim == 1 and mean_a.shape[0] == z_a.shape[1]:
        return z_a * std_a[None, :] + mean_a[None, :]
    if z_a.ndim == 2 and mean_a.ndim == 1:
        return z_a * std_a[:, None] + mean_a[:, None]
    return z_a * std_a + mean_a
