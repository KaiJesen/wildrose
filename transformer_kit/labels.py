"""Market state label builders and threshold utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class MarketStateTargets:
    """Unified multi-task targets for market-state learning."""

    future_log_ret: torch.Tensor
    direction_label: torch.Tensor
    volatility: torch.Tensor
    risk_label: torch.Tensor
    move_label: torch.Tensor | None = None


@dataclass(frozen=True)
class MarketStateThresholds:
    direction_threshold: float
    risk_vol_threshold: float


def _direction_3class(future_raw_log_ret: np.ndarray, *, threshold: float) -> np.ndarray:
    label = np.full_like(future_raw_log_ret, 1, dtype=np.int64)  # flat
    label[future_raw_log_ret > threshold] = 2  # up
    label[future_raw_log_ret < -threshold] = 0  # down
    return label


def build_market_state_targets(
    future_raw_log_ret: np.ndarray,
    *,
    direction_threshold: float,
    risk_vol_threshold: float,
) -> MarketStateTargets:
    """Build market-state targets from future raw log-return window.

    Args:
        future_raw_log_ret: shape [H]
    """
    if future_raw_log_ret.ndim != 1:
        raise ValueError(f"future_raw_log_ret must be [H], got {future_raw_log_ret.shape}")
    h = future_raw_log_ret.shape[0]
    direction = _direction_3class(future_raw_log_ret, threshold=direction_threshold)
    volatility = np.sqrt(np.cumsum(np.square(future_raw_log_ret)) / np.arange(1, h + 1, dtype=np.float32))
    risk = (volatility >= risk_vol_threshold).astype(np.float32)
    move = (np.abs(future_raw_log_ret) > direction_threshold).astype(np.float32)
    return MarketStateTargets(
        future_log_ret=torch.from_numpy(future_raw_log_ret.astype(np.float32)),
        direction_label=torch.from_numpy(direction.astype(np.int64)),
        volatility=torch.from_numpy(volatility.astype(np.float32)),
        risk_label=torch.from_numpy(risk),
        move_label=torch.from_numpy(move),
    )


def estimate_market_state_thresholds(
    train_future_windows: np.ndarray,
    *,
    direction_quantile: float = 0.35,
    risk_quantile: float = 0.8,
    min_direction_threshold: float = 5e-5,
) -> MarketStateThresholds:
    """Estimate thresholds from train split only.

    Args:
        train_future_windows: [N, H] raw future log_ret windows from train split.
    """
    if train_future_windows.ndim != 2:
        raise ValueError(f"train_future_windows must be [N,H], got {train_future_windows.shape}")
    abs_step = np.abs(train_future_windows).reshape(-1)
    direction_threshold = float(np.quantile(abs_step, direction_quantile))
    direction_threshold = max(direction_threshold, min_direction_threshold)
    rv = np.sqrt(np.mean(np.square(train_future_windows), axis=1))
    risk_vol_threshold = float(np.quantile(rv, risk_quantile))
    return MarketStateThresholds(
        direction_threshold=direction_threshold,
        risk_vol_threshold=risk_vol_threshold,
    )

