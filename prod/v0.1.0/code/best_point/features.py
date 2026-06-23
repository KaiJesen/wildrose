from __future__ import annotations

import numpy as np
import pandas as pd

from market_data.schema import COL_CLOSE, COL_HIGH, COL_LOW, COL_OPEN, COL_VOLUME


def compute_causal_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df[COL_CLOSE].to_numpy(dtype=np.float64)
    high = df[COL_HIGH].to_numpy(dtype=np.float64)
    low = df[COL_LOW].to_numpy(dtype=np.float64)
    vol = df[COL_VOLUME].to_numpy(dtype=np.float64)
    open_ = df[COL_OPEN].to_numpy(dtype=np.float64)

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))

    atr14 = np.empty_like(tr)
    atr14[:14] = tr[:14].mean()
    alpha = 1.0 / 14.0
    for i in range(14, len(tr)):
        atr14[i] = alpha * tr[i] + (1.0 - alpha) * atr14[i - 1]

    log_ret = np.zeros_like(close)
    log_ret[1:] = np.log(np.clip(close[1:] / np.clip(close[:-1], 1e-12, None), 1e-12, None))

    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().to_numpy(dtype=np.float64)
    ema24 = pd.Series(close).ewm(span=24, adjust=False).mean().to_numpy(dtype=np.float64)
    ema72 = pd.Series(close).ewm(span=72, adjust=False).mean().to_numpy(dtype=np.float64)

    volz = (vol - pd.Series(vol).rolling(48, min_periods=8).mean().to_numpy()) / np.clip(
        pd.Series(vol).rolling(48, min_periods=8).std().to_numpy(), 1e-12, None
    )
    volz = np.nan_to_num(volz, nan=0.0)

    feat = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
            "log_ret": log_ret,
            "atr14": atr14,
            "atr_ratio": atr14 / np.clip(close, 1e-12, None),
            "ema12_dist_atr": (close - ema12) / np.clip(atr14, 1e-12, None),
            "ema24_dist_atr": (close - ema24) / np.clip(atr14, 1e-12, None),
            "ema72_dist_atr": (close - ema72) / np.clip(atr14, 1e-12, None),
            "vol_z": volz,
        }
    )
    return feat

