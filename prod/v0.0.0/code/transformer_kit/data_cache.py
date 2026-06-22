"""K 线 CSV 本地缓存：下载一次，后续训练直接读取。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from market_data.schema import COL_TIME, normalize_ohlcv_df

DEFAULT_CACHE_DIR = "data/cache/kline"


def kline_cache_filename(
    *,
    source: str,
    symbol: str,
    interval: str,
    days: int,
    end: datetime | None = None,
) -> str:
    """根据数据源与参数生成稳定的缓存文件名。"""
    end_tag = (end or datetime.now()).strftime("%Y%m%d")
    safe_symbol = symbol.replace("/", "-").replace("\\", "-")
    return f"{source}_{safe_symbol}_{interval}_{days}d_end{end_tag}.csv"


def resolve_kline_csv_path(
    *,
    source: str,
    symbol: str,
    interval: str,
    days: int,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    csv_path: str | Path | None = None,
    end: datetime | None = None,
) -> Path:
    if csv_path is not None:
        return Path(csv_path)
    return Path(cache_dir) / kline_cache_filename(
        source=source,
        symbol=symbol,
        interval=interval,
        days=days,
        end=end,
    )


def load_kline_csv(path: str | Path) -> pd.DataFrame:
    """从 CSV 读取并规范化为标准 OHLCV 列。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"cache csv not found: {path}")
    df = pd.read_csv(path)
    if COL_TIME in df.columns:
        df[COL_TIME] = pd.to_datetime(df[COL_TIME])
    return normalize_ohlcv_df(df)


def save_kline_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """将 K 线 DataFrame 写入 CSV（自动创建父目录）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if COL_TIME in out.columns:
        out[COL_TIME] = pd.to_datetime(out[COL_TIME])
    out.to_csv(path, index=False)
    return path
