"""统一 K 线字段名，便于在不同数据源之间切换。"""

from __future__ import annotations

import pandas as pd

# 标准列（英文）
COL_TIME = "time"
COL_OPEN = "open"
COL_HIGH = "high"
COL_LOW = "low"
COL_CLOSE = "close"
COL_VOLUME = "volume"
# 常见扩展（部分数据源可能没有）
COL_AMOUNT = "amount"
COL_AMPLITUDE = "amplitude"
COL_PCT_CHANGE = "pct_change"
COL_CHANGE = "change"
COL_TURNOVER = "turnover"
# 加密货币交易所常见的扩展字段
COL_CLOSE_TIME = "close_time"
COL_QUOTE_VOLUME = "quote_volume"
COL_TRADES = "trades"
COL_TAKER_BUY_BASE = "taker_buy_base"
COL_TAKER_BUY_QUOTE = "taker_buy_quote"

CORE_COLUMNS = (COL_TIME, COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME)

# 东方财富/AkShare 中文列 → 标准列
CN_TO_STANDARD = {
    "时间": COL_TIME,
    "日期": COL_TIME,
    "开盘": COL_OPEN,
    "收盘": COL_CLOSE,
    "最高": COL_HIGH,
    "最低": COL_LOW,
    "成交量": COL_VOLUME,
    "成交额": COL_AMOUNT,
    "振幅": COL_AMPLITUDE,
    "涨跌幅": COL_PCT_CHANGE,
    "涨跌额": COL_CHANGE,
    "换手率": COL_TURNOVER,
}


def normalize_ohlcv_df(raw: pd.DataFrame) -> pd.DataFrame:
    """将带中文列名的行情 DataFrame 转为标准列名并排序。"""
    if raw.empty:
        return pd.DataFrame(columns=list(CORE_COLUMNS))
    rename = {k: v for k, v in CN_TO_STANDARD.items() if k in raw.columns}
    out = raw.rename(columns=rename)
    if COL_TIME not in out.columns:
        raise ValueError(f"缺少时间列，当前列: {list(out.columns)}")
    out[COL_TIME] = pd.to_datetime(out[COL_TIME])
    out = out.sort_values(COL_TIME).reset_index(drop=True)
    ordered = [c for c in CORE_COLUMNS if c in out.columns]
    rest = [c for c in out.columns if c not in ordered]
    return out[ordered + rest]
