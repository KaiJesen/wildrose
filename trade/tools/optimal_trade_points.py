"""全局最优多空点标注器（事后标注，使用未来 K 线信息）。

定位
----
给定一段**完整**的 OHLCV K 线，在“已知全部未来”的上帝视角下，找出一组
**互不重叠**的做多/做空交易，使总净收益最大；并且每一笔交易都必须满足
最低净收益门槛（扣除双边手续费、计入杠杆后）。

适用场景：策略研究 / 模型训练标签 / 复盘分析。
**不可用于实盘实时信号**——它依赖未来数据。

收益模型
--------
单边手续费 ``fee_rate``，开平共两次，杠杆 ``leverage``：

    做多净收益率  net = leverage * ((p_exit / p_entry - 1) - 2 * fee_rate)
    做空净收益率  net = leverage * ((p_entry / p_exit - 1) - 2 * fee_rate)

只有 ``net >= min_net_roi`` 的交易才会被采纳。

最优性
------
用动态规划在所有“互不重叠交易序列”上求 **总净收益最大**，
因此结果是该收益模型下的全局最优解（O(n^2)，n 为 K 线根数）。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

COL_TIME = "time"
COL_CLOSE = "close"

_NEG = -1.0e18


@dataclass
class Trade:
    """一笔事后最优交易。"""

    entry_index: int
    exit_index: int
    direction: str  # "long" | "short"
    entry_time: object
    exit_time: object
    entry_price: float
    exit_price: float
    raw_return: float   # 未加杠杆、未扣费的价格变动幅度
    net_roi: float      # 扣双边手续费、计杠杆后的净收益率
    holding_bars: int


def trade_net_roi(
    direction: str,
    entry_price: float,
    exit_price: float,
    *,
    fee_rate: float = 0.0001,
    leverage: float = 20.0,
) -> float:
    """单笔交易净收益率（计杠杆、扣双边手续费）。"""
    if direction == "long":
        gross = exit_price / entry_price - 1.0
    elif direction == "short":
        gross = entry_price / exit_price - 1.0
    else:
        raise ValueError(f"direction 仅支持 long/short，收到 {direction!r}")
    return leverage * (gross - 2.0 * fee_rate)


def min_price_move(
    *,
    fee_rate: float = 0.0001,
    leverage: float = 20.0,
    min_net_roi: float = 0.002,
) -> float:
    """达到 min_net_roi 所需的最小原始价格变动幅度。"""
    return min_net_roi / leverage + 2.0 * fee_rate


def find_optimal_trades(
    df: pd.DataFrame,
    *,
    fee_rate: float = 0.0001,
    leverage: float = 20.0,
    min_net_roi: float = 0.002,
    price_field: str = COL_CLOSE,
    time_field: str = COL_TIME,
    allow_long: bool = True,
    allow_short: bool = True,
    max_holding_bars: int | None = None,
) -> list[Trade]:
    """求全局最优、互不重叠的交易序列。

    :param df: 标准 OHLCV DataFrame（至少含 ``price_field``；有 ``time_field`` 更好）
    :param fee_rate: 单边手续费率（万分之一 = 0.0001）
    :param leverage: 杠杆倍数
    :param min_net_roi: 每笔交易的最低净收益率门槛
    :param price_field: 用作进出场价的列（默认收盘价，避免单根 K 线内的未来函数）
    :param allow_long / allow_short: 是否允许做多 / 做空
    :param max_holding_bars: 单笔最长持仓 K 线数；None 表示不限制
    :return: 按时间排序的 Trade 列表
    """
    if not allow_long and not allow_short:
        raise ValueError("allow_long 与 allow_short 不能同时为 False")
    if price_field not in df.columns:
        raise ValueError(f"缺少价格列 {price_field!r}；现有列：{list(df.columns)}")

    prices = df[price_field].to_numpy(dtype=float)
    n = len(prices)
    if n < 2:
        return []

    times = (
        df[time_field].to_numpy()
        if time_field in df.columns
        else np.arange(n)
    )

    fee_round = 2.0 * fee_rate

    # dp[i]: 从第 i 根 K 线起、当前空仓，能取得的最大累计净收益
    dp = np.zeros(n, dtype=float)
    # 重建用：choice_exit[i] = 选择的平仓下标（-1 表示在 i 处不开仓、顺延到 i+1）
    choice_exit = np.full(n, -1, dtype=np.int64)
    choice_dir = np.empty(n, dtype=object)

    for i in range(n - 2, -1, -1):
        pi = prices[i]
        # 候选平仓窗口 j ∈ [i+1, j_max]
        j_hi = n - 1 if max_holding_bars is None else min(n - 1, i + max_holding_bars)
        j_idx = np.arange(i + 1, j_hi + 1)
        pj = prices[j_idx]
        future_dp = dp[j_idx]

        roi_long = leverage * ((pj / pi - 1.0) - fee_round) if allow_long else None
        roi_short = leverage * ((pi / pj - 1.0) - fee_round) if allow_short else None

        # 先比同一 j 下多 / 空哪个更优，再把不达门槛的置为 -inf
        if roi_long is not None and roi_short is not None:
            roi_best = np.where(roi_long >= roi_short, roi_long, roi_short)
            dir_best = np.where(roi_long >= roi_short, "long", "short")
        elif roi_long is not None:
            roi_best, dir_best = roi_long, np.full(j_idx.shape, "long", dtype=object)
        else:
            roi_best, dir_best = roi_short, np.full(j_idx.shape, "short", dtype=object)

        valid = roi_best >= min_net_roi
        totals = np.where(valid, roi_best + future_dp, _NEG)

        best_skip = dp[i + 1]
        if totals.size > 0:
            k = int(np.argmax(totals))
            if totals[k] > best_skip:
                dp[i] = totals[k]
                choice_exit[i] = int(j_idx[k])
                choice_dir[i] = str(dir_best[k])
                continue
        dp[i] = best_skip  # 在 i 处不开仓

    # 重建交易序列
    trades: list[Trade] = []
    i = 0
    while i < n - 1:
        j = int(choice_exit[i])
        if j < 0:
            i += 1
            continue
        direction = str(choice_dir[i])
        entry_price = float(prices[i])
        exit_price = float(prices[j])
        if direction == "long":
            raw_return = exit_price / entry_price - 1.0
        else:
            raw_return = entry_price / exit_price - 1.0
        net = trade_net_roi(
            direction, entry_price, exit_price, fee_rate=fee_rate, leverage=leverage
        )
        trades.append(
            Trade(
                entry_index=i,
                exit_index=j,
                direction=direction,
                entry_time=times[i],
                exit_time=times[j],
                entry_price=entry_price,
                exit_price=exit_price,
                raw_return=raw_return,
                net_roi=net,
                holding_bars=j - i,
            )
        )
        i = j  # 允许在平仓那根 K 线立即反手开下一笔
    return trades


def trades_to_dataframe(trades: list[Trade]) -> pd.DataFrame:
    """把 Trade 列表转成 DataFrame，便于落盘 / 展示。"""
    if not trades:
        return pd.DataFrame(
            columns=[f.name for f in Trade.__dataclass_fields__.values()]
        )
    return pd.DataFrame([asdict(t) for t in trades])


def summarize_trades(
    trades: list[Trade],
    *,
    leverage: float = 20.0,
) -> dict:
    """汇总：交易数、多空分布、累计净收益、胜率等。"""
    df = trades_to_dataframe(trades)
    if df.empty:
        return {
            "num_trades": 0,
            "num_long": 0,
            "num_short": 0,
            "total_net_roi": 0.0,
            "avg_net_roi": 0.0,
            "max_net_roi": 0.0,
            "total_holding_bars": 0,
        }
    return {
        "num_trades": int(len(df)),
        "num_long": int((df["direction"] == "long").sum()),
        "num_short": int((df["direction"] == "short").sum()),
        "total_net_roi": float(df["net_roi"].sum()),
        "avg_net_roi": float(df["net_roi"].mean()),
        "max_net_roi": float(df["net_roi"].max()),
        "total_holding_bars": int(df["holding_bars"].sum()),
    }
