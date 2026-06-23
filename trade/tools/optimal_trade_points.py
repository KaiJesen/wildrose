"""全局最优多空点标注器（事后标注，使用未来 K 线信息）。

定位
----
给定一段**完整**的 OHLCV K 线，在“已知全部未来”的上帝视角下标注多空交易。
**不可用于实盘实时信号**——它依赖未来数据。

两种模式（``mode``）：

``dp``（默认，兼容旧行为）
    动态规划求 **总净收益最大** 的互不重叠交易序列。
    容易在明显趋势段内频繁反手，图上出现大段涨跌却未持仓的空白。

``major_legs``（推荐用于图表 / 主波段标签）
    用 hindsight ZigZag 识别主要波段高低点，每段主涨/主跌腿生成一笔交易，
    覆盖整段趋势，避免趋势内碎片化空仓。

收益模型
--------
单边手续费 ``fee_rate``，开平共两次，杠杆 ``leverage``：

    做多净收益率  net = leverage * ((p_exit / p_entry - 1) - 2 * fee_rate)
    做空净收益率  net = leverage * ((p_entry / p_exit - 1) - 2 * fee_rate)

只有 ``net >= min_net_roi`` 的交易才会被采纳。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

COL_TIME = "time"
COL_CLOSE = "close"
COL_HIGH = "high"
COL_LOW = "low"

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


def _atr_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Wilder ATR（因果滚动，首段用 expanding）。"""
    n = len(close)
    tr = np.empty(n, dtype=float)
    tr[0] = float(high[0] - low[0])
    for i in range(1, n):
        tr[i] = max(
            float(high[i] - low[i]),
            abs(float(high[i] - close[i - 1])),
            abs(float(low[i] - close[i - 1])),
        )
    atr = np.empty(n, dtype=float)
    if n == 0:
        return atr
    atr[0] = tr[0]
    alpha = 1.0 / float(period)
    for i in range(1, n):
        atr[i] = (1.0 - alpha) * atr[i - 1] + alpha * tr[i]
    return np.maximum(atr, 1e-12)


def find_zigzag_pivots(
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    *,
    min_move_atr: float = 1.2,
) -> list[tuple[int, str, float]]:
    """Hindsight ZigZag 拐点：交替 ``L``（低点）/ ``H``（高点）。

    反转幅度须达到 ``min_move_atr * ATR`` 才确认拐点，用于过滤小波动。
    """
    n = len(high)
    if n == 0:
        return []
    if n == 1:
        return [(0, "L", float(low[0]))]

    # trend: 0 未定，1 自低点找高点，-1 自高点找低点
    trend = 0
    pivots: list[tuple[int, str, float]] = []
    ext_i = 0
    ext_px = float(low[0])

    for i in range(1, n):
        thresh = float(min_move_atr * atr[i])
        hi = float(high[i])
        lo = float(low[i])

        if trend <= 0:
            if trend == 0 or lo < ext_px:
                ext_px, ext_i = lo, i
            if hi - ext_px >= thresh:
                pivots.append((ext_i, "L", ext_px))
                trend = 1
                ext_px, ext_i = hi, i
        else:
            if hi > ext_px:
                ext_px, ext_i = hi, i
            if ext_px - lo >= thresh:
                pivots.append((ext_i, "H", ext_px))
                trend = -1
                ext_px, ext_i = lo, i

    if not pivots:
        return [(0, "L", float(low[0])), (n - 1, "H", float(high[-1]))]

    if pivots[-1][1] == "L":
        tail_i = int(np.argmax(high[pivots[-1][0] :])) + pivots[-1][0]
        pivots.append((tail_i, "H", float(high[tail_i])))
    else:
        tail_i = int(np.argmin(low[pivots[-1][0] :])) + pivots[-1][0]
        pivots.append((tail_i, "L", float(low[tail_i])))

    if pivots[0][1] != "L":
        start_i = int(np.argmin(low[: pivots[0][0] + 1]))
        pivots.insert(0, (start_i, "L", float(low[start_i])))

    deduped: list[tuple[int, str, float]] = [pivots[0]]
    for idx, kind, px in pivots[1:]:
        if kind == deduped[-1][1]:
            if kind == "H" and (px > deduped[-1][2] or idx > deduped[-1][0]):
                deduped[-1] = (idx, kind, px)
            elif kind == "L" and (px < deduped[-1][2] or idx > deduped[-1][0]):
                deduped[-1] = (idx, kind, px)
        elif idx > deduped[-1][0]:
            deduped.append((idx, kind, px))
    return deduped


def _merge_zigzag_pivots(
    pivots: list[tuple[int, str, float]],
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    *,
    merge_pullback_atr: float,
) -> list[tuple[int, str, float]]:
    """合并趋势内浅回撤：L-H-L 中若 H 相对两侧抬升不足，则合并为 L…L。"""
    if len(pivots) < 3 or merge_pullback_atr <= 0:
        return pivots

    merged = [pivots[0]]
    i = 1
    while i < len(pivots):
        if (
            len(merged) >= 1
            and i + 1 < len(pivots)
            and merged[-1][1] == "L"
            and pivots[i][1] == "H"
            and pivots[i + 1][1] == "L"
        ):
            l0 = merged[-1]
            h1 = pivots[i]
            l2 = pivots[i + 1]
            atr_ref = float(atr[h1[0]])
            pullback = h1[2] - l2[2]
            if pullback < merge_pullback_atr * atr_ref and l2[2] < l0[2]:
                # 下跌腿内浅反弹：合并 L0 -> L2
                merged[-1] = l2
                i += 2
                continue
        if (
            len(merged) >= 1
            and i + 1 < len(pivots)
            and merged[-1][1] == "H"
            and pivots[i][1] == "L"
            and pivots[i + 1][1] == "H"
        ):
            h0 = merged[-1]
            l1 = pivots[i]
            h2 = pivots[i + 1]
            atr_ref = float(atr[l1[0]])
            pullback = h2[2] - l1[2]
            if pullback < merge_pullback_atr * atr_ref and h2[2] > h0[2]:
                merged[-1] = h2
                i += 2
                continue
        merged.append(pivots[i])
        i += 1
    return merged


def find_major_leg_trades(
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
    zigzag_min_move_atr: float = 1.8,
    zigzag_atr_period: int = 14,
    merge_pullback_atr: float = 2.0,
    min_leg_bars: int = 2,
) -> list[Trade]:
    """主波段标注：ZigZag 拐点间整段做多/做空，覆盖主要涨跌腿。"""
    if not allow_long and not allow_short:
        raise ValueError("allow_long 与 allow_short 不能同时为 False")
    if price_field not in df.columns:
        raise ValueError(f"缺少价格列 {price_field!r}；现有列：{list(df.columns)}")

    high_col = COL_HIGH if COL_HIGH in df.columns else price_field
    low_col = COL_LOW if COL_LOW in df.columns else price_field
    high = df[high_col].to_numpy(dtype=float)
    low = df[low_col].to_numpy(dtype=float)
    close = df[price_field].to_numpy(dtype=float)
    n = len(close)
    if n < 2:
        return []

    times = df[time_field].to_numpy() if time_field in df.columns else np.arange(n)
    atr = _atr_series(high, low, close, period=zigzag_atr_period)

    pivots = find_zigzag_pivots(high, low, atr, min_move_atr=zigzag_min_move_atr)
    pivots = _merge_zigzag_pivots(
        pivots, high, low, atr, merge_pullback_atr=merge_pullback_atr
    )

    trades: list[Trade] = []
    for k in range(len(pivots) - 1):
        i0, k0, _ = pivots[k]
        i1, k1, _ = pivots[k + 1]
        if i1 <= i0:
            continue
        holding = i1 - i0
        if holding < min_leg_bars:
            continue
        if max_holding_bars is not None and holding > max_holding_bars:
            continue

        if k0 == "L" and k1 == "H":
            direction = "long"
        elif k0 == "H" and k1 == "L":
            direction = "short"
        else:
            continue

        if direction == "long" and not allow_long:
            continue
        if direction == "short" and not allow_short:
            continue

        entry_price = float(close[i0])
        exit_price = float(close[i1])
        net = trade_net_roi(
            direction, entry_price, exit_price, fee_rate=fee_rate, leverage=leverage
        )
        if net < min_net_roi:
            continue

        if direction == "long":
            raw_return = exit_price / entry_price - 1.0
        else:
            raw_return = entry_price / exit_price - 1.0

        trades.append(
            Trade(
                entry_index=i0,
                exit_index=i1,
                direction=direction,
                entry_time=times[i0],
                exit_time=times[i1],
                entry_price=entry_price,
                exit_price=exit_price,
                raw_return=raw_return,
                net_roi=net,
                holding_bars=holding,
            )
        )
    return trades


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
    mode: str = "dp",
    zigzag_min_move_atr: float = 1.8,
    zigzag_atr_period: int = 14,
    merge_pullback_atr: float = 2.0,
    min_leg_bars: int = 2,
) -> list[Trade]:
    """标注最优/主波段交易序列。

    :param df: 标准 OHLCV DataFrame（至少含 ``price_field``；``major_legs`` 需 high/low）
    :param fee_rate: 单边手续费率（万分之一 = 0.0001）
    :param leverage: 杠杆倍数
    :param min_net_roi: 每笔交易的最低净收益率门槛
    :param price_field: 用作进出场价的列（默认收盘价，避免单根 K 线内的未来函数）
    :param allow_long / allow_short: 是否允许做多 / 做空
    :param max_holding_bars: 单笔最长持仓 K 线数；None 表示不限制
    :param mode: ``dp`` 动态规划最大总收益；``major_legs`` 主波段 ZigZag（推荐看图）
    :param zigzag_min_move_atr: ``major_legs`` 反转确认幅度（ATR 倍数）
    :param zigzag_atr_period: ATR 周期
    :param merge_pullback_atr: 趋势内浅回撤合并阈值（ATR 倍数）
    :param min_leg_bars: 最短波段 K 线数
    :return: 按时间排序的 Trade 列表
    """
    if mode == "major_legs":
        return find_major_leg_trades(
            df,
            fee_rate=fee_rate,
            leverage=leverage,
            min_net_roi=min_net_roi,
            price_field=price_field,
            time_field=time_field,
            allow_long=allow_long,
            allow_short=allow_short,
            max_holding_bars=max_holding_bars,
            zigzag_min_move_atr=zigzag_min_move_atr,
            zigzag_atr_period=zigzag_atr_period,
            merge_pullback_atr=merge_pullback_atr,
            min_leg_bars=min_leg_bars,
        )
    if mode != "dp":
        raise ValueError(f"mode 仅支持 'dp' 或 'major_legs'，收到 {mode!r}")

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
