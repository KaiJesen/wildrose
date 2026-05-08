#!/usr/bin/env python3
"""
K 线示例：通过 market_data 套件切换数据源；默认 AkShare→东方财富。

依赖：pip install -e ".[all]"（在项目根 02_python 下），或把项目根加入 PYTHONPATH。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 未 pip install -e 时，允许直接 python examples/xxx.py
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from market_data import (
    get_kline_provider,
    list_kline_providers,
)
from market_data.plotting import plot_candlestick_volume


def main() -> None:
    parser = argparse.ArgumentParser(description="K-line demo（market_data 多数据源）")
    parser.add_argument("--list-sources", action="store_true", help="列出已注册数据源及周期")
    parser.add_argument("--source", default="akshare_em", help="数据源 id 或别名（如 eastmoney）")
    parser.add_argument("--symbol", default="600519", help="标的代码（含义依数据源而定）")
    parser.add_argument(
        "--interval",
        default="60m",
        help="K 线周期，如 1m,5m,15m,30m,60m,1d（取决于数据源）",
    )
    parser.add_argument("--days", type=int, default=30, help="从当前时刻往前推的自然日区间")
    parser.add_argument(
        "--adjust",
        default="",
        choices=["", "qfq", "hfq"],
        help="复权: 不复权 | 前复权 qfq | 后复权 hfq",
    )
    parser.add_argument("--save", default="", help="保存图片路径（给定则不弹窗）")
    parser.add_argument("--retries", type=int, default=5, help="网络瞬时失败时的重试次数")
    args = parser.parse_args()

    if args.list_sources:
        for pid, desc, intervals in list_kline_providers():
            iv = ", ".join(sorted(intervals)) if intervals else "—"
            print(f"{pid}\t{iv}\t{desc}")
        return

    end = datetime.now()
    start = end - timedelta(days=args.days)

    provider = get_kline_provider(args.source, retries=max(1, args.retries))
    print(
        f"Fetching [{provider.id}] interval={args.interval!r} symbol={args.symbol} "
        f"{start} ~ {end} adjust={args.adjust!r}"
    )
    df = provider.fetch_kline(
        args.symbol,
        args.interval,
        start,
        end,
        adjust=args.adjust,
    )

    if df.empty:
        print("No rows returned. Try shorter --days, another interval/symbol, or trading hours.")
        return

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print("\nHead:")
    print(df.head(8))
    print("\nTail:")
    print(df.tail(8))
    print(f"\nRows: {len(df)}")

    title = f"{args.symbol} — {args.interval} — {provider.id}"
    plot_candlestick_volume(
        df,
        title=title,
        save_path=args.save or None,
        color_style="ashare",  # 红涨绿跌
        price_label="Price (OHLC)",
    )


if __name__ == "__main__":
    main()
