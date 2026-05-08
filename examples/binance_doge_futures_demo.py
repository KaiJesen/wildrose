#!/usr/bin/env python3
"""
DOGEUSDT 合约 K 线示例（market_data → binance_vision / binance_futures）。

默认从 Binance Vision 历史归档拉取最近 14 天 1h K 线（直连可用、免代理），
打印关键统计 + 画蜡烛图 + 成交量。

绘图函数已迁移到 `market_data.plotting`，独立可复用：
    from market_data.plotting import plot_candlestick, plot_candlestick_volume
    plot_candlestick_volume(df, title="DOGEUSDT 1h", save_path="doge.png")

用法示例：
    # 默认：vision 历史归档 + 1h + 14 天，弹窗显示
    python examples/binance_doge_futures_demo.py

    # 切换到实时 fapi（需要能直连 fapi.binance.com，国内通常需代理）
    python examples/binance_doge_futures_demo.py --source binance_futures \\
        --proxy http://127.0.0.1:7890

    # 30 天 4h 线、保存图片：
    python examples/binance_doge_futures_demo.py --interval 4h --days 30 \\
        --save /tmp/doge_4h.png

    # 只抓数据不绘图（适合 SSH/CI）
    python examples/binance_doge_futures_demo.py --no-plot

依赖：requests + pandas（套件核心） + matplotlib（绘图）。
建议安装：pip install -e ".[plot]"
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from market_data import (
    COL_CLOSE,
    COL_HIGH,
    COL_LOW,
    COL_OPEN,
    COL_TIME,
    COL_VOLUME,
    get_kline_provider,
)
from market_data.plotting import plot_candlestick_volume
from market_data.schema import (
    COL_AMOUNT,
    COL_QUOTE_VOLUME,
    COL_TAKER_BUY_BASE,
    COL_TAKER_BUY_QUOTE,
    COL_TRADES,
)


def print_summary(df: pd.DataFrame, symbol: str, interval: str) -> None:
    """打印关键快照：区间、首末价、累计成交量/成交额、主买占比。"""
    if df.empty:
        print("No rows returned.")
        return

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)

    first, last = df.iloc[0], df.iloc[-1]
    chg = float(last[COL_CLOSE]) - float(first[COL_OPEN])
    pct = chg / float(first[COL_OPEN]) * 100 if float(first[COL_OPEN]) else float("nan")

    print(f"\n=== {symbol} {interval} | {len(df)} bars ===")
    print(f"Time range : {first[COL_TIME]}  →  {last[COL_TIME]}")
    print(f"Open/Close : {first[COL_OPEN]}  →  {last[COL_CLOSE]}  ({chg:+.6f}, {pct:+.2f}%)")
    print(f"High / Low : {df[COL_HIGH].max()} / {df[COL_LOW].min()}")
    print(f"Volume sum : {df[COL_VOLUME].sum():,.0f}")
    if COL_QUOTE_VOLUME in df.columns:
        print(f"QuoteVol   : {df[COL_QUOTE_VOLUME].sum():,.2f} USDT")
    if COL_TRADES in df.columns:
        print(f"Trades sum : {int(df[COL_TRADES].sum()):,}")
    if COL_TAKER_BUY_QUOTE in df.columns and COL_QUOTE_VOLUME in df.columns:
        taker = df[COL_TAKER_BUY_QUOTE].sum()
        total = df[COL_QUOTE_VOLUME].sum()
        ratio = (taker / total * 100) if total else float("nan")
        print(f"Taker buy %: {ratio:.2f}%  (taker_buy_quote / quote_volume)")

    cols_to_show = [
        c
        for c in (COL_TIME, COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME, COL_AMOUNT, COL_TRADES)
        if c in df.columns
    ]
    print("\nHead:")
    print(df[cols_to_show].head(5).to_string(index=False))
    print("\nTail:")
    print(df[cols_to_show].tail(5).to_string(index=False))


def _build_provider(args: argparse.Namespace):
    name = args.source.lower()
    if name in {"binance_vision", "vision", "binance_archive", "binance_history"}:
        return get_kline_provider(
            "binance_vision",
            retries=max(1, args.retries),
            proxies=({"http": args.proxy, "https": args.proxy} if args.proxy else None),
        )
    # fapi 实时
    return get_kline_provider(
        "binance_futures",
        retries=max(1, args.retries),
        base_url=args.base_url or None,
        proxies=({"http": args.proxy, "https": args.proxy} if args.proxy else None),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Binance DOGEUSDT K-line demo (vision/fapi 双源)")
    parser.add_argument("--symbol", default="DOGEUSDT", help="合约代码，如 DOGEUSDT、BTCUSDT")
    parser.add_argument(
        "--source",
        default="binance_vision",
        choices=[
            "binance_vision", "vision",
            "binance_futures", "binance", "fapi",
        ],
        help="数据源：默认 vision（公开归档免代理），实时数据用 binance_futures",
    )
    parser.add_argument(
        "--interval",
        default="1h",
        help="K 线周期，如 1m,5m,15m,30m,1h,4h,1d（也接受 60m / 240m / 1day）",
    )
    parser.add_argument("--days", type=int, default=14, help="从当前时刻往前推的自然日区间")
    parser.add_argument("--save", default="", help="保存图片路径（给定则不弹窗）")
    parser.add_argument("--retries", type=int, default=5, help="网络瞬时失败时的重试次数")
    parser.add_argument(
        "--base-url",
        default="",
        help="自定义 API base url（仅对 binance_futures 有效）",
    )
    parser.add_argument(
        "--proxy",
        default="",
        help="HTTP/HTTPS 代理地址，例如 http://127.0.0.1:7890",
    )
    parser.add_argument(
        "--color-style",
        default="crypto",
        choices=["crypto", "ashare"],
        help="蜡烛配色：crypto 绿涨红跌（默认），ashare 红涨绿跌",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="只抓数据 + 打印，不绘图（适合无 GUI 环境）",
    )
    args = parser.parse_args()

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=args.days)

    provider = _build_provider(args)
    print(
        f"Fetching [{provider.id}] symbol={args.symbol} interval={args.interval!r} "
        f"{start.isoformat()} ~ {end.isoformat()}"
    )

    df = provider.fetch_kline(args.symbol.upper(), args.interval, start, end)
    print_summary(df, args.symbol.upper(), args.interval)

    if df.empty or args.no_plot:
        return

    title = f"{args.symbol.upper()} — {args.interval} — {provider.id}"
    plot_candlestick_volume(
        df,
        title=title,
        save_path=args.save or None,
        color_style=args.color_style,
        price_label="Price (USDT)",
        volume_label=f"Volume ({args.symbol.replace('USDT', '').upper()})",
    )


if __name__ == "__main__":
    main()
