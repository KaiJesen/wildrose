"""抓取真实 DOGE 永续合约 K 线 → 跑全局最优多空点标注 → 出带标注蜡烛图 + 报告。

报告写到 ``trade/report/<流水号>_<标题>/``，含：
    chart.png        带开/平仓标注的蜡烛图
    trades.csv       每笔交易（方向 / 进出场下标 / 价格 / 时间 / 净收益）
    ohlcv.csv        本次使用的原始 K 线
    report.md        参数、统计与结论

数据源：Binance Vision 历史归档（真实 OHLCV，符合“实验只用真实数据”规则）。

用法::
    python trade/tools/run_doge_report.py
    python trade/tools/run_doge_report.py --symbol DOGEUSDT --interval 1h \
        --start 2025-05-01 --end 2025-05-10 --title "DOGE永续合约全局最优多空点标注"
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import sys

import pandas as pd

# 让脚本能直接 import 仓库内的 market_data 与同目录算法模块
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOLS_DIR = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT), str(_TOOLS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from market_data import COL_CLOSE, COL_HIGH, COL_LOW, COL_TIME  # noqa: E402
from market_data.plotting import plot_candlestick  # noqa: E402
from market_data.sources.binance_vision import BinanceVisionKlineProvider  # noqa: E402

from optimal_trade_points import (  # noqa: E402
    find_optimal_trades,
    min_price_move,
    summarize_trades,
    trades_to_dataframe,
)

_REPORT_ROOT = _REPO_ROOT / "trade" / "report"


def _next_serial(report_root: Path) -> str:
    """扫描已有 ``NNNN_*`` 文件夹，返回下一个 4 位流水号。"""
    report_root.mkdir(parents=True, exist_ok=True)
    used = []
    for child in report_root.iterdir():
        if child.is_dir():
            m = re.match(r"^(\d{4})_", child.name)
            if m:
                used.append(int(m.group(1)))
    return f"{(max(used) + 1) if used else 1:04d}"


def _slug(title: str) -> str:
    """把标题清洗成可作目录名的片段（保留中英文与数字）。"""
    s = re.sub(r"[\s/\\:*?\"<>|]+", "_", title.strip())
    return s.strip("_") or "report"


def fetch_doge(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    provider = BinanceVisionKlineProvider(contract_type="um", verbose=True)
    df = provider.fetch_kline(symbol, interval, start, end)
    if df.empty:
        raise RuntimeError(
            f"未取到任何 K 线：symbol={symbol} interval={interval} "
            f"{start:%Y-%m-%d}~{end:%Y-%m-%d}（Vision daily 通常延迟 1~2 天，换更早区间试试）"
        )
    return df.reset_index(drop=True)


def plot_with_trades(
    df: pd.DataFrame,
    trades,
    *,
    title: str,
    save_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(16, 8))
    plot_candlestick(df, ax=ax, color_style="crypto", title=title, ylabel="Price (USDT)")

    times_num = mdates.date2num(df[COL_TIME].to_numpy())
    span = float(df[COL_HIGH].max() - df[COL_LOW].min()) or 1.0
    pad = span * 0.018

    long_color = "#1565c0"   # 做多：蓝
    short_color = "#f9a825"  # 做空：橙

    # 交易太多时关闭逐笔文字标注，避免糊成一团；只保留标记 + 连线
    show_labels = len(trades) <= 30
    marker_size = 130 if len(trades) <= 30 else 60

    for tr in trades:
        c = long_color if tr.direction == "long" else short_color
        x_in, x_out = times_num[tr.entry_index], times_num[tr.exit_index]
        y_in, y_out = tr.entry_price, tr.exit_price

        # 进出场连线
        ax.plot([x_in, x_out], [y_in, y_out], color=c, ls="--", lw=1.1, alpha=0.85, zorder=3)

        # 开仓标记：多 ▲ 在低点下方，空 ▼ 在高点上方
        if tr.direction == "long":
            ax.scatter(x_in, y_in - pad, marker="^", s=marker_size, color=c,
                       edgecolors="white", linewidths=0.5, zorder=5)
        else:
            ax.scatter(x_in, y_in + pad, marker="v", s=marker_size, color=c,
                       edgecolors="white", linewidths=0.5, zorder=5)

        # 平仓标记：黑色 X
        ax.scatter(x_out, y_out, marker="x", s=marker_size * 0.6, color="black",
                   linewidths=1.0, zorder=4)

        if show_labels:
            mid_x = (x_in + x_out) / 2.0
            mid_y = max(y_in, y_out) + pad
            ax.annotate(
                f"{tr.direction[0].upper()} +{tr.net_roi * 100:.2f}%",
                xy=(mid_x, mid_y),
                ha="center", va="bottom", fontsize=8, color=c, zorder=6,
            )

    legend_handles = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor=long_color,
               markeredgecolor="black", markersize=11, label="Long entry"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor=short_color,
               markeredgecolor="black", markersize=11, label="Short entry"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="black",
               markersize=11, label="Exit"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_report_md(
    path: Path,
    *,
    title: str,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    df: pd.DataFrame,
    trades,
    params: dict,
) -> None:
    stats = summarize_trades(trades, leverage=params["leverage"])
    move = min_price_move(
        fee_rate=params["fee_rate"],
        leverage=params["leverage"],
        min_net_roi=params["min_net_roi"],
    )
    tdf = trades_to_dataframe(trades)

    lines = [
        f"# {title}",
        "",
        "> 事后视角的全局最优多空点标注（使用未来 K 线，仅用于研究 / 标签 / 复盘，**非实时信号**）。",
        "",
        "## 数据",
        "",
        f"- 数据源：Binance Vision 永续合约归档（真实 OHLCV）",
        f"- 标的 / 周期：`{symbol}` / `{interval}`",
        f"- 区间：`{start:%Y-%m-%d}` ~ `{end:%Y-%m-%d}`（UTC）",
        f"- K 线根数：{len(df)}",
        f"- 价格范围：{df[COL_LOW].min():.6f} ~ {df[COL_HIGH].max():.6f} USDT",
        "",
        "## 收益模型与参数",
        "",
        f"- 单边手续费 `fee_rate` = {params['fee_rate']}（万分之一）",
        f"- 杠杆 `leverage` = {params['leverage']}x",
        f"- 最低净收益门槛 `min_net_roi` = {params['min_net_roi'] * 100:.2f}%",
        f"- 入场/出场用价：`{params['price_field']}`",
        "",
        "净收益率公式（计杠杆、扣双边手续费）：",
        "",
        "```",
        "long :  net = leverage * ((p_exit / p_entry - 1) - 2 * fee_rate)",
        "short:  net = leverage * ((p_entry / p_exit - 1) - 2 * fee_rate)",
        "```",
        "",
        f"达到 {params['min_net_roi'] * 100:.2f}% 净收益所需的最小原始价格变动 ≈ **{move * 100:.4f}%**。",
        "",
        "## 结果总览",
        "",
        f"- 交易笔数：**{stats['num_trades']}**（多 {stats['num_long']} / 空 {stats['num_short']}）",
        f"- 累计净收益率（各笔相加）：**{stats['total_net_roi'] * 100:.2f}%**",
        f"- 单笔平均净收益率：{stats['avg_net_roi'] * 100:.2f}%",
        f"- 单笔最大净收益率：{stats['max_net_roi'] * 100:.2f}%",
        f"- 总持仓 K 线数：{stats['total_holding_bars']}",
        "",
        "## 标注图",
        "",
        "![chart](./chart.png)",
        "",
        "## 交易明细",
        "",
    ]

    if tdf.empty:
        lines.append("（本区间内无满足门槛的交易）")
    else:
        lines.append("| # | 方向 | 开仓时间 | 开仓价 | 平仓时间 | 平仓价 | 原始波动 | 净收益率 | 持仓K线 |")
        lines.append("|---|------|----------|--------|----------|--------|----------|----------|---------|")
        for k, tr in enumerate(trades, 1):
            et = pd.Timestamp(tr.entry_time).strftime("%Y-%m-%d %H:%M")
            xt = pd.Timestamp(tr.exit_time).strftime("%Y-%m-%d %H:%M")
            dir_cn = "做多" if tr.direction == "long" else "做空"
            lines.append(
                f"| {k} | {dir_cn} | {et} | {tr.entry_price:.6f} | {xt} | "
                f"{tr.exit_price:.6f} | {tr.raw_return * 100:.3f}% | "
                f"{tr.net_roi * 100:.2f}% | {tr.holding_bars} |"
            )

    lines += [
        "",
        "## 说明",
        "",
        "- 交易序列由动态规划在“互不重叠交易”空间内求总净收益最大，是该收益模型下的全局最优解。",
        "- 每笔交易均已满足最低净收益门槛；不达标的波段不会被标注。",
        "- 入场/出场默认取收盘价，避免单根 K 线内部的未来函数。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="DOGE 全局最优多空点标注报告")
    ap.add_argument("--symbol", default="DOGEUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--start", default="2025-05-01")
    ap.add_argument("--end", default="2025-05-10")
    ap.add_argument("--fee-rate", type=float, default=0.0001)
    ap.add_argument("--leverage", type=float, default=20.0)
    ap.add_argument("--min-net-roi", type=float, default=0.002)
    ap.add_argument("--price-field", default=COL_CLOSE)
    ap.add_argument("--max-holding-bars", type=int, default=None)
    ap.add_argument("--title", default="DOGE永续合约全局最优多空点标注")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    print(f"[1/4] 抓取真实 K 线 {args.symbol} {args.interval} {args.start}~{args.end} ...")
    df = fetch_doge(args.symbol, args.interval, start, end)
    print(f"      得到 {len(df)} 根 K 线")

    params = {
        "fee_rate": args.fee_rate,
        "leverage": args.leverage,
        "min_net_roi": args.min_net_roi,
        "price_field": args.price_field,
    }

    print("[2/4] 求全局最优多空点 ...")
    trades = find_optimal_trades(
        df,
        fee_rate=args.fee_rate,
        leverage=args.leverage,
        min_net_roi=args.min_net_roi,
        price_field=args.price_field,
        max_holding_bars=args.max_holding_bars,
    )
    stats = summarize_trades(trades, leverage=args.leverage)
    print(f"      交易 {stats['num_trades']} 笔，累计净收益 {stats['total_net_roi'] * 100:.2f}%")

    serial = _next_serial(_REPORT_ROOT)
    out_dir = _REPORT_ROOT / f"{serial}_{_slug(args.title)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[3/4] 写报告到 {out_dir} ...")

    df.to_csv(out_dir / "ohlcv.csv", index=False)
    trades_to_dataframe(trades).to_csv(out_dir / "trades.csv", index=False)

    chart_title = f"{args.symbol} {args.interval}  global optimal long/short points"
    plot_with_trades(df, trades, title=chart_title, save_path=out_dir / "chart.png")

    write_report_md(
        out_dir / "report.md",
        title=args.title,
        symbol=args.symbol,
        interval=args.interval,
        start=start,
        end=end,
        df=df,
        trades=trades,
        params=params,
    )

    print(f"[4/4] 完成。报告目录：{out_dir}")
    print(f"      图：{out_dir / 'chart.png'}")
    print(f"      明细：{out_dir / 'trades.csv'}")
    print(f"      报告：{out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
