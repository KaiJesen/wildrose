#!/usr/bin/env python3
"""对比 dp 与 major_legs 标注效果，输出并排蜡烛图。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_REPO = Path(__file__).resolve().parents[2]
_TOOLS = Path(__file__).resolve().parent
for _p in (str(_REPO), str(_TOOLS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from market_data import COL_CLOSE, COL_HIGH, COL_LOW, COL_TIME
from market_data.plotting import plot_candlestick
from market_data.sources.binance_vision import BinanceVisionKlineProvider
from optimal_trade_points import find_optimal_trades, summarize_trades


def _overlay_trades(ax, df, trades, *, show_labels: bool = False) -> None:
    times_num = mdates.date2num(df[COL_TIME].to_numpy())
    span = float(df[COL_HIGH].max() - df[COL_LOW].min()) or 1.0
    pad = span * 0.015
    long_c, short_c = "#1565c0", "#f9a825"
    marker_size = 90 if len(trades) <= 40 else 50

    for tr in trades:
        c = long_c if tr.direction == "long" else short_c
        x_in, x_out = times_num[tr.entry_index], times_num[tr.exit_index]
        y_in, y_out = tr.entry_price, tr.exit_price
        ax.axvspan(x_in, x_out, color=c, alpha=0.10, zorder=1)
        ax.plot([x_in, x_out], [y_in, y_out], color=c, ls="--", lw=1.2, alpha=0.85, zorder=3)
        if tr.direction == "long":
            ax.scatter(x_in, y_in - pad, marker="^", s=marker_size, color=c, edgecolors="white", linewidths=0.4, zorder=5)
        else:
            ax.scatter(x_in, y_in + pad, marker="v", s=marker_size, color=c, edgecolors="white", linewidths=0.4, zorder=5)
        ax.scatter(x_out, y_out, marker="x", s=marker_size * 0.55, color="black", linewidths=0.9, zorder=4)
        if show_labels and len(trades) <= 20:
            ax.annotate(
                f"{tr.direction[0].upper()}+{tr.net_roi*100:.1f}%",
                xy=((x_in + x_out) / 2, max(y_in, y_out) + pad),
                ha="center", va="bottom", fontsize=7, color=c,
            )

    ax.legend(
        handles=[
            Line2D([0], [0], marker="^", color="w", markerfacecolor=long_c, markersize=9, label="Long"),
            Line2D([0], [0], marker="v", color="w", markerfacecolor=short_c, markersize=9, label="Short"),
            Line2D([0], [0], color=long_c, lw=6, alpha=0.25, label="Holding zone"),
        ],
        loc="upper left",
        fontsize=8,
        framealpha=0.9,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--start", default="2025-06-01")
    ap.add_argument("--end", default="2025-06-15")
    ap.add_argument("--min-net-roi", type=float, default=0.002)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    df = BinanceVisionKlineProvider(contract_type="um", verbose=True).fetch_kline(
        args.symbol, args.interval, start, end
    )
    if df.empty:
        raise SystemExit("no kline data")
    df = df.reset_index(drop=True)

    dp_trades = find_optimal_trades(df, mode="dp", min_net_roi=args.min_net_roi)
    leg_trades = find_optimal_trades(df, mode="major_legs", min_net_roi=args.min_net_roi)
    dp_stats = summarize_trades(dp_trades)
    leg_stats = summarize_trades(leg_trades)

    out = Path(args.output) if args.output else _REPO / "trade" / "report" / f"compare_{args.symbol}_{args.start}_{args.end}.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    period = f"{args.symbol} {args.interval}  {args.start} ~ {args.end} (UTC)"

    for ax, mode, trades, stats in (
        (axes[0], "dp", dp_trades, dp_stats),
        (axes[1], "major_legs", leg_trades, leg_stats),
    ):
        hold = stats["total_holding_bars"]
        cov = 100.0 * hold / len(df)
        title = (
            f"{mode}  |  trades={stats['num_trades']}  "
            f"hold={hold}/{len(df)} bars ({cov:.0f}%)  "
            f"avg_hold={hold/max(stats['num_trades'],1):.1f} bars"
        )
        plot_candlestick(df, ax=ax, color_style="crypto", title=title, ylabel="USDT")
        _overlay_trades(ax, df, trades)

    fig.suptitle(f"Optimal trade labeling compare\n{period}", fontsize=12, fontweight="bold")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")
    print(f"dp:         {dp_stats['num_trades']} trades, hold {dp_stats['total_holding_bars']} bars")
    print(f"major_legs: {leg_stats['num_trades']} trades, hold {leg_stats['total_holding_bars']} bars")


if __name__ == "__main__":
    main()
