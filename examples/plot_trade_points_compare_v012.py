#!/usr/bin/env python3
"""Plot long/short trade points for two v012 backtests."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import sys

_EX = Path(__file__).resolve().parent
_ROOT = _EX.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from market_data.schema import COL_CLOSE, COL_HIGH, COL_LOW, COL_OPEN, COL_TIME


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare long/short points between two v012 runs")
    p.add_argument("--data-csv", default="data/cache/kline/binance_vision_BTCUSDT_1h_365d_end20260618.csv")
    p.add_argument("--run-a-trades", default="backtest/backtest_rule_v012_0062e_opt_best/trades.csv")
    p.add_argument("--run-a-label", default="Balanced")
    p.add_argument("--run-b-trades", default="backtest/backtest_rule_v012_0062e_mdd10_maxret/trades.csv")
    p.add_argument("--run-b-label", default="Aggressive")
    p.add_argument("--output", default="backtest/backtest_rule_v012_compare/trade_points_compare.png")
    p.add_argument("--dpi", type=int, default=160)
    return p.parse_args()


def _load_kline(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if COL_TIME not in df.columns:
        if "time" in df.columns:
            df[COL_TIME] = pd.to_datetime(df["time"], utc=True)
        else:
            raise KeyError(f"time column missing in {path}")
    else:
        df[COL_TIME] = pd.to_datetime(df[COL_TIME], utc=True)
    return df


def _draw_candles(ax, times, open_, high, low, close) -> None:
    x = mdates.date2num(times)
    width = 0.03 if len(x) < 2 else min(0.03, (x[1] - x[0]) * 0.65)
    for xi, o, h, l, c in zip(x, open_, high, low, close):
        color = "#2ca02c" if c >= o else "#d62728"
        ax.vlines(xi, l, h, color=color, linewidth=0.9, alpha=0.8)
        bottom = min(o, c)
        height = max(abs(c - o), max(float(np.mean(close)) * 1e-5, 1e-8))
        ax.add_patch(
            Rectangle(
                (xi - width / 2, bottom),
                width,
                height,
                facecolor=color,
                edgecolor=color,
                alpha=0.75,
            )
        )


def _plot_trade_points(ax, df: pd.DataFrame, trades: pd.DataFrame, label: str) -> None:
    t = df[COL_TIME].to_numpy()
    o = df[COL_OPEN].to_numpy(dtype=np.float64)
    h = df[COL_HIGH].to_numpy(dtype=np.float64)
    l = df[COL_LOW].to_numpy(dtype=np.float64)
    c = df[COL_CLOSE].to_numpy(dtype=np.float64)
    _draw_candles(ax, t, o, h, l, c)

    # Entry/exit markers.
    long_entries = trades[trades["side"] == "long"]
    short_entries = trades[trades["side"] == "short"]
    long_exit = long_entries
    short_exit = short_entries

    ax.scatter(
        pd.to_datetime(long_entries["entry_time"], utc=True),
        long_entries["entry_price"].astype(float),
        marker="^",
        s=46,
        color="#1a9850",
        label="Long Entry",
        zorder=5,
    )
    ax.scatter(
        pd.to_datetime(short_entries["entry_time"], utc=True),
        short_entries["entry_price"].astype(float),
        marker="v",
        s=46,
        color="#d7301f",
        label="Short Entry",
        zorder=5,
    )
    ax.scatter(
        pd.to_datetime(long_exit["exit_time"], utc=True),
        long_exit["exit_price"].astype(float),
        marker="x",
        s=36,
        color="#006d2c",
        label="Long Exit",
        zorder=5,
    )
    ax.scatter(
        pd.to_datetime(short_exit["exit_time"], utc=True),
        short_exit["exit_price"].astype(float),
        marker="x",
        s=36,
        color="#99000d",
        label="Short Exit",
        zorder=5,
    )
    ax.set_title(label)
    ax.grid(True, alpha=0.2)
    ax.set_ylabel("Price")


def main() -> int:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = _load_kline(args.data_csv)
    ta = pd.read_csv(args.run_a_trades)
    tb = pd.read_csv(args.run_b_trades)
    for col in ("entry_price", "exit_price"):
        ta[col] = ta[col].astype(float)
        tb[col] = tb[col].astype(float)

    # Shared visible window: union of all trade timestamps.
    all_times = pd.concat(
        [
            pd.to_datetime(ta["entry_time"], utc=True),
            pd.to_datetime(ta["exit_time"], utc=True),
            pd.to_datetime(tb["entry_time"], utc=True),
            pd.to_datetime(tb["exit_time"], utc=True),
        ],
        ignore_index=True,
    )
    tmin = all_times.min() - pd.Timedelta(hours=24)
    tmax = all_times.max() + pd.Timedelta(hours=24)
    view = df[(df[COL_TIME] >= tmin) & (df[COL_TIME] <= tmax)].copy()

    fig, axes = plt.subplots(2, 1, figsize=(18, 11), sharex=True)
    _plot_trade_points(axes[0], view, ta, f"{args.run_a_label} ({len(ta)} trades)")
    _plot_trade_points(axes[1], view, tb, f"{args.run_b_label} ({len(tb)} trades)")
    axes[0].legend(loc="upper left", ncol=4, fontsize=8)
    axes[1].legend(loc="upper left", ncol=4, fontsize=8)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.suptitle("v012 Trade Points Comparison (Long/Short Entries & Exits)", y=0.995)
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

