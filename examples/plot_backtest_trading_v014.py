#!/usr/bin/env python3
"""Plot v014 backtest: price, strategy trades, hindsight optimal points, equity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

_EX = Path(__file__).resolve().parent
_ROOT = _EX.parent
_TOOLS = _ROOT / "trade" / "tools"
for _p in (str(_ROOT), str(_EX), str(_TOOLS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _train_common import add_data_args, apply_real_data_defaults, fetch_ohlcv_df
from market_data.schema import COL_CLOSE, COL_HIGH, COL_LOW, COL_TIME
from optimal_trade_points import find_optimal_trades, trades_to_dataframe

plt.rcParams["axes.unicode_minus"] = False
for _font in ("Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei", "Microsoft YaHei"):
    try:
        plt.rcParams["font.sans-serif"] = [_font, "DejaVu Sans"]
        break
    except Exception:
        pass

DATE_FMT = "%Y-%m-%d\n%H:%M"
DATE_FMT_COMPACT = "%Y-%m-%d"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot v014 backtest trades and equity")
    add_data_args(p)
    p.add_argument(
        "--backtest-dir",
        default="backtest/backtest_rule_v014b_trend_hold_tuned_0062e_test",
    )
    p.add_argument("--output", default="", help="png path; default <backtest-dir>/backtest_plot.png")
    p.add_argument("--title", default="")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--show-optimal", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fee-bps", type=float, default=4.0, help="one-way fee bps for optimal DP")
    p.add_argument("--leverage", type=float, default=20.0)
    p.add_argument("--min-net-roi", type=float, default=0.002, help="min net roi for optimal_trades_hindsight.csv")
    p.add_argument("--min-net-roi-display", type=float, default=0.10, help="higher bar for chart overlay")
    p.add_argument("--max-optimal-display", type=int, default=20)
    p.add_argument("--max-holding-bars", type=int, default=None)
    p.set_defaults(source="binance_vision", symbol="BTCUSDT", interval="1h", days=365)
    return p.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ("ts", "entry_ts", "exit_ts"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df


def _load_ohlcv_window(t0: pd.Timestamp, t1: pd.Timestamp, args: argparse.Namespace) -> pd.DataFrame:
    apply_real_data_defaults(args)
    df = fetch_ohlcv_df(args)
    df = df.copy()
    df[COL_TIME] = pd.to_datetime(df[COL_TIME], utc=True)
    out = df[(df[COL_TIME] >= t0) & (df[COL_TIME] <= t1)].reset_index(drop=True)
    return out


def _format_ts(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")


def _select_trades_for_display(trades: list, max_show: int) -> list:
    if len(trades) <= max_show:
        return trades
    best_long = max((t for t in trades if t.direction == "long"), key=lambda t: t.net_roi, default=None)
    best_short = max((t for t in trades if t.direction == "short"), key=lambda t: t.net_roi, default=None)
    picked: list = []
    seen: set[int] = set()
    for t in (best_long, best_short):
        if t is not None and id(t) not in seen:
            picked.append(t)
            seen.add(id(t))
    rest = sorted((t for t in trades if id(t) not in seen), key=lambda t: t.net_roi, reverse=True)
    picked.extend(rest[: max(0, max_show - len(picked))])
    return picked


def _plot_optimal_trades(ax, ohlcv: pd.DataFrame, trades: list) -> None:
    """Overlay hindsight optimal long/short from trade/tools/optimal_trade_points."""
    if not trades or ohlcv.empty:
        return

    times = pd.to_datetime(ohlcv[COL_TIME], utc=True)
    times_num = mdates.date2num(times.to_numpy())
    span = float(ohlcv[COL_HIGH].max() - ohlcv[COL_LOW].min()) if COL_HIGH in ohlcv.columns else float(ohlcv[COL_CLOSE].max() - ohlcv[COL_CLOSE].min())
    pad = max(span * 0.022, 1.0)

    long_color = "#1565c0"
    short_color = "#f9a825"
    show_labels = len(trades) <= 24

    best_long = max((t for t in trades if t.direction == "long"), key=lambda t: t.net_roi, default=None)
    best_short = max((t for t in trades if t.direction == "short"), key=lambda t: t.net_roi, default=None)

    for tr in trades:
        is_best = tr is best_long or tr is best_short
        c = long_color if tr.direction == "long" else short_color
        x_in = times_num[tr.entry_index]
        x_out = times_num[tr.exit_index]
        y_in, y_out = tr.entry_price, tr.exit_price
        lw = 2.0 if is_best else 1.0
        alpha = 0.95 if is_best else 0.55
        ms = 220 if is_best else 90

        ax.plot([x_in, x_out], [y_in, y_out], color=c, ls=":", lw=lw, alpha=alpha, zorder=2)

        if tr.direction == "long":
            ax.scatter(x_in, y_in - pad, marker="^", s=ms, color=c, edgecolors="white", linewidths=0.8, zorder=6, alpha=alpha)
        else:
            ax.scatter(x_in, y_in + pad, marker="v", s=ms, color=c, edgecolors="white", linewidths=0.8, zorder=6, alpha=alpha)

        ax.scatter(x_out, y_out, marker="x", s=ms * 0.45, color=c, linewidths=1.2, zorder=5, alpha=alpha)

        if is_best or show_labels:
            mid_x = (x_in + x_out) / 2.0
            mid_y = max(y_in, y_out) + pad * (1.3 if tr.direction == "short" else 0.9)
            prefix = "BEST " if is_best else ""
            ax.annotate(
                f"{prefix}{tr.direction.upper()}\n{_format_ts(tr.entry_time)}\n+{tr.net_roi * 100:.2f}%",
                xy=(mid_x, mid_y),
                ha="center",
                va="bottom",
                fontsize=7 if is_best else 6,
                color=c,
                fontweight="bold" if is_best else "normal",
                zorder=7,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor=c),
            )


def _apply_date_axis(ax, *, major_days: int = 7) -> None:
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, major_days)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(DATE_FMT))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)


def main() -> int:
    args = parse_args()
    bt_dir = Path(args.backtest_dir)
    out_path = Path(args.output) if args.output else bt_dir / "backtest_plot.png"

    equity = _read_csv(bt_dir / "equity_curve.csv")
    trades = _read_csv(bt_dir / "trades.csv")
    decisions = _read_csv(bt_dir / "decisions.csv")
    if equity.empty:
        raise SystemExit(f"missing or empty equity_curve.csv in {bt_dir}")

    equity = equity.sort_values("ts").reset_index(drop=True)
    equity["return_pct"] = (equity["equity"] - 1.0) * 100.0
    peak = equity["equity"].cummax()
    equity["drawdown_pct"] = (equity["equity"] / peak - 1.0) * 100.0

    price_df = decisions[["ts", "price"]].copy() if not decisions.empty else pd.DataFrame()
    if not price_df.empty:
        price_df = price_df.sort_values("ts").reset_index(drop=True)
        price_df["bh_equity"] = price_df["price"] / price_df["price"].iloc[0]
        price_df["bh_return_pct"] = (price_df["bh_equity"] - 1.0) * 100.0

    t0, t1 = equity["ts"].iloc[0], equity["ts"].iloc[-1]
    ohlcv = _load_ohlcv_window(t0, t1, args)

    optimal_trades: list = []
    optimal_display: list = []
    if args.show_optimal and not ohlcv.empty:
        fee_rate = args.fee_bps / 10000.0
        optimal_trades = find_optimal_trades(
            ohlcv,
            fee_rate=fee_rate,
            leverage=args.leverage,
            min_net_roi=args.min_net_roi,
            price_field=COL_CLOSE,
            time_field=COL_TIME,
            max_holding_bars=args.max_holding_bars,
        )
        opt_df = trades_to_dataframe(optimal_trades)
        if not opt_df.empty:
            opt_df.to_csv(bt_dir / "optimal_trades_hindsight.csv", index=False)

        optimal_display = find_optimal_trades(
            ohlcv,
            fee_rate=fee_rate,
            leverage=args.leverage,
            min_net_roi=args.min_net_roi_display,
            price_field=COL_CLOSE,
            time_field=COL_TIME,
            max_holding_bars=args.max_holding_bars,
        )
        optimal_display = _select_trades_for_display(optimal_display, args.max_optimal_display)

    title = args.title or f"v014 Backtest — {bt_dir.name}"
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True, gridspec_kw={"height_ratios": [3.2, 1.4, 1.2]})
    ax_price, ax_eq, ax_dd = axes

    if not ohlcv.empty and COL_HIGH in ohlcv.columns:
        ax_price.plot(ohlcv[COL_TIME], ohlcv[COL_CLOSE], color="#90a4ae", linewidth=0.8, alpha=0.6, label="Close (OHLCV)")
    elif not price_df.empty:
        ax_price.plot(price_df["ts"], price_df["price"], color="#78909c", linewidth=0.9, alpha=0.55, label="Close")

    ax_price.set_ylabel("BTC Price (USDT)")
    ax_price.grid(True, alpha=0.25)
    period = f"{_format_ts(t0)}  ~  {_format_ts(t1)}"
    ax_price.set_title(f"{title}\n{period}", fontsize=11, fontweight="bold")

    if args.show_optimal:
        _plot_optimal_trades(ax_price, ohlcv, optimal_display)

    legend_handles: list[Line2D] = []
    if not trades.empty:
        for row in trades.itertuples(index=False):
            side = str(row.side).upper()
            pnl = float(row.net_pnl)
            trade_color = "#2e7d32" if pnl >= 0 else "#c62828"
            if side == "LONG":
                ax_price.scatter(row.entry_ts, row.entry_price, marker="^", s=140, color="#1b5e20", edgecolors="white", linewidths=0.7, zorder=8)
                ax_price.scatter(row.exit_ts, row.exit_price, marker="v", s=140, color="#66bb6a", edgecolors="white", linewidths=0.7, zorder=8)
            else:
                ax_price.scatter(row.entry_ts, row.entry_price, marker="v", s=140, color="#b71c1c", edgecolors="white", linewidths=0.7, zorder=8)
                ax_price.scatter(row.exit_ts, row.exit_price, marker="^", s=140, color="#ef5350", edgecolors="white", linewidths=0.7, zorder=8)
            ax_price.plot([row.entry_ts, row.exit_ts], [row.entry_price, row.exit_price], color=trade_color, linewidth=1.4, alpha=0.7, linestyle="--", zorder=7)
            mid_ts = row.entry_ts + (row.exit_ts - row.entry_ts) / 2
            mid_px = (row.entry_price + row.exit_price) / 2
            ax_price.annotate(
                f"STRAT {pnl * 100:+.2f}%\n{_format_ts(row.entry_ts)}",
                xy=(mid_ts, mid_px),
                fontsize=6,
                color=trade_color,
                ha="center",
                va="bottom",
                alpha=0.95,
            )

        legend_handles.extend(
            [
                Line2D([0], [0], marker="^", color="w", markerfacecolor="#1b5e20", markersize=9, label="Strategy long entry"),
                Line2D([0], [0], marker="v", color="w", markerfacecolor="#66bb6a", markersize=9, label="Strategy long exit"),
                Line2D([0], [0], marker="v", color="w", markerfacecolor="#b71c1c", markersize=9, label="Strategy short entry"),
                Line2D([0], [0], marker="^", color="w", markerfacecolor="#ef5350", markersize=9, label="Strategy short exit"),
            ]
        )

    if args.show_optimal and optimal_display:
        legend_handles.extend(
            [
                Line2D([0], [0], marker="^", color="w", markerfacecolor="#1565c0", markersize=10, label="Optimal long (hindsight)"),
                Line2D([0], [0], marker="v", color="w", markerfacecolor="#f9a825", markersize=10, label="Optimal short (hindsight)"),
                Line2D([0], [0], color="#1565c0", ls=":", lw=1.5, label="Optimal trade path (DP)"),
            ]
        )

    if legend_handles:
        ax_price.legend(handles=legend_handles, loc="upper left", fontsize=7, framealpha=0.92, ncol=2)

    ax_eq.plot(equity["ts"], equity["return_pct"], color="#1565c0", linewidth=1.6, label="Strategy return %")
    if not price_df.empty:
        aligned = price_df.set_index("ts").reindex(equity["ts"], method="ffill")
        ax_eq.plot(
            equity["ts"],
            aligned["bh_return_pct"].to_numpy(),
            color="#9e9e9e",
            linewidth=1.0,
            linestyle="--",
            alpha=0.85,
            label="Buy & hold %",
        )
    ax_eq.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    final_ret = equity["return_pct"].iloc[-1]
    ax_eq.set_ylabel("Cumulative return %")
    ax_eq.legend(loc="upper left", fontsize=8)
    ax_eq.grid(True, alpha=0.25)
    ax_eq.text(
        0.99,
        0.05,
        f"Final: {final_ret:+.2f}%  |  Trades: {len(trades)}  |  Optimal(DP): {len(optimal_trades)} (show {len(optimal_display)})",
        transform=ax_eq.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax_dd.fill_between(equity["ts"], equity["drawdown_pct"], 0, color="#ef5350", alpha=0.35)
    ax_dd.plot(equity["ts"], equity["drawdown_pct"], color="#c62828", linewidth=1.0)
    ax_dd.set_ylabel("Drawdown %")
    ax_dd.set_xlabel(f"Date (UTC)  |  {period}")
    ax_dd.grid(True, alpha=0.25)
    mdd = equity["drawdown_pct"].min()
    ax_dd.text(
        0.01,
        0.08,
        f"Max drawdown: {mdd:.2f}%",
        transform=ax_dd.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    span_days = max(1, int((t1 - t0).total_seconds() // 86400))
    tick_step = 3 if span_days <= 45 else 7 if span_days <= 120 else 14
    for ax in axes:
        _apply_date_axis(ax, major_days=tick_step)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved plot: {out_path}")
    if optimal_trades:
        print(f"saved optimal trades: {bt_dir / 'optimal_trades_hindsight.csv'} ({len(optimal_trades)} trades)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
