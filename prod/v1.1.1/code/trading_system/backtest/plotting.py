from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from market_data.schema import COL_CLOSE, COL_HIGH, COL_LOW, COL_OPEN, COL_TIME


def _require_matplotlib():
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for backtest plots") from exc
    return mdates, plt, Rectangle


def _normalize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    out = trades.copy()
    if "entry_time" not in out.columns and "entry_ts" in out.columns:
        out["entry_time"] = out["entry_ts"]
    if "exit_time" not in out.columns and "exit_ts" in out.columns:
        out["exit_time"] = out["exit_ts"]
    for col in ("entry_time", "exit_time"):
        out[col] = pd.to_datetime(out[col], utc=True)
    for col in ("entry_price", "exit_price"):
        out[col] = out[col].astype(float)
    return out


def plot_equity_curve(
    path: Path,
    *,
    strategy_eq: np.ndarray,
    benchmark_eq: np.ndarray | None = None,
    title: str = "Backtest equity curve",
    dpi: int = 160,
) -> None:
    _, plt, _ = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(strategy_eq, label="strategy", linewidth=2.0, color="#1f77b4")
    if benchmark_eq is not None and len(benchmark_eq) == len(strategy_eq):
        ax.plot(benchmark_eq, label="buy & hold", linewidth=1.4, alpha=0.85, color="#7f7f7f")
    ax.set_title(title)
    ax.set_xlabel("bar")
    ax.set_ylabel("equity")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_trade_points(
    path: Path,
    *,
    df: pd.DataFrame,
    trades: pd.DataFrame,
    title: str = "Backtest trade points",
    dpi: int = 160,
) -> None:
    mdates, plt, Rectangle = _require_matplotlib()
    trades = _normalize_trades(trades)
    view = df.copy()
    view[COL_TIME] = pd.to_datetime(view[COL_TIME], utc=True)
    if not trades.empty:
        all_times = pd.concat(
            [pd.to_datetime(trades["entry_time"], utc=True), pd.to_datetime(trades["exit_time"], utc=True)],
            ignore_index=True,
        )
        tmin = all_times.min() - pd.Timedelta(hours=24)
        tmax = all_times.max() + pd.Timedelta(hours=24)
        view = view[(view[COL_TIME] >= tmin) & (view[COL_TIME] <= tmax)].copy()
        if view.empty:
            view = df.copy()
            view[COL_TIME] = pd.to_datetime(view[COL_TIME], utc=True)

    fig, ax = plt.subplots(figsize=(16, 7))
    t = view[COL_TIME].to_numpy()
    o = view[COL_OPEN].to_numpy(dtype=np.float64)
    h = view[COL_HIGH].to_numpy(dtype=np.float64)
    l = view[COL_LOW].to_numpy(dtype=np.float64)
    c = view[COL_CLOSE].to_numpy(dtype=np.float64)
    x = mdates.date2num(t)
    width = 0.03 if len(x) < 2 else min(0.03, (x[1] - x[0]) * 0.65)
    for xi, oi, hi, li, ci in zip(x, o, h, l, c):
        color = "#2ca02c" if ci >= oi else "#d62728"
        ax.vlines(xi, li, hi, color=color, linewidth=0.9, alpha=0.8)
        bottom = min(oi, ci)
        height = max(abs(ci - oi), max(float(np.mean(c)) * 1e-5, 1e-8))
        ax.add_patch(Rectangle((xi - width / 2, bottom), width, height, facecolor=color, edgecolor=color, alpha=0.75))

    if not trades.empty:
        longs = trades[trades["side"].astype(str).str.upper() == "LONG"]
        shorts = trades[trades["side"].astype(str).str.upper() == "SHORT"]
        ax.scatter(pd.to_datetime(longs["entry_time"], utc=True), longs["entry_price"], marker="^", s=52, color="#1a9850", label="Long entry", zorder=5)
        ax.scatter(pd.to_datetime(shorts["entry_time"], utc=True), shorts["entry_price"], marker="v", s=52, color="#d7301f", label="Short entry", zorder=5)
        ax.scatter(pd.to_datetime(longs["exit_time"], utc=True), longs["exit_price"], marker="x", s=40, color="#006d2c", label="Long exit", zorder=5)
        ax.scatter(pd.to_datetime(shorts["exit_time"], utc=True), shorts["exit_price"], marker="x", s=40, color="#99000d", label="Short exit", zorder=5)

    ax.set_title(f"{title} ({len(trades)} trades)")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", ncol=4, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
