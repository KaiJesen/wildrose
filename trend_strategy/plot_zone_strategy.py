#!/usr/bin/env python3
"""Plot zone-strategy candlesticks with zones, rails, and entry/exit markers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from market_data.plotting import plot_candlestick
from market_data.schema import COL_CLOSE, COL_TIME
from trend_strategy.rails import TrendLine
from trend_strategy.zone_engine import Trade, Zone, ZoneEngine, ZoneSnapshot

ZONE_STYLE: dict[Zone, dict[str, Any]] = {
    Zone.IDLE: {"color": "#eceff1", "alpha": 0.15, "label": "idle"},
    Zone.LONG_OBS: {"color": "#a5d6a7", "alpha": 0.22, "label": "long obs"},
    Zone.SHORT_OBS: {"color": "#ef9a9a", "alpha": 0.22, "label": "short obs"},
    Zone.LONG_POS: {"color": "#66bb6a", "alpha": 0.30, "label": "long pos"},
    Zone.SHORT_POS: {"color": "#e57373", "alpha": 0.30, "label": "short pos"},
}

LINE_STYLE = {
    "upper_rail": {"color": "#1b5e20", "ls": "--", "lw": 1.4, "label": "upper rail"},
    "lower_rail": {"color": "#b71c1c", "ls": "--", "lw": 1.4, "label": "lower rail"},
    "long_close": {"color": "#0d47a1", "ls": ":", "lw": 1.6, "label": "long close"},
    "short_close": {"color": "#4a148c", "ls": ":", "lw": 1.6, "label": "short close"},
}


def _line_key(line: TrendLine | None) -> tuple | None:
    if line is None:
        return None
    return (line.i0, line.i1, round(line.slope, 12), round(line.intercept, 8))


def _zone_spans(snapshots: list[ZoneSnapshot]) -> list[tuple[int, int, Zone]]:
    if not snapshots:
        return []
    spans: list[tuple[int, int, Zone]] = []
    start = snapshots[0].idx
    cur = snapshots[0].zone
    for snap in snapshots[1:]:
        if snap.zone != cur:
            spans.append((start, snap.idx - 1, cur))
            start = snap.idx
            cur = snap.zone
    spans.append((start, snapshots[-1].idx, cur))
    return spans


def _line_segments(
    snapshots: list[ZoneSnapshot],
    attr: str,
) -> list[tuple[int, int, TrendLine]]:
    segments: list[tuple[int, int, TrendLine]] = []
    start: int | None = None
    cur_key: tuple | None = None
    cur_line: TrendLine | None = None

    for snap in snapshots:
        line = getattr(snap, attr)
        key = _line_key(line)
        if key != cur_key:
            if cur_line is not None and start is not None:
                segments.append((start, snap.idx - 1, cur_line))
            if key is not None and line is not None:
                start = snap.idx
                cur_line = line
            else:
                start = None
                cur_line = None
            cur_key = key

    if cur_line is not None and start is not None:
        segments.append((start, snapshots[-1].idx, cur_line))
    return segments


def _shade_zones(ax, xs_dates: np.ndarray, spans: list[tuple[int, int, Zone]]) -> None:
    for t0, t1, zone in spans:
        if zone == Zone.IDLE or t1 < t0:
            continue
        style = ZONE_STYLE[zone]
        ax.axvspan(
            xs_dates[t0],
            xs_dates[t1],
            color=style["color"],
            alpha=style["alpha"],
            zorder=0,
        )


def _plot_line_segment(
    ax,
    line: TrendLine,
    t0: int,
    t1: int,
    xs_dates: np.ndarray,
    *,
    color: str,
    ls: str,
    lw: float,
) -> None:
    if t1 < t0:
        return
    idx = np.arange(t0, t1 + 1)
    ys = line.values_at(idx.astype(np.float64))
    ax.plot(xs_dates[idx], ys, color=color, ls=ls, lw=lw, zorder=3)


def plot_zone_strategy(
    df: pd.DataFrame,
    engine: ZoneEngine,
    trades: list[Trade],
    *,
    title: str = "Zone Strategy",
    save_path: str | Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(16, 8))
    times = pd.to_datetime(df[COL_TIME], utc=True)
    xs_dates = mdates.date2num(times.dt.to_pydatetime())

    snapshots = engine.snapshots
    _shade_zones(ax, xs_dates, _zone_spans(snapshots))

    plot_candlestick(df, ax=ax, color_style="crypto")

    for attr in ("upper_rail", "lower_rail", "long_close", "short_close"):
        style = LINE_STYLE[attr]
        for t0, t1, line in _line_segments(snapshots, attr):
            _plot_line_segment(
                ax, line, t0, t1, xs_dates,
                color=style["color"], ls=style["ls"], lw=style["lw"],
            )

    closes = df[COL_CLOSE].values.astype(float)
    for tr in trades:
        if tr.side == "long":
            ax.scatter(
                xs_dates[tr.entry_idx], closes[tr.entry_idx],
                marker="^", s=130, c="#00c853", edgecolors="black", linewidths=0.8, zorder=7,
            )
            ax.scatter(
                xs_dates[tr.exit_idx], closes[tr.exit_idx],
                marker="v", s=130, c="#ff6d00", edgecolors="black", linewidths=0.8, zorder=7,
            )
        else:
            ax.scatter(
                xs_dates[tr.entry_idx], closes[tr.entry_idx],
                marker="v", s=130, c="#d50000", edgecolors="black", linewidths=0.8, zorder=7,
            )
            ax.scatter(
                xs_dates[tr.exit_idx], closes[tr.exit_idx],
                marker="^", s=130, c="#00b8d4", edgecolors="black", linewidths=0.8, zorder=7,
            )

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.set_title(title)
    ax.grid(True, alpha=0.2, zorder=1)

    legend_items = [
        Patch(facecolor=ZONE_STYLE[Zone.LONG_OBS]["color"], alpha=0.4, label="long obs"),
        Patch(facecolor=ZONE_STYLE[Zone.SHORT_OBS]["color"], alpha=0.4, label="short obs"),
        Patch(facecolor=ZONE_STYLE[Zone.LONG_POS]["color"], alpha=0.5, label="long pos"),
        Patch(facecolor=ZONE_STYLE[Zone.SHORT_POS]["color"], alpha=0.5, label="short pos"),
        plt.Line2D([0], [0], color=LINE_STYLE["upper_rail"]["color"], ls="--", label="upper rail"),
        plt.Line2D([0], [0], color=LINE_STYLE["lower_rail"]["color"], ls="--", label="lower rail"),
        plt.Line2D([0], [0], color=LINE_STYLE["long_close"]["color"], ls=":", label="long close"),
        plt.Line2D([0], [0], color=LINE_STYLE["short_close"]["color"], ls=":", label="short close"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#00c853", markersize=9, label="long entry / cover"),
        plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="#d50000", markersize=9, label="short entry / exit long"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=8, framealpha=0.9)

    fig.autofmt_xdate()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def main() -> int:
    from trend_strategy.run_zone_strategy import load_ohlcv, parse_args, run

    args = parse_args()
    df = load_ohlcv(args)
    trades, engine, summary = run(df, args)
    out = Path(args.plot)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_zone_strategy(
        df, engine, trades,
        title=f"Zone Strategy {args.symbol} {args.interval} ({summary['trade_count']} trades)",
        save_path=out,
    )
    if args.show:
        plt.show()
    else:
        plt.close()
    print(f"Saved plot -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
