#!/usr/bin/env python3
"""Plot participation overlay: confirmed legs + strategy holds (023 Phase 0)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from trading_system.participation import compute_participation_metrics

plt.rcParams["axes.unicode_minus"] = False


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("ts", "entry_ts", "exit_ts"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="023 participation overlay plot")
    p.add_argument("--backtest-dir", default="backtest/v023_baseline/test")
    p.add_argument("--output", default="")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bt_dir = Path(args.backtest_dir)
    out = Path(args.output) if args.output else bt_dir / "participation_overlay.png"

    decisions = _read_csv(bt_dir / "decisions.csv")
    trades = _read_csv(bt_dir / "trades.csv")
    if decisions.empty:
        raise SystemExit(f"missing {bt_dir / 'decisions.csv'}")

    part = compute_participation_metrics(decisions, trades)
    effective_ids = {lg.leg_id for lg in part.legs if lg.effective_covered}

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(decisions["ts"], decisions["price"], color="#90a4ae", lw=0.9, label="Close")

    # shade confirmed leg intervals
    leg_colors = {
        "FAST_UP_LEG": "#a5d6a7",
        "SLOW_UP_LEG": "#c8e6c9",
        "FAST_DOWN_LEG": "#ef9a9a",
        "SLOW_DOWN_LEG": "#ffcdd2",
    }
    for leg in part.legs:
        rows = decisions[
            (decisions["leg_id"] == leg.leg_id)
            & (decisions["is_leg_confirmed"].astype(int) == 1)
        ]
        if rows.empty:
            continue
        t0, t1 = rows["ts"].iloc[0], rows["ts"].iloc[-1]
        color = leg_colors.get(leg.leg_type, "#e0e0e0")
        alpha = 0.45 if leg.leg_id in effective_ids else 0.2
        ax.axvspan(t0, t1, color=color, alpha=alpha, linewidth=0)

    # strategy hold spans from state
    for state, color in (("LONG", "#1b5e20"), ("SHORT", "#b71c1c")):
        mask = decisions["state"] == state
        if not mask.any():
            continue
        grp = (mask != mask.shift()).cumsum()
        for _, block in decisions.loc[mask].groupby(grp):
            if block.empty:
                continue
            ax.axvspan(block["ts"].iloc[0], block["ts"].iloc[-1], color=color, alpha=0.12)

    if not trades.empty:
        for row in trades.itertuples(index=False):
            c = "#2e7d32" if float(row.net_pnl) >= 0 else "#c62828"
            ax.scatter(row.entry_ts, row.entry_price, marker="^" if row.side == "LONG" else "v", s=60, color=c, zorder=5)
            ax.scatter(row.exit_ts, row.exit_price, marker="x", s=40, color=c, zorder=5)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.set_title(
        f"023 Participation Overlay — effective legs {len(effective_ids)}/{part.leg_count} "
        f"| bar coverage {part.leg_bar_coverage_ratio:.1%}"
    )
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    ax.legend(
        handles=[
            Patch(facecolor="#a5d6a7", alpha=0.5, label="FAST_UP leg"),
            Patch(facecolor="#ef9a9a", alpha=0.5, label="FAST_DOWN leg"),
            Patch(facecolor="#1b5e20", alpha=0.2, label="LONG hold"),
            Patch(facecolor="#b71c1c", alpha=0.2, label="SHORT hold"),
        ],
        loc="upper left",
        fontsize=8,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
