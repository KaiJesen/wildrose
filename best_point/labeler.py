from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trade.tools.optimal_trade_points import find_optimal_trades, trades_to_dataframe

AMBIGUOUS = -100


@dataclass(frozen=True)
class LabelerConfig:
    pre_entry_bars: int = 2
    post_entry_bars: int = 2
    pre_exit_bars: int = 2
    post_exit_bars: int = 1
    min_label_roi_gap: float = 0.01


def _assign_with_priority(target: np.ndarray, start: int, end: int, cls: int, priority: int, prio: np.ndarray) -> None:
    s = max(0, start)
    e = min(len(target) - 1, end)
    if e < s:
        return
    idx = np.arange(s, e + 1)
    better = priority > prio[idx]
    target[idx[better]] = cls
    prio[idx[better]] = priority


def build_best_point_labels(
    df: pd.DataFrame,
    *,
    fee_rate: float,
    leverage: float,
    min_net_roi: float,
    max_holding_bars: int | None,
    allow_long: bool,
    allow_short: bool,
    price_field: str,
    cfg: LabelerConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    trades = find_optimal_trades(
        df,
        fee_rate=fee_rate,
        leverage=leverage,
        min_net_roi=min_net_roi,
        max_holding_bars=max_holding_bars,
        allow_long=allow_long,
        allow_short=allow_short,
        price_field=price_field,
    )
    trades_df = trades_to_dataframe(trades)
    n = len(df)
    entry = np.zeros(n, dtype=np.int64)
    hold = np.zeros(n, dtype=np.int64)
    exit_ = np.zeros(n, dtype=np.int64)
    entry_prio = np.zeros(n, dtype=np.int64)
    hold_prio = np.zeros(n, dtype=np.int64)
    exit_prio = np.zeros(n, dtype=np.int64)
    opp = np.zeros(n, dtype=np.float64)

    for tr in trades:
        is_long = tr.direction == "long"
        ecls = 1 if is_long else 2
        hcls = 1 if is_long else 2
        xcls = 1 if is_long else 2
        _assign_with_priority(entry, tr.entry_index - cfg.pre_entry_bars, tr.entry_index + cfg.post_entry_bars, ecls, 3, entry_prio)
        _assign_with_priority(hold, tr.entry_index + 1, tr.exit_index - 1, hcls, 2, hold_prio)
        _assign_with_priority(exit_, tr.exit_index - cfg.pre_exit_bars, tr.exit_index + cfg.post_exit_bars, xcls, 4, exit_prio)
        opp[max(0, tr.entry_index - cfg.pre_entry_bars) : min(n, tr.exit_index + cfg.post_exit_bars + 1)] = np.maximum(
            opp[max(0, tr.entry_index - cfg.pre_entry_bars) : min(n, tr.exit_index + cfg.post_exit_bars + 1)],
            tr.net_roi,
        )

    # Ambiguous: long/short conflicts with near-equal net_roi.
    ambiguous = np.zeros(n, dtype=bool)
    if len(trades) >= 2:
        for i in range(len(trades)):
            for j in range(i + 1, len(trades)):
                a, b = trades[i], trades[j]
                if a.direction == b.direction:
                    continue
                overlap_s = max(a.entry_index - cfg.pre_entry_bars, b.entry_index - cfg.pre_entry_bars, 0)
                overlap_e = min(a.exit_index + cfg.post_exit_bars, b.exit_index + cfg.post_exit_bars, n - 1)
                if overlap_s <= overlap_e and abs(a.net_roi - b.net_roi) < cfg.min_label_roi_gap:
                    ambiguous[overlap_s : overlap_e + 1] = True

    entry[ambiguous] = AMBIGUOUS
    hold[ambiguous] = AMBIGUOUS
    exit_[ambiguous] = AMBIGUOUS

    label_df = pd.DataFrame(
        {
            "entry_label": entry,
            "hold_label": hold,
            "exit_label": exit_,
            "future_best_net_roi": opp,
            "ambiguous": ambiguous.astype(np.int64),
        }
    )
    summary = {
        "num_rows": int(n),
        "num_trades": int(len(trades)),
        "entry_distribution": {str(k): int(v) for k, v in zip(*np.unique(entry, return_counts=True))},
        "hold_distribution": {str(k): int(v) for k, v in zip(*np.unique(hold, return_counts=True))},
        "exit_distribution": {str(k): int(v) for k, v in zip(*np.unique(exit_, return_counts=True))},
        "ambiguous_ratio": float(ambiguous.mean()),
        "avg_holding_bars": float(np.mean([t.holding_bars for t in trades])) if trades else 0.0,
        "avg_net_roi": float(np.mean([t.net_roi for t in trades])) if trades else 0.0,
        "label_coverage": float(((entry > 0) | (hold > 0) | (exit_ > 0)).mean()),
    }
    return label_df, trades_df, summary


def save_label_outputs(
    *,
    labels: pd.DataFrame,
    trades: pd.DataFrame,
    summary: dict,
    out_dir: str | Path,
    prefix: str,
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels_path = out / f"{prefix}_labels.parquet"
    trades_path = out / f"{prefix}_trades.parquet"
    summary_path = out / "label_summary.json"
    try:
        labels.to_parquet(labels_path, index=False)
        trades.to_parquet(trades_path, index=False)
    except Exception:
        labels_path = out / f"{prefix}_labels.csv"
        trades_path = out / f"{prefix}_trades.csv"
        labels.to_csv(labels_path, index=False)
        trades.to_csv(trades_path, index=False)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"labels": str(labels_path), "trades": str(trades_path), "summary": str(summary_path)}

