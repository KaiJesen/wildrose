#!/usr/bin/env python3
"""Offline Teacher: hindsight trend-leg labels with causal backfill for Student training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from market_data.schema import COL_CLOSE, COL_HIGH, COL_LOW, COL_TIME
from trading_system.config import TrendSegmentConfig, load_config
from trading_system.trend_segment import SubLegPhase, TrendLegType, TrendSegmentEngine


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    prev = np.roll(close, 1)
    prev[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    atr = np.empty_like(tr)
    atr[:period] = tr[:period].mean()
    alpha = 1.0 / period
    for i in range(period, len(tr)):
        atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1]
    return atr


def build_teacher_labels(df: pd.DataFrame, cfg: TrendSegmentConfig) -> tuple[pd.DataFrame, dict]:
    close = df[COL_CLOSE].to_numpy(dtype=np.float64)
    high = df[COL_HIGH].to_numpy(dtype=np.float64)
    low = df[COL_LOW].to_numpy(dtype=np.float64)
    atr = _compute_atr(high, low, close)
    engine = TrendSegmentEngine(cfg)
    rows: list[dict] = []
    r_large = cfg.swing_large_right_bars
    m_end = 3
    for i in range(len(df)):
        seg = engine.update(
            bar_idx=i,
            high=float(high[i]),
            low=float(low[i]),
            close=float(close[i]),
            atr=float(atr[i]),
            trend_signal=None,
            slow_ctx=None,
            crash_ctx=None,
            is_model_blind=False,
        )
        leg = seg.active_leg
        leg_type = seg.leg_type.value if seg.leg_type != TrendLegType.NONE else TrendLegType.TRANSITION_LEG.value
        sub_phase = seg.sub_phase.value if seg.sub_phase != SubLegPhase.NONE else SubLegPhase.BASE.value
        is_confirmed = 0
        bars_since = seg.bars_since_leg_start
        leg_progress = seg.leg_progress_ratio
        teacher_conf = 0.5
        if leg is not None:
            if bars_since < r_large:
                leg_type = TrendLegType.TRANSITION_LEG.value
                is_confirmed = 0
                teacher_conf = 0.4
            elif leg.is_confirmed:
                is_confirmed = 1
                teacher_conf = min(1.0, 0.5 + leg.leg_efficiency)
            if leg.end_bar_idx is not None and leg.end_bar_idx - i <= m_end:
                sub_phase = SubLegPhase.LEG_END.value
        rows.append(
            {
                COL_TIME: df[COL_TIME].iloc[i],
                "trend_leg_type": leg_type,
                "sub_phase": sub_phase,
                "leg_progress": float(leg_progress),
                "is_leg_confirmed": int(is_confirmed),
                "bars_since_leg_start": int(bars_since),
                "leg_id": int(leg.leg_id) if leg is not None else -1,
                "teacher_confidence": float(teacher_conf),
            }
        )
    out = pd.DataFrame(rows)
    summary = {
        "num_rows": int(len(out)),
        "confirmed_ratio": float(out["is_leg_confirmed"].mean()),
        "leg_type_distribution": out["trend_leg_type"].value_counts().to_dict(),
        "sub_phase_distribution": out["sub_phase"].value_counts().to_dict(),
        "avg_bars_since_leg_start": float(out["bars_since_leg_start"].mean()),
    }
    return out, summary


def main() -> int:
    p = argparse.ArgumentParser(description="Build v020 teacher trend-leg labels")
    p.add_argument("--input", required=True, help="OHLCV csv/parquet path")
    p.add_argument("--config", default="configs/trading_rule_v020_trend_segment_0062e.json")
    p.add_argument("--out-dir", default="data/labels/trend_leg_v020_teacher")
    args = p.parse_args()
    path = Path(args.input)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    cfg = load_config(args.config).trend_segment
    labels, summary = build_teacher_labels(df, cfg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "teacher_labels.csv"
    labels.to_csv(labels_path, index=False)
    summary_path = out_dir / "label_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print({"labels": str(labels_path), "summary": str(summary_path)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
