#!/usr/bin/env python3
"""023 Phase 0: participation metrics from backtest artifacts (§5.3)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from trading_system.participation import compute_participation_metrics, compare_runner_metrics


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ("ts", "entry_ts", "exit_ts"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df


def _read_metrics(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def eval_backtest_dir(bt_dir: Path, *, split: str) -> dict:
    decisions = _read_csv(bt_dir / "decisions.csv")
    trades = _read_csv(bt_dir / "trades.csv")
    runner = _read_metrics(bt_dir / "metrics.json")
    if decisions.empty:
        raise FileNotFoundError(f"missing decisions.csv in {bt_dir}")

    part = compute_participation_metrics(decisions, trades)
    legacy_cmp = compare_runner_metrics(runner, part)

    out = {
        "split": split,
        "backtest_dir": str(bt_dir),
        "bar_count": float(len(decisions)),
        "runner_metrics": {k: runner[k] for k in runner if k in (
            "total_return", "max_drawdown", "trade_count", "missed_confirmed_trend_bars",
            "leg_coverage_ratio", "slow_up_open_count", "watch_slow_uptrend_count",
            "long_trend_capture_ratio", "short_trend_capture_ratio",
        )},
        "participation_metrics": part.to_dict(),
        "legacy_comparison": legacy_cmp,
        "legs_summary": [
            {
                "leg_id": lg.leg_id,
                "leg_type": lg.leg_type,
                "leg_bars": lg.leg_bars,
                "aligned_overlap_bars": lg.aligned_overlap_bars,
                "counter_overlap_bars": lg.counter_overlap_bars,
                "effective_covered": lg.effective_covered,
                "captured_pnl": lg.captured_pnl,
            }
            for lg in part.legs
        ],
    }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate 023 participation metrics")
    p.add_argument("--backtest-dir", action="append", default=[], help="repeatable; infers split from path if contains valid/test")
    p.add_argument("--output", default="backtest/v023_baseline/participation_metrics.json")
    p.add_argument("--split", default="", help="optional split label when single dir")
    return p.parse_args()


def _infer_split(path: Path) -> str:
    name = path.as_posix().lower()
    if "valid" in name:
        return "valid"
    if "test" in name:
        return "test"
    if "train" in name:
        return "train"
    return "unknown"


def main() -> int:
    args = parse_args()
    dirs = [Path(d) for d in args.backtest_dir]
    if not dirs:
        dirs = [
            Path("backtest/v023_baseline/test"),
            Path("backtest/v023_baseline/valid"),
        ]

    results: dict[str, dict] = {}
    for d in dirs:
        split = args.split or _infer_split(d)
        results[split] = eval_backtest_dir(d, split=split)
        print(f"[{split}] leg_count_coverage={results[split]['participation_metrics']['leg_count_coverage_ratio']:.4f} "
              f"leg_bar_coverage={results[split]['participation_metrics']['leg_bar_coverage_ratio']:.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
