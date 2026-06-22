#!/usr/bin/env python3
"""Aggregate test metrics from multiple market-state training runs (multi-seed stability)."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


METRIC_KEYS = (
    "return_ic",
    "cum_return_ic",
    "direction_macro_f1",
    "risk_f1",
    "cum_direction_from_return_acc",
    "cum_direction_acc",
    "volatility_mae",
)


ACCEPTANCE_TRACK_BY_STAGE = {
    "usable": "A",
    "balanced_mature": "A-ext",
    "cum_return_recovery": "A-ext",
    "return_direction_branch": "B",
}


def load_run(report_dir: Path) -> dict:
    metrics_path = report_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"missing metrics.json: {metrics_path}")
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    test = data.get("test_metrics") or {}
    stage = data.get("target_stage")
    track = data.get("acceptance_track") or ACCEPTANCE_TRACK_BY_STAGE.get(stage)
    return {
        "run_id": data.get("run_id", report_dir.name),
        "seed": data.get("args", {}).get("seed"),
        "target_stage": stage,
        "acceptance_track": track,
        "decision": data.get("decision"),
        "gates_passed": data.get("gates_passed"),
        "no_valid_checkpoint": data.get("no_valid_checkpoint"),
        "collapse_gates_test": data.get("collapse_gates_test"),
        "test_metrics": {k: test.get(k) for k in METRIC_KEYS if k in test},
    }


def aggregate_numeric(runs: list[dict], key: str) -> dict[str, float]:
    values = [r["test_metrics"][key] for r in runs if r["test_metrics"].get(key) is not None]
    if not values:
        return {}
    return {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Summarize multi-seed market-state reports")
    p.add_argument("report_dirs", nargs="+", type=Path, help="paths to reports/<run_id>/")
    p.add_argument("-o", "--output", type=Path, help="write JSON summary to path")
    args = p.parse_args()

    runs = [load_run(d.resolve()) for d in args.report_dirs]
    agg = {k: aggregate_numeric(runs, k) for k in METRIC_KEYS}
    accept_count = sum(1 for r in runs if r.get("decision") == "accept")
    collapse_pass = sum(
        1
        for r in runs
        if r.get("collapse_gates_test") and all(r["collapse_gates_test"].values())
    )

    payload = {
        "runs": runs,
        "aggregate": agg,
        "summary": {
            "run_count": len(runs),
            "decision_accept_count": accept_count,
            "collapse_gates_pass_count": collapse_pass,
            "all_decision_accept": accept_count == len(runs),
            "all_collapse_gates_pass": collapse_pass == len(runs),
        },
    }

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
