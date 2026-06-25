#!/usr/bin/env python3
"""Grid tune 0065a constraint profiles (drift vs participation)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
OUT = _ROOT / "backtest/v024_0065a_constraint_tune"

GRID = [
    {
        "name": "legacy_unconstrained",
        "args": ["--constraint-profile", "none", "--participation-weight", "1.5", "--epochs", "10"],
    },
    {
        "name": "constrained",
        "args": [
            "--constraint-profile",
            "constrained",
            "--auto-baseline-ic",
            "--participation-weight",
            "1.5",
            "--early-stop-metric",
            "composite",
            "--epochs",
            "12",
        ],
    },
    {
        "name": "soft_drift",
        "args": [
            "--constraint-profile",
            "soft",
            "--auto-baseline-ic",
            "--participation-weight",
            "1.5",
            "--early-stop-metric",
            "composite",
            "--epochs",
            "12",
        ],
    },
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for cfg in GRID:
        name = cfg["name"]
        ckpt_dir = OUT / name / "checkpoint"
        report_dir = OUT / name / "report"
        cmd = [
            sys.executable,
            "examples/train_market_state_0065a.py",
            "--variant",
            "0",
            "--stride",
            "1",
            "--positive-oversample",
            "30",
            "--batch-size",
            "32",
            "--early-stop-patience",
            "5",
            "--checkpoint-dir",
            str(ckpt_dir.relative_to(_ROOT)),
            "--report-dir",
            str(report_dir.relative_to(_ROOT)),
            *cfg["args"],
        ]
        subprocess.check_call(cmd, cwd=_ROOT)
        metrics = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "name": name,
                "valid_best_score": metrics.get("valid_best_score"),
                "baseline_cum_return_ic": metrics.get("baseline_cum_return_ic"),
                "test_participation_auc": metrics.get("test_metrics", {}).get("participation_auc"),
                "test_cum_return_ic": metrics.get("test_metrics", {}).get("cum_return_ic"),
                "constraint": metrics.get("constraint"),
            }
        )
    (OUT / "tune_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    best = max(rows, key=lambda r: (r.get("valid_best_score") or 0))
    print(f"best by valid part_auc: {best['name']} score={best.get('valid_best_score')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
