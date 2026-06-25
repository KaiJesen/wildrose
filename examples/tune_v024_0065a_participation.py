#!/usr/bin/env python3
"""024 Phase 1: grid tune 0065a-0 participation_auc on valid."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
OUT = _ROOT / "backtest/v024_phase1_tune"

GRID = [
    {"name": "s8_baseline", "args": ["--stride", "8", "--epochs", "8"]},
    {"name": "s1_os30_pw1", "args": ["--stride", "1", "--positive-oversample", "30", "--participation-weight", "1.0", "--epochs", "15", "--early-stop-patience", "5"]},
    {"name": "s1_os50_pw2", "args": ["--stride", "1", "--positive-oversample", "50", "--participation-weight", "2.0", "--epochs", "15", "--early-stop-patience", "5"]},
    {"name": "s1_os30_freeze", "args": ["--stride", "1", "--positive-oversample", "30", "--participation-weight", "1.5", "--freeze-encoder", "--epochs", "15", "--early-stop-patience", "5"]},
    {"name": "s2_os30_pw1", "args": ["--stride", "2", "--positive-oversample", "30", "--participation-weight", "1.0", "--epochs", "15", "--early-stop-patience", "5"]},
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for cfg in GRID:
        name = cfg["name"]
        ckpt_dir = OUT / name / "checkpoint"
        report_dir = OUT / name / "report"
        cmd = [
            sys.executable,
            "examples/train_market_state_0065a.py",
            "--variant",
            "0",
            "--checkpoint-dir",
            str(ckpt_dir.relative_to(_ROOT)),
            "--report-dir",
            str(report_dir.relative_to(_ROOT)),
            "--batch-size",
            "32",
            *cfg["args"],
        ]
        print("+", " ".join(cmd))
        subprocess.run(cmd, cwd=_ROOT, check=False)
        metrics_path = report_dir / "metrics.json"
        if not metrics_path.is_file():
            results.append({"name": name, "error": "no metrics"})
            continue
        m = json.loads(metrics_path.read_text())
        hist = m.get("history", [])
        best_auc = max((h.get("valid_participation_auc", 0) for h in hist), default=0)
        best_long = max((h.get("valid_participation_auc_long", 0) for h in hist), default=0)
        results.append(
            {
                "name": name,
                "best_valid_participation_auc": best_auc,
                "best_valid_participation_auc_long": best_long,
                "best_epoch": m.get("best_epoch"),
                "tuning": m.get("tuning", {}),
                "gate_pass_055": best_auc >= 0.55,
            }
        )
        print(f"  -> {name} best_auc={best_auc:.4f} long={best_long:.4f}")

    results.sort(key=lambda r: r.get("best_valid_participation_auc", 0), reverse=True)
    out_path = OUT / "tune_results.json"
    out_path.write_text(json.dumps({"grid": results, "target": 0.55}, indent=2), encoding="utf-8")
    print(f"\nsaved: {out_path}")
    best = results[0] if results else {}
    print(f"best: {best.get('name')} auc={best.get('best_valid_participation_auc', 0):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
