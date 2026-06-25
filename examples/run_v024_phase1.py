#!/usr/bin/env python3
"""024 Phase 1 orchestrator: train labels + 0065a-0 + model-track eval."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
OUT = _ROOT / "backtest/v024_phase1"
RECIPE = _ROOT / "configs/training_recipe_0065a_leg_align.json"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=_ROOT)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _write_report(train_metrics: dict, eval_doc: dict) -> None:
    valid_best = train_metrics.get("history", [])
    last_valid = {}
    for row in reversed(valid_best):
        if any(k.startswith("valid_") for k in row):
            last_valid = row
            break
    lines = [
        "# 024 Phase 1 Report (0065a-0)",
        "",
        f"- timestamp: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC`",
        f"- variant: `0`",
        f"- checkpoint: `{train_metrics.get('checkpoint', '')}`",
        "",
        "## Model-track valid (last epoch)",
        "",
        f"- participation_auc: `{last_valid.get('valid_participation_auc', 0):.4f}`",
        f"- cum_return_ic: `{last_valid.get('valid_cum_return_ic', 0):.4f}`",
        f"- confirmed_leg_flat_edge_p50_long: `{last_valid.get('valid_confirmed_leg_flat_edge_p50_long', 0):.4f}`",
        "",
        "## Phase 1 gate (探索)",
        "",
        f"- participation_auc >= 0.55: **{'PASS' if float(last_valid.get('valid_participation_auc', 0)) >= 0.55 else 'FAIL'}**",
        "",
        "## eval_model_participation",
        "",
    ]
    for split, row in eval_doc.get("splits", {}).items():
        mm = row.get("model_metrics", {})
        lines.append(f"### {split}")
        lines.append("")
        lines.append(f"- participation_auc: `{mm.get('participation_auc')}`")
        lines.append(f"- cum_return_ic: `{mm.get('cum_return_ic')}`")
        lines.append("")
    report = OUT / "REPORT_024_PHASE1.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {report}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    recipe = _read_json(RECIPE)

    _run([
        sys.executable,
        "examples/build_leg_participation_labels.py",
        "--split",
        "train",
        "--split",
        "valid",
        "--split",
        "test",
    ])

    baseline_doc = _read_json(_ROOT / "backtest/v024_phase0/eval_model_participation.json")
    baseline_ic = 0.0
    if "splits" in baseline_doc and "valid" in baseline_doc["splits"]:
        mm = baseline_doc["splits"]["valid"].get("model_metrics") or {}
        baseline_ic = float(mm.get("cum_return_ic") or 0.0)

    _run([
        sys.executable,
        "examples/train_market_state_0065a.py",
        "--variant",
        "0",
        "--epochs",
        "8",
        "--early-stop-patience",
        "4",
        "--baseline-cum-return-ic",
        str(baseline_ic),
    ])

    ckpt = _ROOT / "checkpoints/0065a_leg_align_v0/market_state_best.pt"
    _run([
        sys.executable,
        "examples/eval_model_participation.py",
        "--checkpoint",
        str(ckpt.relative_to(_ROOT)),
        "--output",
        str((OUT / "eval_model_participation.json").relative_to(_ROOT)),
    ])

    train_metrics = _read_json(_ROOT / "reports/0065a_leg_align_v0/metrics.json")
    eval_doc = _read_json(OUT / "eval_model_participation.json")
    _write_report(train_metrics, eval_doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
