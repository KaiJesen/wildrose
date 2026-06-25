#!/usr/bin/env python3
"""024 Phase 0 orchestrator: freeze 023 stack + labels + model-track eval gate."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
OUT = _ROOT / "backtest/v024_phase0"
LABELS_DIR = _ROOT / "data/labels/leg_participation"
FROZEN_REPORT = _ROOT / "backtest/v023_rule_stack_frozen/REPORT_023_RULE_STACK_FROZEN.md"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=_ROOT)


def _write_phase0_report(eval_doc: dict) -> None:
    lines = [
        "# 024 Phase 0 Report",
        "",
        f"- timestamp: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC`",
        f"- frozen 023 report: `{FROZEN_REPORT.relative_to(_ROOT)}`",
        f"- labels dir: `{LABELS_DIR.relative_to(_ROOT)}`",
        "",
        "## Leg-count alignment (gate < 2%)",
        "",
        "| split | label legs | participation legs | deviation | gate |",
        "|-------|------------|--------------------|-----------|------|",
    ]
    for split, row in eval_doc.get("splits", {}).items():
        dev = row.get("leg_count_deviation_ratio")
        gate = row.get("leg_count_alignment_gate_pass")
        lines.append(
            f"| {split} | {int(row.get('label_confirmed_leg_count', 0))} | "
            f"{row.get('participation_leg_count', '—')} | "
            f"{dev * 100:.2f}% | {'PASS' if gate else 'FAIL'} |"
            if dev is not None
            else f"| {split} | {int(row.get('label_confirmed_leg_count', 0))} | — | — | SKIP |"
        )

    lines.extend([
        "",
        "## Label summary",
        "",
    ])
    for split, row in eval_doc.get("splits", {}).items():
        ls = row.get("label_summary", {})
        lines.append(f"### {split}")
        lines.append("")
        lines.append(f"- ideal_participate_long_rate: `{ls.get('ideal_participate_long_rate', 0):.4f}`")
        lines.append(f"- ideal_participate_short_rate: `{ls.get('ideal_participate_short_rate', 0):.4f}`")
        lines.append(f"- confirmed_leg_count: `{int(ls.get('confirmed_leg_count', 0))}`")
        lines.append("")

    all_pass = eval_doc.get("leg_count_alignment_all_pass", False)
    lines.extend([
        "## Phase 0 exit",
        "",
        f"- leg_count_alignment_all_pass: **{'PASS' if all_pass else 'FAIL'}**",
        "",
        "Phase 1 may start only after PASS + frozen 023 report present.",
        "",
    ])
    report = OUT / "REPORT_024_PHASE0.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {report}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    _run([sys.executable, "examples/freeze_v023_rule_stack.py"])
    _run([
        sys.executable,
        "examples/build_leg_participation_labels.py",
        "--split",
        "valid",
        "--split",
        "test",
    ])
    _run([
        sys.executable,
        "examples/eval_model_participation.py",
        "--output",
        str((OUT / "eval_model_participation.json").relative_to(_ROOT)),
    ])

    eval_doc = json.loads((OUT / "eval_model_participation.json").read_text(encoding="utf-8"))
    _write_phase0_report(eval_doc)
    return 0 if eval_doc.get("leg_count_alignment_all_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
