#!/usr/bin/env python3
"""026 Phase 1: C1+D1 ablation train/eval vs Phase 0 C3 baseline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(_ROOT / "examples"))

from _v025_common import sha256_prefix
from run_v026_phase0 import _eval_split_metrics, _read_json

RECIPE = _ROOT / "configs/training_recipe_026_phase1_c1d1.json"
OUT = _ROOT / "backtest/v026_phase1"
P1_CKPT = _ROOT / "checkpoints/026_phase1_c1d1/market_state_best.pt"
P1_REPORT = _ROOT / "reports/026_phase1_c1d1/metrics.json"
C3_EVAL = _ROOT / "backtest/v026_phase0/eval_model_participation_c3.json"
EVAL_JSON = OUT / "eval_model_participation_c1d1.json"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=_ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description="026 Phase 1 C1+D1 pipeline")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    recipe = _read_json(RECIPE)
    gates = recipe.get("gates", {})

    if not args.skip_train or not P1_CKPT.is_file():
        _run([
            sys.executable,
            "examples/train_v026_phase1_c1d1.py",
            "--recipe",
            str(RECIPE.relative_to(_ROOT)),
        ])
    else:
        print(f"reusing checkpoint: {P1_CKPT}")

    _run([
        sys.executable,
        "examples/eval_model_participation.py",
        "--checkpoint",
        str(P1_CKPT.relative_to(_ROOT)),
        "--output",
        str(EVAL_JSON.relative_to(_ROOT)),
    ])

    eval_out = _read_json(EVAL_JSON)
    train_metrics = _read_json(P1_REPORT)
    c3_eval = _read_json(C3_EVAL)
    valid_mm = _eval_split_metrics(eval_out, "valid")
    c3_mm = _eval_split_metrics(c3_eval, "valid")

    valid_auc = float(valid_mm.get("participation_auc", 0.0))
    c3_auc = float(c3_mm.get("participation_auc", gates.get("c3_baseline_part_auc", 0.695)))
    baseline_ic = float(train_metrics.get("baseline_cum_return_ic", 0.0))
    valid_ic = float(valid_mm.get("cum_return_ic", train_metrics.get("valid_cum_return_ic", 0.0)))
    ic_deg = float(train_metrics.get("valid_ic_degradation", 0.0))
    if baseline_ic > 0 and ic_deg == 0.0:
        ic_deg = max(0.0, (baseline_ic - valid_ic) / baseline_ic)

    part_min = float(gates.get("participation_auc_valid_min", 0.65))
    ic_max = float(gates.get("cum_return_ic_degradation_max", 0.10))
    delta = valid_auc - c3_auc
    stack_ok = valid_auc >= part_min and ic_deg <= ic_max
    phase1_pass = stack_ok and delta >= -0.02

    lines = [
        "# 026 Phase 1 — C1 Leg Context + D1 Sampling (ablation)",
        "",
        f"- init: `checkpoints/026_phase0_c3/market_state_best.pt`",
        f"- checkpoint: `{P1_CKPT.relative_to(_ROOT)}` (`{sha256_prefix(P1_CKPT)}`)",
        f"- recipe: `{RECIPE.relative_to(_ROOT)}`",
        "",
        "## vs Phase 0 C3",
        "",
        "| metric | C3 baseline | C1+D1 | Δ | gate |",
        "|--------|-------------|-------|---|------|",
        f"| valid part_auc | {c3_auc:.4f} | {valid_auc:.4f} | {delta:+.4f} | ≥ {part_min:.2f} |",
        f"| cum_IC degradation | — | {ic_deg*100:.2f}% | — | ≤ {ic_max*100:.0f}% |",
        "",
        f"**Stack value: {'PASS' if stack_ok else 'FAIL'}** (maintain ≥0.65 & IC gate)",
        f"**Phase 1 ablation: {'PASS' if phase1_pass else 'FAIL'}** (no >2pp regression vs C3)",
        "",
        "## Routing",
        "",
    ]
    if valid_auc >= 0.65 and delta >= 0:
        lines.append("- C1+D1 叠加有增益 → 可跳过 Phase 2 主路径，进入 Phase 3 全链路（仍建议 A1 并行准备）")
    elif valid_auc >= 0.62:
        lines.append("- 模型轨 ≥0.62 → 可进入 Phase 3 全链路；Phase 2 A1 按 §4.1.2 可选")
    else:
        lines.append("- 未达 0.62 → Phase 2 A1 必做")
    lines.extend([
        "",
        "## Reproduction",
        "```bash",
        "python examples/run_v026_phase1.py",
        "```",
    ])
    report = OUT / "REPORT_026_PHASE1.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "phase1_pass": phase1_pass,
        "stack_ok": stack_ok,
        "valid_participation_auc": valid_auc,
        "c3_participation_auc": c3_auc,
        "delta_part_auc": delta,
        "ic_degradation": ic_deg,
    }
    (OUT / "phase1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {report}")
    return 0 if phase1_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
