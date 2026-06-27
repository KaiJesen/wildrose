#!/usr/bin/env python3
"""026 Phase 2 A1: build labels, CORAL train/eval vs Phase 1 baseline."""

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

RECIPE = _ROOT / "configs/training_recipe_026_phase2_a1.json"
LABELS_DIR = _ROOT / "data/labels/leg_participation_a1"
OUT = _ROOT / "backtest/v026_phase2"
P2_CKPT = _ROOT / "checkpoints/026_phase2_a1/market_state_best.pt"
P2_REPORT = _ROOT / "reports/026_phase2_a1/metrics.json"
P1_EVAL = _ROOT / "backtest/v026_phase1/eval_model_participation_c1d1.json"
EVAL_JSON = OUT / "eval_model_participation_a1.json"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=_ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description="026 Phase 2 A1 pipeline")
    ap.add_argument("--skip-labels", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    recipe = _read_json(RECIPE)
    gates = recipe.get("gates", {})

    if not args.skip_labels and not (LABELS_DIR / "leg_participation_train.csv").is_file():
        _run([sys.executable, "examples/build_v026_a1_labels.py"])
    elif (LABELS_DIR / "leg_participation_train.csv").is_file():
        print(f"reusing A1 labels: {LABELS_DIR}")

    if not args.skip_train or not P2_CKPT.is_file():
        _run([
            sys.executable,
            "examples/train_v026_phase2_a1.py",
            "--recipe",
            str(RECIPE.relative_to(_ROOT)),
        ])
    else:
        print(f"reusing checkpoint: {P2_CKPT}")

    _run([
        sys.executable,
        "examples/eval_model_participation.py",
        "--checkpoint",
        str(P2_CKPT.relative_to(_ROOT)),
        "--labels-dir",
        str(LABELS_DIR.relative_to(_ROOT)),
        "--output",
        str(EVAL_JSON.relative_to(_ROOT)),
    ])

    eval_out = _read_json(EVAL_JSON)
    train_metrics = _read_json(P2_REPORT)
    p1_eval = _read_json(P1_EVAL) if P1_EVAL.is_file() else {}
    valid_mm = _eval_split_metrics(eval_out, "valid")
    p1_mm = _eval_split_metrics(p1_eval, "valid")

    valid_auc = float(valid_mm.get("participation_auc", 0.0))
    valid_tier1 = float(valid_mm.get("participation_auc_tier1", train_metrics.get("valid_participation_auc_tier1", 0.0)))
    p1_auc = float(p1_mm.get("participation_auc", gates.get("phase1_baseline_part_auc", 0.697)))
    baseline_ic = float(train_metrics.get("baseline_cum_return_ic", 0.0))
    valid_ic = float(valid_mm.get("cum_return_ic", train_metrics.get("valid_cum_return_ic", 0.0)))
    ic_deg = float(train_metrics.get("valid_ic_degradation", 0.0))
    if baseline_ic > 0 and ic_deg == 0.0:
        ic_deg = max(0.0, (baseline_ic - valid_ic) / baseline_ic)

    part_min = float(gates.get("participation_auc_valid_min", 0.62))
    ic_max = float(gates.get("cum_return_ic_degradation_max", 0.10))
    delta = valid_auc - p1_auc
    model_ok = valid_auc >= part_min and ic_deg <= ic_max

    lines = [
        "# 026 Phase 2 — A1 Three-Tier CORAL Labels",
        "",
        f"- init: `checkpoints/026_phase1_c1d1/market_state_best.pt`",
        f"- labels: `{LABELS_DIR.relative_to(_ROOT)}`",
        f"- checkpoint: `{P2_CKPT.relative_to(_ROOT)}` (`{sha256_prefix(P2_CKPT)}`)",
        f"- recipe: `{RECIPE.relative_to(_ROOT)}`",
        "",
        "## vs Phase 1 C1+D1",
        "",
        "| metric | Phase1 | A1 CORAL | Δ | gate |",
        "|--------|--------|----------|---|------|",
        f"| valid part_auc (tier2) | {p1_auc:.4f} | {valid_auc:.4f} | {delta:+.4f} | ≥ {part_min:.2f} |",
        f"| valid part_auc_tier1 | — | {valid_tier1:.4f} | — | — |",
        f"| cum_IC degradation | — | {ic_deg*100:.2f}% | — | ≤ {ic_max*100:.0f}% |",
        "",
        f"**Phase 2 model gate: {'PASS' if model_ok else 'FAIL'}**",
        "",
        "## Routing",
        "",
    ]
    if model_ok:
        lines.append("- 模型轨达标 → 可重跑 Phase 3 全链路（M3 臂 + valid TEQ 重调）")
    else:
        lines.append("- 未达 0.62 或 IC 退化 → 考虑 B1 渐进解冻或架构师裁定")
    lines.extend([
        "",
        "## Reproduction",
        "```bash",
        "python examples/run_v026_phase2.py",
        "```",
    ])
    report = OUT / "REPORT_026_PHASE2.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "phase2_pass": model_ok,
        "valid_participation_auc": valid_auc,
        "valid_participation_auc_tier1": valid_tier1,
        "phase1_participation_auc": p1_auc,
        "delta_part_auc": delta,
        "ic_degradation": ic_deg,
    }
    (OUT / "phase2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {report}")
    return 0 if model_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
