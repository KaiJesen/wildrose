#!/usr/bin/env python3
"""027 Phase 1: Core rule ablation on valid (independent dimension sweeps, M2 checkpoint)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(_ROOT / "examples"))

from _v027_common import (
    B0_CONFIG,
    CORE_BASELINE,
    M2_CKPT,
    OUT_PHASE1,
    SLOW_UP_CHANNEL_PATCH,
    deep_merge,
    passes_phase1_gates,
    run_valid_backtest,
    slow_up_force_disable,
    verify_pw20_checkpoint,
)

# Independent sweeps from M2 Core baseline (one patch per arm).
SWEEPS: list[tuple[str, dict]] = [
    (
        "slow_up_on",
        {
            "slow_up_position": {"enabled": True},
            **SLOW_UP_CHANNEL_PATCH,
        },
    ),
    ("crash_regime_repeat", {"crash_short": {"same_regime_once": False}}),
    ("crash_hold_30", {"crash_short": {"max_hold_bars": 30, "strong_max_hold_bars": 48}}),
    ("crash_hold_48", {"crash_short": {"max_hold_bars": 48, "strong_max_hold_bars": 72}}),
    ("crash_hold_72", {"crash_short": {"max_hold_bars": 72, "strong_max_hold_bars": 96}}),
    (
        "trend_hold_p50",
        {"trend_position": {"max_trend_hold_bars": 72, "strong_trend_hold_bars": 108}},
    ),
]

QUICK_SWEEPS = ["slow_up_on", "crash_regime_repeat", "trend_hold_p50"]


def _write_config(base: dict, patch: dict, path: Path) -> None:
    cfg = deep_merge(base, patch) if patch else base
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _pick_best(rows: list[dict], *, b0_return: float, b0_coverage: float) -> dict | None:
    candidates = [r for r in rows if passes_phase1_gates(r, b0_return=b0_return, b0_coverage=b0_coverage)]
    if not candidates:
        return None
    return max(candidates, key=lambda r: (r["coverage"], r["return"]))


def main() -> int:
    p = argparse.ArgumentParser(description="027 Phase 1 Core valid ablation")
    p.add_argument("--quick", action="store_true", help="run subset of sweeps")
    p.add_argument("--dual-slot", action="store_true", help="use DualSlotEngine path")
    p.add_argument("--skip-b0", action="store_true", help="skip B0 valid reference run")
    args = p.parse_args()

    verify_pw20_checkpoint()
    if not M2_CKPT.is_file():
        raise FileNotFoundError(M2_CKPT)
    if not CORE_BASELINE.is_file():
        raise FileNotFoundError(CORE_BASELINE)

    OUT_PHASE1.mkdir(parents=True, exist_ok=True)
    cfg_dir = OUT_PHASE1 / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    base = json.loads(CORE_BASELINE.read_text(encoding="utf-8"))
    rows: list[dict] = []

    if not args.skip_b0:
        b0 = run_valid_backtest(B0_CONFIG, OUT_PHASE1 / "b0_valid", checkpoint=M2_CKPT, dual_slot=args.dual_slot)
        rows.append({"tag": "b0_valid", "role": "reference", **b0})

    baseline = run_valid_backtest(
        CORE_BASELINE,
        OUT_PHASE1 / "baseline_valid",
        checkpoint=M2_CKPT,
        dual_slot=args.dual_slot,
    )
    rows.append({"tag": "baseline", "role": "core_m2", **baseline})

    if not args.skip_b0:
        ref_return = rows[0]["return"]
        ref_coverage = rows[0]["coverage"]
        ref_tag = "b0_valid"
    else:
        ref_return = baseline["return"]
        ref_coverage = baseline["coverage"]
        ref_tag = "baseline"

    sweeps = SWEEPS
    if args.quick:
        sweeps = [s for s in SWEEPS if s[0] in QUICK_SWEEPS]

    for tag, patch in sweeps:
        cfg_path = cfg_dir / f"{tag}.json"
        _write_config(base, patch, cfg_path)
        row = run_valid_backtest(cfg_path, OUT_PHASE1 / f"{tag}_valid", checkpoint=M2_CKPT, dual_slot=args.dual_slot)
        force_off = slow_up_force_disable(row) if tag == "slow_up_on" else False
        gate = passes_phase1_gates(row, b0_return=ref_return, b0_coverage=ref_coverage)
        rows.append(
            {
                "tag": tag,
                "role": "sweep",
                "config": str(cfg_path.relative_to(_ROOT)),
                "phase1_gate": gate,
                "slow_up_force_disable": force_off,
                **row,
            }
        )
        status = "PASS" if gate else ("FORCE_OFF" if force_off else "FAIL")
        print(f"{tag}: cov={row['coverage']*100:.2f}% ret={row['return']*100:.2f}% -> {status}")

    best = _pick_best([r for r in rows if r.get("role") == "sweep"], b0_return=ref_return, b0_coverage=ref_coverage)
    slow_up_row = next((r for r in rows if r.get("tag") == "slow_up_on"), None)
    slow_up_verdict = "disabled"
    if slow_up_row:
        if slow_up_row.get("slow_up_open", 0) == 0:
            slow_up_verdict = "no_opens_on_valid"
        elif slow_up_row.get("slow_up_force_disable"):
            slow_up_verdict = "force_disabled"
        else:
            slow_up_verdict = "enabled_ok"

    summary = {
        "checkpoint": str(M2_CKPT),
        "baseline_config": str(CORE_BASELINE),
        "b0_valid_reference": rows[0] if not args.skip_b0 else None,
        "baseline_valid": baseline,
        "gates": {
            "return_floor_delta": -0.005,
            "coverage_boost_delta": 0.015,
            "reference": ref_tag,
            "ref_return": ref_return,
            "ref_coverage": ref_coverage,
        },
        "rows": rows,
        "best_valid": best,
        "slow_up_verdict": slow_up_verdict,
        "phase1_ablation_pass": best is not None,
    }
    summary_path = OUT_PHASE1 / "phase1_ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# 027 Phase 1 Core Ablation (valid)",
        "",
        f"| baseline return | {baseline['return']*100:.2f}% |",
        f"| baseline coverage | {baseline['coverage']*100:.2f}% |",
        f"| gate return ≥ | {(ref_return - 0.005)*100:.2f}% |",
        f"| gate coverage ≥ | {(ref_coverage + 0.015)*100:.2f}% |",
        "",
        "| tag | return | coverage | slow_up | crash | gate |",
        "|-----|--------|----------|---------|-------|------|",
    ]
    for r in rows:
        if r.get("role") != "sweep":
            continue
        gate = "PASS" if r.get("phase1_gate") else ("OFF" if r.get("slow_up_force_disable") else "FAIL")
        lines.append(
            f"| {r['tag']} | {r['return']*100:.2f}% | {r['coverage']*100:.2f}% | "
            f"{r.get('slow_up_open', 0)} | {r.get('crash_short_count', 0)} | {gate} |"
        )
    lines.extend(
        [
            "",
            f"**slow_up 裁定**: {slow_up_verdict}",
            f"**best valid**: {best['tag'] if best else 'none'}",
            f"**Phase 1 ablation gate**: {'PASS' if summary['phase1_ablation_pass'] else 'FAIL'}",
            "",
            "## 复现",
            "```bash",
            "python examples/run_v027_core_ablation.py",
            "```",
        ]
    )
    (OUT_PHASE1 / "REPORT_027_PHASE1_ABLATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
