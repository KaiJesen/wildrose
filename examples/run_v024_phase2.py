#!/usr/bin/env python3
"""024 Phase 2: calibrate TEQ edge + A0/A2 backtest + participation gates."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
PHASE1C_CONFIG = _ROOT / "configs/trading_rule_v023_phase1c_0062e.json"
TEQ_CONFIG = _ROOT / "configs/trading_rule_v024_phase1c_teq_0065a1.json"
BASELINE_CKPT = _ROOT / "prod/v0.0.0/checkpoint/market_state_best.pt"
TEQ_CKPT = _ROOT / "checkpoints/0065a_leg_align_v1/market_state_best.pt"
CALIBRATION = _ROOT / "backtest/v024_phase2/teq_edge_calibration.json"
OUT = _ROOT / "backtest/v024_phase2"
PHASE1C_BASELINE = _ROOT / "backtest/v023_phase1c/test/metrics.json"
PHASE1C_PART = _ROOT / "backtest/v023_phase1c/participation_metrics.json"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=_ROOT)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _part_split_metrics(part: dict, split: str) -> dict:
    if split in part and isinstance(part[split], dict):
        return part[split].get("participation_metrics", {})
    return part


def _gate_report(a0: dict, a2: dict, a2_part: dict, baseline_teq: float, baseline_part: dict) -> list[str]:
    lines = ["## Phase 2 gates (test)"]
    teq_a0 = float(a0.get("trend_qualified_open_count", 0))
    teq_a2 = float(a2.get("trend_qualified_open_count", 0))
    ratio = teq_a2 / max(1.0, baseline_teq)
    lines.append(f"- baseline phase1c teq opens (frozen): **{baseline_teq:.0f}**")
    lines.append(f"- A0 teq opens (0062e, teq off): **{teq_a0:.0f}**")
    lines.append(f"- A2 teq opens (0065a-1 + teq): **{teq_a2:.0f}** (ratio vs baseline {ratio:.2f}x)")
    lines.append(
        f"- gate teq trigger ≥2x baseline: **{'PASS' if ratio >= 2.0 else 'FAIL'}** (need ≥{baseline_teq * 2:.0f})"
    )
    counter_base = float(baseline_part.get("counter_leg_participation_count", 3))
    a2_test_part = _part_split_metrics(a2_part, "test")
    counter_a2 = float(a2_test_part.get("counter_leg_participation_count", 0))
    teq_counter = float(a2_part.get("teq_open_on_counter_leg_count", 0))
    lines.append(
        f"- counter_leg_participation_count A2={counter_a2:.0f} "
        f"(gate ≤ phase1c+2={counter_base + 2:.0f}): "
        f"**{'PASS' if counter_a2 <= counter_base + 2 else 'FAIL'}**"
    )
    lines.append(
        f"- teq_open_on_counter_leg_count A2={teq_counter:.0f} "
        f"(gate ≤ phase1c+1={counter_base + 1:.0f}): "
        f"**{'PASS' if teq_counter <= counter_base + 1 else 'FAIL'}**"
    )
    ret_a2 = float(a2.get("total_return", 0.0))
    ret_a0 = float(a0.get("total_return", 0.0))
    lines.append(f"- test return A0={ret_a0:.4f} A2={ret_a2:.4f}")
    cov_a2 = float(a2_test_part.get("leg_count_coverage_ratio", 0.0))
    lines.append(f"- leg_count_coverage_ratio A2={cov_a2:.4f}")
    return lines


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not TEQ_CKPT.is_file():
        raise FileNotFoundError(f"missing 0065a-1 checkpoint: {TEQ_CKPT}")
    if not TEQ_CONFIG.is_file():
        raise FileNotFoundError(f"missing teq config: {TEQ_CONFIG}")

    _run([
        sys.executable,
        "examples/calibrate_teq_edge.py",
        "--checkpoint",
        str(TEQ_CKPT.relative_to(_ROOT)),
        "--config",
        str(PHASE1C_CONFIG.relative_to(_ROOT)),
        "--output",
        str(CALIBRATION.relative_to(_ROOT)),
    ])

    for split in ("valid", "test"):
        _run([
            sys.executable,
            "examples/backtest_trading_system_v014.py",
            "--config",
            str(PHASE1C_CONFIG.relative_to(_ROOT)),
            "--checkpoint",
            str(BASELINE_CKPT.relative_to(_ROOT)),
            "--split",
            split,
            "--output-dir",
            str((OUT / f"a0_0062e_{split}").relative_to(_ROOT)),
        ])
        _run([
            sys.executable,
            "examples/backtest_trading_system_v014.py",
            "--config",
            str(TEQ_CONFIG.relative_to(_ROOT)),
            "--checkpoint",
            str(TEQ_CKPT.relative_to(_ROOT)),
            "--split",
            split,
            "--output-dir",
            str((OUT / f"a2_teq_{split}").relative_to(_ROOT)),
        ])

    a2_part_path = OUT / "participation_metrics.json"
    _run([
        sys.executable,
        "examples/eval_participation.py",
        "--backtest-dir",
        str((OUT / "a2_teq_valid").relative_to(_ROOT)),
        "--backtest-dir",
        str((OUT / "a2_teq_test").relative_to(_ROOT)),
        "--output",
        str(a2_part_path.relative_to(_ROOT)),
    ])

    _run([
        sys.executable,
        "examples/eval_model_participation.py",
        "--checkpoint",
        str(TEQ_CKPT.relative_to(_ROOT)),
        "--split",
        "valid",
        "--split",
        "test",
        "--backtest-dir",
        str((OUT / "a2_teq_valid").relative_to(_ROOT)),
        "--backtest-dir",
        str((OUT / "a2_teq_test").relative_to(_ROOT)),
        "--output",
        str((OUT / "eval_model_participation_a2.json").relative_to(_ROOT)),
    ])

    frozen = _read_json(PHASE1C_BASELINE)
    baseline_part = _read_json(PHASE1C_PART)
    baseline_teq = float(frozen.get("trend_qualified_open_count", 0))
    a0_test = _read_json(OUT / "a0_0062e_test" / "metrics.json")
    a2_test = _read_json(OUT / "a2_teq_test" / "metrics.json")
    a0_valid = _read_json(OUT / "a0_0062e_valid" / "metrics.json")
    a2_valid = _read_json(OUT / "a2_teq_valid" / "metrics.json")
    a2_part = _read_json(a2_part_path)
    model_eval = _read_json(OUT / "eval_model_participation_a2.json")

    report_lines = [
        "# 024 Phase 2 Report (TEQ edge wiring)",
        "",
        f"- teq config: `{TEQ_CONFIG.name}`",
        f"- calibration: `{CALIBRATION.relative_to(_ROOT)}`",
        f"- checkpoint A2: `{TEQ_CKPT.relative_to(_ROOT)}`",
        "",
        "## Valid split",
        f"- A0 teq opens: {a0_valid.get('trend_qualified_open_count', 'n/a')}",
        f"- A2 teq opens: {a2_valid.get('trend_qualified_open_count', 'n/a')}",
        "",
    ]
    if model_eval.get("splits"):
        for split in ("valid", "test"):
            sm = model_eval["splits"].get(split, {}).get("model_metrics", {})
            if sm:
                report_lines.append(
                    f"- model [{split}] part_auc={sm.get('participation_auc', 'n/a')} "
                    f"recall@5%={sm.get('leg_entry_recall_at_k', 'n/a')}"
                )
    report_lines.append("")
    report_lines.append("## Test split")
    report_lines.extend(_gate_report(a0_test, a2_test, a2_part, baseline_teq, baseline_part))
    report_path = OUT / "REPORT_024_PHASE2.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
