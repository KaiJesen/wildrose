#!/usr/bin/env python3
"""026 Phase 0: B0 reproduction gate + C3 ParticipationAttn training/eval."""

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

from _v025_common import PW20_CKPT, sha256_prefix, verify_pw20_checkpoint

RECIPE = _ROOT / "configs/training_recipe_026_phase0_c3.json"
OUT = _ROOT / "backtest/v026_phase0"
C3_CKPT = _ROOT / "checkpoints/026_phase0_c3/market_state_best.pt"
C3_REPORT = _ROOT / "reports/026_phase0_c3/metrics.json"
EVAL_JSON = OUT / "eval_model_participation_c3.json"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=_ROOT)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _run_b0_gate() -> dict:
    _run(["bash", "prod/v1.1.1/scripts/verify_phase0.sh", str(OUT / "b0_smoke")])
    verify_pw20_checkpoint()
    smoke = OUT / "b0_smoke"
    metrics = _read_json(smoke / "metrics.json")
    part = _read_json(smoke / "participation.json")
    cov = float(part["test"]["participation_metrics"]["leg_count_coverage_ratio"])
    ret = float(metrics["total_return"])
    teq = int(metrics["trend_qualified_open_count"])
    gates = _read_json(RECIPE).get("gates", {})
    ret_ok = abs(ret - float(gates.get("b0_return", 0.0901))) <= 0.002
    cov_ok = abs(cov - float(gates.get("b0_coverage", 0.267))) <= 0.01
    teq_ok = teq == int(gates.get("b0_teq", 3))
    return {
        "return": ret,
        "coverage": cov,
        "teq_open": teq,
        "pass": ret_ok and cov_ok and teq_ok,
        "ret_ok": ret_ok,
        "cov_ok": cov_ok,
        "teq_ok": teq_ok,
        "checkpoint_hash": sha256_prefix(PW20_CKPT),
    }


def _train_c3(skip_train: bool) -> None:
    if skip_train and C3_CKPT.is_file():
        print(f"reusing C3 checkpoint: {C3_CKPT}")
        return
    _run([sys.executable, "examples/train_v026_phase0_c3.py", "--recipe", str(RECIPE.relative_to(_ROOT))])


def _eval_c3() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    _run([
        sys.executable,
        "examples/eval_model_participation.py",
        "--checkpoint",
        str(C3_CKPT.relative_to(_ROOT)),
        "--output",
        str(EVAL_JSON.relative_to(_ROOT)),
    ])
    return _read_json(EVAL_JSON)


def _eval_split_metrics(eval_out: dict, split: str = "valid") -> dict:
    splits = eval_out.get("splits", eval_out)
    block = splits.get(split, {})
    return block.get("model_metrics", block)


def _baseline_ic_from_ckpt() -> float:
    if not C3_CKPT.is_file():
        return 0.0
    import torch
    from transformer_kit.train_utils import load_checkpoint
    ck = load_checkpoint(C3_CKPT, map_location="cpu")
    if "baseline_cum_return_ic" in ck:
        return float(ck["baseline_cum_return_ic"])
    return float(ck.get("metrics", {}).get("cum_return_ic", 0.0))


def main() -> int:
    ap = argparse.ArgumentParser(description="026 Phase 0 pipeline")
    ap.add_argument("--skip-train", action="store_true", help="reuse existing C3 checkpoint")
    ap.add_argument("--skip-b0", action="store_true", help="skip B0 verify (dev only)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    recipe = _read_json(RECIPE)
    gates = recipe.get("gates", {})

    b0 = {"pass": True, "return": 0.0, "coverage": 0.0, "teq_open": 0, "checkpoint_hash": ""}
    if not args.skip_b0:
        b0 = _run_b0_gate()

    _train_c3(args.skip_train)
    if not C3_CKPT.is_file():
        raise FileNotFoundError(f"missing C3 checkpoint: {C3_CKPT}")

    eval_out = _eval_c3()
    train_metrics = _read_json(C3_REPORT)
    valid_mm = _eval_split_metrics(eval_out, "valid")
    valid_auc = float(valid_mm.get("participation_auc", 0.0))
    baseline_ic = float(train_metrics.get("baseline_cum_return_ic", 0.0))
    if baseline_ic <= 0:
        baseline_ic = _baseline_ic_from_ckpt()
    valid_ic = float(train_metrics.get("valid_cum_return_ic", valid_mm.get("cum_return_ic", 0.0)))
    ic_deg = float(train_metrics.get("valid_ic_degradation", 0.0))
    if baseline_ic > 0 and (ic_deg == 0.0 or valid_ic > 0):
        ic_deg = max(0.0, (baseline_ic - valid_ic) / baseline_ic)

    part_min = float(gates.get("participation_auc_valid_min", 0.60))
    ic_max_deg = float(gates.get("cum_return_ic_degradation_max", 0.08))
    c3_part_ok = valid_auc >= part_min
    c3_ic_ok = ic_deg <= ic_max_deg
    c3_pass = c3_part_ok and c3_ic_ok
    phase0_pass = b0["pass"] and c3_pass

    ckpt_hash = sha256_prefix(C3_CKPT)
    lines = [
        "# 026 Phase 0 — B0 Reproduction + C3 ParticipationAttn",
        "",
        "## B0 gate",
        "",
        "| metric | expected | actual | status |",
        "|--------|----------|--------|--------|",
        f"| total_return | {_pct(float(gates.get('b0_return', 0.0901)))} ±0.20pp | {_pct(b0['return'])} | **{'PASS' if b0.get('ret_ok', b0['pass']) else 'FAIL'}** |",
        f"| leg_count_coverage | {_pct(float(gates.get('b0_coverage', 0.267)))} ±1.00pp | {_pct(b0['coverage'])} | **{'PASS' if b0.get('cov_ok', b0['pass']) else 'FAIL'}** |",
        f"| teq_open | {gates.get('b0_teq', 3)} | {b0['teq_open']} | **{'PASS' if b0.get('teq_ok', b0['pass']) else 'FAIL'}** |",
        f"| B0 checkpoint hash | — | `{b0.get('checkpoint_hash', '')}` | — |",
        "",
        f"**B0 overall: {'PASS' if b0['pass'] else 'FAIL'}**",
        "",
        "## C3 model track",
        "",
        f"- init: `prod/v1.1.1/checkpoint/market_state_best.pt`",
        f"- C3 checkpoint: `{C3_CKPT.relative_to(_ROOT)}` (`{ckpt_hash}`)",
        f"- recipe: `{RECIPE.relative_to(_ROOT)}`",
        "",
        "| metric | gate | actual | status |",
        "|--------|------|--------|--------|",
        f"| valid participation_auc | ≥ {part_min:.2f} | {valid_auc:.4f} | **{'PASS' if c3_part_ok else 'FAIL'}** |",
        f"| cum_return_ic degradation | ≤ {_pct(ic_max_deg)} | {_pct(ic_deg)} | **{'PASS' if c3_ic_ok else 'FAIL'}** |",
        f"| baseline cum_return_ic | — | {baseline_ic:.4f} | — |",
        f"| valid cum_return_ic | — | {valid_ic:.4f} | — |",
        "",
        f"**C3 overall: {'PASS' if c3_pass else 'FAIL'}**",
        "",
        "## Phase 0 routing (§4.1.1)",
        "",
    ]
    if valid_auc >= 0.65:
        lines.append("- C3 ≥ 0.65 → 可跳过 Phase 1 主路径（建议 1 个 C1+D1 消融对照臂）")
    elif valid_auc >= 0.62:
        lines.append("- C3 ∈ [0.62, 0.65) → **必须** Phase 1 C1+D1")
    elif valid_auc >= 0.60:
        lines.append("- C3 ∈ [0.60, 0.62) → **必须** Phase 1 C1+D1 + Phase 2 A1")
    else:
        lines.append("- C3 < 0.60 → 并行 A1 重评或架构师裁定 F2/结案")
    lines.extend([
        "",
        f"**Phase 0 overall: {'PASS' if phase0_pass else 'FAIL'}**",
        "",
        "## Reproduction",
        "```bash",
        "python examples/run_v026_phase0.py",
        "```",
    ])
    report = OUT / "REPORT_026_PHASE0.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "phase0_pass": phase0_pass,
        "b0": b0,
        "c3": {
            "valid_participation_auc": valid_auc,
            "ic_degradation": ic_deg,
            "pass": c3_pass,
            "checkpoint_hash": ckpt_hash,
        },
    }
    (OUT / "phase0_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {report}")
    return 0 if phase0_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
