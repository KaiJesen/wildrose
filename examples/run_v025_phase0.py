#!/usr/bin/env python3
"""025 Phase 0: restore frozen B0 artifacts + reproduction gate."""

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

from _v025_common import (
    FROZEN_KLINE_CSV,
    PW20_CKPT,
    TEQ_CALIBRATION,
    kline_backtest_args,
    sha256_prefix,
    verify_pw20_checkpoint,
)

RECIPE = _ROOT / "configs/training_recipe_0065a_constrained.json"
PHASE1C = _ROOT / "configs/trading_rule_v023_phase1c_0062e.json"
B0_CONFIG = _ROOT / "configs/trading_rule_v024_phase1c_teq_0065a_c1_pw20.json"
PROD_CKPT = _ROOT / "prod/v0.0.0/checkpoint/market_state_best.pt"
C0_CKPT = _ROOT / "checkpoints/0065a_leg_align_c0/market_state_best.pt"
CALIBRATION = TEQ_CALIBRATION
OUT = _ROOT / "backtest/v025_phase0"
B0_RETURN = 0.0901
B0_COVERAGE = 0.267
B0_TEQ = 3
RETURN_TOL = 0.002
COV_TOL = 0.01


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=_ROOT)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _part_metrics(part: dict) -> dict:
    if "test" in part and isinstance(part["test"], dict):
        return part["test"].get("participation_metrics", {})
    return part.get("participation_metrics", {})


def _sha256(path: Path) -> str:
    return sha256_prefix(path)


def _train_c0(recipe: dict) -> None:
    v0 = recipe["variants"]["0"]
    td = recipe["train_defaults"]
    _run([
        sys.executable,
        "examples/train_market_state_0065a.py",
        "--variant",
        "0",
        "--init-checkpoint",
        recipe.get("init_checkpoint", str(PROD_CKPT.relative_to(_ROOT))),
        "--constraint-profile",
        recipe.get("constraint_profile", "constrained"),
        "--checkpoint-dir",
        v0["checkpoint_dir"],
        "--report-dir",
        v0["report_dir"],
        "--stride",
        str(td["stride"]),
        "--positive-oversample",
        str(td["positive_oversample"]),
        "--participation-weight",
        str(td["participation_weight"]),
        "--epochs",
        str(td["epochs"]),
        "--early-stop-patience",
        str(td["early_stop_patience"]),
        "--batch-size",
        str(td["batch_size"]),
        "--early-stop-metric",
        str(td.get("early_stop_metric", "composite")),
        "--cum-ic-min-ratio",
        str(td.get("cum_ic_min_ratio", 0.95)),
        "--auto-baseline-ic",
    ])


def _train_pw20() -> None:
    _run([
        sys.executable,
        "examples/train_market_state_0065a.py",
        "--variant",
        "1",
        "--init-checkpoint",
        str(C0_CKPT.relative_to(_ROOT)),
        "--constraint-profile",
        "constrained",
        "--checkpoint-dir",
        "checkpoints/0065a_leg_align_c1_pw20",
        "--report-dir",
        "reports/0065a_leg_align_c1_pw20",
        "--participation-weight",
        "2.0",
        "--stride",
        "1",
        "--positive-oversample",
        "30",
        "--epochs",
        "15",
        "--early-stop-patience",
        "6",
        "--batch-size",
        "32",
        "--early-stop-metric",
        "composite",
        "--cum-ic-min-ratio",
        "0.95",
        "--auto-baseline-ic",
    ])


def _calibrate_teq() -> None:
    CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
    _run([
        sys.executable,
        "examples/calibrate_teq_edge.py",
        "--checkpoint",
        str(PW20_CKPT.relative_to(_ROOT)),
        "--config",
        str(B0_CONFIG.relative_to(_ROOT)),
        "--output",
        str(CALIBRATION.relative_to(_ROOT)),
    ])


def _backtest_b0(split: str) -> Path:
    out = OUT / f"b0_{split}"
    _run([
        sys.executable,
        "examples/backtest_trading_system_v014.py",
        "--config",
        str(B0_CONFIG.relative_to(_ROOT)),
        "--checkpoint",
        str(PW20_CKPT.relative_to(_ROOT)),
        "--split",
        split,
        "--output-dir",
        str(out.relative_to(_ROOT)),
        *kline_backtest_args(),
    ])
    return out


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description="025 Phase 0 B0 reproduction")
    ap.add_argument("--skip-train", action="store_true", help="reuse existing checkpoint/calibration")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    recipe = _read_json(RECIPE)
    if not PROD_CKPT.is_file():
        raise FileNotFoundError(f"missing prod baseline checkpoint: {PROD_CKPT}")

    if not args.skip_train:
        if not C0_CKPT.is_file():
            print("restoring c0 checkpoint...")
            _train_c0(recipe)
        if not PW20_CKPT.is_file():
            print("restoring c1_pw20 checkpoint...")
            _train_pw20()
        if not CALIBRATION.is_file():
            print("fitting TEQ calibration...")
            _calibrate_teq()

    verify_pw20_checkpoint()
    if not CALIBRATION.is_file():
        raise FileNotFoundError(f"missing frozen TEQ calibration: {CALIBRATION}")

    b0_test = _backtest_b0("test")
    _run([
        sys.executable,
        "examples/eval_participation.py",
        "--backtest-dir",
        str(b0_test.relative_to(_ROOT)),
        "--output",
        str((OUT / "participation_b0_test.json").relative_to(_ROOT)),
    ])

    metrics = _read_json(b0_test / "metrics.json")
    part = _read_json(OUT / "participation_b0_test.json")
    cov = float(_part_metrics(part).get("leg_count_coverage_ratio", 0.0))
    ret = float(metrics.get("total_return", 0.0))
    teq = int(metrics.get("trend_qualified_open_count", 0))

    ret_ok = abs(ret - B0_RETURN) <= RETURN_TOL
    cov_ok = abs(cov - B0_COVERAGE) <= COV_TOL
    teq_ok = teq == B0_TEQ
    phase0_pass = ret_ok and cov_ok and teq_ok

    ckpt_hash = _sha256(PW20_CKPT)
    cfg_hash = _sha256(B0_CONFIG)

    lines = [
        "# 025 Phase 0 — B0 Reproduction",
        "",
        f"- checkpoint: `{PW20_CKPT.relative_to(_ROOT)}` (`{ckpt_hash}`)",
        f"- config: `{B0_CONFIG.relative_to(_ROOT)}` (`{cfg_hash}`)",
        f"- calibration: `{CALIBRATION.relative_to(_ROOT)}`",
        f"- frozen kline: `{FROZEN_KLINE_CSV.relative_to(_ROOT)}`",
        "",
        "## B0 test gate",
        "",
        "| metric | expected | actual | status |",
        "|--------|----------|--------|--------|",
        f"| total_return | {_pct(B0_RETURN)} ±{_pct(RETURN_TOL)} | {_pct(ret)} | **{'PASS' if ret_ok else 'FAIL'}** |",
        f"| leg_count_coverage | {_pct(B0_COVERAGE)} ±{_pct(COV_TOL)} | {_pct(cov)} | **{'PASS' if cov_ok else 'FAIL'}** |",
        f"| teq_open | {B0_TEQ} | {teq} | **{'PASS' if teq_ok else 'FAIL'}** |",
        "",
        f"**Phase 0 overall: {'PASS' if phase0_pass else 'FAIL'}**",
        "",
        "## Field对照（decisions.csv ↔ eval_participation）",
        "",
        "| decisions.csv | eval / metrics |",
        "|---------------|----------------|",
        "| edge_source | 通道归因（teq / slow_up / legacy） |",
        "| channel_threshold_snapshot | τ_slow 或 w_part 快照 |",
        "| participate_score_long | participation 主门 |",
        "| slow_up_edge_long | 慢涨通道 edge |",
        "| trend_qualified_open_count | metrics.json teq 开仓数 |",
        "| leg_count_coverage_ratio | participation_metrics |",
        "",
        "## Reproduction",
        "```bash",
        "python examples/run_v025_phase0.py",
        "```",
    ]
    report = OUT / "REPORT_025_PHASE0.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "phase0_summary.json").write_text(
        json.dumps(
            {
                "phase0_pass": phase0_pass,
                "return": ret,
                "coverage": cov,
                "teq_open": teq,
                "checkpoint_hash": ckpt_hash,
                "config_hash": cfg_hash,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {report}")
    return 0 if phase0_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
