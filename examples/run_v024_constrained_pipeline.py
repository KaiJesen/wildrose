#!/usr/bin/env python3
"""024 constrained pipeline: c1 train → TEQ calibrate → Phase 2/3 gates."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
RECIPE = _ROOT / "configs/training_recipe_0065a_constrained.json"
PHASE1C = _ROOT / "configs/trading_rule_v023_phase1c_0062e.json"
TEQ_CONFIG = _ROOT / "configs/trading_rule_v024_phase1c_teq_0065a_c1.json"
BASELINE_CKPT = _ROOT / "prod/v0.0.0/checkpoint/market_state_best.pt"
C0_CKPT = _ROOT / "checkpoints/0065a_leg_align_c0/market_state_best.pt"
C1_CKPT = _ROOT / "checkpoints/0065a_leg_align_c1/market_state_best.pt"
CALIBRATION = _ROOT / "backtest/v024_constrained/teq_edge_calibration.json"
OUT = _ROOT / "backtest/v024_constrained"
PHASE2_OUT = _ROOT / "backtest/v024_phase2_c1"
PHASE3_OUT = _ROOT / "backtest/v024_phase3_constrained"
EXPLORE_RETURN = 0.0884
EXPLORE_COVERAGE = 0.28


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=_ROOT)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _part_test(part: dict) -> dict:
    if "test" in part and isinstance(part["test"], dict):
        return part["test"].get("participation_metrics", {})
    return part


def _train_c1(recipe: dict) -> None:
    v1 = recipe["variants"]["1"]
    td = recipe["train_defaults"]
    init = v1.get("init_from", str(C0_CKPT.relative_to(_ROOT)))
    cmd = [
        sys.executable,
        "examples/train_market_state_0065a.py",
        "--variant",
        "1",
        "--init-checkpoint",
        init,
        "--constraint-profile",
        recipe.get("constraint_profile", "constrained"),
        "--checkpoint-dir",
        v1["checkpoint_dir"],
        "--report-dir",
        v1["report_dir"],
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
    ]
    _run(cmd)


def _ensure_teq_config() -> None:
    base = _read_json(PHASE1C)
    base["_024_meta"] = {
        "phase": "2c",
        "recipe": "teq_edge_0065a_c1_constrained",
        "source": PHASE1C.name,
    }
    base["teq_edge"] = {
        "enabled": True,
        "weight_edge_5": 0.35,
        "weight_edge_24": 0.45,
        "weight_participation": 0.20,
        "calibration_path": str(CALIBRATION.relative_to(_ROOT)),
        "use_calibrated": True,
        "model_checkpoint": str(C1_CKPT.relative_to(_ROOT)),
    }
    TEQ_CONFIG.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")


def _smoke_backtest(name: str, *, config: Path, ckpt: Path, split: str = "test") -> dict:
    out = OUT / name
    _run([
        sys.executable,
        "examples/backtest_trading_system_v014.py",
        "--config",
        str(config.relative_to(_ROOT)),
        "--checkpoint",
        str(ckpt.relative_to(_ROOT)),
        "--split",
        split,
        "--output-dir",
        str(out.relative_to(_ROOT)),
    ])
    return _read_json(out / "metrics.json")


def main() -> int:
    if not C0_CKPT.is_file():
        raise FileNotFoundError(f"run c0 first: {C0_CKPT}")
    recipe = _read_json(RECIPE)
    OUT.mkdir(parents=True, exist_ok=True)
    PHASE2_OUT.mkdir(parents=True, exist_ok=True)
    PHASE3_OUT.mkdir(parents=True, exist_ok=True)

    _train_c1(recipe)
    if not C1_CKPT.is_file():
        raise FileNotFoundError(C1_CKPT)

    _run([
        sys.executable,
        "examples/eval_model_participation.py",
        "--checkpoint",
        str(C1_CKPT.relative_to(_ROOT)),
        "--output",
        str((OUT / "eval_c1.json").relative_to(_ROOT)),
    ])

    a0 = _smoke_backtest("a0_0062e_test", config=PHASE1C, ckpt=BASELINE_CKPT)
    a1_c0 = _smoke_backtest("a1_c0_test", config=PHASE1C, ckpt=C0_CKPT)
    a1_c1 = _smoke_backtest("a1_c1_test", config=PHASE1C, ckpt=C1_CKPT)

    _ensure_teq_config()
    CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
    _run([
        sys.executable,
        "examples/calibrate_teq_edge.py",
        "--checkpoint",
        str(C1_CKPT.relative_to(_ROOT)),
        "--config",
        str(PHASE1C.relative_to(_ROOT)),
        "--output",
        str(CALIBRATION.relative_to(_ROOT)),
    ])

    a2_teq = _smoke_backtest("a2_c1_teq_test", config=TEQ_CONFIG, ckpt=C1_CKPT)

    for split in ("valid", "test"):
        _smoke_backtest(f"a2_teq_{split}", config=TEQ_CONFIG, ckpt=C1_CKPT, split=split)
        dst = PHASE2_OUT / f"a2_teq_{split}"
        src = OUT / f"a2_teq_{split}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    part_path = PHASE2_OUT / "participation_metrics.json"
    _run([
        sys.executable,
        "examples/eval_participation.py",
        "--backtest-dir",
        str((PHASE2_OUT / "a2_teq_valid").relative_to(_ROOT)),
        "--backtest-dir",
        str((PHASE2_OUT / "a2_teq_test").relative_to(_ROOT)),
        "--output",
        str(part_path.relative_to(_ROOT)),
    ])

    c1_metrics = _read_json(_ROOT / recipe["variants"]["1"]["report_dir"] / "metrics.json")
    a2_part = _read_json(part_path)
    a2_part_test = _part_test(a2_part)
    frozen_teq = float(_read_json(_ROOT / "backtest/v023_phase1c/test/metrics.json").get("trend_qualified_open_count", 1))
    max_drop = float(recipe["gates"].get("a1_backtest_return_max_drop", 0.02))

    ret_a0 = float(a0.get("total_return", 0))
    ret_c0 = float(a1_c0.get("total_return", 0))
    ret_c1 = float(a1_c1.get("total_return", 0))
    ret_a2 = float(a2_teq.get("total_return", 0))
    teq_a2 = int(a2_teq.get("trend_qualified_open_count", 0))
    cov_a2 = float(a2_part_test.get("leg_count_coverage_ratio", 0))

    drift_c0 = ret_c0 >= ret_a0 - max_drop
    drift_c1 = ret_c1 >= ret_a0 - max_drop
    teq_ratio = teq_a2 / max(1.0, frozen_teq)
    explore_pass = ret_a2 >= EXPLORE_RETURN and cov_a2 >= EXPLORE_COVERAGE
    part_auc = float(c1_metrics.get("valid_best_score", 0))

    lines = [
        "# 024 Constrained Pipeline Report (c1 + TEQ)",
        "",
        f"- c1 checkpoint: `{C1_CKPT.relative_to(_ROOT)}`",
        f"- teq config: `{TEQ_CONFIG.name}`",
        "",
        "## Model gates",
        f"- valid participation_auc: **{part_auc:.4f}** ({'PASS' if part_auc >= 0.55 else 'FAIL'} ≥0.55)",
        f"- baseline cum_return_ic: {c1_metrics.get('baseline_cum_return_ic', 'n/a')}",
        "",
        "## Drift gate (A1 vs A0, test)",
        f"| arm | return | trades | teq | drift ≤2pp |",
        f"|-----|--------|--------|-----|------------|",
        f"| A0 0062e | {_pct(ret_a0)} | {int(a0.get('trade_count',0))} | {int(a0.get('trend_qualified_open_count',0))} | baseline |",
        f"| A1 c0 | {_pct(ret_c0)} | {int(a1_c0.get('trade_count',0))} | {int(a1_c0.get('trend_qualified_open_count',0))} | **{'PASS' if drift_c0 else 'FAIL'}** |",
        f"| A1 c1 | {_pct(ret_c1)} | {int(a1_c1.get('trade_count',0))} | {int(a1_c1.get('trend_qualified_open_count',0))} | **{'PASS' if drift_c1 else 'FAIL'}** |",
        "",
        "## Phase 2 TEQ (test)",
        f"- A2 teq opens: **{teq_a2}** vs frozen baseline **{int(frozen_teq)}** ({teq_ratio:.2f}x, gate ≥2x: **{'PASS' if teq_ratio >= 2 else 'FAIL'}**)",
        f"- A2 return: **{_pct(ret_a2)}** (A0 {_pct(ret_a0)})",
        f"- counter_leg_participation: {int(a2_part_test.get('counter_leg_participation_count',0))}",
        "",
        "## Phase 3 exploration line (A2)",
        f"- return ≥ {_pct(EXPLORE_RETURN)}: **{'PASS' if ret_a2 >= EXPLORE_RETURN else 'FAIL'}** ({_pct(ret_a2)})",
        f"- coverage ≥ {_pct(EXPLORE_COVERAGE)}: **{'PASS' if cov_a2 >= EXPLORE_COVERAGE else 'FAIL'}** ({_pct(cov_a2)})",
        f"- overall: **{'PASS' if explore_pass else 'FAIL'}**",
        "",
        "## Reproduction",
        "```bash",
        "python examples/run_v024_constrained_pipeline.py",
        "```",
    ]
    report = OUT / "REPORT_024_CONSTRAINED_PIPELINE.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "ret_a0": ret_a0,
        "ret_c0": ret_c0,
        "ret_c1": ret_c1,
        "ret_a2_teq": ret_a2,
        "drift_c0": drift_c0,
        "drift_c1": drift_c1,
        "teq_ratio": teq_ratio,
        "explore_pass": explore_pass,
        "c1_metrics": c1_metrics,
    }
    (OUT / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
