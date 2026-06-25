#!/usr/bin/env python3
"""024: train 0065a with drift constraints + smoke A1 backtest vs 0062e."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
RECIPE = _ROOT / "configs/training_recipe_0065a_constrained.json"
PHASE1C = _ROOT / "configs/trading_rule_v023_phase1c_0062e.json"
BASELINE_CKPT = _ROOT / "prod/v0.0.0/checkpoint/market_state_best.pt"
OUT = _ROOT / "backtest/v024_0065a_constrained"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=_ROOT)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _train_variant(recipe: dict, variant: str, *, init_ckpt: str) -> Path:
    v = recipe["variants"][variant]
    td = recipe["train_defaults"]
    ckpt = _ROOT / v["checkpoint_dir"] / "market_state_best.pt"
    cmd = [
        sys.executable,
        "examples/train_market_state_0065a.py",
        "--variant",
        variant,
        "--init-checkpoint",
        init_ckpt,
        "--constraint-profile",
        recipe.get("constraint_profile", "constrained"),
        "--checkpoint-dir",
        v["checkpoint_dir"],
        "--report-dir",
        v.get("report_dir", f"reports/0065a_leg_align_c{variant}"),
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
    ]
    if td.get("auto_baseline_ic", True):
        cmd.append("--auto-baseline-ic")
    _run(cmd)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    return ckpt


def main() -> int:
    recipe = _read_json(RECIPE)
    OUT.mkdir(parents=True, exist_ok=True)
    init = recipe.get("init_checkpoint", str(BASELINE_CKPT.relative_to(_ROOT)))

    c0 = _train_variant(recipe, "0", init_ckpt=init)
    _run([
        sys.executable,
        "examples/eval_model_participation.py",
        "--checkpoint",
        str(c0.relative_to(_ROOT)),
        "--output",
        str((OUT / "eval_c0.json").relative_to(_ROOT)),
    ])

    _run([
        sys.executable,
        "examples/backtest_trading_system_v014.py",
        "--config",
        str(PHASE1C.relative_to(_ROOT)),
        "--checkpoint",
        str(BASELINE_CKPT.relative_to(_ROOT)),
        "--split",
        "test",
        "--output-dir",
        str((OUT / "a0_baseline_test").relative_to(_ROOT)),
    ])
    _run([
        sys.executable,
        "examples/backtest_trading_system_v014.py",
        "--config",
        str(PHASE1C.relative_to(_ROOT)),
        "--checkpoint",
        str(c0.relative_to(_ROOT)),
        "--split",
        "test",
        "--output-dir",
        str((OUT / "a1_constrained_c0_test").relative_to(_ROOT)),
    ])

    a0 = _read_json(OUT / "a0_baseline_test" / "metrics.json")
    a1 = _read_json(OUT / "a1_constrained_c0_test" / "metrics.json")
    c0_metrics = _read_json(_ROOT / recipe["variants"]["0"]["report_dir"] / "metrics.json")
    eval_c0 = _read_json(OUT / "eval_c0.json")

    ret_a0 = float(a0.get("total_return", 0.0))
    ret_a1 = float(a1.get("total_return", 0.0))
    max_drop = float(recipe["gates"].get("a1_backtest_return_max_drop", 0.02))
    drift_ok = ret_a1 >= ret_a0 - max_drop

    lines = [
        "# 024 0065a Constrained Training Report",
        "",
        f"- profile: `{recipe.get('constraint_profile')}`",
        f"- checkpoint: `{c0.relative_to(_ROOT)}`",
        "",
        "## Model (valid)",
        f"- participation_auc: {c0_metrics.get('valid_best_score', 'n/a')}",
        f"- baseline cum_return_ic: {c0_metrics.get('baseline_cum_return_ic', 'n/a')}",
        f"- test part_auc: {c0_metrics.get('test_metrics', {}).get('participation_auc', 'n/a')}",
        "",
        "## A1 smoke backtest (phase1c, test)",
        f"- A0 (0062e) return: {ret_a0:.4f} trades: {int(a0.get('trade_count', 0))}",
        f"- A1 (c0) return: {ret_a1:.4f} trades: {int(a1.get('trade_count', 0))} teq: {int(a1.get('trend_qualified_open_count', 0))}",
        f"- drift gate (A1 >= A0 - 2pp): **{'PASS' if drift_ok else 'FAIL'}**",
        "",
        "## Reproduction",
        "```bash",
        "python examples/run_v024_0065a_constrained.py",
        "```",
    ]
    report = OUT / "REPORT_0065a_CONSTRAINED.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(
        json.dumps(
            {
                "a0_test_return": ret_a0,
                "a1_test_return": ret_a1,
                "drift_gate_pass": drift_ok,
                "c0_metrics": c0_metrics,
                "eval_c0": eval_c0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {report}")
    return 0 if drift_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
