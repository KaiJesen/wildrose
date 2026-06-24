#!/usr/bin/env python3
"""Full-system integration test: v022 trend module + trading PnL validation."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_EX = Path(__file__).resolve().parent
_ROOT = _EX.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))

BASELINE_CONFIG = _ROOT / "configs/trading_rule_v021_full_bias_0062e.json"
CANDIDATE_CONFIG = _ROOT / "configs/trading_rule_v022_trend_quality_0062e.json"
CHECKPOINT = "prod/v0.0.0/checkpoint/market_state_best.pt"


@dataclass
class Candidate:
    name: str
    trend_signal: dict = field(default_factory=dict)
    trend_bias: dict = field(default_factory=dict)


def _candidates() -> list[Candidate]:
    """System-level tuning grid on top of v021-hybrid base."""
    return [
        Candidate("current_p10_ir7", trend_signal={}),
        Candidate(
            "p10_ir7_chop",
            trend_signal={
                "chop_guard_enabled": True,
                "chop_efficiency_min": 0.3,
                "chop_range_atr_max": 2.5,
                "chop_flip_max": 4,
            },
        ),
        Candidate(
            "p10_ir8",
            trend_signal={"invalid_reset_bars": 8},
        ),
        Candidate(
            "p10_ir7_h3",
            trend_signal={"hold_confirm_score": 3},
        ),
        Candidate(
            "p10_ir7_bias_soft",
            trend_bias={"chop_soft_micro_weight": 0.35, "light_counter_size_penalty": 0.75},
        ),
    ]


def _write_config(base: dict, cand: Candidate, path: Path) -> None:
    cfg = copy.deepcopy(base)
    cfg["trend_signal"].update(cand.trend_signal)
    cfg["trend_bias"].update(cand.trend_bias)
    meta = cfg.setdefault("_022_meta", {})
    meta["system_test_candidate"] = cand.name
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _run_backtest(config: Path, split: str, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_ROOT / "examples/backtest_trading_system_v014.py"),
        "--config",
        str(config),
        "--checkpoint",
        CHECKPOINT,
        "--output-dir",
        str(out),
        "--split",
        split,
        "--symbol",
        "BTCUSDT",
        "--interval",
        "1h",
        "--days",
        "365",
    ]
    subprocess.check_call(cmd, cwd=_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.loads((out / "metrics.json").read_text(encoding="utf-8"))


def _run_module_eval(config: Path, split: str, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_ROOT / "examples/eval_trend_modules.py"),
        "--config",
        str(config),
        "--split",
        split,
        "--output-dir",
        str(out),
    ]
    subprocess.check_call(cmd, cwd=_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    payload = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    return payload.get("metrics", payload)


def _business_pass(base_m: dict, cand_m: dict) -> bool:
    tr_base = float(base_m.get("total_return", 0))
    tr_cand = float(cand_m.get("total_return", 0))
    dd_base = float(base_m.get("max_drawdown", 0))
    dd_cand = float(cand_m.get("max_drawdown", 0))
    tc_base = float(base_m.get("trade_count", 0))
    tc_cand = float(cand_m.get("trade_count", 0))
    return (
        tr_cand >= 0.7 * tr_base
        and dd_cand >= dd_base - 0.002
        and dd_cand >= dd_base * 1.2
        and tc_cand <= 1.5 * max(tc_base, 1)
    )


def _module_gate_count(m: dict) -> int:
    gates = 0
    if m.get("teacher_trend_coverage", 0) >= 0.65:
        gates += 1
    if m.get("confirmed_precision_vs_teacher", 0) >= 0.55:
        gates += 1
    if m.get("false_confirm_on_range_teacher", 1) <= 0.20:
        gates += 1
    if m.get("choppy_false_confirm_rate", 1) <= 0.05:
        gates += 1
    if m.get("broken_ratio", 1) <= 0.45:
        gates += 1
    if m.get("hard_block_long_ratio", 1) <= 0.30:
        gates += 1
    return gates


def _system_score(
    valid_bt: dict,
    test_bt: dict,
    v021_valid: dict,
    v021_test: dict,
    mod_valid: dict,
    mod_test: dict,
) -> float:
    ret_ratio = (
        float(valid_bt.get("total_return", 0)) / max(float(v021_valid.get("total_return", 0)), 1e-6)
        + float(test_bt.get("total_return", 0)) / max(float(v021_test.get("total_return", 0)), 1e-6)
    )
    mod = _module_gate_count(mod_valid) + _module_gate_count(mod_test)
    business = int(_business_pass(v021_valid, valid_bt)) + int(_business_pass(v021_test, test_bt))
    return 3.0 * ret_ratio + 0.5 * mod + 2.0 * business


def _integration_smoke() -> list[str]:
    import tempfile

    from trading_system.config import load_config
    from trading_system.engine import TradingEngine
    from trading_system.logger import TradeLogger

    cfg = load_config(CANDIDATE_CONFIG)
    with tempfile.TemporaryDirectory() as tmp:
        engine = TradingEngine(cfg, TradeLogger(Path(tmp)))
    checks = [
        engine.trend_signal_provider is not None,
        engine.trend_segment_engine is not None,
        engine.trend_bias_builder is not None,
        cfg.trend_signal.invalid_reset_bars == 7,
        cfg.trend_signal.persistence_lookback == 10,
        cfg.trend_bias.legacy_down_hard_block is False,
    ]
    return ["integration_smoke: PASS" if all(checks) else f"integration_smoke: FAIL {checks}"]


def _run_unit_tests() -> tuple[bool, str]:
    test_modules = [
        "tests.test_trend_signal_provider",
        "tests.test_trend_bias",
        "tests.test_trend_segment",
        "tests.test_trend_segment_golden",
    ]
    import importlib

    failed: list[str] = []
    count = 0
    for mod_name in test_modules:
        mod = importlib.import_module(mod_name)
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            count += 1
            try:
                fn()
            except Exception as exc:
                failed.append(f"{mod_name}.{name}: {exc}")
    if failed:
        return False, "unit_tests: FAIL\n" + "\n".join(failed)
    return True, f"unit_tests: PASS ({count} tests)"


def main() -> int:
    p = argparse.ArgumentParser(description="v022 full-system integration test")
    p.add_argument("--out-dir", default="backtest/v022_system_test")
    p.add_argument("--skip-tune", action="store_true", help="only run current config vs v021")
    p.add_argument("--skip-unit", action="store_true")
    args = p.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    lines = ["# v022 System Integration Test", "", f"- timestamp: `{time.strftime('%Y-%m-%d %H:%M:%S')}`", ""]
    lines.extend(_integration_smoke())

    if not args.skip_unit:
        ok, msg = _run_unit_tests()
        lines.append(msg)
        if not ok:
            lines.append("```")
            lines.append(msg)
            lines.append("```")

    base_cfg = json.loads(CANDIDATE_CONFIG.read_text(encoding="utf-8"))
    cfg_dir = out_root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    v021_valid = _run_backtest(BASELINE_CONFIG, "valid", out_root / "baseline/v021_valid")
    v021_test = _run_backtest(BASELINE_CONFIG, "test", out_root / "baseline/v021_test")

    candidates = [_candidates()[0]] if args.skip_tune else _candidates()
    results: list[dict] = []

    for cand in candidates:
        cfg_path = cfg_dir / f"{cand.name}.json"
        _write_config(base_cfg, cand, cfg_path)
        print(f"evaluating {cand.name}...")
        valid_bt = _run_backtest(cfg_path, "valid", out_root / cand.name / "valid_bt")
        test_bt = _run_backtest(cfg_path, "test", out_root / cand.name / "test_bt")
        mod_valid = _run_module_eval(cfg_path, "valid", out_root / cand.name / "mod_valid")
        mod_test = _run_module_eval(cfg_path, "test", out_root / cand.name / "mod_test")
        score = _system_score(valid_bt, test_bt, v021_valid, v021_test, mod_valid, mod_test)
        row = {
            "name": cand.name,
            "score": score,
            "valid_return": valid_bt.get("total_return"),
            "test_return": test_bt.get("total_return"),
            "valid_dd": valid_bt.get("max_drawdown"),
            "test_dd": test_bt.get("max_drawdown"),
            "valid_trades": valid_bt.get("trade_count"),
            "test_trades": test_bt.get("trade_count"),
            "business_valid": _business_pass(v021_valid, valid_bt),
            "business_test": _business_pass(v021_test, test_bt),
            "module_gates_valid": _module_gate_count(mod_valid),
            "module_gates_test": _module_gate_count(mod_test),
            "mod_valid": mod_valid,
            "mod_test": mod_test,
        }
        results.append(row)

    best = max(results, key=lambda r: r["score"])
    lines.extend(["", "## Baseline v021", ""])
    lines.append(f"| split | return | max_dd | trades |")
    lines.append(f"|-------|--------|--------|--------|")
    lines.append(
        f"| valid | {100*float(v021_valid['total_return']):.2f}% | {100*float(v021_valid['max_drawdown']):.2f}% | {int(v021_valid['trade_count'])} |"
    )
    lines.append(
        f"| test | {100*float(v021_test['total_return']):.2f}% | {100*float(v021_test['max_drawdown']):.2f}% | {int(v021_test['trade_count'])} |"
    )

    lines.extend(["", "## Candidate comparison", ""])
    lines.append("| candidate | valid ret | test ret | business | mod gates (v/t) | score |")
    lines.append("|-----------|-----------|----------|----------|-----------------|-------|")
    for r in sorted(results, key=lambda x: -x["score"]):
        biz = "PASS" if r["business_valid"] and r["business_test"] else "FAIL"
        lines.append(
            f"| {r['name']} | {100*float(r['valid_return']):.2f}% | {100*float(r['test_return']):.2f}% | {biz} | {r['module_gates_valid']}/{r['module_gates_test']} | {r['score']:.2f} |"
        )

    lines.extend(["", f"## Recommended: `{best['name']}`", ""])
    lines.append(
        f"- valid: {100*float(best['valid_return']):.2f}% (v021 {100*float(v021_valid['total_return']):.2f}%), "
        f"test: {100*float(best['test_return']):.2f}% (v021 {100*float(v021_test['total_return']):.2f}%)"
    )
    lines.append(f"- module gates valid/test: {best['module_gates_valid']}/6, {best['module_gates_test']}/6")
    lines.append(f"- elapsed: {time.time() - t0:.0f}s")

    summary = {
        "baseline": {"valid": v021_valid, "test": v021_test},
        "candidates": results,
        "best": best["name"],
        "elapsed_sec": time.time() - t0,
    }
    (out_root / "SYSTEM_TEST_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines[-8:]))
    print(f"saved: {out_root / 'SYSTEM_TEST_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
