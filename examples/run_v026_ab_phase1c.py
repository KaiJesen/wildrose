#!/usr/bin/env python3
"""026 Phase 3: B0 vs M2 full-chain A/B on frozen phase1c rule stack."""

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

from _v025_common import PW20_CKPT, kline_backtest_args, sha256_prefix, verify_pw20_checkpoint

PHASE1_SUMMARY = _ROOT / "backtest/v026_phase1/phase1_summary.json"
B0_CONFIG = _ROOT / "configs/trading_rule_v024_phase1c_teq_0065a_c1_pw20.json"
M2_CKPT = _ROOT / "checkpoints/026_phase1_c1d1/market_state_best.pt"
OUT = _ROOT / "backtest/v026_phase3"
SWEEP = OUT / "teq_wp_sweep.json"
EXPLORE_RETURN = 0.0884
EXPLORE_COVERAGE = 0.28
B0_TEQ = 3
COUNTER_LEG_MAX_DELTA = 2


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=_ROOT)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _part_metrics(part: dict, split: str = "test") -> dict:
    if split in part and isinstance(part[split], dict):
        return part[split].get("participation_metrics", {})
    return part.get("participation_metrics", {})


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_ROOT.resolve()))
    except ValueError:
        return str(path)


def _backtest(name: str, *, config: Path, checkpoint: Path, split: str) -> Path:
    out = OUT / f"{name}_{split}"
    _run([
        sys.executable,
        "examples/backtest_trading_system_v014.py",
        "--config",
        _repo_rel(config),
        "--checkpoint",
        _repo_rel(checkpoint),
        "--split",
        split,
        "--output-dir",
        str(out.relative_to(_ROOT)),
        *kline_backtest_args(),
    ])
    return out


def _arm_row(name: str, bt_dir: Path, part: dict) -> dict:
    m = _read_json(bt_dir / "metrics.json")
    pm = _part_metrics(part, "test")
    return {
        "arm": name,
        "total_return": float(m.get("total_return", 0.0)),
        "max_drawdown": float(m.get("max_drawdown", 0.0)),
        "trade_count": int(m.get("trade_count", 0)),
        "teq_open": int(m.get("trend_qualified_open_count", 0)),
        "teq_pnl": float(m.get("trend_qualified_pnl", 0.0)),
        "coverage": float(pm.get("leg_count_coverage_ratio", 0.0)),
        "counter_leg": int(pm.get("counter_leg_participation_count", 0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="026 Phase 3 B0 vs M2 A/B")
    ap.add_argument("--skip-tune", action="store_true", help="reuse teq_wp_sweep.json")
    args = ap.parse_args()

    p1 = _read_json(PHASE1_SUMMARY)
    if not p1.get("phase1_pass"):
        raise SystemExit("Phase 1 gate not PASS — run examples/run_v026_phase1.py first")
    if not M2_CKPT.is_file():
        raise FileNotFoundError(M2_CKPT)
    if not PW20_CKPT.is_file():
        raise FileNotFoundError(PW20_CKPT)
    verify_pw20_checkpoint()

    OUT.mkdir(parents=True, exist_ok=True)
    if not args.skip_tune or not SWEEP.is_file():
        _run([sys.executable, "examples/tune_v026_teq_weights.py"])

    sweep = _read_json(SWEEP)
    best = sweep.get("best", {})
    m2_cfg = _ROOT / str(best.get("config", "backtest/v026_phase3/configs/m2_wp0.35.json"))
    if not m2_cfg.is_file():
        raise FileNotFoundError(f"missing M2 config after TEQ tune: {m2_cfg}")

    arms = {
        "b0": (B0_CONFIG, PW20_CKPT),
        "m2": (m2_cfg, M2_CKPT),
    }
    part_all: dict[str, dict] = {}
    for name, (cfg, ckpt) in arms.items():
        for split in ("valid", "test"):
            bt = _backtest(name, config=cfg, checkpoint=ckpt, split=split)
            part_path = OUT / f"participation_{name}_{split}.json"
            _run([
                sys.executable,
                "examples/eval_participation.py",
                "--backtest-dir",
                str(bt.relative_to(_ROOT)),
                "--output",
                str(part_path.relative_to(_ROOT)),
            ])
            part_all[f"{name}_{split}"] = _read_json(part_path)

    b0 = _arm_row("b0", OUT / "b0_test", part_all["b0_test"])
    m2 = _arm_row("m2", OUT / "m2_test", part_all["m2_test"])

    ret_ok = m2["total_return"] >= EXPLORE_RETURN
    cov_ok = m2["coverage"] >= EXPLORE_COVERAGE
    teq_ok = m2["teq_pnl"] >= 0
    dd_ok = m2["max_drawdown"] >= -0.025
    counter_ok = m2["counter_leg"] <= b0["counter_leg"] + COUNTER_LEG_MAX_DELTA
    explore_pass = ret_ok and cov_ok
    phase3_pass = explore_pass and teq_ok and dd_ok and counter_ok

    lines = [
        "# 026 Phase 3 — B0 vs M2 Full-Chain A/B",
        "",
        f"- B0 checkpoint: `{PW20_CKPT.relative_to(_ROOT)}` (`{sha256_prefix(PW20_CKPT)}`)",
        f"- M2 checkpoint: `{M2_CKPT.relative_to(_ROOT)}` (`{sha256_prefix(M2_CKPT)}`)",
        f"- M2 config: `{m2_cfg.relative_to(_ROOT)}` (w_part={best.get('w_part', 'n/a')})",
        f"- TEQ calibration: `{sweep.get('calibration', OUT / 'teq_edge_calibration.json')}`",
        "",
        "## Test metrics",
        "",
        "| arm | return | max_dd | trades | teq_open | teq_pnl | coverage | counter_leg |",
        "|-----|--------|--------|--------|----------|---------|----------|-------------|",
    ]
    for row in (b0, m2):
        lines.append(
            f"| {row['arm']} | {_pct(row['total_return'])} | {_pct(row['max_drawdown'])} | "
            f"{row['trade_count']} | {row['teq_open']} | {_pct(row['teq_pnl'])} | "
            f"{_pct(row['coverage'])} | {row['counter_leg']} |"
        )
    lines.extend([
        "",
        "## Exploration gate (M2 test)",
        "",
        f"| check | gate | M2 | status |",
        f"|-------|------|-----|--------|",
        f"| total_return | ≥ {_pct(EXPLORE_RETURN)} | {_pct(m2['total_return'])} | **{'PASS' if ret_ok else 'FAIL'}** |",
        f"| leg_count_coverage | ≥ {_pct(EXPLORE_COVERAGE)} | {_pct(m2['coverage'])} | **{'PASS' if cov_ok else 'FAIL'}** |",
        f"| teq_pnl | ≥ 0 | {_pct(m2['teq_pnl'])} | **{'PASS' if teq_ok else 'FAIL'}** |",
        f"| max_drawdown | ≥ -2.50% | {_pct(m2['max_drawdown'])} | **{'PASS' if dd_ok else 'FAIL'}** |",
        f"| counter_leg | ≤ B0+{COUNTER_LEG_MAX_DELTA} ({b0['counter_leg']}+{COUNTER_LEG_MAX_DELTA}) | {m2['counter_leg']} | **{'PASS' if counter_ok else 'FAIL'}** |",
        "",
        f"**Explore dual gate: {'PASS' if explore_pass else 'FAIL'}**",
        f"**Phase 3 overall: {'PASS' if phase3_pass else 'FAIL'}**",
        "",
        "## Reproduction",
        "```bash",
        "python examples/run_v026_ab_phase1c.py",
        "```",
    ])
    report = OUT / "REPORT_026_PHASE3.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "phase3_pass": phase3_pass,
        "explore_pass": explore_pass,
        "best_w_part": best.get("w_part"),
        "b0": b0,
        "m2": m2,
    }
    (OUT / "phase3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {report}")
    return 0 if phase3_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
