#!/usr/bin/env python3
"""025 Phase 2: B0 vs A3b matrix on frozen phase1c stack."""

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

from _v025_common import PW20_CKPT, kline_backtest_args, verify_pw20_checkpoint

PHASE0_SUMMARY = _ROOT / "backtest/v025_phase0/phase0_summary.json"
B0_CONFIG = _ROOT / "configs/trading_rule_v024_phase1c_teq_0065a_c1_pw20.json"
A3B_DEFAULT = _ROOT / "configs/trading_rule_v025_a3b_std_trend.json"
A3B_TUNED = _ROOT / "configs/trading_rule_v025_a3b_std_trend_tuned.json"
CKPT = PW20_CKPT
OUT = _ROOT / "backtest/v025_ab_a3b"
EXPLORE_RETURN = 0.0884
EXPLORE_COVERAGE = 0.28
INCR_COV_SUGGEST = 0.008


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=_ROOT)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _part_split_metrics(part: dict, split: str) -> dict:
    if split in part and isinstance(part[split], dict):
        return part[split].get("participation_metrics", {})
    return part.get("participation_metrics", {})


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _backtest(name: str, *, config: Path, split: str) -> Path:
    out = OUT / f"{name}_{split}"
    _run([
        sys.executable,
        "examples/backtest_trading_system_v014.py",
        "--config",
        str(config.relative_to(_ROOT)),
        "--checkpoint",
        str(CKPT.relative_to(_ROOT)),
        "--split",
        split,
        "--output-dir",
        str(out.relative_to(_ROOT)),
        *kline_backtest_args(),
    ])
    return out


def _arm_row(name: str, bt_dir: Path, part: dict) -> dict:
    m = _read_json(bt_dir / "metrics.json")
    pm = _part_split_metrics(part, "test")
    return {
        "arm": name,
        "total_return": float(m.get("total_return", 0.0)),
        "max_drawdown": float(m.get("max_drawdown", 0.0)),
        "trade_count": int(m.get("trade_count", 0)),
        "teq_open": int(m.get("trend_qualified_open_count", 0)),
        "slow_up_open": int(m.get("slow_up_open_count", 0)),
        "teq_pnl": float(m.get("trend_qualified_pnl", 0.0)),
        "coverage": float(pm.get("leg_count_coverage_ratio", 0.0)),
        "counter_leg": int(pm.get("counter_leg_participation_count", 0)),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="A3b config (default: tuned if present else a3b_std_trend.json)",
    )
    args = p.parse_args()

    p0 = _read_json(PHASE0_SUMMARY)
    if not p0.get("phase0_pass"):
        raise SystemExit("Phase 0 B0 reproduction gate not PASS — run examples/run_v025_phase0.py first")
    if not CKPT.is_file():
        raise FileNotFoundError(CKPT)
    verify_pw20_checkpoint()

    a3b_config = args.config
    if a3b_config is None:
        a3b_config = A3B_TUNED if A3B_TUNED.is_file() else A3B_DEFAULT

    OUT.mkdir(parents=True, exist_ok=True)
    arms = {"b0": B0_CONFIG, "a3b": a3b_config}
    part_all: dict[str, dict] = {}
    for name, cfg in arms.items():
        for split in ("valid", "test"):
            bt = _backtest(name, config=cfg, split=split)
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
    a3b = _arm_row("a3b", OUT / "a3b_test", part_all["a3b_test"])
    incr_cov = a3b["coverage"] - b0["coverage"]
    explore_pass = a3b["total_return"] >= EXPLORE_RETURN and a3b["coverage"] >= EXPLORE_COVERAGE

    lines = [
        "# 025 A/B Report (B0 vs A3b)",
        "",
        f"A3b config: `{a3b_config.relative_to(_ROOT)}`",
        "",
        "## Test metrics",
        "",
        "| arm | return | coverage | teq | trades | teq_pnl | counter_leg |",
        "|-----|--------|----------|-----|--------|---------|-------------|",
    ]
    for r in (b0, a3b):
        lines.append(
            f"| {r['arm']} | {_pct(r['total_return'])} | {_pct(r['coverage'])} | {r['teq_open']} | "
            f"{r['trade_count']} | {_pct(r['teq_pnl'])} | {r['counter_leg']} |"
        )
    lines.extend([
        "",
        "## A3b std_trend gate",
        f"- std_trend_incremental_coverage_pp: **{_pct(incr_cov)}** (suggest ≥ {_pct(INCR_COV_SUGGEST)})",
        f"- explore return ≥ {_pct(EXPLORE_RETURN)}: **{'PASS' if a3b['total_return'] >= EXPLORE_RETURN else 'FAIL'}**",
        f"- explore coverage ≥ {_pct(EXPLORE_COVERAGE)}: **{'PASS' if a3b['coverage'] >= EXPLORE_COVERAGE else 'FAIL'}**",
        f"- explore dual gate: **{'PASS' if explore_pass else 'FAIL'}**",
        "",
        "## Reproduction",
        "```bash",
        "python examples/run_v025_phase0.py --skip-train",
        "python examples/tune_v025_std_trend.py",
        "python examples/run_v025_ab_a3b.py",
        "```",
    ])
    report = OUT / "REPORT_025_A3B.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "ab_summary.json").write_text(
        json.dumps(
            {
                "b0": b0,
                "a3b": a3b,
                "incr_cov": incr_cov,
                "explore_pass": explore_pass,
                "a3b_config": str(a3b_config.relative_to(_ROOT)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {report}")
    return 0 if explore_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
