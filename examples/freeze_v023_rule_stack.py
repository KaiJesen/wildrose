#!/usr/bin/env python3
"""024 Phase 0: freeze 023 phase1c + teq_ceiling baselines for reproducible metrics."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PHASE1C_CONFIG = _ROOT / "configs/trading_rule_v023_phase1c_0062e.json"
TEQ_CEILING_CONFIG = _ROOT / "configs/trading_rule_v023_teq_ceiling_0062e.json"
PHASE1C_OUT = _ROOT / "backtest/v023_phase1c"
TEQ_CEILING_OUT = _ROOT / "backtest/v023_teq_ceiling"
FROZEN_OUT = _ROOT / "backtest/v023_rule_stack_frozen"
CHECKPOINT = "prod/v0.0.0/checkpoint/market_state_best.pt"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=_ROOT)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _ensure_phase1c() -> None:
    for split in ("valid", "test"):
        out = PHASE1C_OUT / split
        if not (out / "decisions.csv").is_file():
            _run([
                sys.executable,
                "examples/backtest_trading_system_v023.py",
                "--config",
                str(PHASE1C_CONFIG.relative_to(_ROOT)),
                "--checkpoint",
                CHECKPOINT,
                "--split",
                split,
                "--output-dir",
                str(out.relative_to(_ROOT)),
            ])
    part_path = PHASE1C_OUT / "participation_metrics.json"
    if not part_path.is_file():
        _run([
            sys.executable,
            "examples/eval_participation.py",
            "--backtest-dir",
            str((PHASE1C_OUT / "valid").relative_to(_ROOT)),
            "--backtest-dir",
            str((PHASE1C_OUT / "test").relative_to(_ROOT)),
            "--output",
            str(part_path.relative_to(_ROOT)),
        ])


def _ensure_teq_ceiling() -> None:
    out = TEQ_CEILING_OUT / "test"
    if not (out / "decisions.csv").is_file():
        _run([
            sys.executable,
            "examples/backtest_trading_system_v023.py",
            "--config",
            str(TEQ_CEILING_CONFIG.relative_to(_ROOT)),
            "--checkpoint",
            CHECKPOINT,
            "--split",
            "test",
            "--output-dir",
            str(out.relative_to(_ROOT)),
        ])
    part_path = TEQ_CEILING_OUT / "participation_metrics.json"
    if not part_path.is_file():
        _run([
            sys.executable,
            "examples/eval_participation.py",
            "--backtest-dir",
            str(out.relative_to(_ROOT)),
            "--output",
            str(part_path.relative_to(_ROOT)),
        ])


def _write_report() -> Path:
    phase1c_part = _read_json(PHASE1C_OUT / "participation_metrics.json")
    teq_part = _read_json(TEQ_CEILING_OUT / "participation_metrics.json")
    p1c_test = phase1c_part.get("test", {})
    p1c_rm = p1c_test.get("runner_metrics", {})
    p1c_pm = p1c_test.get("participation_metrics", {})
    teq_test = teq_part.get("test", {})
    teq_rm = teq_test.get("runner_metrics", {})
    teq_pm = teq_test.get("participation_metrics", {})

    FROZEN_OUT.mkdir(parents=True, exist_ok=True)
    frozen_part = FROZEN_OUT / "participation_metrics.json"
    frozen_part.write_text(
        json.dumps(
            {
                "phase1c": phase1c_part,
                "teq_ceiling": teq_part,
                "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    lines = [
        "# 023 Rule Stack Frozen Report",
        "",
        f"- timestamp: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC`",
        f"- checkpoint: `{CHECKPOINT}`",
        "",
        "## Config hashes",
        "",
        f"- phase1c: `{PHASE1C_CONFIG.relative_to(_ROOT)}` → `{_sha256(PHASE1C_CONFIG)}`",
        f"- teq_ceiling: `{TEQ_CEILING_CONFIG.relative_to(_ROOT)}` → `{_sha256(TEQ_CEILING_CONFIG)}`",
        "",
        "## phase1c test metrics",
        "",
        "| metric | value |",
        "|--------|-------|",
        f"| total_return | {_pct(float(p1c_rm.get('total_return', 0)))} |",
        f"| max_drawdown | {_pct(float(p1c_rm.get('max_drawdown', 0)))} |",
        f"| trade_count | {int(p1c_rm.get('trade_count', 0))} |",
        f"| trend_qualified_open_count | {int(p1c_rm.get('trend_qualified_open_count', 0))} |",
        f"| trend_qualified_pnl | {_pct(float(p1c_rm.get('trend_qualified_pnl', 0)))} |",
        f"| leg_count_coverage_ratio | {_pct(float(p1c_pm.get('leg_count_coverage_ratio', 0)))} |",
        f"| counter_leg_participation_count | {int(p1c_pm.get('counter_leg_participation_count', 0))} |",
        "",
        "## teq_ceiling test metrics",
        "",
        "| metric | value |",
        "|--------|-------|",
        f"| total_return | {_pct(float(teq_rm.get('total_return', 0)))} |",
        f"| max_drawdown | {_pct(float(teq_rm.get('max_drawdown', 0)))} |",
        f"| trade_count | {int(teq_rm.get('trade_count', 0))} |",
        f"| trend_qualified_open_count | {int(teq_rm.get('trend_qualified_open_count', 0))} |",
        f"| trend_qualified_pnl | {_pct(float(teq_rm.get('trend_qualified_pnl', 0)))} |",
        f"| leg_count_coverage_ratio | {_pct(float(teq_pm.get('leg_count_coverage_ratio', 0)))} |",
        f"| counter_leg_participation_count | {int(teq_pm.get('counter_leg_participation_count', 0))} |",
        "",
        "## Participation artifacts",
        "",
        f"- phase1c: `backtest/v023_phase1c/participation_metrics.json`",
        f"- teq_ceiling: `backtest/v023_teq_ceiling/participation_metrics.json`",
        f"- frozen bundle: `backtest/v023_rule_stack_frozen/participation_metrics.json`",
        "",
        "## Reproduction commands",
        "",
        "```bash",
        "python examples/run_v023_phase1c.py",
        "python examples/diagnose_v023_teq_ceiling.py",
        "python examples/freeze_v023_rule_stack.py",
        "```",
        "",
        "024 references **only** this report (or hash-aligned metrics) for 023 rule-stack numbers.",
        "",
    ]
    report = FROZEN_OUT / "REPORT_023_RULE_STACK_FROZEN.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {report}")
    return report


def main() -> int:
    _ensure_phase1c()
    _ensure_teq_ceiling()
    _write_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
