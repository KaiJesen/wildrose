#!/usr/bin/env python3
"""023 Phase 3.1: regime relax on trend-qualified only (+ phase2 hold extension)."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
CONFIG = _ROOT / "configs/trading_rule_v023_phase31_0062e.json"
OUT = _ROOT / "backtest/v023_phase31"
BASELINE = _ROOT / "backtest/v023_baseline/participation_metrics.json"
PHASE1C = _ROOT / "backtest/v023_phase1c/participation_metrics.json"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=_ROOT)


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def main() -> int:
    for split in ("valid", "test"):
        _run([
            sys.executable, "examples/backtest_trading_system_v023.py",
            "--config", str(CONFIG.relative_to(_ROOT)),
            "--split", split,
            "--output-dir", str((OUT / split).relative_to(_ROOT)),
        ])
    _run([
        sys.executable, "examples/eval_participation.py",
        "--backtest-dir", str((OUT / "valid").relative_to(_ROOT)),
        "--backtest-dir", str((OUT / "test").relative_to(_ROOT)),
        "--output", str((OUT / "participation_metrics.json").relative_to(_ROOT)),
    ])
    results = json.loads((OUT / "participation_metrics.json").read_text())
    base = json.loads(BASELINE.read_text()) if BASELINE.is_file() else {}
    p1c = json.loads(PHASE1C.read_text()) if PHASE1C.is_file() else {}
    test = results.get("test", {}).get("runner_metrics", {})
    part = results.get("test", {}).get("participation_metrics", {})
    bm = base.get("test", {}).get("runner_metrics", {})
    c1 = p1c.get("test", {}).get("runner_metrics", {})
    leg_cov = float(part.get("leg_count_coverage_ratio", 0))
    lines = [
        "# 023 Phase 3.1 Report",
        "",
        f"- timestamp: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC`",
        f"- config: `{CONFIG.relative_to(_ROOT)}`",
        "- note: `regime_threshold.apply_to_standard_opens=false`",
        "",
        "| metric | baseline | phase1c | phase3.1 |",
        "|--------|----------|---------|----------|",
    ]
    for k in ("total_return", "max_drawdown", "trade_count", "trend_qualified_open_count",
              "trend_qualified_pnl", "leg_coverage_ratio"):
        a, b, c = float(bm.get(k, 0)), float(c1.get(k, 0)), float(test.get(k, 0))
        if k == "trend_qualified_pnl":
            lines.append(f"| {k} | — | {_pct(b)} | {_pct(c)} |")
        elif "ratio" in k or k in ("total_return", "max_drawdown"):
            lines.append(f"| {k} | {_pct(a)} | {_pct(b)} | {_pct(c)} |")
        else:
            lines.append(f"| {k} | {a:.0f} | {b:.0f} | {c:.0f} |")
    lines += [
        "",
        f"- leg_count_coverage_ratio: {_pct(leg_cov)} (target ≥ 35%)",
        f"- coverage gate: {'PASS' if leg_cov >= 0.35 else 'FAIL'}",
        f"- return gate (≥95% baseline): {'PASS' if float(test.get('total_return',0)) >= float(bm.get('total_return',0))*0.95 else 'FAIL'}",
    ]
    (OUT / "REPORT_023_PHASE31.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {OUT / 'REPORT_023_PHASE31.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
