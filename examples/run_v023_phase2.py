#!/usr/bin/env python3
"""023 Phase 2: hold extension + leg alignment on phase1c baseline."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
CONFIG = _ROOT / "configs/trading_rule_v023_phase2_0062e.json"
OUT = _ROOT / "backtest/v023_phase2"
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
    leg_cov = float(part.get("leg_count_coverage_ratio", test.get("leg_coverage_ratio", 0.0)))
    lines = [
        "# 023 Phase 2 Report",
        "",
        f"- timestamp: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC`",
        f"- config: `{CONFIG.relative_to(_ROOT)}`",
        "",
        "| metric | baseline | phase1c | phase2 |",
        "|--------|----------|---------|--------|",
    ]
    keys = (
        "total_return", "max_drawdown", "trade_count", "avg_trend_hold_bars",
        "trend_qualified_open_count", "trend_qualified_pnl", "leg_coverage_ratio",
        "short_trend_capture_ratio",
    )
    for k in keys:
        a = float(bm.get(k, 0))
        b = float(c1.get(k, 0))
        c = float(test.get(k, 0))
        if k == "trend_qualified_pnl":
            lines.append(f"| {k} | — | {_pct(b)} | {_pct(c)} |")
        elif "ratio" in k or k in ("total_return", "max_drawdown"):
            lines.append(f"| {k} | {_pct(a)} | {_pct(b)} | {_pct(c)} |")
        else:
            lines.append(f"| {k} | {a:.1f} | {b:.1f} | {c:.1f} |")
    lines.extend([
        "",
        f"- leg_count_coverage_ratio: {_pct(leg_cov)} (target ≥ 35%)",
        f"- leg_count_coverage gate: {'PASS' if leg_cov >= 0.35 else 'FAIL'}",
        f"- total_return ≥ baseline×0.95: {'PASS' if float(test.get('total_return', 0)) >= float(bm.get('total_return', 0)) * 0.95 else 'FAIL'}",
    ])
    (OUT / "REPORT_023_PHASE2.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {OUT / 'REPORT_023_PHASE2.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
