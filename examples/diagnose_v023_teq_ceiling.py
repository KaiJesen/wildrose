#!/usr/bin/env python3
"""Estimate teq participation ceiling on test split (diagnostic, not for production)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
CONFIG = _ROOT / "configs/trading_rule_v023_teq_ceiling_0062e.json"
OUT = _ROOT / "backtest/v023_teq_ceiling"


def main() -> int:
    subprocess.check_call([
        sys.executable, "examples/backtest_trading_system_v023.py",
        "--config", str(CONFIG.relative_to(_ROOT)),
        "--split", "test",
        "--output-dir", str((OUT / "test").relative_to(_ROOT)),
    ], cwd=_ROOT)
    subprocess.check_call([
        sys.executable, "examples/eval_participation.py",
        "--backtest-dir", str((OUT / "test").relative_to(_ROOT)),
        "--output", str((OUT / "participation_metrics.json").relative_to(_ROOT)),
    ], cwd=_ROOT)
    m = json.loads((OUT / "test" / "metrics.json").read_text())
    p = json.loads((OUT / "participation_metrics.json").read_text())["test"]
    leg_cov = float(p["participation_metrics"]["leg_count_coverage_ratio"])
    lines = [
        "# 023 TEQ Ceiling Diagnostic",
        "",
        f"- config: `{CONFIG.name}` (aggressive teq thresholds, standard opens unchanged)",
        "",
        "| metric | value |",
        "|--------|-------|",
        f"| total_return | {float(m['total_return'])*100:.2f}% |",
        f"| max_drawdown | {float(m['max_drawdown'])*100:.2f}% |",
        f"| trade_count | {int(m['trade_count'])} |",
        f"| trend_qualified_open_count | {int(m.get('trend_qualified_open_count',0))} |",
        f"| trend_qualified_pnl | {float(m.get('trend_qualified_pnl',0))*100:.2f}% |",
        f"| leg_count_coverage_ratio | {leg_cov*100:.2f}% |",
        "",
        "If ceiling config still misses 35% coverage or has negative return,",
        "tuning alone is unlikely to meet 023 exit gates on this model window.",
    ]
    (OUT / "CEILING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print((OUT / "CEILING_REPORT.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
