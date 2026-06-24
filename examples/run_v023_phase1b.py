#!/usr/bin/env python3
"""023 Phase 1b: crash → trend upgrade on frozen baseline."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
CONFIG = _ROOT / "configs/trading_rule_v023_phase1b_0062e.json"
OUT = _ROOT / "backtest/v023_phase1b"
BASELINE = _ROOT / "backtest/v023_baseline/participation_metrics.json"


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
    test = results.get("test", {}).get("runner_metrics", {})
    bm = base.get("test", {}).get("runner_metrics", {})
    lines = [
        "# 023 Phase 1b Report",
        "",
        f"- timestamp: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC`",
        f"- config: `{CONFIG.relative_to(_ROOT)}`",
        "",
        "| metric | baseline | phase1b |",
        "|--------|----------|---------|",
    ]
    for k in ("total_return", "max_drawdown", "trade_count", "trend_upgrade_count",
              "crash_short_count", "short_trend_capture_ratio", "leg_coverage_ratio"):
        a, b = float(bm.get(k, 0)), float(test.get(k, 0))
        if "ratio" in k or k in ("total_return", "max_drawdown"):
            lines.append(f"| {k} | {_pct(a)} | {_pct(b)} |")
        else:
            lines.append(f"| {k} | {a:.0f} | {b:.0f} |")
    lines.append("")
    lines.append(f"- trend_upgrade_count > 0: {'PASS' if float(test.get('trend_upgrade_count', 0)) > 0 else 'FAIL'}")
    (OUT / "REPORT_023_PHASE1B.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {OUT / 'REPORT_023_PHASE1B.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
