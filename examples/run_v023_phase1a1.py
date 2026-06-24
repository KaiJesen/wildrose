#!/usr/bin/env python3
"""023 Phase 1a.1: run tuned config backtest + participation eval."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
BASELINE_BT = _ROOT / "backtest/v023_baseline"
PHASE1A_BT = _ROOT / "backtest/v023_phase1a"
OUT_ROOT = _ROOT / "backtest/v023_phase1a1"
CONFIG = _ROOT / "configs/trading_rule_v023_phase1a1_0062e.json"
CHECKPOINT = "prod/v0.0.0/checkpoint/market_state_best.pt"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=_ROOT)


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _write_report(results: dict, baseline: dict, phase1a: dict) -> None:
    test = results.get("test", {})
    rm = test.get("runner_metrics", {})
    bm = baseline.get("test", {}).get("runner_metrics", {})
    p1 = phase1a.get("test", {}).get("runner_metrics", {})
    lines = [
        "# 023 Phase 1a.1 Report",
        "",
        f"- timestamp: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC`",
        f"- config: `{CONFIG.relative_to(_ROOT)}`",
        "",
        "## vs baseline / phase1a (test)",
        "",
        "| metric | baseline | phase1a | phase1a1 |",
        "|--------|----------|---------|----------|",
    ]
    for key in (
        "total_return", "max_drawdown", "trade_count", "slow_up_open_count",
        "long_trend_capture_ratio", "leg_coverage_ratio", "missed_confirmed_trend_bars",
    ):
        a, b, c = float(bm.get(key, 0)), float(p1.get(key, 0)), float(rm.get(key, 0))
        if key.endswith("_ratio") or key in ("total_return", "max_drawdown"):
            lines.append(f"| {key} | {_pct(a)} | {_pct(b)} | {_pct(c)} |")
        else:
            lines.append(f"| {key} | {a:.0f} | {b:.0f} | {c:.0f} |")

    lines.extend(["", "## Participation (test)", ""])
    pm = test.get("participation_metrics", {})
    for k, v in pm.items():
        if k.endswith("_ratio"):
            lines.append(f"- {k}: `{_pct(float(v))}`")
        else:
            lines.append(f"- {k}: `{v}`")

    lines.extend([
        "",
        "## Exit",
        "",
        f"- slow_up_open_count >= 1: {'PASS' if float(rm.get('slow_up_open_count', 0)) >= 1 else 'FAIL'}",
        f"- total_return >= baseline*0.95: {'PASS' if float(rm.get('total_return', 0)) >= float(bm.get('total_return', 0)) * 0.95 else 'FAIL'}",
        f"- max_drawdown >= -2%: {'PASS' if float(rm.get('max_drawdown', 0)) >= -0.02 else 'FAIL'}",
        "",
    ])
    report = OUT_ROOT / "REPORT_023_PHASE1A1.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {report}")


def main() -> int:
    if not CONFIG.is_file():
        raise SystemExit(f"run tune first: examples/tune_v023_phase1a.py (missing {CONFIG})")

    for split in ("valid", "test"):
        _run([
            sys.executable, "examples/backtest_trading_system_v023.py",
            "--config", str(CONFIG.relative_to(_ROOT)),
            "--split", split,
            "--output-dir", str((OUT_ROOT / split).relative_to(_ROOT)),
        ])

    _run([
        sys.executable, "examples/eval_participation.py",
        "--backtest-dir", str((OUT_ROOT / "valid").relative_to(_ROOT)),
        "--backtest-dir", str((OUT_ROOT / "test").relative_to(_ROOT)),
        "--output", str((OUT_ROOT / "participation_metrics.json").relative_to(_ROOT)),
    ])
    _run([
        sys.executable, "examples/plot_participation_overlay.py",
        "--backtest-dir", str((OUT_ROOT / "test").relative_to(_ROOT)),
    ])

    results = json.loads((OUT_ROOT / "participation_metrics.json").read_text())
    baseline = json.loads((BASELINE_BT / "participation_metrics.json").read_text())
    phase1a = json.loads((PHASE1A_BT / "participation_metrics.json").read_text())
    _write_report(results, baseline, phase1a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
