#!/usr/bin/env python3
"""023 Phase 1a: slow-up watch → probe backtest + participation eval + report."""

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

BASELINE_CONFIG = _ROOT / "configs/trading_rule_v023_baseline_0062e.json"
PHASE1A_CONFIG = _ROOT / "configs/trading_rule_v023_phase1a_0062e.json"
OUT_ROOT = _ROOT / "backtest/v023_phase1a"
BASELINE_BT = _ROOT / "backtest/v023_baseline"
CHECKPOINT = "prod/v0.0.0/checkpoint/market_state_best.pt"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=_ROOT)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _write_report(results: dict, baseline: dict) -> None:
    lines = [
        "# 023 Phase 1a Report (slow-up watch → probe)",
        "",
        f"- timestamp: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC`",
        f"- checkpoint: `{CHECKPOINT}`",
        f"- baseline config: `{BASELINE_CONFIG.relative_to(_ROOT)}`",
        f"- phase1a config: `{PHASE1A_CONFIG.relative_to(_ROOT)}`",
        f"- phase1a hash: `{_sha256(PHASE1A_CONFIG)}`",
        "",
        "## vs Phase 0 baseline (test)",
        "",
    ]
    test = results.get("test", {})
    rm = test.get("runner_metrics", {})
    bm = baseline.get("test", {}).get("runner_metrics", {})
    lines.append("| metric | baseline | phase1a | delta |")
    lines.append("|--------|----------|---------|-------|")
    keys = (
        "total_return",
        "max_drawdown",
        "trade_count",
        "slow_up_open_count",
        "slow_up_trade_count",
        "long_trend_capture_ratio",
        "missed_confirmed_trend_bars",
        "leg_coverage_ratio",
    )
    for key in keys:
        a = float(bm.get(key, 0))
        b = float(rm.get(key, 0))
        if key.endswith("_ratio") or key in ("total_return", "max_drawdown"):
            lines.append(f"| {key} | {_pct(a)} | {_pct(b)} | {_pct(b - a)} |")
        else:
            lines.append(f"| {key} | {a:.0f} | {b:.0f} | {b - a:+.0f} |")

    lines.extend(["", "## Participation (§5.3)", ""])
    for split in ("valid", "test"):
        if split not in results:
            continue
        pm = results[split].get("participation_metrics", {})
        lines.append(f"### {split}")
        lines.append("")
        for k, v in pm.items():
            if k.endswith("_ratio"):
                lines.append(f"- {k}: `{_pct(float(v))}`")
            else:
                lines.append(f"- {k}: `{v}`")
        lines.append("")

    lines.extend([
        "## Phase 1a exit checklist",
        "",
        "- [ ] test `slow_up_open_count >= 1`",
        "- [ ] valid `slow_up_false_entry_count` controlled",
        "- [ ] test `total_return` >= baseline × 0.95",
        "",
    ])
    report = OUT_ROOT / "REPORT_023_PHASE1A.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {report}")


def main() -> int:
    if not PHASE1A_CONFIG.is_file():
        raise SystemExit(f"missing config: {PHASE1A_CONFIG}")

    for split in ("valid", "test"):
        out = OUT_ROOT / split
        _run([
            sys.executable,
            "examples/backtest_trading_system_v023.py",
            "--config",
            str(PHASE1A_CONFIG.relative_to(_ROOT)),
            "--split",
            split,
            "--output-dir",
            str(out.relative_to(_ROOT)),
        ])

    _run([
        sys.executable,
        "examples/eval_participation.py",
        "--backtest-dir",
        str((OUT_ROOT / "valid").relative_to(_ROOT)),
        "--backtest-dir",
        str((OUT_ROOT / "test").relative_to(_ROOT)),
        "--output",
        str((OUT_ROOT / "participation_metrics.json").relative_to(_ROOT)),
    ])
    _run([
        sys.executable,
        "examples/plot_participation_overlay.py",
        "--backtest-dir",
        str((OUT_ROOT / "test").relative_to(_ROOT)),
    ])

    results = _read_json(OUT_ROOT / "participation_metrics.json")
    baseline_part = _read_json(BASELINE_BT / "participation_metrics.json")
    _write_report(results, baseline_part)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
