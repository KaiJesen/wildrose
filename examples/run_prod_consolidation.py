#!/usr/bin/env python3
"""P0 prod consolidation: health monitor + FLAT decision attribution (architect §6)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from trading_system.decision_attribution import analyze_decisions, compare_to_baseline

PROD = _ROOT / "prod/v1.1.0"
BASELINE_METRICS = PROD / "metrics/backtest_test_oos.json"
OUT = _ROOT / "backtest/prod_monitor"


def _run_prod_backtest(split: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "bash",
        str(PROD / "scripts/run_backtest.sh"),
        "--split",
        split,
        "--output-dir",
        str(out_dir.relative_to(_ROOT)),
    ]
    subprocess.run(cmd, check=True, cwd=_ROOT)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser(description="Prod v1.1.0 health + attribution")
    p.add_argument("--split", default="test", choices=["train", "valid", "test"])
    p.add_argument("--skip-backtest", action="store_true", help="Only analyze existing out dir")
    p.add_argument("--out-dir", default="")
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUT / f"v1.1.0_{args.split}"
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir

    if not args.skip_backtest:
        _run_prod_backtest(args.split, out_dir)

    metrics_path = out_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    current_metrics = _load_json(metrics_path)

    decisions = pd.read_csv(out_dir / "decisions.csv")
    trades_path = out_dir / "trades.csv"
    trades = pd.read_csv(trades_path) if trades_path.is_file() and trades_path.stat().st_size else pd.DataFrame()
    attribution = analyze_decisions(decisions, trades=trades)

    baseline = _load_json(BASELINE_METRICS) if args.split == "test" else {}
    health = (
        compare_to_baseline(current_metrics, baseline)
        if baseline
        else {"pass": None, "checks": {}, "note": "no frozen baseline for this split"}
    )

    summary = {
        "prod_version": "v1.1.0",
        "split": args.split,
        "out_dir": str(out_dir.relative_to(_ROOT)),
        "metrics": current_metrics,
        "health_vs_baseline": health,
        "attribution": attribution,
        "architect_p0": {
            "monitor": health.get("pass"),
            "watch_slow_blind_spot": {
                "watch_bars": attribution["watch_slow_uptrend_bars"],
                "opens": attribution["slow_up_open_count"],
                "watch_to_open_ratio": attribution["slow_up_watch_to_open_ratio"],
            },
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prod_consolidation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "# Prod v1.1.0 巩固报告（架构师 P0）",
        "",
        f"**Split**: {args.split}",
        f"**Health vs baseline**: {'PASS' if health.get('pass') else health.get('pass') if health.get('pass') is not None else 'N/A'}",
        "",
        "## 指标",
        "",
        f"| 指标 | 当前 | 基线 |",
        f"|------|------|------|",
    ]
    if health.get("checks"):
        for key, chk in health["checks"].items():
            base = chk.get("baseline", "—")
            cur = chk.get("current", chk)
            if key in ("total_return", "max_drawdown") and isinstance(cur, (int, float)):
                cur_s = f"{cur*100:.2f}%"
                base_s = f"{base*100:.2f}%" if isinstance(base, (int, float)) else base
            else:
                cur_s, base_s = cur, base
            lines.append(f"| {key} | {cur_s} | {base_s} |")
    lines.extend(
        [
            "",
            "## 未成交归因（FLAT）",
            "",
            f"- FLAT 占比: **{attribution['flat_ratio']*100:.1f}%** ({attribution['flat_bars']}/{attribution['total_bars']} bars)",
            f"- WATCH_SLOW_UPTREND: **{attribution['watch_slow_uptrend_bars']}** bars，"
            f"{attribution['watch_slow_episode_count']} 段（最长 {attribution['watch_slow_episode_max_bars']} bars）",
            f"- slow_up 开仓: **{attribution['slow_up_open_count']}**（watch→open 比 {attribution['slow_up_watch_to_open_ratio']:.4f}）",
            f"- HOLD_NO_ENTRY: {attribution['hold_no_entry_bars']} bars",
            f"- TEQ reject bars: {attribution['teq_reject_bars']}",
            "",
            "### FLAT Top reasons",
            "",
        ]
    )
    for reason, cnt in list(attribution["flat_top_reasons"].items())[:10]:
        lines.append(f"- `{reason}`: {cnt}")
    if attribution["teq_reject_reason_tokens"]:
        lines.extend(["", "### TEQ reject tokens", ""])
        for tok, cnt in list(attribution["teq_reject_reason_tokens"].items())[:8]:
            lines.append(f"- `{tok}`: {cnt}")
    lines.extend(
        [
            "",
            "```bash",
            "bash scripts/prod_health_monitor.sh",
            "```",
        ]
    )
    (out_dir / "REPORT_PROD_CONSOLIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if health.get("pass") is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
