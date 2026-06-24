#!/usr/bin/env python3
"""Phase 3 linkage validation: v021 full_bias vs v022 trend_quality."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
BASELINE = "configs/trading_rule_v021_full_bias_0062e.json"
CANDIDATE = "configs/trading_rule_v022_trend_quality_0062e.json"
CHECKPOINT = "prod/v0.0.0/checkpoint/market_state_best.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="022 Phase 3 linkage validation")
    p.add_argument("--split", default="test")
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--out-dir", default="backtest/v022_phase3_validation")
    p.add_argument("--skip-run", action="store_true")
    return p.parse_args()


def _run(config: str, out: Path, split: str, checkpoint: str) -> None:
    cmd = [
        sys.executable,
        "examples/backtest_trading_system_v014.py",
        "--config",
        config,
        "--checkpoint",
        checkpoint,
        "--output-dir",
        str(out),
        "--split",
        split,
        "--symbol",
        "BTCUSDT",
        "--interval",
        "1h",
        "--days",
        "365",
    ]
    subprocess.check_call(cmd, cwd=_ROOT)


def _pct(v: float) -> str:
    return f"{100.0 * v:.2f}%"


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_dir)
    rows: dict[str, dict] = {}

    if not args.skip_run:
        _run(BASELINE, out_root / "v021_full_bias", args.split, args.checkpoint)
        _run(CANDIDATE, out_root / "v022_trend_quality", args.split, args.checkpoint)

    for name, sub in (("v021_full_bias", "v021_full_bias"), ("v022_trend_quality", "v022_trend_quality")):
        mpath = out_root / sub / "metrics.json"
        if mpath.is_file():
            rows[name] = json.loads(mpath.read_text(encoding="utf-8"))

    base = rows.get("v021_full_bias", {})
    cand = rows.get("v022_trend_quality", {})

    lines = [
        "# 022 Phase 3 Linkage Validation",
        "",
        f"- split: `{args.split}`",
        f"- baseline: `{BASELINE}`",
        f"- candidate: `{CANDIDATE}`",
        "",
        "| metric | v021 full_bias | v022 trend_quality |",
        "|--------|----------------|---------------------|",
    ]
    keys = [
        "annualized_return",
        "total_return",
        "max_drawdown",
        "trade_count",
        "missed_confirmed_trend_bars",
        "trend_upgrade_count",
        "avg_bars_held",
        "hard_counter_open_count",
    ]
    for k in keys:
        lines.append(f"| {k} | {base.get(k, '—')} | {cand.get(k, '—')} |")

    business_ok = True
    if base and cand:
        tr_base = float(base.get("total_return", 0))
        tr_cand = float(cand.get("total_return", 0))
        dd_base = float(base.get("max_drawdown", 0))
        dd_cand = float(cand.get("max_drawdown", 0))
        tc_base = float(base.get("trade_count", 0))
        tc_cand = float(cand.get("trade_count", 0))
        business_ok = (
            tr_cand >= 0.7 * tr_base
            and dd_cand >= dd_base - 0.002
            and dd_cand >= dd_base * 1.2
            and tc_cand <= 1.5 * max(tc_base, 1)
        )
        lines.extend(
            [
                "",
                "## Business底线",
                "",
                f"- total_return ≥ 70% baseline: **{'PASS' if tr_cand >= 0.7 * tr_base else 'FAIL'}** ({_pct(tr_cand)} vs {_pct(tr_base)})",
                f"- max_drawdown not worse: **{'PASS' if dd_cand >= dd_base - 0.002 and dd_cand >= dd_base * 1.2 else 'FAIL'}**",
                f"- trade_count ≤ 1.5× baseline: **{'PASS' if tc_cand <= 1.5 * max(tc_base, 1) else 'FAIL'}** ({tc_cand:.0f} vs {tc_base:.0f})",
                f"- overall: **{'PASS' if business_ok else 'FAIL'}**",
            ]
        )

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "VALIDATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_root / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"saved: {out_root / 'VALIDATION_REPORT.md'}")
    return 0 if business_ok or not (base and cand) else 1


if __name__ == "__main__":
    raise SystemExit(main())
