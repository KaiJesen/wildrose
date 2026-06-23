#!/usr/bin/env python3
"""Phase C/D deep validation: train/valid full-window backtests + June decline bias spot check."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CHECKPOINT = "prod/v0.0.0/checkpoint/market_state_best.pt"
JUNE_DECLINE_START = "2026-06-01"
JUNE_DECLINE_END = "2026-06-18"

RUNS = [
    ("v020", "examples/backtest_trading_system_v020.py", "configs/trading_rule_v020_trend_segment_0062e.json", None),
    ("open_size_bias", "examples/backtest_trading_system_v021.py", None, "open_size_bias"),
    ("full_bias", "examples/backtest_trading_system_v021.py", None, "full_bias"),
]
SPLITS = ("train", "valid")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate v021 Phase C/D on train/valid")
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--skip-run", action="store_true", help="only summarize existing outputs")
    p.add_argument("--out-dir", default="backtest/v021_cd_validation")
    p.add_argument("--june-spot-check-split", default="test", choices=["train", "valid", "test"])
    return p.parse_args()


def _run_backtest(script: str, out_dir: Path, split: str, checkpoint: str, config: str | None, variant: str | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, script, "--checkpoint", checkpoint, "--output-dir", str(out_dir), "--split", split]
    if variant:
        cmd.extend(["--variant", variant])
    elif config:
        cmd.extend(["--config", config])
    print("RUN", " ".join(cmd))
    subprocess.check_call(cmd, cwd=_ROOT)


def _load_metrics(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    return {k: float(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def _pct(v: float) -> str:
    return f"{100.0 * v:.2f}%"


def _gate_c(m: dict[str, float], base: dict[str, float]) -> list[str]:
    notes: list[str] = []
    if m.get("max_position_ratio_observed", 0.0) > 0.20 + 1e-9:
        notes.append("FAIL max_position_ratio > 20%")
    else:
        notes.append("PASS max_position_ratio <= 20%")
    cand = m.get("trend_add_candidate_count", 0.0)
    evaluated = m.get("trend_add_risk_evaluated_count", 0.0)
    if cand > 0 and evaluated != cand:
        notes.append(f"FAIL trend_add_risk_evaluated ({evaluated}) != candidate ({cand})")
    elif cand > 0:
        notes.append("PASS trend_add_risk_evaluated == candidate")
    else:
        notes.append("INFO trend_add_candidate_count=0")
    base_hold = base.get("avg_trend_hold_bars", 0.0)
    hold = m.get("avg_trend_hold_bars", 0.0)
    if base_hold > 0 and hold < base_hold * 0.95:
        notes.append(f"WARN avg_trend_hold_bars {hold:.1f} < v020*{0.95:.2f} ({base_hold*0.95:.1f})")
    else:
        notes.append("PASS avg_trend_hold_bars vs v020")
    return notes


def _gate_d(m: dict[str, float], base: dict[str, float]) -> list[str]:
    notes = _gate_c(m, base)
    legacy_read = m.get("legacy_trend_direct_read_count", 0.0)
    if legacy_read > 0:
        notes.append(f"WARN legacy_trend_direct_read_count={legacy_read:.0f}")
    else:
        notes.append("PASS legacy_trend_direct_read_count=0")
    hard = m.get("hard_counter_open_count", 0.0)
    if hard > 0:
        notes.append(f"FAIL hard_counter_open_count={hard:.0f}")
    else:
        notes.append("PASS hard_counter_open_count=0")
    cov = m.get("bias_reason_codes_coverage", 0.0)
    if cov < 1.0:
        notes.append(f"WARN bias_reason_codes_coverage={cov:.2f}")
    else:
        notes.append("PASS bias_reason_codes_coverage=1.0")
    return notes


def _spot_check_june(decisions_csv: Path) -> dict:
    rows = list(csv.DictReader(decisions_csv.open(encoding="utf-8")))
    window = [
        r
        for r in rows
        if JUNE_DECLINE_START <= (r.get("ts") or "")[:10] <= JUNE_DECLINE_END
    ]
    crash_bars = [r for r in window if int(r.get("is_crash", 0)) == 1 or int(r.get("is_model_blind_crash", 0)) == 1]
    long_opens = [r for r in window if r.get("action") == "OPEN_LONG"]
    short_opens = [r for r in window if r.get("action") == "OPEN_SHORT"]
    blocks_long = [r for r in window if "LONG" in r.get("reason_code", "") and r.get("action") == "BLOCK"]
    reason_counter = Counter()
    for r in window:
        for code in (r.get("bias_reason_codes") or "").split("|"):
            if code:
                reason_counter[code] += 1
    samples: list[dict] = []
    for r in crash_bars[:8]:
        samples.append(
            {
                "ts": r.get("ts"),
                "action": r.get("action"),
                "reason_code": r.get("reason_code"),
                "allow_open_long": r.get("allow_open_long"),
                "allow_open_short": r.get("allow_open_short"),
                "open_bias_short": r.get("open_bias_short"),
                "counter_level_long": r.get("counter_level_long"),
                "leg_type": r.get("leg_type"),
                "sub_phase": r.get("sub_phase"),
                "bias_reason_codes": r.get("bias_reason_codes"),
            }
        )
    expectations = {
        "crash_p1_block_long_present": "CRASH_P1_BLOCK_LONG" in reason_counter,
        "crash_short_boost_or_block_present": (
            reason_counter.get("CRASH_P1_BOOST_SHORT_OPEN", 0) > 0
            or reason_counter.get("CRASH_P1_BLOCK_LONG", 0) > 0
        ),
        "crash_bars_block_long": all(int(r.get("allow_open_long", 1)) == 0 for r in crash_bars) if crash_bars else True,
    }
    return {
        "window": f"{JUNE_DECLINE_START} .. {JUNE_DECLINE_END}",
        "bars": len(window),
        "crash_bars": len(crash_bars),
        "long_opens": len(long_opens),
        "short_opens": len(short_opens),
        "long_blocks": len(blocks_long),
        "top_bias_reasons": reason_counter.most_common(12),
        "crash_samples": samples,
        "checks": expectations,
    }


def main() -> int:
    args = parse_args()
    root = Path(args.out_dir)
    summary: dict = {"runs": {}, "gates": {}, "june_spot_check": {}}

    if not args.skip_run:
        for split in SPLITS:
            for name, script, config, variant in RUNS:
                out = root / f"{name}_{split}"
                _run_backtest(script, out, split, args.checkpoint, config, variant)
        june_out = root / f"full_bias_{args.june_spot_check_split}"
        if not (june_out / "metrics.json").is_file():
            _run_backtest(
                "examples/backtest_trading_system_v021.py",
                june_out,
                args.june_spot_check_split,
                args.checkpoint,
                None,
                "full_bias",
            )

    lines = [
        "# v021 Phase C/D Validation (annualized return)",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- splits: train / valid (full window each)",
        f"- June spot-check split: `{args.june_spot_check_split}`",
        "",
        "## Metrics (annualized primary)",
        "",
        "| variant | split | ann_return | bench_ann | excess_ann | max_dd | trades | max_pos | hard_counter |",
        "|---------|-------|------------|-----------|------------|--------|--------|---------|--------------|",
    ]

    for split in SPLITS:
        base = _load_metrics(root / f"v020_{split}" / "metrics.json")
        summary["runs"][f"v020_{split}"] = base
        lines.append(
            f"| v020 | {split} | {_pct(base.get('annualized_return', 0))} | "
            f"{_pct(base.get('benchmark_annualized_return', 0))} | "
            f"{_pct(base.get('excess_annualized_return', 0))} | "
            f"{_pct(base.get('max_drawdown', 0))} | {base.get('trade_count', 0):.0f} | "
            f"{base.get('max_position_ratio_observed', 0):.2%} | {base.get('hard_counter_open_count', 0):.0f} |"
        )
        for name, _, _, _ in RUNS[1:]:
            m = _load_metrics(root / f"{name}_{split}" / "metrics.json")
            summary["runs"][f"{name}_{split}"] = m
            gates = _gate_c(m, base) if name == "open_size_bias" else _gate_d(m, base)
            summary["gates"][f"{name}_{split}"] = gates
            lines.append(
                f"| {name} | {split} | {_pct(m.get('annualized_return', 0))} | "
                f"{_pct(m.get('benchmark_annualized_return', 0))} | "
                f"{_pct(m.get('excess_annualized_return', 0))} | "
                f"{_pct(m.get('max_drawdown', 0))} | {m.get('trade_count', 0):.0f} | "
                f"{m.get('max_position_ratio_observed', 0):.2%} | {m.get('hard_counter_open_count', 0):.0f} |"
            )

    lines.extend(["", "## Acceptance gates", ""])
    for key, gates in summary["gates"].items():
        lines.append(f"### {key}")
        for g in gates:
            lines.append(f"- {g}")
        lines.append("")

    june_path = root / f"full_bias_{args.june_spot_check_split}" / "decisions.csv"
    if june_path.is_file():
        spot = _spot_check_june(june_path)
        summary["june_spot_check"] = spot
        lines.extend(["", "## June decline bias spot check", ""])
        lines.append(f"- window: `{spot['window']}` ({spot['bars']} bars)")
        lines.append(f"- crash bars: {spot['crash_bars']}, long opens: {spot['long_opens']}, short opens: {spot['short_opens']}")
        lines.append("")
        lines.append("### Checks")
        for k, v in spot["checks"].items():
            lines.append(f"- {k}: {'PASS' if v else 'FAIL'}")
        lines.append("")
        lines.append("### Top bias reason codes")
        for code, cnt in spot["top_bias_reasons"]:
            lines.append(f"- `{code}`: {cnt}")
        lines.append("")
        lines.append("### Crash-window samples")
        for s in spot["crash_samples"]:
            lines.append(f"- `{s['ts']}` action={s['action']} reason={s['reason_code']} allow_L={s['allow_open_long']} open_bias_S={s['open_bias_short']} leg={s['leg_type']}/{s['sub_phase']} reasons={s['bias_reason_codes']}")

    root.mkdir(parents=True, exist_ok=True)
    (root / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = "\n".join(lines) + "\n"
    (root / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"saved: {root / 'VALIDATION_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
