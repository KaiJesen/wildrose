#!/usr/bin/env python3
"""Run v020 baseline and v021 phases B/C/D, print acceptance summary."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = "prod/v0.0.0/checkpoint/market_state_best.pt"
SPLIT = "test"

PHASES = [
    ("v020", "examples/backtest_trading_system_v020.py", "backtest/backtest_v020_phase_a_compare_test"),
    ("observe", "examples/backtest_trading_system_v021.py", "backtest/backtest_v021_observe_regress_test"),
    ("open_bias", "examples/backtest_trading_system_v021.py", "backtest/backtest_v021_open_bias_test"),
    ("open_size_bias", "examples/backtest_trading_system_v021.py", "backtest/backtest_v021_open_size_bias_test"),
    ("full_bias", "examples/backtest_trading_system_v021.py", "backtest/backtest_v021_full_bias_test"),
]

KEY_METRICS = [
    "annualized_return",
    "benchmark_annualized_return",
    "excess_annualized_return",
    "trade_count",
    "total_return",
    "max_drawdown",
    "legacy_trend_direct_block_count",
    "legacy_trend_direct_read_count",
    "hard_counter_open_count",
    "bias_reason_codes_coverage",
    "max_position_ratio_observed",
    "avg_trend_hold_bars",
    "leg_coverage_ratio",
    "trend_add_candidate_count",
    "trend_add_risk_evaluated_count",
    "trend_add_rejected_by_risk_count",
    "trend_add_allowed_count",
    "bias_field_nonempty_ratio",
    "bias_reason_nonempty_ratio",
]


def _run(script: str, out_dir: str, extra: list[str]) -> None:
    cmd = [
        sys.executable,
        script,
        "--checkpoint",
        CHECKPOINT,
        "--output-dir",
        out_dir,
        "--split",
        SPLIT,
    ]
    if script.endswith("v021.py"):
        cmd[1:1] = []  # no-op placeholder
    cmd.extend(extra)
    if "v021" in script:
        cmd = [
            sys.executable,
            script,
            "--variant",
            extra[0] if extra and not extra[0].startswith("--") else "observe",
            "--checkpoint",
            CHECKPOINT,
            "--output-dir",
            out_dir,
            "--split",
            SPLIT,
        ]
    else:
        cmd = [
            sys.executable,
            script,
            "--checkpoint",
            CHECKPOINT,
            "--output-dir",
            out_dir,
            "--split",
            SPLIT,
        ]
    print("RUN", " ".join(cmd))
    subprocess.check_call(cmd, cwd=_ROOT)


def _load_metrics(out_dir: Path) -> dict[str, float]:
    path = out_dir / "metrics.json"
    if not path.exists():
        return {}
    return {k: float(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def main() -> int:
    run_backtests = "--skip-run" not in sys.argv
    if run_backtests:
        _run("examples/backtest_trading_system_v020.py", "backtest/backtest_v020_phase_a_compare_test", [])
        for variant, script, out in [
            ("observe", "examples/backtest_trading_system_v021.py", "backtest/backtest_v021_observe_regress_test"),
            ("open_bias", "examples/backtest_trading_system_v021.py", "backtest/backtest_v021_open_bias_test"),
            ("open_size_bias", "examples/backtest_trading_system_v021.py", "backtest/backtest_v021_open_size_bias_test"),
            ("full_bias", "examples/backtest_trading_system_v021.py", "backtest/backtest_v021_full_bias_test"),
        ]:
            cmd = [
                sys.executable,
                script,
                "--variant",
                variant,
                "--checkpoint",
                CHECKPOINT,
                "--output-dir",
                out,
                "--split",
                SPLIT,
            ]
            print("RUN", " ".join(cmd))
            subprocess.check_call(cmd, cwd=_ROOT)

    rows: dict[str, dict[str, float]] = {}
    for name, _, out in PHASES:
        rows[name] = _load_metrics(_ROOT / out)

    base = rows.get("v020", {})
    print("\n=== v021 phase comparison (test split) ===")
    header = f"{'metric':<36}" + "".join(f"{n:>14}" for n, _, _ in PHASES)
    print(header)
    for key in KEY_METRICS:
        line = f"{key:<36}"
        for name, _, _ in PHASES:
            val = rows.get(name, {}).get(key, float("nan"))
            line += f"{val:14.4f}"
        print(line)

    print("\n=== acceptance checks vs v020 ===")
    b = rows.get("open_bias", {})
    if base and b:
        tc_delta = abs(b.get("trade_count", 0) - base.get("trade_count", 0)) / max(1, base.get("trade_count", 1))
        dd_ratio = abs(b.get("max_drawdown", 0)) / max(1e-9, abs(base.get("max_drawdown", 0)))
        print(f"Phase B trade_count delta ratio: {tc_delta:.2%} (limit 15%)")
        print(f"Phase B max_drawdown ratio vs base: {dd_ratio:.2%} (limit 110%)")
        print(f"Phase B legacy blocks: v020={base.get('legacy_trend_direct_block_count',0):.0f} -> open_bias={b.get('legacy_trend_direct_block_count',0):.0f}")
        print(f"Phase B hard_counter_open: {b.get('hard_counter_open_count',0):.0f}")

    obs = rows.get("observe", {})
    if base and obs:
        match = (
            obs.get("trade_count") == base.get("trade_count")
            and abs(obs.get("annualized_return", 0) - base.get("annualized_return", 0)) < 1e-6
        )
        print(f"Phase A regression match v020: {match}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
