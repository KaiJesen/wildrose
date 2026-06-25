#!/usr/bin/env python3
"""Valid-only TEQ w_part sweep (no test leakage for selection)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
CKPT = _ROOT / "checkpoints/0065a_leg_align_c1_pw20/market_state_best.pt"
BASE_CFG = _ROOT / "configs/trading_rule_v023_phase1c_0062e.json"
OUT = _ROOT / "backtest/v024_constrained/teq_wp_sweep"
EXPLORE_RETURN = 0.0884
EXPLORE_COVERAGE = 0.28
CALIBS = {
    "c1": _ROOT / "backtest/v024_constrained/teq_edge_calibration.json",
    "pw20": _ROOT / "backtest/v024_constrained/teq_edge_calibration_pw20.json",
}
WP_GRID = [0.30, 0.32, 0.34, 0.35, 0.36, 0.37, 0.38, 0.40]


def _run_backtest(cfg_path: Path, split: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            sys.executable,
            "examples/backtest_trading_system_v014.py",
            "--config",
            str(cfg_path.relative_to(_ROOT)),
            "--checkpoint",
            str(CKPT.relative_to(_ROOT)),
            "--split",
            split,
            "--output-dir",
            str(out_dir.relative_to(_ROOT)),
        ],
        cwd=_ROOT,
        stdout=subprocess.DEVNULL,
    )
    return json.loads((out_dir / "metrics.json").read_text())


def _run_part(backtest_dir: Path) -> float:
    part_path = backtest_dir / "part.json"
    subprocess.check_call(
        [
            sys.executable,
            "examples/eval_participation.py",
            "--backtest-dir",
            str(backtest_dir.relative_to(_ROOT)),
            "--output",
            str(part_path.relative_to(_ROOT)),
        ],
        cwd=_ROOT,
        stdout=subprocess.DEVNULL,
    )
    part = json.loads(part_path.read_text())
    split = list(part.keys())[0]
    return float(part[split]["participation_metrics"]["leg_count_coverage_ratio"])


def main() -> int:
    base = json.loads(BASE_CFG.read_text())
    frozen_teq = float(
        json.loads((_ROOT / "backtest/v023_phase1c/test/metrics.json").read_text()).get(
            "trend_qualified_open_count", 1
        )
    )
    rows: list[dict] = []
    OUT.mkdir(parents=True, exist_ok=True)
    for cal_name, cal_path in CALIBS.items():
        for wp in WP_GRID:
                cfg = dict(base)
                cfg["teq_edge"] = {
                    "enabled": True,
                    "weight_edge_5": 0.25,
                    "weight_edge_24": 0.35,
                    "weight_participation": wp,
                    "calibration_path": str(cal_path.relative_to(_ROOT)),
                    "use_calibrated": True,
                }
                cfg_path = OUT / "configs" / f"teq_{cal_name}_wp{wp}.json"
                cfg_path.parent.mkdir(parents=True, exist_ok=True)
                cfg_path.write_text(json.dumps(cfg))
                tag = f"{cal_name}_wp{wp}"
                valid_dir = OUT / f"{tag}_valid"
                test_dir = OUT / f"{tag}_test"
                m_valid = _run_backtest(cfg_path, "valid", valid_dir)
                m_test = _run_backtest(cfg_path, "test", test_dir)
                cov_valid = _run_part(valid_dir)
                cov_test = _run_part(test_dir)
                teq_valid = int(m_valid.get("trend_qualified_open_count", 0))
                teq_test = int(m_test.get("trend_qualified_open_count", 0))
                row = {
                    "tag": tag,
                    "cal": cal_name,
                    "w_part": wp,
                    "valid_return": m_valid.get("total_return", 0),
                    "valid_coverage": cov_valid,
                    "valid_teq": teq_valid,
                    "valid_teq_ratio": teq_valid / max(1.0, frozen_teq),
                    "valid_explore_pass": (
                        m_valid.get("total_return", 0) >= EXPLORE_RETURN
                        and cov_valid >= EXPLORE_COVERAGE
                    ),
                    "test_return": m_test.get("total_return", 0),
                    "test_coverage": cov_test,
                    "test_teq": teq_test,
                    "test_explore_pass": (
                        m_test.get("total_return", 0) >= EXPLORE_RETURN
                        and cov_test >= EXPLORE_COVERAGE
                    ),
                }
                rows.append(row)
                print(
                    f"{tag}: valid ret={row['valid_return']*100:.2f}% cov={cov_valid*100:.1f}% "
                    f"teq={teq_valid} | test ret={row['test_return']*100:.2f}% cov={cov_test*100:.1f}%"
                )

    valid_pass = [r for r in rows if r["valid_explore_pass"]]
    test_pass = [r for r in rows if r["test_explore_pass"]]
    best_valid = max(rows, key=lambda r: (r["valid_explore_pass"], r["valid_return"]))
    summary = {
        "frozen_teq_baseline": frozen_teq,
        "explore_return": EXPLORE_RETURN,
        "explore_coverage": EXPLORE_COVERAGE,
        "valid_explore_pass_count": len(valid_pass),
        "test_explore_pass_count": len(test_pass),
        "best_by_valid_return": best_valid,
        "rows": rows,
    }
    (OUT / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nvalid explore PASS: {len(valid_pass)}/{len(rows)}")
    print(f"test explore PASS: {len(test_pass)}/{len(rows)}")
    if valid_pass:
        print("valid-pass configs:", [r["tag"] for r in valid_pass])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
