#!/usr/bin/env python3
"""026 Phase 3: TEQ valid calibration + w_part sweep for M2 checkpoint."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(_ROOT / "examples"))

from _v025_common import kline_backtest_args
M2_CKPT = _ROOT / "checkpoints/026_phase1_c1d1/market_state_best.pt"
B0_CONFIG = _ROOT / "configs/trading_rule_v024_phase1c_teq_0065a_c1_pw20.json"
BASE_RULE = _ROOT / "configs/trading_rule_v023_phase1c_0062e.json"
OUT = _ROOT / "backtest/v026_phase3"
CALIBRATION = OUT / "teq_edge_calibration.json"
WP_GRID = [0.30, 0.32, 0.34, 0.35, 0.36, 0.37, 0.38, 0.40]
EXPLORE_RETURN = 0.0884
EXPLORE_COVERAGE = 0.28


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=_ROOT)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _kline_args() -> list[str]:
    return kline_backtest_args()


def _backtest(cfg: Path, ckpt: Path, split: str, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    _run([
        sys.executable,
        "examples/backtest_trading_system_v014.py",
        "--config",
        str(cfg.relative_to(_ROOT)),
        "--checkpoint",
        str(ckpt.relative_to(_ROOT)),
        "--split",
        split,
        "--output-dir",
        str(out.relative_to(_ROOT)),
        *_kline_args(),
    ])
    return _read_json(out / "metrics.json")


def _coverage(backtest_dir: Path) -> float:
    part_path = backtest_dir / "part.json"
    _run([
        sys.executable,
        "examples/eval_participation.py",
        "--backtest-dir",
        str(backtest_dir.relative_to(_ROOT)),
        "--output",
        str(part_path.relative_to(_ROOT)),
    ])
    part = _read_json(part_path)
    split = list(part.keys())[0]
    return float(part[split]["participation_metrics"]["leg_count_coverage_ratio"])


def _write_m2_config(wp: float) -> Path:
    base = _read_json(B0_CONFIG)
    base["_026_meta"] = {
        "phase": "3_m2",
        "recipe": "c3_c1d1_teq",
        "checkpoint": str(M2_CKPT.relative_to(_ROOT)),
    }
    base["teq_edge"] = {
        "enabled": True,
        "weight_edge_5": 0.25,
        "weight_edge_24": 0.35,
        "weight_participation": wp,
        "calibration_path": str(CALIBRATION.relative_to(_ROOT)),
        "use_calibrated": True,
        "model_checkpoint": str(M2_CKPT.relative_to(_ROOT)),
    }
    cfg_path = OUT / "configs" / f"m2_wp{wp:.2f}.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
    return cfg_path


def main() -> int:
    if not M2_CKPT.is_file():
        raise FileNotFoundError(M2_CKPT)
    OUT.mkdir(parents=True, exist_ok=True)

    _run([
        sys.executable,
        "examples/calibrate_teq_edge.py",
        "--checkpoint",
        str(M2_CKPT.relative_to(_ROOT)),
        "--config",
        str(BASE_RULE.relative_to(_ROOT)),
        "--output",
        str(CALIBRATION.relative_to(_ROOT)),
        *_kline_args(),
    ])

    rows: list[dict] = []
    for wp in WP_GRID:
        cfg = _write_m2_config(wp)
        valid_dir = OUT / f"sweep_wp{wp:.2f}_valid"
        m = _backtest(cfg, M2_CKPT, "valid", valid_dir)
        cov = _coverage(valid_dir)
        row = {
            "w_part": wp,
            "config": str(cfg.relative_to(_ROOT)),
            "valid_return": float(m.get("total_return", 0.0)),
            "valid_coverage": cov,
            "valid_teq": int(m.get("trend_qualified_open_count", 0)),
            "valid_explore_pass": (
                float(m.get("total_return", 0.0)) >= EXPLORE_RETURN and cov >= EXPLORE_COVERAGE
            ),
        }
        rows.append(row)
        print(
            f"wp={wp:.2f} valid ret={row['valid_return']*100:.2f}% "
            f"cov={cov*100:.1f}% teq={row['valid_teq']}"
        )

    best = max(rows, key=lambda r: (r["valid_explore_pass"], r["valid_coverage"], r["valid_return"]))
    summary = {"rows": rows, "best": best, "calibration": str(CALIBRATION.relative_to(_ROOT))}
    (OUT / "teq_wp_sweep.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"best w_part={best['w_part']:.2f} (valid explore={'PASS' if best['valid_explore_pass'] else 'FAIL'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
