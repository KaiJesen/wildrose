"""027 shared paths, gates, and backtest helpers."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(_ROOT / "examples"))

from _v025_common import kline_backtest_args, verify_pw20_checkpoint

M2_CKPT = _ROOT / "checkpoints/026_phase1_c1d1/market_state_best.pt"
B0_CONFIG = _ROOT / "configs/trading_rule_v024_phase1c_teq_0065a_c1_pw20.json"
CORE_BASELINE = _ROOT / "configs/trading_rule_v027_core_m2_baseline.json"
OUT_PHASE1 = _ROOT / "backtest/v027_phase1"

# Phase 1 valid gates (vs B0 valid baseline, measured at runtime)
RETURN_FLOOR_DELTA = -0.005
COVERAGE_BOOST_DELTA = 0.015
SLOW_UP_FALSE_MAX = 3

SLOW_UP_CHANNEL_PATCH: dict[str, Any] = {
    "participation_channel": {
        "enabled": True,
        "slow_up_gate": {
            "enabled": True,
            "tau_slow": 0.55,
            "edge_threshold_slow": -0.15,
            "probe_ratio": 0.5,
            "weight_legacy": 0.6,
            "weight_teq": 0.25,
            "weight_part": 0.15,
            "calibration_path": "backtest/v025/channel_edge_slow_up_calibration.json",
            "use_calibrated": False,
        },
    },
}


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in patch.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_ROOT.resolve()))
    except ValueError:
        return str(path)


def run_valid_backtest(
    cfg_path: Path,
    out_dir: Path,
    *,
    checkpoint: Path = M2_CKPT,
    dual_slot: bool = False,
    quiet: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "examples/backtest_trading_system_v014.py",
        "--config",
        repo_rel(cfg_path),
        "--checkpoint",
        repo_rel(checkpoint),
        "--split",
        "valid",
        "--output-dir",
        repo_rel(out_dir),
        *kline_backtest_args(),
    ]
    if dual_slot:
        cmd.append("--dual-slot")
    stdout = subprocess.DEVNULL if quiet else None
    subprocess.check_call(cmd, cwd=_ROOT, stdout=stdout)
    part_path = out_dir / "participation.json"
    subprocess.check_call(
        [
            sys.executable,
            "examples/eval_participation.py",
            "--backtest-dir",
            repo_rel(out_dir),
            "--split",
            "valid",
            "--output",
            repo_rel(part_path),
        ],
        cwd=_ROOT,
        stdout=stdout,
    )
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    part = json.loads(part_path.read_text(encoding="utf-8"))["valid"]["participation_metrics"]
    return {
        "return": float(metrics.get("total_return", 0.0)),
        "coverage": float(part.get("leg_count_coverage_ratio", 0.0)),
        "slow_up_open": int(metrics.get("slow_up_open_count", 0)),
        "slow_up_pnl": float(metrics.get("slow_up_trade_total_return", 0.0)),
        "teq_open": int(metrics.get("trend_qualified_open_count", 0)),
        "slow_up_false": int(part.get("slow_up_false_entry_count", 0)),
        "counter_leg": int(part.get("counter_leg_participation_count", 0)),
        "crash_short_count": int(metrics.get("crash_short_count", 0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
    }


def slow_up_force_disable(row: dict[str, Any]) -> bool:
    return row.get("slow_up_open", 0) > 0 and (
        row.get("slow_up_pnl", 0.0) < 0.0 and row.get("slow_up_false", 0) >= SLOW_UP_FALSE_MAX
    )


def passes_phase1_gates(row: dict[str, Any], *, b0_return: float, b0_coverage: float) -> bool:
    if slow_up_force_disable(row) and row.get("tag", "").startswith("slow_up"):
        return False
    return (
        row["return"] >= b0_return + RETURN_FLOOR_DELTA
        and row["coverage"] >= b0_coverage + COVERAGE_BOOST_DELTA
    )
