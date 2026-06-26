#!/usr/bin/env python3
"""025: valid-only grid for slow-up participation gate (tau, edge threshold)."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(_ROOT / "examples"))

from _v025_common import PW20_CKPT, kline_backtest_args, verify_pw20_checkpoint

A3A_BASE = _ROOT / "configs/trading_rule_v025_a3a_slow_up.json"
B0_CONFIG = _ROOT / "configs/trading_rule_v024_phase1c_teq_0065a_c1_pw20.json"
OUT = _ROOT / "backtest/v025_tune"
TAU_GRID = [0.35, 0.45, 0.55]
EDGE_GRID = [-0.35, -0.25, -0.15]


def _run_backtest(cfg_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            sys.executable,
            "examples/backtest_trading_system_v014.py",
            "--config",
            str(cfg_path.relative_to(_ROOT)),
            "--checkpoint",
            str(PW20_CKPT.relative_to(_ROOT)),
            "--split",
            "valid",
            "--output-dir",
            str(out_dir.relative_to(_ROOT)),
            *kline_backtest_args(),
        ],
        cwd=_ROOT,
        stdout=subprocess.DEVNULL,
    )
    subprocess.check_call(
        [
            sys.executable,
            "examples/eval_participation.py",
            "--backtest-dir",
            str(out_dir.relative_to(_ROOT)),
            "--split",
            "valid",
            "--output",
            str((out_dir / "part.json").relative_to(_ROOT)),
        ],
        cwd=_ROOT,
        stdout=subprocess.DEVNULL,
    )
    m = json.loads((out_dir / "metrics.json").read_text())
    part = json.loads((out_dir / "part.json").read_text())["valid"]["participation_metrics"]
    return {
        "return": float(m.get("total_return", 0)),
        "coverage": float(part.get("leg_count_coverage_ratio", 0)),
        "slow_up_open": int(m.get("slow_up_open_count", 0)),
        "teq_open": int(m.get("trend_qualified_open_count", 0)),
        "slow_up_false": int(part.get("slow_up_false_entry_count", 0)),
        "counter_leg": int(part.get("counter_leg_participation_count", 0)),
    }


def main() -> int:
    verify_pw20_checkpoint()
    base = json.loads(A3A_BASE.read_text())
    b0 = _run_backtest(B0_CONFIG, OUT / "b0_valid")
    rows: list[dict] = [{"tag": "b0", "tau": None, "edge": None, **b0}]
    OUT.mkdir(parents=True, exist_ok=True)
    cfg_dir = OUT / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    for tau in TAU_GRID:
        for edge in EDGE_GRID:
            cfg = copy.deepcopy(base)
            gate = cfg["participation_channel"]["slow_up_gate"]
            gate["tau_slow"] = tau
            gate["edge_threshold_slow"] = edge
            tag = f"tau{tau}_edge{edge}"
            cfg_path = cfg_dir / f"a3a_{tag}.json"
            cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
            row = _run_backtest(cfg_path, OUT / f"{tag}_valid")
            row.update({"tag": tag, "tau": tau, "edge": edge})
            row["incr_cov"] = row["coverage"] - b0["coverage"]
            rows.append(row)
            print(
                f"{tag}: ret={row['return']*100:.2f}% cov={row['coverage']*100:.1f}% "
                f"slow_up={row['slow_up_open']} incr_cov={row['incr_cov']*100:.2f}pp"
            )

    candidates = [
        r
        for r in rows
        if r["tag"] != "b0"
        and r["slow_up_open"] > 0
        and r["incr_cov"] >= 0.008
        and r["return"] >= b0["return"] - 0.01
        and r["slow_up_false"] <= 3
    ]
    best = max(candidates, key=lambda r: (r["incr_cov"], r["return"])) if candidates else None
    summary = {"b0_valid": b0, "rows": rows, "best": best}
    (OUT / "tune_summary.json").write_text(json.dumps(summary, indent=2))
    if best:
        best_cfg = json.loads((cfg_dir / f"a3a_{best['tag']}.json").read_text())
        out_cfg = _ROOT / "configs/trading_rule_v025_a3a_slow_up_tuned.json"
        out_cfg.write_text(json.dumps(best_cfg, indent=2) + "\n")
        print(f"best: {best['tag']} -> {out_cfg.relative_to(_ROOT)}")
    else:
        print("no candidate passed valid gates; see tune_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
