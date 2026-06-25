#!/usr/bin/env python3
"""024 Phase 3: full-chain A/B on frozen phase1c rule stack (A0/A1/A2)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
PHASE1C_CONFIG = _ROOT / "configs/trading_rule_v023_phase1c_0062e.json"
TEQ_CONFIG = _ROOT / "configs/trading_rule_v024_phase1c_teq_0065a1.json"
CKPT_A0 = _ROOT / "prod/v0.0.0/checkpoint/market_state_best.pt"
CKPT_A1 = _ROOT / "backtest/v024_phase1_tune/s1_os30_pw15/checkpoint/market_state_best.pt"
CKPT_A1_FALLBACK = _ROOT / "checkpoints/0065a_leg_align_v0/market_state_best.pt"
CKPT_A2 = _ROOT / "checkpoints/0065a_leg_align_v1/market_state_best.pt"
OUT = _ROOT / "backtest/v024_phase3_ab"
EXPLORE_RETURN = 0.0884
EXPLORE_COVERAGE = 0.28


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=_ROOT)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _part_split_metrics(part: dict, split: str) -> dict:
    if split in part and isinstance(part[split], dict):
        return part[split].get("participation_metrics", {})
    return part


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _backtest_arm(name: str, *, config: Path, checkpoint: Path, split: str) -> Path:
    out = OUT / f"{name}_{split}"
    _run([
        sys.executable,
        "examples/backtest_trading_system_v014.py",
        "--config",
        str(config.relative_to(_ROOT)),
        "--checkpoint",
        str(checkpoint.relative_to(_ROOT)),
        "--split",
        split,
        "--output-dir",
        str(out.relative_to(_ROOT)),
    ])
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt_a1 = CKPT_A1 if CKPT_A1.is_file() else CKPT_A1_FALLBACK
    for p, label in [(CKPT_A0, "A0"), (ckpt_a1, "A1"), (CKPT_A2, "A2")]:
        if not p.is_file():
            raise FileNotFoundError(f"missing {label} checkpoint: {p}")

    if not (OUT.parent / "v024_phase2/teq_edge_calibration.json").is_file():
        _run([sys.executable, "examples/run_v024_phase2.py"])

    arms = {
        "a0_0062e": (PHASE1C_CONFIG, CKPT_A0),
        "a1_0065a0": (PHASE1C_CONFIG, ckpt_a1),
        "a2_teq": (TEQ_CONFIG, CKPT_A2),
    }
    for split in ("valid", "test"):
        for name, (cfg, ckpt) in arms.items():
            _backtest_arm(name, config=cfg, checkpoint=ckpt, split=split)

    part_path = OUT / "participation_metrics.json"
    _run([
        sys.executable,
        "examples/eval_participation.py",
        "--backtest-dir",
        str((OUT / "a0_0062e_test").relative_to(_ROOT)),
        "--backtest-dir",
        str((OUT / "a1_0065a0_test").relative_to(_ROOT)),
        "--backtest-dir",
        str((OUT / "a2_teq_test").relative_to(_ROOT)),
        "--output",
        str(part_path.relative_to(_ROOT)),
    ])

    rows: list[dict] = []
    for name in arms:
        m = _read_json(OUT / f"{name}_test" / "metrics.json")
        rows.append({
            "arm": name,
            "total_return": float(m.get("total_return", 0.0)),
            "max_drawdown": float(m.get("max_drawdown", 0.0)),
            "trade_count": float(m.get("trade_count", 0.0)),
            "trend_qualified_open_count": float(m.get("trend_qualified_open_count", 0.0)),
            "trend_qualified_pnl": float(m.get("trend_qualified_pnl", 0.0)),
        })

    part = _read_json(part_path)
    a2_part_test = _part_split_metrics(part, "test")
    a2_cov = float(a2_part_test.get("leg_count_coverage_ratio", 0.0))
    a2_row = next((r for r in rows if r["arm"] == "a2_teq"), {})
    explore_pass = a2_row.get("total_return", 0.0) >= EXPLORE_RETURN and a2_cov >= EXPLORE_COVERAGE

    lines = [
        "# 024 Phase 3 A/B Report (frozen phase1c)",
        "",
        "## Test metrics",
        "",
        "| arm | return | max_dd | trades | teq_opens | teq_pnl |",
        "|-----|--------|--------|--------|-----------|---------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['arm']} | {_pct(r['total_return'])} | {_pct(r['max_drawdown'])} | "
            f"{int(r['trade_count'])} | {int(r['trend_qualified_open_count'])} | {_pct(r['trend_qualified_pnl'])} |"
        )
    lines.extend([
        "",
        "## Exploration gate (A2)",
        f"- return ≥ {_pct(EXPLORE_RETURN)}: **{'PASS' if a2_row.get('total_return', 0) >= EXPLORE_RETURN else 'FAIL'}** "
        f"(A2={_pct(a2_row.get('total_return', 0.0))})",
        f"- leg_count_coverage ≥ {_pct(EXPLORE_COVERAGE)}: **{'PASS' if a2_cov >= EXPLORE_COVERAGE else 'FAIL'}** "
        f"(A2={_pct(a2_cov)})",
        f"- overall explore line: **{'PASS' if explore_pass else 'FAIL'}**",
        "",
        "## Reproduction",
        "```bash",
        "python examples/run_v024_ab_phase1c.py",
        "```",
    ])
    report = OUT / "REPORT_024_AB.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "ab_summary.json").write_text(json.dumps({"arms": rows, "participation": part, "explore_pass": explore_pass}, indent=2), encoding="utf-8")
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
