#!/usr/bin/env python3
"""027 Phase 1 orchestrator: Core ablation + June case on baseline/best."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from _v027_common import CORE_BASELINE, M2_CKPT, OUT_PHASE1, repo_rel

OUT = OUT_PHASE1


def _run_test_baseline() -> Path:
    """June case is on test split — run M2 baseline backtest once for attribution."""
    out = OUT / "baseline_test"
    if (out / "decisions.csv").is_file():
        return out
    subprocess.check_call(
        [
            sys.executable,
            "examples/backtest_trading_system_v014.py",
            "--config",
            repo_rel(CORE_BASELINE),
            "--checkpoint",
            repo_rel(M2_CKPT),
            "--split",
            "test",
            "--output-dir",
            repo_rel(out),
            "--csv",
            "data/cache/kline/binance_vision_BTCUSDT_1h_365d_end20260625.csv",
        ],
        cwd=_ROOT,
    )
    return out


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=_ROOT)


def main() -> int:
    p = argparse.ArgumentParser(description="027 Phase 1")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--dual-slot", action="store_true")
    p.add_argument("--skip-ablation", action="store_true")
    args = p.parse_args()

    if not args.skip_ablation:
        cmd = [sys.executable, "examples/run_v027_core_ablation.py"]
        if args.quick:
            cmd.append("--quick")
        if args.dual_slot:
            cmd.append("--dual-slot")
        _run(cmd)

    summary_path = OUT / "phase1_ablation_summary.json"
    if not summary_path.is_file():
        print("missing ablation summary")
        return 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    targets: list[tuple[str, Path]] = [("baseline_valid", OUT / "baseline_valid")]
    test_dir = _run_test_baseline()
    targets.append(("baseline_test", test_dir))
    best = summary.get("best_valid")
    if best and best.get("tag"):
        targets.append((best["tag"], OUT / f"{best['tag']}_valid"))

    june_reports = {}
    for name, bt_dir in targets:
        if not bt_dir.is_dir():
            continue
        out_json = bt_dir / "june_case_diagnosis.json"
        _run([
            sys.executable,
            "examples/diagnose_june_case.py",
            "--backtest-dir",
            str(bt_dir.relative_to(_ROOT)),
            "--output",
            str(out_json.relative_to(_ROOT)),
        ])
        june_reports[name] = json.loads(out_json.read_text(encoding="utf-8"))

    phase1_pass = bool(summary.get("phase1_ablation_pass"))
    report = [
        "# 027 Phase 1 报告",
        "",
        f"| Core ablation (valid) | {'PASS' if phase1_pass else 'FAIL'} |",
        f"| slow_up 裁定 | {summary.get('slow_up_verdict', 'n/a')} |",
        f"| best valid arm | {best.get('tag') if best else 'none'} |",
        "",
        "详见 `REPORT_027_PHASE1_ABLATION.md`、`phase1_ablation_summary.json`",
        "",
        "```bash",
        "python examples/run_v027_phase1.py",
        "```",
    ]
    (OUT / "REPORT_027_PHASE1.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUT / "phase1_summary.json").write_text(
        json.dumps({"phase1_pass": phase1_pass, "ablation": summary, "june": june_reports}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
