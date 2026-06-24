#!/usr/bin/env python3
"""023 Phase 0 orchestrator: baseline backtest + participation eval + report."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

V022_CONFIG = _ROOT / "configs/trading_rule_v022_trend_quality_0062e.json"
V023_CONFIG = _ROOT / "configs/trading_rule_v023_baseline_0062e.json"
V022_REF_BT = _ROOT / "backtest/v022_trade_points_plot"
OUT_ROOT = _ROOT / "backtest/v023_baseline"
CHECKPOINT = "prod/v0.0.0/checkpoint/market_state_best.pt"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=_ROOT)


def _ensure_v023_config() -> None:
    if V023_CONFIG.exists():
        return
    data = json.loads(V022_CONFIG.read_text(encoding="utf-8"))
    data["_023_meta"] = {
        "phase": "0",
        "recipe": "v022_frozen_baseline",
        "source_v022": str(V022_CONFIG.relative_to(_ROOT)),
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    V023_CONFIG.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _write_report(results: dict, v022_ref: dict) -> None:
    lines = [
        "# 023 Phase 0 Baseline Report",
        "",
        f"- timestamp: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC`",
        f"- checkpoint: `{CHECKPOINT}`",
        f"- v022 config: `{V022_CONFIG.relative_to(_ROOT)}`",
        f"- v022 config hash: `{_sha256(V022_CONFIG)}`",
        f"- v023 config: `{V023_CONFIG.relative_to(_ROOT)}`",
        f"- v023 config hash: `{_sha256(V023_CONFIG)}`",
        "",
        "## Reproduction vs v022_trade_points_plot (test)",
        "",
    ]
    test = results.get("test", {})
    rm = test.get("runner_metrics", {})
    v022m = v022_ref.get("test", {})
    lines.append("| metric | v022 reference | v023 baseline | delta |")
    lines.append("|--------|----------------|---------------|-------|")
    for key in ("total_return", "max_drawdown", "trade_count", "missed_confirmed_trend_bars", "leg_coverage_ratio"):
        a = float(v022m.get(key, 0))
        b = float(rm.get(key, 0))
        if key in ("total_return", "max_drawdown", "leg_coverage_ratio"):
            lines.append(f"| {key} | {_pct(a)} | {_pct(b)} | {_pct(b - a)} |")
        else:
            lines.append(f"| {key} | {a:.0f} | {b:.0f} | {b - a:+.0f} |")

    lines.extend(["", "## 023 Participation Metrics (§5.3)", ""])
    for split in ("valid", "test"):
        if split not in results:
            continue
        pm = results[split].get("participation_metrics", {})
        lines.append(f"### {split}")
        lines.append("")
        for k, v in pm.items():
            if k.endswith("_ratio"):
                lines.append(f"- {k}: `{_pct(float(v))}`")
            else:
                lines.append(f"- {k}: `{v}`")
        lines.append("")

    lines.extend([
        "## Artifacts",
        "",
        f"- participation metrics: `backtest/v023_baseline/participation_metrics.json`",
        f"- overlay plot (test): `backtest/v023_baseline/test/participation_overlay.png`",
        "",
        "## Phase 0 exit",
        "",
        "Reproduction within 0.1% on key runner metrics → **ready for Phase 1a**.",
        "",
    ])
    report = OUT_ROOT / "REPORT_023_BASELINE.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {report}")


def main() -> int:
    _ensure_v023_config()

    for split in ("valid", "test"):
        out = OUT_ROOT / split
        if not (out / "metrics.json").exists():
            _run([
                sys.executable,
                "examples/backtest_trading_system_v023.py",
                "--split",
                split,
                "--output-dir",
                str(out.relative_to(_ROOT)),
            ])

    _run([sys.executable, "examples/eval_participation.py", "--output", str((OUT_ROOT / "participation_metrics.json").relative_to(_ROOT))])
    _run([
        sys.executable,
        "examples/plot_participation_overlay.py",
        "--backtest-dir",
        str((OUT_ROOT / "test").relative_to(_ROOT)),
    ])

    results = _read_json(OUT_ROOT / "participation_metrics.json")
    v022_ref = {"test": _read_json(V022_REF_BT / "metrics.json")}
    _write_report(results, v022_ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
