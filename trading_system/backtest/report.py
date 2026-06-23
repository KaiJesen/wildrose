from __future__ import annotations

import json
from pathlib import Path


def write_report(out_dir: Path, *, metrics: dict[str, float], config_path: str, checkpoint: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Backtest Report v014",
        "",
        f"- config: `{config_path}`",
        f"- checkpoint: `{checkpoint}`",
        "",
        "## Metrics",
        "",
    ]
    priority = (
        "annualized_return",
        "benchmark_annualized_return",
        "excess_annualized_return",
        "max_drawdown",
        "trade_count",
    )
    shown = set()
    for k in priority:
        if k in metrics:
            v = metrics[k]
            lines.append(f"- {k}: `{v:.6f}` ({_pct(v)})")
            shown.add(k)
    for k, v in metrics.items():
        if k in shown:
            continue
        if isinstance(v, float):
            if (k.endswith("_return") or k.endswith("_ratio")) and "position" not in k:
                lines.append(f"- {k}: `{v:.6f}` ({_pct(v)})")
            else:
                lines.append(f"- {k}: `{v:.6f}`")
        else:
            lines.append(f"- {k}: `{v}`")
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "metrics.txt").write_text("\n".join(f"{k}={v}" for k, v in metrics.items()) + "\n", encoding="utf-8")


def _pct(v: float) -> str:
    return f"{100.0 * v:.2f}%"

