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
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"- {k}: `{v:.6f}`")
        else:
            lines.append(f"- {k}: `{v}`")
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "metrics.txt").write_text("\n".join(f"{k}={v}" for k, v in metrics.items()) + "\n", encoding="utf-8")

