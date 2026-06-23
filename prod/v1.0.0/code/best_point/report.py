from __future__ import annotations

import json
from pathlib import Path


def write_report(out_dir: str | Path, metrics: dict, title: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# {title}", "", "## Metrics", ""]
    for k, v in metrics.items():
        lines.append(f"- {k}: `{v}`")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

