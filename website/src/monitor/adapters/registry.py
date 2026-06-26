from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from monitor.adapters.archive_import import import_archive
from monitor.db import connect, upsert_backend
from monitor.schemas import BackendProfile
from monitor.settings import BACKENDS_YAML, REPO_ROOT


def _resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p


def load_backends_yaml(path: Path | None = None) -> list[BackendProfile]:
    cfg_path = path or BACKENDS_YAML
    if not cfg_path.exists():
        return []
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    items = raw.get("backends") or []
    return [BackendProfile.model_validate(x) for x in items]


def register_backends(profiles: list[BackendProfile] | None = None) -> list[BackendProfile]:
    profiles = profiles or load_backends_yaml()
    with connect() as conn:
        for p in profiles:
            upsert_backend(
                conn,
                {
                    "id": p.id,
                    "display_name": p.display_name,
                    "symbol": p.symbol,
                    "interval": p.interval,
                    "mode": p.mode,
                    "manifest_path": p.manifest_path,
                    "config_path": p.config_path,
                    "data_root": p.data_root,
                    "checkpoint": p.checkpoint,
                },
            )
    return profiles


def bootstrap_archives(profiles: list[BackendProfile] | None = None) -> list[dict[str, Any]]:
    profiles = profiles or load_backends_yaml()
    results = []
    for p in profiles:
        if not p.data_root:
            continue
        if p.mode not in ("archive", "live"):
            continue
        root = _resolve(p.data_root)
        if root is None or not root.exists():
            continue
        r = import_archive(
            p.id,
            root,
            display_name=p.display_name,
            symbol=p.symbol,
            interval=p.interval,
            mode=p.mode,
            manifest_path=p.manifest_path,
            config_path=p.config_path,
        )
        results.append(r.__dict__)
    return results


def load_reference_metrics(manifest_path: str | None) -> dict[str, Any]:
    mp = _resolve(manifest_path)
    if mp is None or not mp.exists():
        return {}
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    ref = manifest.get("backtest_reference") or {}
    metrics_file = ref.get("metrics_file")
    if not metrics_file:
        return {}
    metrics_path = mp.parent / metrics_file
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))
