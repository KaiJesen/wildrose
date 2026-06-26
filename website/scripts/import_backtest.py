#!/usr/bin/env python3
"""Import configured archive backends into monitor SQLite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "website" / "src"))
sys.path.insert(0, str(ROOT))

from monitor.adapters.registry import bootstrap_archives, load_backends_yaml, register_backends
from monitor.db import init_db
from monitor.settings import BACKENDS_YAML, DB_PATH, REPO_ROOT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend-id", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--config", default=str(BACKENDS_YAML))
    args = ap.parse_args()

    init_db(DB_PATH)
    profiles = register_backends(load_backends_yaml(Path(args.config)))

    if args.all:
        results = bootstrap_archives(profiles)
        for r in results:
            print(f"imported {r['backend_id']}: decisions={r['decisions']} trades={r['trades']}")
        return 0

    if not args.backend_id:
        print("specify --backend-id or --all")
        return 1

    prof = next((p for p in profiles if p.id == args.backend_id), None)
    if not prof or not prof.data_root:
        print("backend not found or no data_root")
        return 1
    from monitor.adapters.archive_import import import_archive

    root = Path(prof.data_root)
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    r = import_archive(
        prof.id, root,
        display_name=prof.display_name,
        symbol=prof.symbol,
        interval=prof.interval,
        mode=prof.mode,
        manifest_path=prof.manifest_path,
        config_path=prof.config_path,
    )
    print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
