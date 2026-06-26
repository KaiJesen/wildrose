from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WEBSITE_ROOT = _REPO_ROOT / "website"
_RUNTIME_ROOT = Path(os.environ.get("MONITOR_RUNTIME", _WEBSITE_ROOT / "runtime"))

DB_PATH = Path(os.environ.get("MONITOR_DB", _RUNTIME_ROOT / "data" / "monitor.db"))
BACKENDS_YAML = Path(os.environ.get("MONITOR_BACKENDS", _RUNTIME_ROOT / "configs" / "backends.yaml"))
API_HOST = os.environ.get("MONITOR_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("MONITOR_PORT", "8765"))
POLL_INTERVAL_S = int(os.environ.get("MONITOR_POLL_INTERVAL", "45"))
REPO_ROOT = _REPO_ROOT
