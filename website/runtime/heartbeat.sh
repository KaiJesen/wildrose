#!/usr/bin/env bash
# 发送 runner 心跳到监控网站（配合 bar_runner 或单独运行）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MONITOR_RUNTIME="$RUNTIME"
export MONITOR_DB="$RUNTIME/data/monitor.db"
export MONITOR_BACKENDS="$RUNTIME/configs/backends.yaml"
export PYTHONPATH="$ROOT/website/src:$ROOT:${PYTHONPATH:-}"

BACKEND_ID="${1:-prod-v1.0.0-live}"
API_URL="${MONITOR_API_URL:-http://127.0.0.1:8765}"

"$RUNTIME/venv/bin/python" - <<PY
import os, sys
sys.path.insert(0, "$ROOT/website/src")
from monitor.exporter import MonitorExporter
ok = MonitorExporter("$API_URL", "$BACKEND_ID").heartbeat()
print("heartbeat", "ok" if ok else "failed")
PY
