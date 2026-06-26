#!/usr/bin/env bash
# 一键启动 wildrose 交易监控网站（API + 行情轮询 + bar_runner）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBSITE="$ROOT/website"

export MONITOR_RUNTIME="$RUNTIME"
export MONITOR_DB="$RUNTIME/data/monitor.db"
export MONITOR_BACKENDS="$RUNTIME/configs/backends.yaml"
export PYTHONPATH="$WEBSITE/src:$ROOT:${PYTHONPATH:-}"

mkdir -p "$RUNTIME/data" "$RUNTIME/logs"

if [[ ! -d "$RUNTIME/venv" ]]; then
  echo "[setup] creating venv..."
  python3 -m venv "$RUNTIME/venv"
  "$RUNTIME/venv/bin/pip" install -U pip -q
  "$RUNTIME/venv/bin/pip" install -r "$RUNTIME/requirements.txt" -q
  "$RUNTIME/venv/bin/pip" install -e "$ROOT" -q 2>/dev/null || true
fi

PY="$RUNTIME/venv/bin/python"
PIP="$RUNTIME/venv/bin/pip"

# ensure deps
$PIP install -i https://pypi.tuna.tsinghua.edu.cn/simple -r "$RUNTIME/requirements.txt" -q

echo "[import] bootstrap archive backends..."
$PY "$WEBSITE/scripts/import_backtest.py" --all || true

API_PORT="${MONITOR_PORT:-8765}"
if command -v fuser >/dev/null 2>&1; then
  fuser -k "${API_PORT}/tcp" 2>/dev/null || true
  sleep 1
fi
echo "[api] starting on :$API_PORT ..."
$PY -m uvicorn monitor.api.app:app --host 0.0.0.0 --port "$API_PORT" \
  > "$RUNTIME/logs/api.log" 2>&1 &
API_PID=$!

sleep 2
echo "[runner] starting bar_runner..."
$PY "$RUNTIME/bar_runner.py" --api-url "http://127.0.0.1:$API_PORT" \
  > "$RUNTIME/logs/bar_runner.log" 2>&1 &
RUNNER_PID=$!

echo ""
echo "=========================================="
echo " wildrose monitor running"
echo " API:      http://127.0.0.1:$API_PORT"
echo " Frontend: http://127.0.0.1:$API_PORT/"
echo " logs:     $RUNTIME/logs/"
echo " stop:     kill $API_PID $RUNNER_PID"
echo "=========================================="

wait $API_PID
