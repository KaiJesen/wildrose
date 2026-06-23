#!/usr/bin/env bash
set -euo pipefail
PROD_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$PROD_ROOT/../.." && pwd)"
export PYTHONPATH="$PROD_ROOT/code:${PYTHONPATH:-}"
cd "$REPO_ROOT"
exec python3 "$PROD_ROOT/scripts/backtest.py" --config "$PROD_ROOT/config/trading_rule.json" --checkpoint "$PROD_ROOT/checkpoint/market_state_best.pt" "$@"
