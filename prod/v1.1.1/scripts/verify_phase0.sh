#!/usr/bin/env bash
set -euo pipefail
PROD_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$PROD_ROOT/../.." && pwd)"
export PYTHONPATH="$PROD_ROOT/code:${PYTHONPATH:-}"
OUT="${1:-$REPO_ROOT/backtest/prod_v1.1.1_smoke}"
cd "$REPO_ROOT"

python3 "$PROD_ROOT/scripts/backtest.py" \
  --config "$PROD_ROOT/config/trading_rule.json" \
  --checkpoint "$PROD_ROOT/checkpoint/market_state_best.pt" \
  --split test \
  --output-dir "$OUT" \
  --csv "$PROD_ROOT/data/kline/binance_vision_BTCUSDT_1h_365d_end20260625.csv"

python3 "$REPO_ROOT/examples/eval_participation.py" \
  --backtest-dir "$OUT" \
  --output "$OUT/participation.json"

python3 - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
m = json.loads((out / "metrics.json").read_text())
p = json.loads((out / "participation.json").read_text())
cov = p["test"]["participation_metrics"]["leg_count_coverage_ratio"]
ret = m["total_return"]
teq = int(m["trend_qualified_open_count"])
ok = abs(ret - 0.0901) <= 0.002 and abs(cov - 0.267) <= 0.01 and teq == 3
print(f"return={ret*100:.2f}% coverage={cov*100:.2f}% teq={teq} -> {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
