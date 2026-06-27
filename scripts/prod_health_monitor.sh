#!/usr/bin/env bash
# P0: prod v1.1.0 health check + decision attribution (architect 027 post-close §6)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== prod v1.1.0 test OOS health + attribution =="
python examples/run_prod_consolidation.py --split test

echo ""
echo "== prod v1.1.0 valid blind-spot attribution (no baseline gate) =="
python examples/run_prod_consolidation.py --split valid || true

echo ""
echo "Reports:"
echo "  backtest/prod_monitor/v1.1.0_test/REPORT_PROD_CONSOLIDATION.md"
echo "  backtest/prod_monitor/v1.1.0_valid/REPORT_PROD_CONSOLIDATION.md"
