#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROD="$(cd "$(dirname "$0")" && pwd)"

rm -rf "$PROD/code"
mkdir -p "$PROD/code" "$PROD/config" "$PROD/checkpoint" "$PROD/metrics" "$PROD/scripts" "$PROD/docs"

for pkg in trading_system transformer_kit market_data best_point; do
  rsync -a --exclude '__pycache__' "$ROOT/$pkg/" "$PROD/code/$pkg/"
done

cp "$ROOT/examples/_train_common.py" "$PROD/scripts/_train_common.py"
cp "$ROOT/examples/backtest_trading_system_v014.py" "$PROD/scripts/backtest.py"
cp "$ROOT/prod/v1.0.0/scripts/run_backtest.sh" "$PROD/scripts/run_backtest.sh"
sed -i 's/parents\[1\]/parents[2]/' "$PROD/scripts/_train_common.py"

cp "$ROOT/configs/trading_rule_v022_trend_quality_0062e.json" "$PROD/config/trading_rule.json"
cp "$ROOT/prod/v0.0.0/checkpoint/market_state_best.pt" "$PROD/checkpoint/market_state_best.pt"

cp "$ROOT/backtest/v022_system_test/current_p10_ir7/test_bt/metrics.json" "$PROD/metrics/backtest_test_oos.json"
cp "$ROOT/backtest/v022_system_test/current_p10_ir7/test_bt/equity_curve.png" "$PROD/metrics/backtest_plot.png"
cp "$ROOT/backtest/v022_system_test/current_p10_ir7/test_bt/trade_points.png" "$PROD/metrics/backtest_trade_points.png"
cp "$ROOT/backtest/v022_system_test/linkage_test/summary.json" "$PROD/metrics/validation_summary.json"

cp "$ROOT/backtest/v022_system_test/current_p10_ir7/test_bt/REPORT.md" "$PROD/docs/REPORT_022_BACKTEST.md"
cp "$ROOT/backtest/v022_system_test/linkage_test/VALIDATION_REPORT.md" "$PROD/docs/REPORT_022_VALIDATION.md"
cp "$ROOT/backtest/v022_system_test/SYSTEM_TEST_REPORT.md" "$PROD/docs/REPORT_022_SYSTEM_TEST.md"

GIT_SHA="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DATE="$(git -C "$ROOT" log -1 --format=%ci 2>/dev/null || echo unknown)"

cat > "$PROD/MANIFEST.json" <<EOF
{
  "product": "wildrose-trading-system",
  "version": "v1.1.0",
  "release_date": "$(date -u +%Y-%m-%d)",
  "git_commit": "$GIT_SHA",
  "git_commit_date": "$GIT_DATE",
  "strategy": "v022 trend-quality hybrid_v021_mem_bias_p10_ir7",
  "model_checkpoint": "checkpoint/market_state_best.pt",
  "model_source": "0062e_market_state_return_ic_recovery",
  "config": "config/trading_rule.json",
  "code_packages": ["trading_system", "transformer_kit", "market_data", "best_point"],
  "backtest_reference": {
    "symbol": "BTCUSDT",
    "interval": "1h",
    "days": 365,
    "split": "test",
    "metrics_file": "metrics/backtest_test_oos.json"
  },
  "supplemental_references": {
    "phase_validation": "docs/REPORT_022_VALIDATION.md",
    "validation_summary": "metrics/validation_summary.json",
    "primary_report": "docs/REPORT_022_BACKTEST.md",
    "system_test_report": "docs/REPORT_022_SYSTEM_TEST.md",
    "project_log": "docs/PROJECT_LOG.md"
  }
}
EOF

echo "packed prod snapshot at $PROD"
