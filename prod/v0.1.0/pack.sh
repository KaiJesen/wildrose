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
sed -i 's/parents\[1\]/parents[2]/' "$PROD/scripts/_train_common.py"

cp "$ROOT/configs/trading_rule_v020_trend_segment_tuned_0062e.json" "$PROD/config/trading_rule.json"
cp "$ROOT/checkpoints/0065a_multi_seed_s45_market_state_stability/market_state_best.pt" "$PROD/checkpoint/market_state_best.pt"
cp "$ROOT/backtest/020_trend_segment_tuned_test/metrics.json" "$PROD/metrics/backtest_test_oos.json"
cp "$ROOT/backtest/020_best_point_tuned_test/metrics.json" "$PROD/metrics/backtest_bp_filtered_test_oos.json"
cp "$ROOT/backtest/020_trend_segment_tuned_test/backtest_plot.png" "$PROD/metrics/backtest_plot.png"
cp "$ROOT/backtest/020_best_point_tuned_test/backtest_plot.png" "$PROD/metrics/backtest_bp_filtered_plot.png"
cp "$ROOT/backtest/020_trend_segment_tuned_test/REPORT_020_BACKTEST.md" "$PROD/docs/REPORT_020_BACKTEST.md"
cp "$ROOT/backtest/020_best_point_tuned_test/REPORT_020_BEST_POINT.md" "$PROD/docs/REPORT_020_BEST_POINT.md"

GIT_SHA="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DATE="$(git -C "$ROOT" log -1 --format=%ci 2>/dev/null || echo unknown)"

cat > "$PROD/MANIFEST.json" <<EOF
{
  "product": "wildrose-trading-system",
  "version": "v0.1.0",
  "release_date": "$(date -u +%Y-%m-%d)",
  "git_commit": "$GIT_SHA",
  "git_commit_date": "$GIT_DATE",
  "strategy": "v020 trend-segment tuned",
  "model_checkpoint": "checkpoint/market_state_best.pt",
  "model_source": "0065a_multi_seed_s45_market_state_stability",
  "config": "config/trading_rule.json",
  "code_packages": ["trading_system", "transformer_kit", "market_data", "best_point"],
  "backtest_reference": {
    "split": "test",
    "metrics_file": "metrics/backtest_test_oos.json"
  },
  "supplemental_references": {
    "best_point_filtered_metrics": "metrics/backtest_bp_filtered_test_oos.json",
    "primary_report": "docs/REPORT_020_BACKTEST.md",
    "best_point_report": "docs/REPORT_020_BEST_POINT.md"
  }
}
EOF

echo "packed prod snapshot at $PROD"
