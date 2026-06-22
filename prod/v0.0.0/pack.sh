#!/usr/bin/env bash
# Rebuild prod/v0.0.0 snapshot from repository root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROD="$(cd "$(dirname "$0")" && pwd)"

rm -rf "$PROD/code"
mkdir -p "$PROD/code" "$PROD/config" "$PROD/checkpoint" "$PROD/metrics" "$PROD/scripts"

for pkg in trading_system transformer_kit market_data; do
  rsync -a --exclude '__pycache__' "$ROOT/$pkg/" "$PROD/code/$pkg/"
done

cp "$ROOT/examples/_train_common.py" "$PROD/scripts/_train_common.py"
cp "$ROOT/examples/backtest_trading_system_v014.py" "$PROD/scripts/backtest.py"
# prod 脚本位于 prod/v0.0.0/scripts，数据缓存仍使用仓库根目录
sed -i 's/parents\[1\]/parents[2]/' "$PROD/scripts/_train_common.py"
cp "$ROOT/configs/trading_rule_v016_trend_signal_tuned2_0062e.json" "$PROD/config/trading_rule.json"
cp "$ROOT/checkpoints/0062e_market_state_return_ic_recovery/market_state_best.pt" "$PROD/checkpoint/market_state_best.pt"
cp "$ROOT/backtest/backtest_rule_v016_trend_signal_tuned2_0062e_test/metrics.json" "$PROD/metrics/backtest_test_oos.json"

GIT_SHA="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DATE="$(git -C "$ROOT" log -1 --format=%ci 2>/dev/null || echo unknown)"

cat > "$PROD/MANIFEST.json" <<EOF
{
  "product": "wildrose-trading-system",
  "version": "v0.0.0",
  "release_date": "$(date -u +%Y-%m-%d)",
  "git_commit": "$GIT_SHA",
  "git_commit_date": "$GIT_DATE",
  "strategy": "v016 trend-signal tuned2",
  "model_checkpoint": "checkpoint/market_state_best.pt",
  "model_source": "0062e_market_state_return_ic_recovery",
  "config": "config/trading_rule.json",
  "code_packages": ["trading_system", "transformer_kit", "market_data"],
  "backtest_reference": {
    "split": "test",
    "metrics_file": "metrics/backtest_test_oos.json"
  }
}
EOF

echo "packed prod snapshot at $PROD"
