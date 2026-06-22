# Backtest Report v014 — Aggressive Tuned (0062e, OOS Test)

- checkpoint: `checkpoints/0062e_market_state_return_ic_recovery/market_state_best.pt`
- config: `configs/trading_rule_v014_aggressive_tuned_0062e.json`
- tuning: valid grid → `backtest/backtest_rule_v014_aggressive_tuning_0062e/`

## Tuned Thresholds

Same as 0065a grid optimum: edge=0.04, prob=0.34, flat_max=0.40, risk_open=0.45, risk_exit=0.52

## Metrics (test)

| Metric | 0062e | 0065a (ref) |
|--------|-------|-------------|
| total_return | **+11.93%** | +4.24% |
| benchmark_return | -17.82% | -17.82% |
| excess_return | **+29.75%** | +22.06% |
| max_drawdown | **-1.51%** | -4.52% |
| trade_count | 6 | 7 |
| win_rate | **83.3%** | 71.4% |
| profit_factor | 98.3 | 5.42 |

Valid (tune split): +2.80% return, 11 trades, MDD -9.58%

## Baseline

Raw aggressive config on test: **0 trades** (same prob-threshold mismatch).

## Artifacts

- `backtest/backtest_rule_v014_aggressive_tuned_0062e_test/`
- `backtest/backtest_rule_v014_aggressive_tuning_0062e/tuning_results.txt`
