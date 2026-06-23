# Backtest Report v014 — Aggressive Tuned (OOS Test)

- config: `configs/trading_rule_v014_aggressive_tuned.json`
- checkpoint: `checkpoints/0065a_multi_seed_s45_market_state_stability/market_state_best.pt`
- split: **test** (1308 bars)
- tuning: valid grid search via `examples/tune_backtest_trading_system_v014.py`

## Tuned Rule Thresholds

| Parameter | Doc aggressive | Tuned |
|-----------|----------------|-------|
| open_edge_threshold | 0.05 | **0.04** |
| open_prob_threshold | 0.39 | **0.34** |
| open_flat_max | 0.40 | 0.40 |
| risk_open_max | 0.45 | 0.45 |
| risk_exit_threshold | 0.55 | **0.52** |

## Metrics

| Metric | Value |
|--------|-------|
| total_return | **+4.24%** |
| benchmark_return | -17.82% |
| excess_return | **+22.06%** |
| max_drawdown | -4.52% |
| win_rate | 71.4% (5/7) |
| profit_factor | 5.42 |
| trade_count | 7 |
| max_margin_loss_ratio_observed | 15.8% |
| position_limit_violations | 0 |
| risk_rule_violations | 0 |

## Baseline Comparison

Raw `trading_rule_v014_aggressive.json` on the same test split: **0 trades** (p_up/p_down never reach 0.39).

## Artifacts

- `trades.csv` — 7 round trips
- `decisions.csv` — per-bar signal + action log
- `equity_curve.csv` — mark-to-market equity
- Full analysis: `document/014_交易系统功能设计/回测师-014-进攻模式回测报告.md`
