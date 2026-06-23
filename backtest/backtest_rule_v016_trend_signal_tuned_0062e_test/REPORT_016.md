# v016 tuned backtest summary

## Tune setup

- checkpoint: `checkpoints/0062e_market_state_return_ic_recovery/market_state_best.pt`
- tune split: `valid`
- test split: `test`
- tuner: `examples/tune_backtest_trading_system_v016.py`

## Tuned parameters

- `trend_signal.confirmed_score = 3`
- `trend_signal.strong_score = 4`
- `trend_signal.extreme_score = 5`
- `trend_position.upgrade_profit_atr = 0.5`
- `trend_position.add_profit_atr = 1.5`
- rule thresholds unchanged from tuned v014b baseline

## OOS test metrics

- total_return: `9.49%`
- excess_return: `27.31%`
- max_drawdown: `-1.05%`
- trade_count: `10`
- profit_factor: `8.08`
- trend_upgrade_count: `3`
- trend_trade_total_return: `8.24%`
- avg_trend_hold_bars: `16.75`
- add_trend_continuation_count: `2`
- reduce_trend_exhaustion_count: `6`
- short_trend_capture_ratio: `20.46%`
- position_limit_violations: `0`
- risk_rule_violations: `0`

## Comparison

| strategy | return | max drawdown | trades | trend upgrades | trend trade return |
| --- | ---: | ---: | ---: | ---: | ---: |
| v014b tuned | 8.35% | -0.63% | 4 | n/a | n/a |
| v015 | 8.87% | -0.63% | 11 | n/a | n/a |
| v016 default | 9.66% | -0.58% | 11 | 0 | 5.70% |
| v016 tuned | 9.49% | -1.05% | 10 | 3 | 8.24% |

## Acceptance checklist

- pass: `position_limit_violations == 0`
- pass: `risk_rule_violations == 0`
- pass: `max_margin_loss_ratio_observed < 1.0`
- pass: `trend_upgrade_count >= 1`
- pass: `trend_trade_total_return > 0`
- pass: `total_return >= v014b`
- pass: `max_drawdown <= v014b + 2pp`
- near miss: `trade_count <= v014b + 5` target is `<= 9`, actual is `10`

## Interpretation

The tuning accomplished the main 016 objective: it turned the trend module from a passive observer into an active hold-upgrade mechanism. Lowering the confirmation ladder by one notch and cutting `upgrade_profit_atr` from `1.5` to `0.5` unlocked three real trend upgrades on the test set, and those upgraded trend trades contributed most of the strategy return.

The remaining weakness is trade count discipline. The tuned config removes one trade versus default v016, but it still finishes one trade above the design cap.
