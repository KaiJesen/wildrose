# Backtest Report

- symbol: `BTCUSDT`
- split: `valid`
- config: `configs/trading_rule_v023_baseline_0062e.json`
- checkpoint: `prod/v0.0.0/checkpoint/market_state_best.pt`

## 核心指标

- 年化收益率: `30.73%`
- 总收益率: `4.09%`
- 基准收益率: `8.64%`
- 超额收益率: `-4.55%`
- 最大回撤: `-1.95%`
- 胜率: `58.82%`
- 盈亏比: `6.022326`
- 交易次数: `17`
- 平均持仓周期: `7.647059`

## 全部指标

- benchmark_annualized_return: `74.11%`
- excess_annualized_return: `-43.37%`
- avg_fee_per_trade: `0.000311`
- bar_count: `1309.000000`
- max_margin_loss_ratio_observed: `0.236488`
- position_limit_violations: `0.000000`
- risk_rule_violations: `0.000000`
- probe_short_count: `0.000000`
- probe_short_win_rate: `0.000000`
- probe_short_total_return: `0.00%`
- sentinel_short_count: `0.000000`
- sentinel_upgrade_count: `0.000000`
- sentinel_not_confirmed_close_count: `0.000000`
- sentinel_short_total_return: `0.00%`
- blocked_long_downtrend_count: `0.000000`
- missed_downtrend_bars: `368.000000`
- short_coverage_downtrend_ratio: `0.062645`
- model_short_trend_hold_count: `0.000000`
- avg_model_short_hold_bars: `1.000000`
- close_max_hold_bars_in_downtrend_count: `2.000000`
- close_short_trend_broken_count: `0.000000`
- crash_short_count: `3.000000`
- crash_upgrade_count: `0.000000`
- same_regime_reentry_count: `0.000000`
- model_blind_crash_count: `3.000000`
- upgrade_crash_to_trend_short_count: `0.000000`
- hold_crash_trend_confirming_count: `23.000000`
- close_trend_exit_confirmed_count: `0.000000`
- reduce_trend_profit_lock_count: `0.000000`
- hold_trend_runner_count: `0.000000`
- trend_upgrade_count: `0.000000`
- trend_trade_count: `0.000000`
- trend_trade_total_return: `0.00%`
- avg_trend_hold_bars: `0.000000`
- close_trend_broken_count: `0.000000`
- reduce_trend_exhaustion_count: `0.000000`
- add_trend_continuation_count: `0.000000`
- short_trend_capture_ratio: `0.195122`
- long_trend_capture_ratio: `0.036585`
- missed_confirmed_trend_bars: `254.000000`
- missed_slow_uptrend_bars: `597.000000`
- slow_up_open_count: `0.000000`
- watch_slow_uptrend_count: `597.000000`
- upgrade_slow_long_to_trend_count: `0.000000`
- close_slow_uptrend_broken_count: `0.000000`
- reduce_slow_up_profit_lock_count: `0.000000`
- hold_slow_up_runner_count: `0.000000`
- slow_up_trade_count: `0.000000`
- slow_up_trade_total_return: `0.00%`
- avg_slow_up_hold_bars: `0.000000`
- leg_coverage_ratio: `0.113043`
- missed_slow_up_legs: `10.000000`
- missed_fast_down_legs: `15.000000`
- avg_hold_vs_leg_duration: `7.625000`
- false_leg_entry_count: `0.000000`
- close_trend_leg_end_count: `0.000000`
- block_counter_trend_count: `0.000000`
- bias_field_nonempty_ratio: `1.000000`
- bias_reason_nonempty_ratio: `1.000000`
- hard_counter_open_count: `0.000000`
- legacy_trend_direct_block_count: `0.000000`
- legacy_trend_direct_read_count: `0.000000`
- bias_reason_codes_coverage: `1.000000`
- max_position_ratio_observed: `0.172306`
- trend_add_candidate_count: `0.000000`
- trend_add_risk_evaluated_count: `0.000000`
- trend_add_rejected_by_risk_count: `0.000000`
- trend_add_allowed_count: `0.000000`

## 图表

- 资金曲线: `backtest/v023_baseline/valid/equity_curve.png`
- 买卖点: `backtest/v023_baseline/valid/trade_points.png`
