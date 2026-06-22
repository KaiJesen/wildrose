# backtest

规则回测与调参输出目录，与 `reports/`（训练实验报告）分离。

## 脚本入口

| 脚本 | 默认输出 |
|------|----------|
| `examples/backtest_market_state_rule_v012.py` | `backtest/backtest_rule_v012` |
| `examples/tune_backtest_rule_v012.py` | `backtest/backtest_rule_v012_*_tuning` |
| `examples/optimize_backtest_rule_v012_return.py` | `backtest/backtest_rule_v012_*_opt_return` |
| `examples/plot_trade_points_compare_v012.py` | `backtest/backtest_rule_v012_compare` |
| `examples/sim_backtest_rule.py` | `backtest/btc_sim_backtest` |
