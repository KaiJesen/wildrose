# backtest

规则回测与调参输出目录，与 `reports/`（训练实验报告）分离。

## 目录命名

| 类型 | 格式 | 示例 |
|------|------|------|
| 交易系统迭代（017+） | `{流水号}_{描述}_{split}` | `020_trend_segment_tuned_test` |
| 调参网格 | `{流水号}_{描述}_tuning_{split}` | `019_slow_uptrend_tuning_valid` |
| 多 split 对比汇总 | `{流水号}_report_summary.json` | `018_report_summary.json` |
| 规则版本 + checkpoint（014–016） | `backtest_rule_v{版本}_{checkpoint}_{split}` | `backtest_rule_v016_trend_signal_tuned2_0062e_test` |

流水号与 `document/{NNN}/` 项目编号、`reports/{NNN}_*` 保持一致。

## 脚本入口

| 脚本 | 默认输出 |
|------|----------|
| `examples/backtest_market_state_rule_v012.py` | `backtest/backtest_rule_v012` |
| `examples/tune_backtest_rule_v012.py` | `backtest/backtest_rule_v012_*_tuning` |
| `examples/optimize_backtest_rule_v012_return.py` | `backtest/backtest_rule_v012_*_opt_return` |
| `examples/plot_trade_points_compare_v012.py` | `backtest/backtest_rule_v012_compare` |
| `examples/sim_backtest_rule.py` | `backtest/btc_sim_backtest` |
| `examples/tune_backtest_trading_system_v018.py` | `backtest/018_lifecycle_tuning_valid` |
| `examples/tune_backtest_trading_system_v019.py` | `backtest/019_slow_uptrend_tuning_valid` |

## 020 最新回测

| 目录 | 说明 |
|------|------|
| `020_trend_segment_test` | v020 初版（slow_up 交易开启） |
| `020_trend_segment_tuned_test` | v020 调参版（推荐，test +13.24%） |
