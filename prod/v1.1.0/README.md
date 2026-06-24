# wildrose prod v1.1.0

生产固化包：当前正式候选的 **v022 trend-quality** 交易系统。

## 版本信息

| 项 | 值 |
|---|---|
| 产品版本 | `v1.1.0` |
| 策略代号 | `v022 trend-quality` |
| 固化配方 | `hybrid_v021_mem_bias_p10_ir7` |
| 模型 checkpoint | `0062e_market_state_return_ic_recovery` |
| 标的 / 周期 | BTCUSDT 1h |
| 数据窗口 | 365 天 |

## 目录结构

```text
prod/v1.1.0/
  README.md
  MANIFEST.json
  config/trading_rule.json
  checkpoint/market_state_best.pt
  metrics/
    backtest_test_oos.json
    backtest_plot.png
    backtest_trade_points.png
    validation_summary.json
  docs/
    REPORT_022_BACKTEST.md
    REPORT_022_VALIDATION.md
    REPORT_022_SYSTEM_TEST.md
    PROJECT_LOG.md
  code/
    trading_system/
    transformer_kit/
    market_data/
    best_point/
  scripts/
    backtest.py
    _train_common.py
    run_backtest.sh
  pack.sh
```

## OOS test 参考指标

来源：`metrics/backtest_test_oos.json`

| 指标 | 值 |
|---|---:|
| total_return | 9.31% |
| annualized_return | 81.55% |
| excess_return | 26.47% |
| max_drawdown | -0.85% |
| win_rate | 75.0% |
| trade_count | 8 |
| profit_factor | 33.70 |
| risk_rule_violations | 0 |

## 关键策略特性

- 延续 `v021` 的 bias 统一调度链，但切换到 `v022` 趋势质量版本
- `TrendSignal` 引入记忆状态机：`invalid_reset_bars=7`、`persistence_lookback=10`
- 集成 `TrendSegment` 增量计算与性能优化
- `TrendBias` 采用 soft legacy block 与 `chop_soft_micro_weight` 重平衡
- 当前生产配方中 `chop_guard_enabled=false`，以保留系统级 PnL
- Business 底线验证通过：test 收益优于 `v021`，回撤不恶化，交易数持平

## 相对 v1.0.0 的变化

| 指标 | v1.0.0 (v021) | v1.1.0 (v022) |
|---|---:|---:|
| test 总收益 | 8.25% | **9.31%** |
| test 最大回撤 | -0.93% | **-0.85%** |
| test 交易笔数 | 8 | 8 |
| test 胜率 | 75.0% | 75.0% |

v022 在维持交易频率的同时，提升了 test OOS 收益并略降回撤；代价是趋势模块评估中的 `false_confirm_on_range_teacher` 仍未达标，当前版本选择优先固化系统交易效果。

## 运行回测

在仓库根目录执行：

```bash
bash prod/v1.1.0/scripts/run_backtest.sh --split test --output-dir backtest/prod_v1.1.0_smoke
```

或使用项目虚拟环境：

```bash
.venv/bin/python prod/v1.1.0/scripts/backtest.py \
  --config prod/v1.1.0/config/trading_rule.json \
  --checkpoint prod/v1.1.0/checkpoint/market_state_best.pt \
  --symbol BTCUSDT --split test \
  --output-dir backtest/prod_v1.1.0_smoke
```

## 重新打包

```bash
bash prod/v1.1.0/pack.sh
```

会同步 `code/`、`config/`、`checkpoint/`、`metrics/` 与 `MANIFEST.json`。

## 版本关系

```text
v016 tuned2 (prod v0.0.0)
    └─ v020 trend-segment tuned (prod v0.1.0)
        └─ v021 trend-bias full_bias (prod v1.0.0)
            └─ v022 trend-quality hybrid_v021_mem_bias_p10_ir7 (prod v1.1.0)
```
