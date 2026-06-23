# wildrose prod v1.0.0

生产固化包：当前正式候选的 **v021 trend-bias full_bias** 交易系统。

## 版本信息

| 项 | 值 |
|---|---|
| 产品版本 | `v1.0.0` |
| 策略代号 | `v021 trend-bias full_bias` |
| 模型 checkpoint | `0062e_market_state_return_ic_recovery` |
| 标的 / 周期 | BTCUSDT 1h |
| 数据窗口 | 365 天（与研发回测一致） |

## 目录结构

```text
prod/v1.0.0/
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
    REPORT_021_BACKTEST.md
    REPORT_021_VALIDATION.md
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

来源：`metrics/backtest_test_oos.json`（BTCUSDT test 段）

| 指标 | 值 |
|---|---:|
| total_return | 8.99% |
| annualized_return | 77.87% |
| excess_return | 25.50% |
| max_drawdown | -0.84% |
| win_rate | 71.43% |
| trade_count | 7 |
| profit_factor | 33.34 |
| risk_rule_violations | 0 |

## 关键策略特性

- 市场状态模型：`0062e`（短周期方向 + 累计收益）
- 趋势分段：继承 `v020 trend-segment`（趋势腿 / 子阶段 / counter-trend block）
- **新增** `TrendBiasContext` 统一调度层（`trend_bias.decision_scope=full`）
- 开仓 / 仓位 / 退出投票由 bias 框架统一裁决，旧趋势规则仅作诊断
- crash 空头：保留 `crash_short_open_boost=1.35`
- 慢升交易：仍关闭（仅观察）
- `hard_counter_open_count = 0`，`legacy_trend_direct_read_count = 0`

## 相对 v0.1.0（v020）的变化

| 指标 | v0.1.0 (v020) | **v1.0.0 (v021)** |
|---|---:|---:|
| test 总收益 | 13.24% | 8.99% |
| test 最大回撤 | -2.86% | **-0.84%** |
| test 交易笔数 | 10 | 7 |
| test 胜率 | 70.0% | **71.4%** |

v021 在 test OOS 上牺牲了部分绝对收益，换取显著更低的回撤与更干净的趋势偏置决策链；train 段上相对 v020 的超额年化收益由 -2.11% 改善到 +10.24%（见 `docs/REPORT_021_VALIDATION.md`）。

## 运行回测

在仓库根目录执行：

```bash
bash prod/v1.0.0/scripts/run_backtest.sh --split test --output-dir backtest/prod_v1.0.0_smoke
```

或使用项目虚拟环境：

```bash
.venv/bin/python prod/v1.0.0/scripts/backtest.py \
  --config prod/v1.0.0/config/trading_rule.json \
  --checkpoint prod/v1.0.0/checkpoint/market_state_best.pt \
  --symbol BTCUSDT --split test \
  --output-dir backtest/prod_v1.0.0_smoke
```

依赖：Python 3.10+、`torch`、`numpy<2`、`pandas`、`matplotlib` 及仓库现有数据拉取依赖。

## 重新打包

当研发分支更新后，在仓库根目录执行：

```bash
bash prod/v1.0.0/pack.sh
```

会同步 `code/`、`config/`、`checkpoint/`、`metrics/` 与 `MANIFEST.json`。

## 版本关系

```text
v016 tuned2 (prod v0.0.0)
    └─ v020 trend-segment tuned (prod v0.1.0)
        └─ v021 observe → open_bias → open_size_bias → full_bias (prod v1.0.0)
```

本版本相对 `v0.1.0` 的主要升级：**引入趋势优先 bias 框架，消除 legacy 趋势硬拦截，在 test OOS 上将最大回撤从 -2.86% 压到 -0.84%。**
