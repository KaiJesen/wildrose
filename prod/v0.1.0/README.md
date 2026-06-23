# wildrose prod v0.1.0

生产固化包：当前回测效果最优的 **v020 trend-segment tuned** 交易系统。

## 版本信息

| 项 | 值 |
|---|---|
| 产品版本 | `v0.1.0` |
| 策略代号 | `v020 trend-segment tuned` |
| 模型 checkpoint | `0065a_multi_seed_s45_market_state_stability` |
| 标的 / 周期 | BTCUSDT 1h |
| 数据窗口 | 365 天（与研发回测一致） |

## 目录结构

```text
prod/v0.1.0/
  README.md
  MANIFEST.json
  config/trading_rule.json
  checkpoint/market_state_best.pt
  metrics/
    backtest_test_oos.json
    backtest_bp_filtered_test_oos.json
    backtest_plot.png
    backtest_bp_filtered_plot.png
  docs/
    REPORT_020_BACKTEST.md
    REPORT_020_BEST_POINT.md
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
| total_return | 13.24% |
| excess_return | 30.21% |
| max_drawdown | -2.86% |
| trade_count | 10 |
| profit_factor | 7.45 |
| risk_rule_violations | 0 |

## 关键策略特性

- 市场状态模型：`0062e`（短周期方向 + 累计收益）
- 趋势分段：`v020 trend-segment`（趋势腿 / 子阶段 / counter-trend block）
- 慢升交易：**关闭**，仅观察 (`slow_up_position.enabled=false`)
- crash 空头：保留
- 趋势持有：保留 48 bar runner 逻辑
- `best_point` 辅助模块代码已随包固化，但**默认不启用开仓确认**

## 为什么不启用 017c 辅助开仓

`metrics/backtest_bp_filtered_test_oos.json` 对应 017c best-point 开仓确认实验：

- total_return：**7.53%**
- max_drawdown：**-1.02%**
- trade_count：**4**

它降低了回撤，但明显削弱了 crash / short 收益来源，因此本正式版本仍以 **纯 020 tuned** 作为生产候选；详情见 `docs/REPORT_020_BEST_POINT.md`。

## 运行回测

在仓库根目录执行：

```bash
bash prod/v0.1.0/scripts/run_backtest.sh --split test   --output-dir backtest/prod_v0.1.0_smoke
```

或直接：

```bash
PYTHONPATH=prod/v0.1.0/code python3 prod/v0.1.0/scripts/backtest.py   --config prod/v0.1.0/config/trading_rule.json   --checkpoint prod/v0.1.0/checkpoint/market_state_best.pt   --split test   --output-dir backtest/prod_v0.1.0_smoke
```

依赖：Python 3.10+、`torch`、`numpy`、`pandas` 及仓库现有数据拉取依赖。

## 重新打包

当研发分支更新后，在仓库根目录执行：

```bash
bash prod/v0.1.0/pack.sh
```

会同步 `code/`、`config/`、`checkpoint/`、`metrics/` 与 `MANIFEST.json`。

## 版本关系

```text
v016 tuned2 (prod v0.0.0)
    └─ v018 lifecycle
        └─ v019 slow-up observe
            └─ v020 trend-segment tuned (prod v0.1.0)
                └─ 017c best-point confirm experiment (rejected for prod default)
```

本版本相对 `v0.0.0` 的主要升级：**从逐 bar 趋势信号升级为趋势区间分段框架，在 test OOS 上提升到 13.24% 收益，同时维持可接受回撤。**
