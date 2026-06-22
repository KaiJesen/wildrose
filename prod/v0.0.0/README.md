# wildrose prod v0.0.0

生产固化包：当前回测效果最优的 **v016 trend-signal tuned2** 交易系统。

## 版本信息

| 项 | 值 |
|---|---|
| 产品版本 | `v0.0.0` |
| 策略代号 | v016 tuned2 |
| 模型 checkpoint | `0062e_market_state_return_ic_recovery` |
| 标的 / 周期 | BTCUSDT 1h |
| 数据窗口 | 365 天（与研发回测一致） |

## 目录结构

```text
prod/v0.0.0/
  README.md
  MANIFEST.json
  config/trading_rule.json
  checkpoint/market_state_best.pt
  metrics/backtest_test_oos.json
  code/
    trading_system/
    transformer_kit/
    market_data/
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
| total_return | 9.66% |
| excess_return | 27.48% |
| max_drawdown | -0.58% |
| trade_count | 11 |
| profit_factor | 10.82 |
| risk_rule_violations | 0 |

## 关键策略特性

- 市场状态模型：`0062e`（短周期方向 + 累计收益）
- 下跌保护：v014a trend filter
- 趋势持有：v014b trend hold
- 崩跌状态机：v015 crash regime
- 趋势输出模块：v016 trend signal
- **生产约束**：`allow_crash_trend_upgrade = false`（crash 空单不升级为 trend 仓）

## 运行回测

在仓库根目录执行：

```bash
bash prod/v0.0.0/scripts/run_backtest.sh --split test \
  --output-dir backtest/prod_v0.0.0_smoke
```

或直接：

```bash
PYTHONPATH=prod/v0.0.0/code python3 prod/v0.0.0/scripts/backtest.py \
  --config prod/v0.0.0/config/trading_rule.json \
  --checkpoint prod/v0.0.0/checkpoint/market_state_best.pt \
  --split test \
  --output-dir backtest/prod_v0.0.0_smoke
```

依赖：Python 3.10+、`torch`、`numpy`、`pandas` 及仓库现有数据拉取依赖。

## 重新打包

当研发分支更新后，在仓库根目录执行：

```bash
bash prod/v0.0.0/pack.sh
```

会同步 `code/`、`config/`、`checkpoint/` 与 `MANIFEST.json`。

## 版本关系

```text
v014b tuned ─┬─ v015 crash regime ─┬─ v016 trend signal
             │                      └─ v016 tuned2 (prod v0.0.0)
             └─ 基线规则阈值 (edge=0.04, prob=0.34)
```

本版本相对 v016 tuned v1 的改进：**禁止 crash 仓 trend 升级**，在 test 上恢复 9.66% 收益与 -0.58% 回撤。
