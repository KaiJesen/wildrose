# 019 慢速上涨趋势 — 调优与回测报告

**日期：** 2026-06-23  
**数据：** BTCUSDT 1h，Binance Vision，365 天  
**模型：** `prod/v0.0.0/checkpoint/market_state_best.pt`（0062e）  
**基线：** `configs/trading_rule_v018_lifecycle_tuned2_0062e.json`（018 tuned2，test +11.05%）

---

## 1. 执行摘要

019 已实现 `SlowUptrendDetector`、慢涨开仓/持有/退出状态机，并在 **v018 tuned2** 基底上完成调参回测。

**结论：**

| 模式 | 配置 | test 收益 | test MDD | 推荐 |
|------|------|----------:|---------:|------|
| **观察模式（推荐）** | `trading_rule_v019_slow_uptrend_tuned_0062e.json` | **+11.05%** | **-2.91%** | ✅ 上线 |
| 交易实验版 | `trading_rule_v019_slow_uptrend_trade_0062e.json` | +1.14% | -6.56% | ⚠️ 仅实验 |
| 019 初版（未调） | `trading_rule_v019_slow_uptrend_0062e.json` | -8.99% | -12.77% | ❌ |

观察模式下 **PnL 与 018 tuned2 完全一致**，同时落盘 `WATCH_SLOW_UPTREND` 与 `missed_slow_uptrend_bars`，满足文档阶段一验收。  
慢涨**实盘开多**在 test 震荡段（5 月）净亏损，暂未达到「收益 ≥ 基线 80%」目标。

---

## 2. 实现与调优改动

### 2.1 已有模块（019 编码）

- `trading_system/slow_trend.py` — 慢涨检测与打分  
- `trading_system/rules.py` — `WATCH` / `OPEN_LONG_SLOW_TREND` / 升级 / 退出投票  
- `trading_system/engine.py` / `risk.py` / `logger.py` / `runner.py` — 集成与指标  

### 2.2 调优期代码修正

| 问题 | 修复 |
|------|------|
| 未配置时 `slow_up_position` 默认 `enabled=true`，污染 018 回测 | v018 tuned2 显式 `slow_uptrend/slow_up_position.enabled=false` |
| 慢涨仓使用 1.4 ATR 硬止损过紧 | 新增 `slow_up_position.stop_atr_mult=2.2` |
| 下跌趋势/反转风险仍开多 | 拦截 `trend_direction!=UP`、`REVERSAL_RISK` |
| 模型预测累计收益为负仍开多 | 要求 `pred_cum_ret_5 >= 0` |

### 2.3 网格调参

**脚本：** `examples/tune_backtest_trading_system_v019.py`  
**集合：** valid，288 组（stable_score、slope、persist、upgrade、仓位、exit_votes）  
**结果：** valid 最优交易配置收益仅 +3.68%，慢涨子策略 -5.39%，**不如观察模式**。

---

## 3. 分区间回测对比

年化 = \((1+r)^{365/\text{天数}}-1\)

### 3.1 推荐：019 观察模式 vs 018 基线

| 区间 | 018 tuned2 | 019 observe | 差异 |
|------|----------:|------------:|-----:|
| train | -33.50% | -33.50% | 0 |
| valid | +10.49% | +10.49% | 0 |
| **test** | **+11.05%** | **+11.05%** | **0** |
| full | -13.21% | -13.21% | 0 |

观察模式额外指标（test）：

| 指标 | 数值 |
|------|-----:|
| `watch_slow_uptrend_count` | 540 |
| `missed_slow_uptrend_bars` | 540 |
| `slow_up_open_count` | 0 |

### 3.2 实验：019 交易模式（trade_c）

| 区间 | 收益 | MDD | 笔数 | 慢涨笔数 | 慢涨收益 | `long_trend_capture_ratio` |
|------|-----:|----:|-----:|--------:|--------:|---------------------------:|
| train | -35.48% | -35.5% | 114 | 37 | -4.72% | 52.2% |
| valid | -0.14% | -4.72% | 25 | 12 | -8.30% | 49.4% |
| **test** | **+1.14%** | **-6.56%** | **21** | **12** | **-6.96%** | **54.3%** |
| full | -34.90% | -39.3% | 159 | 62 | -20.08% | 52.1% |

交易模式提升了多头趋势捕捉率（test 0% → 54%），但慢涨子账户整体为负，拖累总收益。

---

## 4. 文档验收对照（test OOS）

**基线：** 018 tuned2，收益 +11.05%，MDD -2.91%，9 笔

### 4.1 硬性约束

| 指标 | 019 observe | 019 trade | 判定 |
|------|------------:|----------:|------|
| `position_limit_violations` | 0 | 0 | ✅ |
| `risk_rule_violations` | 0 | 0 | ✅ |
| `trade_count` ≤ 13.5（×1.5） | 9 | 21 | observe ✅ / trade ❌ |

### 4.2 观察模式（阶段一）

| 指标 | 目标 | 结果 | 判定 |
|------|------|------|------|
| 慢涨段可标注 | 能标出 | watch=540 | ✅ |
| 收益 ≥ 基线 80% | ≥8.84% | **11.05%** | ✅ |
| MDD ≤ 基线+2pp | ≤-4.91% | **-2.91%** | ✅ |

### 4.3 交易模式（阶段二~三）

| 指标 | 目标 | trade 结果 | 判定 |
|------|------|------------|------|
| `slow_up_open_count` | ≥1 | 12 | ✅ |
| `slow_up_trade_total_return` | >0 | **-6.96%** | ❌ |
| `avg_slow_up_hold_bars` | ≥8 | **20.3** | ✅ |
| `long_trend_capture_ratio` 提升 | 提升 | 0%→54% | ✅ |
| `missed_slow_uptrend_bars` 降 30% | 下降 | 540→298（-45%） | ✅ |
| `total_return` ≥ 80% 基线 | ≥8.84% | **+1.14%** | ❌ |

---

## 5. 失败原因分析（交易模式）

test 段 12 笔慢涨交易中，多笔在 **5 月震荡上行/假突破** 被 `CLOSE_HARD_STOP` 或 `CLOSE_SLOW_UPTREND_BROKEN` 止损，例如：

- 5/1、5/10、5/21、5/24、5/25、5/31 慢涨开多 → 硬止损  
- 5/4~5/9 升级趋势后小盈小亏震荡  

根因：

1. test 区间 BTC 先涨后大跌，慢涨结构在 5 月下旬频繁被破坏  
2. 0062e 对慢涨段 `pred_cum_ret_5` 常为弱/负，与价格结构背离  
3. 放宽止损后仍不足以覆盖假突破频率  

**建议：** 先用 `data/labels/best_point_v019_slow_up_long_horizon/` 校准检测 recall，再考虑阶段四 runner；或仅在 `pred_cum_ret_5>0` 且 valid 通过的更严子集上开多。

---

## 6. 推荐配置

### 生产（观察模式）

`configs/trading_rule_v019_slow_uptrend_tuned_0062e.json`

```json
"slow_uptrend": { "enabled": true, ... },
"slow_up_position": { "enabled": false, ... }
```

### 实验（交易模式）

`configs/trading_rule_v019_slow_uptrend_trade_0062e.json`

---

## 7. 产出路径

| 内容 | 路径 |
|------|------|
| 设计文档 | `document/019/软件设计师-019-慢速上涨趋势捕捉修正方案.md` |
| **推荐配置** | `configs/trading_rule_v019_slow_uptrend_tuned_0062e.json` |
| 实验交易配置 | `configs/trading_rule_v019_slow_uptrend_trade_0062e.json` |
| 调优网格 | `backtest/019_slow_uptrend_tuning_valid/` |
| test 回测 | `backtest/019_slow_uptrend_tuned_test/` |
| 对比汇总 | `backtest/019_report_summary.json` |
| 调优脚本 | `examples/tune_backtest_trading_system_v019.py` |
| 回测图 | `backtest/019_slow_uptrend_tuned_test/backtest_plot.png` |

**复现：**

```bash
python3 examples/backtest_trading_system_v019.py \
  --config configs/trading_rule_v019_slow_uptrend_tuned_0062e.json \
  --checkpoint prod/v0.0.0/checkpoint/market_state_best.pt \
  --split test \
  --output-dir backtest/019_slow_uptrend_tuned_test
```

---

## 8. 结论

1. **019 检测与日志链路可用**；推荐以 **观察模式** 合入主线，收益风险与 018 tuned2 相同。  
2. **慢涨实盘开多** 在当前参数与 test 行情下**未通过**收益验收，不宜默认开启。  
3. 下一步：用 `best_point_v019_slow_up_long_horizon` 标签评估 detector recall，再在 valid 上重训门槛后开启 `slow_up_position.enabled=true`。
