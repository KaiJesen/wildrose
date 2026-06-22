# Backtest Report v014b — Trend Hold + Tuned (0062e, OOS test)

- checkpoint: `checkpoints/0062e_market_state_return_ic_recovery/market_state_best.pt`
- config: `configs/trading_rule_v014b_trend_hold_tuned_0062e.json`（valid 调参后与默认 rule 一致）
- 设计师改版：**关闭 probe**、新增 **trend_hold** 延长模型空头、**sentinel_short**（test 未触发）

## 调参（valid）

| 参数 | 最优值 |
|------|--------|
| open_edge_threshold | 0.04 |
| open_prob_threshold | 0.34 |
| open_flat_max | 0.40 |
| risk_open_max | 0.45 |
| risk_exit_threshold | 0.52 |

valid 表现：+8.78%，MDD -1.25%，9 笔  
明细：`backtest/backtest_rule_v014b_tuning_0062e/tuning_results.txt`

## OOS test 核心指标

| 指标 | v014 tuned | 014a+tuned | **v014b** |
|------|------------|------------|-----------|
| total_return | +11.93% | +2.81% | **+8.35%** |
| excess_return | +29.75% | +20.63% | **+26.17%** |
| max_drawdown | -1.51% | -4.87% | **-0.63%** |
| trade_count | 6 | 28 | **4** |
| win_rate | 83.3% | 35.7% | **75.0%** |
| probe_short | 0 | 24 | **0** |
| blocked_long (downtrend) | 0 | 4 | **4** |
| model_short trend_hold | — | — | **1** |
| avg_model_short_hold_bars | — | — | **7.5** |
| 风控违规 | 0 | 0 | **0** |

> v014b 默认与 tuned 在 test 上 **结果相同**（rule 阈值已与调参最优一致）。

## 交易摘要（4 笔）

| # | 方向 | 出场原因 | 净 ROE | hold_mode |
|---|------|----------|--------|-----------|
| 1 | SHORT | CLOSE_REVERSE_SIGNAL | +0.35% | NORMAL |
| 2 | LONG | CLOSE_REVERSE_SIGNAL | ~0% | NORMAL |
| 3 | LONG | CLOSE_MAX_HOLD_BARS | +3.65% | NORMAL |
| 4 | SHORT | **CLOSE_SHORT_TREND_TRAIL** | **+5.47%** | **TREND**（持仓 14 bars） |

第 4 笔为 6 月下旬主贡献：`trend_hold` 将空头延长至 14 bars，趋势跟踪止盈出场，避免 014a 中 8-bar 强平或 probe 频繁洗出。

## 行为指标

- `sentinel_short_count`: 0（强跌哨兵未触发，阈值 ret6_atr ≤ -2.0 较严）
- `short_coverage_downtrend_ratio`: 2.4%（仍低，但单笔质量高）
- `missed_downtrend_bars`: 490
- `close_short_trend_broken_count`: 0

## 结论

1. **v014b 显著优于 014a/014a+tuned**：去掉 probe 噪声，收益 +8.35%、回撤仅 -0.63%。
2. **相对 v014 tuned**：收益略低（-3.6pp），但回撤减半、交易更精简，风险调整更优。
3. **设计师改版有效**：`trend_hold` + 空头滞后退出使 6/17 空头持有至趋势跟踪止盈（+5.5% ROE）。
4. **推荐配置**：`configs/trading_rule_v014b_trend_hold_0062e.json`（与 tuned 等价）。

## 复现

```bash
# valid 调参
python examples/tune_backtest_trading_system_v014.py \
  --checkpoint checkpoints/0062e_market_state_return_ic_recovery/market_state_best.pt \
  --base-config configs/trading_rule_v014b_trend_hold_0062e.json \
  --split valid \
  --tuned-config-out configs/trading_rule_v014b_trend_hold_tuned_0062e.json

# OOS test
python examples/backtest_trading_system_v014b.py \
  --checkpoint checkpoints/0062e_market_state_return_ic_recovery/market_state_best.pt \
  --split test \
  --output-dir backtest/backtest_rule_v014b_trend_hold_0062e_test
```

## Artifacts

- `backtest/backtest_rule_v014b_trend_hold_0062e_test/backtest_plot.png`
- `backtest/backtest_rule_v014b_tuning_0062e/`
