# Backtest Report v015 — Crash Regime (0062e, OOS test)

- checkpoint: `checkpoints/0062e_market_state_return_ic_recovery/market_state_best.pt`
- config: `configs/trading_rule_v015_crash_regime_0062e.json`
- baseline: `backtest/backtest_rule_v014b_trend_hold_tuned_0062e_test/`

## OOS 核心指标

| 指标 | v014b | **v015** | 验收 |
|------|-------|----------|------|
| total_return | +8.35% | **+8.87%** | ≥80% baseline ✓ |
| excess_return | +26.17% | **+26.69%** | — |
| max_drawdown | -0.63% | **-0.63%** | ≤baseline+2pp ✓ |
| trade_count | 4 | **11** | ≤baseline+3 **✗** |
| win_rate | 75.0% | 63.6% | — |
| profit_factor | 474 | 10.6 | — |
| crash_short_count | 0 | **7** | ≤3 **✗** |
| model_blind_crash_count | — | **8** | — |
| same_regime_reentry | — | **0** | ==0 ✓ |
| 风控违规 | 0 | **0** | ✓ |

## 015 行为指标

| 字段 | 值 |
|------|-----|
| OPEN_SHORT_CRASH 触发 | 7 次 |
| crash 升级模型空头 | 0 |
| 下跌中拦截开多 | 4 |
| model_short trend_hold | 1（6/17 大空头 +5.56% ROE） |
| probe / sentinel | 0 |

## 6 月初专项（6/1~6/5）

| 日期 | 事件 | 净 ROE |
|------|------|--------|
| 06-01 | OPEN_SHORT_CRASH → CLOSE_CRASH_FAILED | ~0% |
| 06-02 | OPEN_SHORT_CRASH → CLOSE_CRASH_FAILED | **+0.90%** |
| 06-04 | OPEN_SHORT_CRASH → CLOSE_CRASH_FAILED | +0.20% |

- `is_model_blind_crash` 在 6/1、6/2、6/4 均触发 ✓
- 同 regime 未重复开仓（`same_regime_reentry_count=0`）✓
- 该段 crash 空单合计净收益为正 ✓

## 交易构成（11 笔）

- 模型标准：3（1 空 + 2 多，含 6/17 trend_hold 空头 +5.47%）
- **崩跌防守空**：7（`OPEN_SHORT_CRASH`，仓位 6%，`hold_mode=CRASH`）

## 结论

1. **收益目标达成**：相对 v014b 收益略升（+8.87% vs +8.35%），回撤持平，6 月初模型失效段已能开防守空。
2. **硬性验收未全过**：`crash_short_count=7` 超过设计上限 3；`trade_count=11` 超过 v014b+3。
3. **原因**：多个独立 crash regime 各触发一次 6% 小仓空（4 月底、5 月中、6 月初/末），未在同一 regime 内重复，但全样本累计次数偏高。
4. **建议**：提高 `min_crash_votes` 或收紧 `ret6_atr_threshold`；延长 `regime_release_bars` 合并相邻崩跌段；或提高 `crash_short.fail_stop_atr` 减少无效 crash 单。

## 复现

```bash
python examples/backtest_trading_system_v015.py \
  --checkpoint checkpoints/0062e_market_state_return_ic_recovery/market_state_best.pt \
  --split test \
  --output-dir backtest/backtest_rule_v015_crash_regime_0062e_test
```

## Artifacts

- `trades.csv` / `decisions.csv`（含 crash 字段）
- `backtest_plot.png`
- `optimal_trades_hindsight.csv`
