# Backtest Report v014a + Tuned — 0062e OOS (test)

- config: `configs/trading_rule_v014a_downside_protection_tuned_0062e.json`
- checkpoint: `checkpoints/0062e_market_state_return_ic_recovery/market_state_best.pt`

## 三方案对照

| 指标 | v014 tuned | 014a 原版 | **014a + tuned** |
|------|------------|-----------|------------------|
| total_return | **+11.93%** | -5.01% | +2.81% |
| excess_return | +29.75% | +12.81% | +20.63% |
| max_drawdown | **-1.51%** | -7.09% | -4.87% |
| trade_count | 6 | 25 | 28 |
| win_rate | **83.3%** | 28.0% | 35.7% |
| profit_factor | **98.3** | 0.69 | 1.51 |
| probe_short_count | 0 | 25 | 24 |
| probe_short PnL | — | -3.26% | **-4.14%** |
| blocked_long_downtrend | 0 | 0 | **4** |
| short_coverage (downtrend) | — | 13.9% | 13.7% |
| 风控违规 | 0 | 0 | 0 |

## 混合方案改进点

1. **恢复模型标准开仓**：4 笔非 probe 交易（2 多 2 空），含 6/17 `OPEN_SHORT_SIGNAL` 大空头 **+5.29%**。
2. **下跌中拦截开多生效**：`blocked_long_downtrend_count=4`。
3. **相对纯 014a 收益回升**：-5.01% → +2.81%（+7.8pp）。

## 仍弱于 baseline 的原因

1. **probe 空单仍过密**：24 笔 probe 净亏约 -4.14%，抵消模型盈利。
2. **模型多头质量下降**：baseline 5/6 笔为多头；混合版在下跌趋势中部分多头被拦截或 timing 变化。
3. **末段空头持仓缩短**：6/17 标准空头因 `CLOSE_MAX_HOLD_BARS`（8 bars）出场 +5.3%，baseline 同段持至 6/18 约 +12.9%。

## 结论

`tuned 阈值 + 014a 保护` 优于纯 014a，但 **未超过 v014 tuned baseline**。当前 probe 规则在震荡下跌中频繁开平，是主要拖累。

## 建议（P1）

- 在已有标准空头/多头持仓时 **禁止重复 probe 开空**。
- 提高 probe 门槛：`min_downtrend_votes=3` 或要求 `is_strong_downtrend`。
- 强下跌趋势中延长 `max_hold_bars` 或放宽空头滞后退出。

## Artifacts

- `backtest/backtest_rule_v014a_tuned_0062e_test/`
- `backtest_plot.png`

```bash
python examples/backtest_trading_system_v014.py \
  --checkpoint checkpoints/0062e_market_state_return_ic_recovery/market_state_best.pt \
  --config configs/trading_rule_v014a_downside_protection_tuned_0062e.json \
  --split test \
  --output-dir backtest/backtest_rule_v014a_tuned_0062e_test
```
