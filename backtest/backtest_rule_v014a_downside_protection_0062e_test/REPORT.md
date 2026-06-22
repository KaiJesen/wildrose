# Backtest Report v014a — Downside Protection (0062e, OOS test)

- checkpoint: `checkpoints/0062e_market_state_return_ic_recovery/market_state_best.pt`
- config: `configs/trading_rule_v014a_downside_protection_0062e.json`
- split: **test**
- baseline ref: `backtest/backtest_rule_v014_aggressive_tuned_0062e_test/`

## vs Baseline (v014 aggressive tuned)

| Metric | Baseline v014 | **014a** | 验收 |
|--------|---------------|----------|------|
| total_return | +11.93% | **-5.01%** | 收益 < baseline 80% |
| excess_return | +29.75% | +12.81% | — |
| max_drawdown | -1.51% | **-7.09%** | 恶化 +5.6pp |
| trade_count | 6 | **25** | **>3× baseline** |
| win_rate | 83.3% | 28.0% | — |
| profit_factor | 98.3 | 0.69 | — |
| position_limit_violations | 0 | 0 | pass |
| risk_rule_violations | 0 | 0 | pass |
| max_margin_loss_ratio | 1.6% | 35.6% | <100% pass |

## 014a 行为指标

| Metric | Value |
|--------|-------|
| probe_short_count | 25（全部为试探空单） |
| probe_short_win_rate | 28.0% |
| probe_short_total_return | -3.26% |
| blocked_long_downtrend_count | 0 |
| missed_downtrend_bars | 407 |
| short_coverage_downtrend_ratio | **13.9%** |

## 结论

1. **趋势识别已生效**：下跌结构中触发 `OPEN_SHORT_PROBE_DOWNTREND`，`short_coverage_downtrend_ratio` 从接近 0 提升至 13.9%。
2. **收益未改善**：试探空单过密（25 笔 vs baseline 6 笔），多数被 `CLOSE_REVERSE_SIGNAL` 或硬止损小亏洗出；probe 合计净亏约 -3.26%。
3. **未拦截错误开多**：`blocked_long_downtrend_count=0`（test 期模型未在下跌趋势中触发标准开多）。
4. **相对 baseline 退步**：总收益由 +11.93% 降至 -5.01%，主因 baseline 在 6 月下旬有一笔模型确认大空头（+12.9% ROE），014a 仅用小仓位 probe 且多次止损，未能等价替代。

## 建议

- 将 **tuned 开仓阈值**（prob=0.34, edge=0.04）并入 014a 配置，保留 probe 作为补充而非唯一空头来源。
- 收紧 probe 触发：提高 `min_downtrend_votes` 或降低 probe 频率上限。
- 延长 probe 空头在 `is_strong_downtrend` 下的 `short_exit_confirm_bars`，减少震荡市被 `CLOSE_REVERSE_SIGNAL` 洗出。

## Artifacts

- `trades.csv` / `decisions.csv`（含 trend 字段）
- `backtest_plot.png`
- Plot: `python examples/plot_backtest_trading_v014.py --backtest-dir backtest/backtest_rule_v014a_downside_protection_0062e_test`
