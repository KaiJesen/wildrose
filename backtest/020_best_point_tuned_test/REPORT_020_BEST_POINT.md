# 020 + 017c 辅助开仓参数调优回测报告

**基础策略**：`configs/trading_rule_v020_trend_segment_tuned_0062e.json`  
**辅助模型**：`checkpoints/017_best_point_signal/017c_best_point_major_legs/best.pt`  
**调参脚本**：`examples/tune_backtest_trading_system_v017_bp.py`  
**test 回测目录**：`backtest/020_best_point_tuned_test/`  
**图像**：`backtest/020_best_point_tuned_test/backtest_plot.png`

---

## 1. 本次改动

将 017c best-point 信号从“只观察”改为“开仓确认 + 退出辅助”：

- 多头开仓需通过 `p_long_entry_zone`
- 空头开仓需通过 `p_short_entry_zone`
- 趋势持仓退出使用 `p_exit_* / p_hold_*`
- crash 空头可选接入 best-point 约束（本轮保持宽松）

valid 网格搜索最优参数：

- `long_entry_confirm_threshold = 0.40`
- `short_entry_confirm_threshold = 0.40`
- `exit_prob_threshold = 0.65`
- `bp_exit_confirm_bars = 2`
- `min_opportunity_roi = 0.0`

---

## 2. valid 调参结果

最佳 valid 结果来自 `backtest/020_best_point_tuning_valid/tuning_results.json`：

| 指标 | 数值 |
|------|-----:|
| total_return | +4.27% |
| max_drawdown | -0.32% |
| trade_count | 3 |
| avg_bars_held | 8.0 |

结论：best-point 开仓确认显著压缩了交易频率，valid 上回撤很小，但也几乎放弃了下跌段捕捉。

---

## 3. test OOS 结果

| 版本 | total_return | max_drawdown | trade_count | avg_bars_held |
|------|-------------:|-------------:|------------:|--------------:|
| 020 trend-segment tuned | **+13.24%** | -2.86% | 10 | 12.3 |
| **020 + 017c best-point tuned** | +7.53% | **-1.02%** | **4** | 16.25 |

附加观察：

- `win_rate = 100%`
- `crash_short_count = 0`
- `trend_trade_total_return = +4.77%`
- `short_trend_capture_ratio = 0.0796`，明显低于基线的 `0.3134`

结论：017c 辅助开仓把策略过滤得过严，**风险下降了，但收益和下跌段覆盖显著变差**。test OOS 上不应替代当前 020 tuned 基线。

---

## 4. 交易明细

本轮 test 仅保留 4 笔交易：

1. 2026-04-29 短空，1 bar，+0.35%
2. 2026-05-11 做多，8 bars，+2.50%
3. 2026-05-24 做多，8 bars，+1.67%
4. 2026-06-17 趋势空头 runner，48 bars，+4.77%

6 月初原本由 crash 规则捕捉的大部分下跌交易被 best-point 确认挡掉，因此收益明显低于 020 tuned。

---

## 5. 买卖点图

已生成：

`backtest/020_best_point_tuned_test/backtest_plot.png`

图中可看到：

- 策略交易点数量明显少于 020 tuned
- 6 月初下跌区间大量 hindsight 最优空点未被执行
- 仅保留了少数置信度较高的多头与一笔长持有空头

---

## 6. 建议

当前 best-point 辅助开仓配置**不建议直接上线**。更合理的下一步是：

1. 保持 `020_trend_segment_tuned_test` 作为主基线；
2. 将 017c 先只用于 **趋势持仓退出辅助**，不要拦截 crash / 标准 short 开仓；
3. 或仅对 **多头开仓** 使用 best-point 确认，避免削弱下跌段收益来源。
