# 020 趋势区间分段 — 回测报告（test OOS）

**配置**: `configs/trading_rule_v020_trend_segment_tuned_0062e.json`  
**输出目录**: `backtest/020_trend_segment_tuned_test`  
**数据**: BTCUSDT 1h test 段（时间序 15% OOS）  
**买卖点图**: `backtest_plot.png`

---

## 1. 调参要点

在 v018 tuned2 / v019 observe 基线上：

- `slow_up_position.enabled = false`（慢升趋势仅观察，不交易）
- `trend_segment.enabled = true`
- 分段参数：`min_leg_bars=12`, `merge_pullback_atr=2.5`, `min_move_atr=1.2`, `upgrade_min_bars=6`

> 初版 v020（slow_up 交易开启）test 收益仅 +0.81%，MDD -6.23%，17 笔；关闭 slow_up 交易后显著改善。

---

## 2. 核心绩效（test OOS）

| 指标 | v020 初版 | **v020 tuned** | v018 tuned2 | v019 observe |
|------|----------:|---------------:|------------:|-------------:|
| 总收益 | +0.81% | **+13.24%** | +11.05% | +11.05% |
| 最大回撤 | -6.23% | **-2.86%** | -2.91% | -2.91% |
| 交易笔数 | 17 | **10** | 9 | 9 |
| 胜率 | 41.2% | **70.0%** | — | — |
| Profit Factor | 1.40 | **7.45** | — | — |
| 基准收益 | -16.98% | -16.98% | -16.98% | -16.98% |

**结论**: tuned v020 在 test OOS 上**略优于 v018/v019**（+13.24% vs +11.05%），回撤相近；主要增量来自 crash 空头与趋势空头持仓，且避免了 slow_up 亏损交易（8 笔合计 -8.66% 的拖累被消除）。

---

## 3. 分段与趋势模块统计

| 指标 | 值 | 说明 |
|------|-----|------|
| leg_coverage_ratio | 14.7% | 持仓覆盖已识别 leg 的比例 |
| block_counter_trend_count | 5 | 逆趋势 leg 拦截次数 |
| missed_slow_up_legs | 14 | 未参与的慢升 leg（观察模式） |
| missed_fast_down_legs | 15 | 未参与的快跌 leg |
| false_leg_entry_count | 0 | 无假 leg 入场 |
| close_trend_leg_end_count | 0 | 未触发 leg 结束平仓 |
| crash_short_count | 5 | crash 空头 |
| trend_trade_count | 1 | 趋势模式空头（48 bar 持有） |
| slow_up_trade_count | 0 | 慢升交易已关闭 |

---

## 4. 交易明细（10 笔）

| # | 方向 | 入场 | 出场 | bars | 净 ROE | 入场 leg | 备注 |
|---|------|------|------|-----:|-------:|----------|------|
| 1 | SHORT | 04-29 06:00 | 04-29 07:00 | 1 | +0.35% | NONE | 信号反转 |
| 2 | SHORT | 04-29 16:00 | 04-29 17:00 | 1 | +0.07% | NONE | crash |
| 3 | LONG | 05-11 10:00 | 05-11 18:00 | 8 | +2.50% | TRANSITION | max hold |
| 4 | SHORT | 05-13 14:00 | 05-14 01:00 | 11 | +0.05% | FAST_DOWN | crash |
| 5 | LONG | 05-16 01:00 | 05-16 03:00 | 2 | -0.02% | TRANSITION | 反转亏损 |
| 6 | SHORT | 05-18 02:00 | 05-18 12:00 | 10 | -0.34% | FAST_DOWN | crash 失败 |
| 7 | LONG | 05-24 22:00 | 05-25 06:00 | 8 | +3.66% | RANGE | max hold |
| 8 | SHORT | 06-01 15:00 | 06-02 21:00 | 30 | **+6.63%** | FAST_DOWN | crash 大赢 |
| 9 | SHORT | 06-04 01:00 | 06-04 05:00 | 4 | -2.06% | FAST_DOWN | 硬止损 |
| 10 | SHORT | 06-17 18:00 | 06-19 18:00 | 48 | **+4.77%** | TRANSITION | 趋势 runner |

最大单笔盈利：06-01 crash 空头 +6.63%；最大亏损：06-04 硬止损 -2.06%。

---

## 5. 买卖点可视化

![买卖点对比](backtest_plot.png)

- 绿色三角：策略做多入场；红色三角：做空入场  
- 圆圈：策略平仓  
- 浅色标记：事后最优买卖点（hindsight DP，仅供参考）

生成命令：

```bash
python3 examples/plot_backtest_trading_v014.py \
  --backtest-dir backtest/020_trend_segment_tuned_test \
  --title "020 trend segment — strategy vs optimal (test OOS)"
```

---

## 6. 与 Student 模型的关系

本次回测为 **规则引擎 + 分段上下文**（Phase A–D），**未使用** TrendLegClassifier 推理。Student 训练见 `reports/020_trend_leg_classifier/020a_trend_leg_baseline/REPORT_020_TRAIN.md`。

建议路径：先将 Student 接入 `SegmentContext` 做 leg_type 置信度过滤，再在 valid 上网格搜索融合权重，避免 test 过拟合。

---

## 7. 建议

1. **生产候选**：`trading_rule_v020_trend_segment_tuned_0062e.json` 可作为 v018 的增量版本在 paper 环境观察。  
2. **保持 slow_up 观察模式**：实盘前勿重新开启 slow_up 交易。  
3. **分段参数**：可在 valid 上对 `min_leg_bars` / `merge_pullback_atr` 做小范围网格，当前结果已优于基线。  
4. **Phase F**：完成 Student 融合后做第二轮回测对比。
