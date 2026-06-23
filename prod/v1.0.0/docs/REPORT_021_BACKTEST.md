# 021 趋势优先 bias 框架 — 回测报告（test OOS）

**配置**: `configs/trading_rule_v021_full_bias_0062e.json`  
**输出目录**: `backtest/backtest_v021_btcusdt_test`  
**数据**: BTCUSDT 1h test 段（时间序 15% OOS）  
**买卖点图**: `metrics/backtest_trade_points.png`

---

## 1. 策略要点

在 v020 trend-segment tuned 基线上，启用 Phase D `full_bias`：

- `trend_bias.enabled = true`
- `trend_bias.decision_scope = full`
- `trend_bias.disable_legacy_trend_rules = true`
- 开仓放松：`long_open_relax / short_open_relax = 1.2`
- crash 空头加成：`crash_short_open_boost = 1.35`
- 逆趋势惩罚：light/medium counter size & tighten 分层
- 慢升交易仍关闭（`slow_up_position.enabled = false`）

---

## 2. 核心绩效（test OOS）

| 指标 | v020 (v0.1.0) | **v021 full_bias** |
|------|-------------:|-------------------:|
| 总收益 | +13.24% | **+8.99%** |
| 年化收益 | — | **+77.87%** |
| 最大回撤 | -2.86% | **-0.84%** |
| 交易笔数 | 10 | **7** |
| 胜率 | 70.0% | **71.4%** |
| Profit Factor | 7.45 | **33.34** |
| 基准收益 | -16.98% | -16.51% |
| 超额收益 | +30.21% | **+25.50%** |

**结论**: v021 在 test OOS 上以更低回撤换取略低的绝对收益；风险调整后表现更稳健，适合作为生产默认候选。

---

## 3. bias 框架验收指标

| 指标 | 值 | 说明 |
|------|-----|------|
| bias_field_nonempty_ratio | 1.0 | 每根 bar 均有 bias 字段 |
| bias_reason_codes_coverage | 1.0 | reason code 全覆盖 |
| hard_counter_open_count | 0 | 无硬逆趋势探针开仓 |
| legacy_trend_direct_read_count | 0 | 旧趋势规则不再直接裁决 |
| max_position_ratio_observed | 17.2% | 低于 20% 上限 |
| crash_short_count | 3 | crash 空头 |
| hold_trend_runner_count | 25 | 趋势 runner 持有 |
| trend_trade_count | 1 | 48 bar 趋势空头 |

---

## 4. 交易明细（7 笔）

| # | 方向 | 入场 | 出场 | bars | 净 ROE | leg | 备注 |
|---|------|------|------|-----:|-------:|-----|------|
| 1 | SHORT | 05-13 14:00 | 05-14 01:00 | 11 | +0.05% | FAST_DOWN | crash |
| 2 | LONG | 05-16 01:00 | 05-16 03:00 | 2 | -0.01% | TRANSITION | 信号反转 |
| 3 | SHORT | 05-18 02:00 | 05-18 12:00 | 10 | -0.34% | FAST_DOWN | crash 失败 |
| 4 | LONG | 05-24 22:00 | 05-25 06:00 | 8 | +3.09% | RANGE | max hold |
| 5 | SHORT | 06-01 15:00 | 06-02 21:00 | 30 | +6.63% | FAST_DOWN | crash 大赢 |
| 6 | LONG | 06-03 07:00 | 06-03 13:00 | 6 | +0.31% | FAST_DOWN | 信号反转 |
| 7 | SHORT | 06-17 18:00 | 06-19 18:00 | 48 | +1.55% | TRANSITION | trend runner |

最大单笔贡献来自 06-01 crash 空头（+6.63% ROE）。

---

## 5. 产物路径

- 指标 JSON: `metrics/backtest_test_oos.json`
- 资金曲线: `metrics/backtest_plot.png`
- 买卖点: `metrics/backtest_trade_points.png`
- Phase C/D 验证: `docs/REPORT_021_VALIDATION.md`
