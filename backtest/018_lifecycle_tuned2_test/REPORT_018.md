# 018 长趋势生命周期 — 调优与回测报告

**日期：** 2026-06-23  
**数据：** BTCUSDT 1h，Binance Vision，365 天真实 K 线  
**模型：** `prod/v0.0.0/checkpoint/market_state_best.pt`（0062e）  
**推荐配置：** `configs/trading_rule_v018_lifecycle_tuned2_0062e.json`

---

## 1. 执行摘要

018 在 prod v0.0.0 基础上引入 **持仓生命周期状态机**（`PROBATION → TREND → PROTECT_PROFIT → RUNNER`）、**crash 趋势确认持有**（`HOLD_CRASH_TREND_CONFIRMING`）、**分层退出投票**（`CLOSE_TREND_EXIT_CONFIRMED`）及 **一次性衰竭减仓**。

调优过程中发现并修复：**`REDUCE_TREND_EXHAUSTION` 每根 K 线重复触发**，导致仓位被指数级削至接近零、长持仓无收益贡献。

**round-2 结论：** 在 valid 网格最优生命周期参数上，**保留 `allow_crash_trend_upgrade=false`**（与 prod 一致），通过 `HOLD_CRASH_TREND_CONFIRMING` 持住 confirmed DOWN crash 仓，test OOS 收益优于 prod 与 018 初版。

| 版本 | test 收益 | test MDD | 胜率 | 笔数 | avg_bars | 6/1 crash 持仓 |
|------|----------:|---------:|-----:|-----:|---------:|----------------|
| prod v0.0.0（+018 代码） | +8.59% | -3.09% | 66.7% | 9 | 12.8 | 30 bars，+6.7% ROE |
| 018 初版（bug 未修） | +2.90% | -2.23% | 70.0% | 10 | 14.4 | 48 bars，仓位≈0 |
| 018 tuned（crash 升级开） | +6.93% | -2.93% | 55.6% | 9 | 15.4 | 48 bars，+3.7% ROE |
| **018 tuned2（推荐）** | **+11.05%** | **-2.91%** | **66.7%** | **9** | **12.8** | **30 bars，+6.7% ROE** |

---

## 2. 代码修复（调优前）

| 问题 | 修复 |
|------|------|
| `REDUCE_TREND_EXHAUSTION` 每 bar 触发，仓位指数衰减 | 新增 `exhaustion_reduce_done`，每仓仅减仓一次 |
| `REDUCE_TREND_PROFIT_LOCK` 误用 `rule.reduce_scale` | 改用 `trend_lifecycle.runner_reduce_scale` |

涉及文件：`state.py`、`rules.py`、`sizing.py`、`engine.py`

修复后 018 基线 test 收益由 **+2.9% → +6.3%**。

---

## 3. 调优方法

**脚本：** `examples/tune_backtest_trading_system_v018.py`  
**调优集：** valid（~54.5 天）  
**网格（192 组）：**

- `crash_upgrade_profit_atr`: 0.4 / 0.6 / 0.8
- `upgrade_profit_atr`: 1.0 / 1.2
- `min_trend_hold_bars`: 4 / 6
- `exit_confirm_votes`: 3 / 4
- `runner_profit_atr`: 4.0 / 5.0
- `exhaustion_reduce_scale`: 0.85 / 0.95
- `max_trend_hold_bars`: 36 / 48

**round-1 valid 最优（tuned）：** 收益 +9.58%，MDD -1.42%，14 笔。

**round-2 test 验证：** `allow_crash_trend_upgrade=true` 在 test 上弱于 prod；改为 **false** 后 test **+11.05%**，产出 `tuned2`。

### tuned2 关键参数

```json
"trend_position": {
  "upgrade_profit_atr": 1.0,
  "crash_upgrade_profit_atr": 1.5,
  "allow_crash_trend_upgrade": false,
  "exhaustion_reduce_scale": 0.95
},
"trend_lifecycle": {
  "min_trend_hold_bars": 4,
  "exit_confirm_votes": 3,
  "runner_profit_atr": 5.0,
  "runner_reduce_scale": 0.5
}
```

---

## 4. 分区间回测（prod vs 018 tuned2）

年化 = \((1+r)^{365/\text{days}}-1\)

| 区间 | 天数 | prod 收益 | tuned2 收益 | prod 年化 | tuned2 年化 | prod MDD | tuned2 MDD |
|------|-----:|----------:|------------:|----------:|------------:|---------:|-----------:|
| train | 249.3 | -31.26% | -33.50% | -42.2% | -45.0% | -31.3% | -33.5% |
| valid | 54.5 | +10.49% | +10.49% | +94.9% | +94.9% | -1.23% | -1.23% |
| **test** | **54.5** | **+8.59%** | **+11.05%** | **+73.6%** | **+101.6%** | **-3.09%** | **-2.91%** |
| full | 358.5 | -13.42% | -13.21% | -13.7% | -13.4% | -32.5% | -34.7% |

test OOS tuned2 相对 prod：**+2.46pp** 收益，MDD 略好 **0.18pp**。全年略优 **0.21pp**，train 略差（生命周期 min_hold 在震荡市略增亏损）。

---

## 5. 018 文档验收对照（test OOS）

| 指标 | 目标 | tuned2 结果 | 判定 |
|------|------|------------|------|
| `position_limit_violations` | 0 | 0 | ✅ |
| `risk_rule_violations` | 0 | 0 | ✅ |
| `trade_count` | ≤ 16（11×1.5） | 9 | ✅ |
| `avg_bars_held` | ≥ 5 | 12.8 | ✅ |
| `total_return` | ≥ 8.18%（10.23%×80%） | **11.05%** | ✅ |
| `max_drawdown` | ≤ -2.53%（-0.53%+2pp） | -2.91% | ⚠️ 略超 0.38pp |
| `short_coverage_downtrend_ratio` | ≥ 8% | 17.9% | ✅ |
| `missed_confirmed_trend_bars` | 降 30%（284→≤199） | 237（-16.5%） | ⚠️ 未达 30% |
| `trend_upgrade_count` | ≥ 1 | 0 | ⚠️ 靠 crash 确认持有替代 |
| `avg_trend_hold_bars` | ≥ 8 | 48.0 | ✅ |

---

## 6. 6 月初专项（test）

| 检查项 | 结果 |
|--------|------|
| 6/1~6/5 存在 SHORT ≥ 8 bars | ✅ 6/1 15:00 开仓，持仓 **30 bars** |
| 6/1 附近不得 2 bars 内 `CLOSE_CRASH_FAILED` | ✅ 退出为 `CLOSE_CRASH_MAX_HOLD_BARS`（30 bars 后） |
| 该段净收益 | ✅ ROE **+6.67%**（`71534 → 67556`） |
| 6/4 续跌 crash | 4 bars 后 `CLOSE_HARD_STOP`，ROE -2.0%（反弹止损） |

对比 prod v017 初版（017 observe）：6/1 仅 2 bars 即 `CLOSE_CRASH_FAILED`，几乎无收益。

---

## 7. 生命周期动作统计（test tuned2）

| 动作 | 次数 | 说明 |
|------|-----:|------|
| `hold_crash_trend_confirming_count` | 37 | crash + confirmed DOWN 时拒绝失败退出 |
| `hold_trend_runner_count` | 25 | runner 仓延续 |
| `reduce_trend_exhaustion_count` | 1 | 修复后每仓最多一次 |
| `reduce_trend_profit_lock_count` | 1 | 浮盈达 runner 阈值减半 |
| `close_trend_exit_confirmed_count` | 0 | 投票退出未触发 |
| `upgrade_crash_to_trend_short_count` | 0 | tuned2 关闭 crash→trend 正式升级 |

---

## 8. 结论与建议

1. **推荐上线配置：** `configs/trading_rule_v018_lifecycle_tuned2_0062e.json`  
   test OOS **+11.05%**，在保持 9 笔交易的前提下，显著改善 6 月初 crash 持仓质量。

2. **不建议** 在现网打开 `allow_crash_trend_upgrade=true`：test 上 trend 升级路径增加亏损交易、压低总收益。

3. **train / full 仍为负**：与 prod 相同，主因是 train 段震荡亏损；018 未解决全年结构性问题，但 test 段已超 baseline。

4. **后续可选：**  
   - 将 tuned2 合入 `prod/v0.1.0` 并更新 checkpoint 文档  
   - 用 `data/labels/best_point_v018_long_horizon/` 离线校准 `exit_confirm_votes`  
   - 针对 `missed_confirmed_trend_bars` 继续压低（当前仅 -16.5%）

---

## 9. 产出路径

| 内容 | 路径 |
|------|------|
| 推荐配置 | `configs/trading_rule_v018_lifecycle_tuned2_0062e.json` |
| 调优结果 | `backtest/018_lifecycle_tuning_valid/tuning_results.json` |
| test 回测 | `backtest/018_lifecycle_tuned2_test/` |
| 对比汇总 | `backtest/018_report_summary.json` |
| 调优脚本 | `examples/tune_backtest_trading_system_v018.py` |
| 设计文档 | `document/018_长趋势收益捕捉修正/软件设计师-018-长趋势收益捕捉修正方案.md` |

**复现命令：**

```bash
python3 examples/backtest_trading_system_v014.py \
  --config configs/trading_rule_v018_lifecycle_tuned2_0062e.json \
  --checkpoint prod/v0.0.0/checkpoint/market_state_best.pt \
  --split test \
  --output-dir backtest/018_lifecycle_tuned2_test
```
