# wildrose prod v1.0.0 — 项目日志

## 2026-06-23 — v1.0.0 发布

### 背景

021 期目标是在不推翻 v020 趋势分段框架的前提下，引入 **TrendBiasContext** 统一调度层，将开仓许可、仓位缩放、退出投票从分散的 legacy 趋势规则收敛到单一 bias 决策链。

### 分阶段 rollout

| 阶段 | variant | decision_scope | 状态 |
|------|---------|----------------|------|
| Phase A | `observe` | observe | 仅记录 bias，不改变交易 |
| Phase B | `open_bias` | open | 开仓许可由 bias 裁决 |
| Phase C | `open_size_bias` | open+size | 开仓 + 仓位缩放 |
| Phase D | `full_bias` | full | **全链路 bias（生产默认）** |

本版本固化 **Phase D `full_bias`**。

### 关键工程变更

- 新增 `trading_system/trend_bias.py`：`TrendBiasBuilder` / `TrendBiasContext`
- `engine.py`：开仓、加仓、减仓、退出均读取 bias 上下文
- `configs/trading_rule_v021_full_bias_0062e.json`：生产配置
- 验收脚本：`examples/validate_v021_cd.py`、`examples/compare_v021_phases.py`

### 验收结论（摘要）

来源：`docs/REPORT_021_VALIDATION.md`

**train 段**

- v020 超额年化：-2.11%，MDD -33.48%，hard_counter_open=21
- v021 full_bias 超额年化：**+10.24%**，MDD **-23.62%**，hard_counter_open=**0**

**valid 段**

- v020 年化：108.55%，MDD -1.03%
- v021 full_bias 年化：87.98%，MDD -1.36%
- 全部 Phase D gates PASS（legacy read=0，bias coverage=1.0）

**test OOS（本版本生产参考）**

- 总收益：**+8.99%**
- 最大回撤：**-0.84%**
- 交易 7 笔，胜率 71.4%

### 相对上一生产版本 v0.1.0

| 维度 | v0.1.0 | v1.0.0 |
|------|--------|--------|
| 策略 | v020 trend-segment | v021 full_bias |
| 决策链 | 分段 + legacy 趋势 | 分段 + bias 统一调度 |
| test 收益 | 13.24% | 8.99% |
| test MDD | -2.86% | **-0.84%** |
| 模型 | 0065a | **0062e** |

### 已知限制

- 模型在 BTC 上训练，跨标的（如 DOGE）迁移效果未验证为生产可用
- valid 段年化收益略低于 v020，需在实盘中持续观察 bias 框架对趋势 runner 的影响
- `best_point` 辅助开仓仍默认关闭（与 v0.1.0 一致）

### 固化清单

- [x] `config/trading_rule.json` ← `trading_rule_v021_full_bias_0062e.json`
- [x] `checkpoint/market_state_best.pt` ← `0062e`（`prod/v0.0.0` 同源）
- [x] `metrics/backtest_test_oos.json` ← `backtest_v021_btcusdt_test`
- [x] 回测图表与验证报告
- [x] 代码快照：`trading_system` / `transformer_kit` / `market_data` / `best_point`

### 下一步建议

1. 纸面/小仓位实盘跟踪 `bias_reason_codes` 分布
2. 评估是否在 DOGE 等品种上单独微调 `trend_bias` 参数
3. 视实盘反馈决定是否重启 `best_point` 开仓确认实验
