# wildrose prod v1.1.0 — 项目日志

## 2026-06-24 — v1.1.0 发布

### 背景

022 期目标是在 `v021 trend-bias full_bias` 的可交易骨架上，继续修正趋势模块的质量与性能问题，包括：

- `TrendSignal` 增加确认记忆与 hold/sustain 状态机
- `TrendSegment` 从全量回扫改为增量计算，解决性能瓶颈
- `TrendBias` 在不破坏系统收益底线的前提下，降低 legacy 趋势规则的硬阻断副作用

### 固化版本

本版本固化配方：

- 配置：`configs/trading_rule_v022_trend_quality_0062e.json`
- recipe：`hybrid_v021_mem_bias_p10_ir7`
- 模型：`0062e_market_state_return_ic_recovery`

### 关键工程变更

- `trading_system/trend_signal.py`：确认记忆、`is_sustained`、`confirm_tier`
- `trading_system/trend_segment.py`：增量指标、bounded buffer、性能优化
- `trading_system/trend_bias.py` / `trend_bias_audit.py`：soft legacy block、audit reason 统计
- `examples/system_test_v022_integration.py`：新增系统级联动验证脚本

### 验收结论（摘要）

**系统集成**

- `TradingEngine` 已完整接入 `TrendSignalProvider` / `TrendSegmentEngine` / `TrendBiasBuilder`
- 趋势相关单元测试通过（12 项）
- valid / test 联动回测均通过 Business 底线

**valid**

- v021：3.73%
- v022：**4.09%**
- 最大回撤：-1.96% -> **-1.95%**

**test OOS（本版本生产参考）**

- 总收益：**+9.31%**
- 最大回撤：**-0.85%**
- 交易 8 笔，胜率 75.0%

### 相对上一生产版本 v1.0.0

| 维度 | v1.0.0 | v1.1.0 |
|------|--------|--------|
| 策略 | v021 trend-bias full_bias | v022 trend-quality |
| test 收益 | 8.25% | **9.31%** |
| test MDD | -0.93% | **-0.85%** |
| test 交易笔数 | 8 | 8 |
| 关键增量 | bias 统一调度 | 趋势信号记忆 + 分段性能优化 + bias 重平衡 |

### 已知限制

- `false_confirm_on_range_teacher` 仍未达到模块门限，当前版本优先固化系统级收益表现
- 当前推荐配方中 `chop_guard_enabled=false`，若后续重启 chop，需要重新做 valid/test 联动回测
- `slow_uptrend` 仍保持观察模式，不作为本版本主收益来源

### 固化清单

- [x] `config/trading_rule.json` <- `trading_rule_v022_trend_quality_0062e.json`
- [x] `checkpoint/market_state_best.pt` <- `0062e`
- [x] `metrics/backtest_test_oos.json` <- `backtest/v022_system_test/current_p10_ir7/test_bt`
- [x] `docs/REPORT_022_VALIDATION.md` <- `backtest/v022_system_test/linkage_test/VALIDATION_REPORT.md`
- [x] `docs/REPORT_022_SYSTEM_TEST.md` <- `backtest/v022_system_test/SYSTEM_TEST_REPORT.md`
- [x] 代码快照：`trading_system` / `transformer_kit` / `market_data` / `best_point`

### 下一步建议

1. 针对 `false_confirm_on_range_teacher` 做逻辑级修正，而不是继续只靠参数调优
2. 在更多标的上验证 `v022` 配方的泛化性
3. 评估是否把 `p10_ir7` 固化为新的长期基线，供后续 v023 继续迭代
