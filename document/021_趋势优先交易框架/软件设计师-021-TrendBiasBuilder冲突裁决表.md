# TrendBiasBuilder 冲突裁决表

## 1. 文档说明

| 项 | 内容 |
|----|------|
| 角色 | 软件设计师 |
| 流水号 | 021 |
| 主题 | 多趋势模块冲突时的确定性裁决 |
| 依据 | `document/021_趋势优先交易框架/架构师-021-趋势优先交易框架架构评审报告.md` 问题三、问题七 |
| 配套 | `软件设计师-021-TrendBiasContext最终字段契约.md` |

`TrendBiasBuilder` 不是简单“汇总”各模块输出，而是按固定优先级覆盖，并把每一步写入 `reason_codes`。

---

## 2. 两层优先级（趋势裁决 vs 动作合并）

### 2.1 趋势模块优先级（仅 `TrendBiasBuilder` 输入，P1–P6）

| 优先级 | 来源 | 用途 | 可覆盖范围 |
|--------|------|------|------------|
| **P1** | `CrashContext` 强确认 | 紧急偏置源；可 block 多头、上调空头 bias；**不绕过 RiskBudget** | P2–P6 |
| **P2** | `SegmentContext.active_leg` | 主交易腿；`leg_direction`、`sub_phase`、`active_leg_id` | P3–P6 |
| **P3** | `TrendSignal` | 微观方向、阶段、破坏信号 | P4–P6 |
| **P4** | `SlowTrendContext` | 多头慢趋势增强器；**只修正 long 侧 bias** | P5–P6 |
| **P5** | 旧 `TrendContext` | 兼容兜底；**不得覆盖 P1–P4** | 仅填充缺失 |
| **P6** | `BestPointSignal` | entry/exit **投票**；**不改主趋势方向** | 无方向覆盖权 |

> **P0 不属于 TrendBiasBuilder。** 熔断、强平、`RiskEvent` 由 `RiskManager` 在消费阶段输出，经 `RuleEngineActionResolver` 与 `TrendBiasContext` 合并。

### 2.2 动作合并层（`RuleEngineActionResolver`，P0–P6）

| 优先级 | 来源 | 用途 |
|--------|------|------|
| **P0** | `RiskManager` / `RiskEvent` | 熔断、强平；`risk_event.force_exit`；禁止一切开仓/加仓/反手 |
| **P1–P6** | 见偏置数学公式说明 §8 | 趋势强退、减仓、加仓、新开、持有、观望 |

字段区分：

```text
trend_bias.force_exit_long    # P1 趋势侧强退建议（LEG_END、crash 极端）
risk_event.force_exit         # P0 风险侧强退命令，最终优先级最高
```

日志以 **最终合并结果** 为准；若 RiskManager 覆盖 Builder，记录 `RISK_OVERRIDES_TREND_BIAS`。

---

## 3. 方向字段裁决规则

### 3.1 `leg_direction`

```text
若 SegmentContext 有已确认 active_leg → 取 leg.direction
否则若 TrendSignal.direction 已确认 → 取 TrendSignal.direction
否则 → NONE
reason: SEGMENT_LEG_UP / SEGMENT_LEG_DOWN / TREND_SIGNAL_FALLBACK / LEG_NONE
```

### 3.2 `micro_direction`

```text
优先 TrendSignal.direction
结合 sub_phase：PULLBACK 时 micro 可与 leg 暂时相反
reason: MICRO_PULLBACK / MICRO_ACCELERATION / MICRO_ALIGNED
```

### 3.3 `macro_direction`

```text
若有独立 regime → 取之
否则长窗口推导 → macro，并打 MACRO_DERIVED_WEAK
否则 → NONE（alignment 该层计 0 分）
```

---

## 4. 典型冲突场景与裁决

### 场景 A：`SLOW_UP_LEG` vs `REVERSAL_RISK`

| 输入 | 裁决 |
|------|------|
| Segment = 慢涨腿 UP，`sub_phase=CONTINUATION` | `leg_direction=UP`，`allow_add_long` 仍可 true |
| TrendSignal = `REVERSAL_RISK` | `is_trend_breaking=true`，`exit_bias_long` 下调，`hold_bias_long` 下调 |
| 合成 | **持有但收紧退出**；不直接 force_exit，除非 P0 风险触发 |
| reason | `SLOW_UP_LEG_ACTIVE` + `REVERSAL_RISK_TIGHTEN_EXIT` |

### 场景 B：Crash 强确认 vs 短周期 `p_up` 高

| 输入 | 裁决 |
|------|------|
| `CrashContext.is_crash=true`（强确认） | P1：`crash_short_active=true`，`allow_open_long=false`，`open_bias_short` 上调 |
| 模型 `p_up` 高 | **不覆盖** crash 对多头的 block |
| 是否开空 | 仍须 `allow_open_short` + 信号 + **`risk_budget.allow_open_short`** |
| reason | `CRASH_OVERRIDES_MODEL_UP` |

### 场景 C：`PULLBACK` vs `best_point_exit` 高

| 输入 | 裁决 |
|------|------|
| `sub_phase=PULLBACK`，顺势持仓 | `hold_bias` 偏高，退出票门槛提高 |
| `best_point_exit` 高 | 贡献 **1 票**（P6），不单独触发平仓 |
| reason | `BEST_POINT_EXIT_HALF_VOTE_ONLY` |

### 场景 D：macro UP，leg DOWN，micro DOWN_ACCELERATION

| 输入 | 裁决 |
|------|------|
| 做空 | `alignment_score_short` 可能 +1（leg+micro），`counter_level_short=NONE` |
| 做多 | `alignment_score_long` 可能 -2 或 -1，`counter_level_long=MEDIUM` 或 `HARD_BLOCK` |
| reason | `MACRO_UP_LEG_DOWN_MICRO_DOWN` |

---

## 5. Crash 紧急通道（P1 细则）

Crash **不是**普通 `size_bias_short` 微调，而是 P1 紧急偏置源：

```text
crash 强确认时：
  allow_open_long = false
  open_bias_short 明显上调（如 1.3~1.5，受配置上限约束）
  exit_bias_long 明显下调（更快平多）
  force_exit_long = 仅当 crash 极端 + 已持多仓且 leg 破坏

仍须遵守：
  新开 crash 空 → risk_budget.allow_open_short（非 allow_add）
  不得绕过 max_position_ratio / worst_case_loss
```

reason 示例：`CRASH_P1_BLOCK_LONG`、`CRASH_P1_BOOST_SHORT_OPEN`、`CRASH_FORCE_EXIT_LONG`。

---

## 6. Slow Up 修正边界（P4）

`SlowTrendContext` **只能**：

- 提高 `size_bias_long`、`hold_bias_long`、`allow_add_long`
- 写入 `slow_up_active=true`

**不能**：

- 覆盖 P1 crash 对多头的 block
- 覆盖 P2 segment 确认的 DOWN leg
- 在 `decision_scope != full` 时绕过旧规则直接 `OPEN_LONG_SLOW_UP`

reason 示例：`SLOW_UP_BOOST_LONG_SIZE`、`SLOW_UP_EXTEND_HOLD`。

---

## 7. BestPoint 边界（P6）

```text
BestPointSignal：
  可影响：开仓二次确认、退出投票 +1/-1
  不可影响：macro/leg/micro_direction、counter_level、force_exit（除非与 P0/P1 一致）
```

reason 示例：`BEST_POINT_ENTRY_CONFIRM`、`BEST_POINT_EXIT_VOTE`。

---

## 8. reason_codes 命名规范

格式：`{SOURCE}_{ACTION}_{DETAIL}`，全大写，下划线分隔。

| 前缀 | 含义 |
|------|------|
| `SEGMENT_` | 来自 TrendSegmentEngine |
| `TREND_` | 来自 TrendSignal |
| `CRASH_` | 来自 CrashContext（P1） |
| `SLOW_UP_` | 来自 SlowTrendContext |
| `LEGACY_` | 来自旧 TrendContext |
| `BEST_POINT_` | 来自 BestPointSignal |
| `MACRO_` | macro 推导相关 |
| `COUNTER_` | 逆向级别映射 |
| `RISK_` | P0 风险层 |

每条非 `HOLD` 交易动作在日志中应能关联至少一条 bias reason（验收 gate：`bias_reason_codes_coverage`）。

---

## 9. Builder 推荐流程（P1–P6，不含 P0）

`TrendBiasBuilder` **不接收** `RiskBudget` / `RiskEvent`。P0 在 `RuleEngineActionResolver` 合并。

```text
Step 0: 初始化默认 TrendBiasContext（中性偏置）
Step 1: P5 旧 TrendContext 填充缺失（仅兜底）
Step 2: P3 TrendSignal → micro、phase、breaking
Step 3: P2 Segment → leg、sub_phase、active_leg_id
Step 4: 计算 alignment_score_long/short → counter_level_*
Step 4b: 若 counter_level_side == HARD_BLOCK → 执行强不变量（见偏置公式 §10.1）
Step 5: P4 SlowUp 修正 long 侧 bias（若有）
Step 6: P1 Crash 紧急偏置（可覆盖 P2–P5 的方向性结论）；可置 trend_bias.force_exit_*
Step 7: P6 BestPoint 仅记入 exit/entry 辅助标记（不改方向）
Step 8: 输出 TrendBiasContext；P0 由 engine → RiskManager → ActionResolver 处理
```

---

## 10. 相关文档

- 字段契约：`软件设计师-021-TrendBiasContext最终字段契约.md`
- 偏置公式：`软件设计师-021-偏置数学公式说明.md`
- 动作优先级：`软件设计师-021-偏置数学公式说明.md` §6
