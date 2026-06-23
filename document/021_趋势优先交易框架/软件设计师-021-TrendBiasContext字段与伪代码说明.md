# TrendBiasContext 字段与伪代码说明

## 1. 文档说明

| 项 | 内容 |
|----|------|
| 角色 | 软件设计师 |
| 流水号 | 021 |
| 主题 | TrendBiasContext 字段定义、构建流程与 RuleEngine 消费伪代码 |
| 依据 | `document/021_趋势优先交易框架/软件设计师-021-TrendBiasContext最终字段契约.md`（冻结 schema）、`document/021_趋势优先交易框架/软件设计师-021-TrendBiasBuilder冲突裁决表.md`、`document/021_趋势优先交易框架/软件设计师-021-偏置数学公式说明.md` |
| 架构复审 | `document/021_趋势优先交易框架/架构师-021-趋势优先交易框架立项材料复审报告.md` |
| 当前代码参考 | `trading_system/engine.py`、`trading_system/rules.py`、`trading_system/trend_signal.py`、`trading_system/trend_segment.py`、`trading_system/slow_trend.py`、`trading_system/crash.py` |
| 目标读者 | 程序员、策略设计师、回测师 |

本文是 021 期实现落地的中间说明。目的不是给出完整代码，而是把：

1. `TrendBiasContext` 应有哪些字段  
2. 这些字段从哪里来  
3. `RuleEngine` 应如何消费这些字段  

讲清楚，使后续实现可以围绕统一结构推进，而不是继续堆零散条件分支。

---

## 2. 设计目标

`TrendBiasContext` 的目标是把当前多个趋势模块的结论收敛成一个统一对象，供交易系统使用。

它不负责：

- 直接训练模型
- 直接输出开平仓动作
- 直接替代风险管理

它负责：

```text
统一描述当前趋势背景，
统一表达顺势 / 逆势程度，
统一把这种背景映射为交易偏置。
```

---

## 3. 字段定义（冻结版）

字段命名与类型以 `软件设计师-021-TrendBiasContext最终字段契约.md` 为唯一权威。以下为摘要（**方向在前、用途在后**）：

```python
@dataclass(frozen=True)
class TrendBiasContext:
    macro_direction: BiasDirection
    leg_direction: BiasDirection
    micro_direction: BiasDirection

    alignment_score_long: int
    alignment_score_short: int
    counter_level_long: CounterTrendLevel
    counter_level_short: CounterTrendLevel

    open_bias_long: float
    open_bias_short: float
    size_bias_long: float
    size_bias_short: float
    hold_bias_long: float
    hold_bias_short: float
    exit_bias_long: float
    exit_bias_short: float
    risk_tolerance_bias_long: float
    risk_tolerance_bias_short: float

    allow_open_long: bool
    allow_open_short: bool
    allow_add_long: bool
    allow_add_short: bool
    force_exit_long: bool
    force_exit_short: bool

    active_leg_id: int | None
    active_leg_type: str
    sub_phase: str
    leg_progress_ratio: float

    regime_strength: str
    regime_phase: str
    is_confirmed: bool
    is_trend_breaking: bool
    slow_up_active: bool
    crash_short_active: bool

    source_confidence: float
    reason_codes: list[str]
```

> 禁止再使用 `long_open_bias`、`allow_long_open`、`counter_trend_level_long` 等历史别名。

---

## 4. 字段来源映射

## 4.1 `macro_direction`

建议来源优先级：

1. 未来若有更高层级 regime 模块，则直接取其输出
2. 当前阶段可临时由长窗口特征 + `TrendSignalProvider` 推导，并打 `MACRO_DERIVED_WEAK`
3. 若无明确信号，则置为 `NONE`（alignment 该层计 **0 分**，不扣分）

目的：

- 表示宏观背景，不直接用于择时
- 用于判断当前交易是否“逆大背景”

---

## 4.2 `leg_direction`

建议主要来源：

- `TrendSegmentEngine` 当前活跃腿方向

若无确认腿：

- 可回退到 `TrendSignal.direction`

目的：

- 表示当前主交易段方向
- 是开仓偏置与持仓延长的核心层

---

## 4.3 `micro_direction`

建议来源：

- `TrendSignal.direction`
- 当前 `sub_phase`
- 近几根 bar 的推进 / 回撤方向

目的：

- 决定当前是顺势推进、正常回撤，还是小级别反向脉冲

---

## 4.4 `is_confirmed`

建议来源：

- `TrendSignal.is_confirmed`
- `SegmentContext.active_leg.is_confirmed`

建议规则：

```text
至少一项确认且没有明确破坏，才视为 confirmed
```

---

## 4.5 `is_trend_breaking`

建议来源：

- `TrendSignal.is_broken`
- `SegmentContext.sub_phase in {LEG_END, EXHAUSTION}`
- `CrashContext` 或 `SlowTrendContext` 中的失效信号

---

## 4.6 `alignment_score_long` / `alignment_score_short`

建议不要只存一个分数，而是分别从**做多视角**和**做空视角**计算。

示例：

### 对多头

| 层 | 同向加分 | 反向减分 |
|----|----------|----------|
| `macro_direction` | +1 | -1 |
| `leg_direction` | +2 | -2 |
| `micro_direction` | +1 | -1 |

则：

```text
alignment_score_long = score(macro vs LONG) + score(leg vs LONG) + score(micro vs LONG)
```

再映射到压缩分值：

- `>= +3` → `+2`
- `+1 ~ +2` → `+1`
- `0` → `0`
- `-1 ~ -2` → `-1`
- `<= -3` → `-2`

空头同理。

这样可以直接回答：

```text
从做多角度看，现在顺不顺？
从做空角度看，现在顺不顺？
```

---

## 4.7 `counter_level_long` / `counter_level_short`

基于对应方向的 `alignment_score_long` / `alignment_score_short` 映射：

| 分数 | 级别 |
|------|------|
| `+2 / +1` | `NONE` |
| `0` | `LIGHT` |
| `-1` | `MEDIUM` |
| `-2` | `HARD_BLOCK` |

必要时允许用附加条件修正：

- 若 `sub_phase == EXHAUSTION`，则允许把 `HARD_BLOCK` 降到 `MEDIUM`
- 若 `CrashContext` 强烈确认，空头可从 `MEDIUM` 提升到 `NONE`

---

## 4.8 开仓 / 仓位 / 持仓 / 退出偏置字段

偏置系数在 `TrendBiasBuilder` 中计算；**数学含义与接入公式**见 `软件设计师-021-偏置数学公式说明.md`。

摘要：

```text
open_bias_long > 1  → 有效 edge/prob 门槛 = base / open_bias_long（放宽）
size_bias_long        → 进入 PositionSizer，受 RiskBudget 硬上限约束
hold_bias_long        → effective_min_hold_bars = base * hold_bias_long
exit_bias_long        → effective_exit_votes = ceil(base * exit_bias_long)
risk_tolerance_bias_long → 仅影响 p_risk 上限，不与 open_bias 混用
```

---

## 5. `TrendBiasBuilder` 构建流程建议

冲突优先级与典型场景见 `软件设计师-021-TrendBiasBuilder冲突裁决表.md`。

## 5.1 输入

建议 `TrendBiasBuilder.build(...)` 接收：

```python
build(
    trend_context: TrendContext | None,
    trend_signal: TrendSignal | None,
    segment_context: SegmentContext | None,
    slow_context: SlowTrendContext | None,
    crash_context: CrashContext | None,
) -> TrendBiasContext
```

## 5.2 推荐流程

### Step 1：读取原始趋势上下文

- 从 `segment_context` 取主腿方向与子阶段
- 从 `trend_signal` 取方向、强度、阶段、是否破坏
- 从 `slow_context` 取慢涨稳定性
- 从 `crash_context` 取 crash 确认性

### Step 2：确定三层方向

```text
macro_direction
leg_direction
micro_direction
```

### Step 3：计算多头 / 空头一致性分数

分别算：

- `alignment_score_long`
- `alignment_score_short`

### Step 4：映射逆向级别

分别得到：

- `counter_level_long`
- `counter_level_short`

### Step 5：映射偏置系数

为多头和空头分别给出 `open_bias_*`、`size_bias_*`、`hold_bias_*`、`exit_bias_*`、`risk_tolerance_bias_*`。

### Step 6：生成行为开关

- `allow_open_long` / `allow_open_short`
- `allow_add_long` / `allow_add_short`
- `force_exit_long` / `force_exit_short`（P0/P1 层可置位）

### Step 7：生成诊断 reason_codes

例如：

- `LEG_CONFIRMED_UP`
- `MICRO_PULLBACK`
- `COUNTER_MEDIUM_LONG`
- `SLOW_UP_ACTIVE`
- `CRASH_SHORT_PRIORITY`

---

## 6. `RuleEngine` 消费方式建议

## 6.1 新函数签名

建议把 `decide()` 签名扩展为：

```python
def decide(
    signal: TradingSignal,
    portfolio: PortfolioState,
    bar_index: int = 0,
    trend_context: TrendContext | None = None,
    crash_context: CrashContext | None = None,
    trend_signal: TrendSignal | None = None,
    best_point_signal: BestPointSignal | None = None,
    slow_context: SlowTrendContext | None = None,
    segment_context: SegmentContext | None = None,
    trend_bias: TrendBiasContext | None = None,
) -> TradingAction:
```

后续逐步把 `trend_context` / `trend_signal` / `segment_context` 的直接读取收敛到 `trend_bias`。

---

## 6.2 空仓时的伪代码

```text
if no position:
    build side-specific effective thresholds from trend_bias

    # 多头开仓
    if trend_bias.allow_open_long:
        if signal satisfies long condition under trend_bias.open_bias_long:
            if risk_budget.allow_open_long:
                if best_point allows:
                    return OPEN_LONG

    # 空头开仓
    if trend_bias.allow_open_short:
        if signal satisfies short condition under trend_bias.open_bias_short:
            if risk_budget.allow_open_short:
            if best_point allows:
                return OPEN_SHORT

    # 慢涨 / crash 等专项，只作为 bias 修正，不直接绕开主流程
    return HOLD_NO_ENTRY
```

### 核心变化

旧逻辑：

```text
先看 signal 是否满足，再看趋势要不要拦
```

新逻辑：

```text
先看 trend_bias 是否允许、门槛应如何变化，再解释 signal
```

---

## 6.3 已持仓时的伪代码

```text
if has position:
    determine current side = LONG / SHORT
    read side-specific hold_bias / exit_bias / add permission

    if hold_mode != TREND:
        if profit enough and trend_bias says same-direction confirmed:
            upgrade to TREND

    if hold_mode == TREND:
        if trend leg says PULLBACK and hold_bias is high:
            HOLD
        if trend phase says ACCELERATION and allow_add_{side}:
            ADD
        if trend phase says EXHAUSTION:
            REDUCE
        if exit votes adjusted by exit_bias exceed threshold:
            CLOSE

    if current position is counter-trend:
        shorten min hold
        prohibit add
        lower exit confirmation threshold
        if any stronger reverse evidence:
            CLOSE or REDUCE
```

---

## 6.4 退出投票如何结合 `exit_bias`

建议不要把趋势偏置写成新的 if/else，而是使用冻结公式（见偏置数学公式说明 §7）：

```text
effective_exit_votes_required = ceil(base_exit_votes * max(exit_bias_side, eps))
```

示例：

- 高一致性顺势：需要 4 票才退
- 轻逆向：需要 2 票
- 中逆向：1~2 票即可退出

这样行为更连续、可解释性更强。

---

## 7. `PositionSizer` 消费方式建议

公式见 `软件设计师-021-偏置数学公式说明.md` §4。

```python
compute_target_position_ratio(
    signal,
    trend_bias,
    risk_budget: RiskBudget,
    portfolio,
)
```

计算思路：

```text
base_position
  -> signal_strength_scale
  -> size_bias_side
  -> min(..., risk_budget.remaining_position_ratio, max_position_ratio)
```

---

## 8. `RiskManager` 与动作合并

输出 `RiskBudget`（动作级权限，见偏置公式 §5）与 `RiskEvent`（P0，`force_exit` 等）。

```text
may_open_long  = trend_bias.allow_open_long  AND risk_budget.allow_open_long
may_add_long   = trend_bias.allow_add_long   AND risk_budget.allow_add_long
may_reverse    = risk_budget.allow_reverse   AND trend_bias.allow_open_{target}
```

`TrendBiasBuilder` 不消费 RiskBudget。`RuleEngineActionResolver` 按 P0–P6 合并 `trend_bias`、`risk_budget`、`risk_event`。

---

## 9. 建议的最小实现顺序

与架构师评审及 `软件设计师-021-A-B回测验收表.md` 对齐：

### Phase 0（契约冻结）

- 字段契约、冲突表、偏置公式、配置 schema
- **不改 `rules.py` 主流程**

### Phase A（observe）

- `TrendBiasContext` + `TrendBiasBuilder`
- `engine.py` 生成 bias；logger 记录
- `decision_scope=observe`，rules **不消费** bias

### Phase B（open_only）

- bias 接管 `allow_open_*`、`open_bias_*`
- `disable_legacy_trend_rules=true`

### Phase C（open_size）

- `size_bias_*` + `RiskBudget`

### Phase D（full）

- add / hold / exit / `force_exit_*`
- 清理 RuleEngine 直接读趋势原始字段

---

## 10. 结论

`TrendBiasContext` 的价值不在于再加一个新对象，而在于：

```text
把当前分散在多个趋势模块中的判断，
统一收敛成“交易可以直接消费的偏置语言”。
```

只有这样，021 才能真正从：

```text
短信号主导 + 趋势补丁
```

升级成：

```text
趋势背景主导 + 短信号执行细节
```

这份文档可作为后续正式改造 `trading_system/` 时的接口说明与伪代码依据。
