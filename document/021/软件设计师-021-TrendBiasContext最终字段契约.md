# TrendBiasContext 最终字段契约

## 1. 文档说明

| 项 | 内容 |
|----|------|
| 角色 | 软件设计师 |
| 流水号 | 021 |
| 主题 | `TrendBiasContext` 冻结字段契约 |
| 依据 | `document/021/架构师-021-趋势优先交易框架架构评审报告.md` 问题一 |
| 状态 | **已冻结**（021 Phase 0 交付物） |

本文冻结 `TrendBiasContext` 的唯一 schema。所有 021 文档、配置与代码必须采用本文命名，禁止再引入 `long_open_bias`、`allow_long_open`、`counter_trend_level_long` 等历史别名。

命名规则：**方向在前、用途在后**（例：`open_bias_long`，非 `long_open_bias`）。

---

## 2. 枚举定义

```python
class BiasDirection(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    NONE = "NONE"


class CounterTrendLevel(str, Enum):
    NONE = "NONE"
    LIGHT = "LIGHT"
    MEDIUM = "MEDIUM"
    HARD_BLOCK = "HARD_BLOCK"
```

---

## 3. 主体对象（冻结版）

```python
@dataclass(frozen=True)
class TrendBiasContext:
    # --- 方向层级 ---
    macro_direction: BiasDirection
    leg_direction: BiasDirection
    micro_direction: BiasDirection

    # --- 一致性 / 逆向级别（按方向拆分）---
    alignment_score_long: int      # ∈ {-2, -1, 0, +1, +2}
    alignment_score_short: int
    counter_level_long: CounterTrendLevel
    counter_level_short: CounterTrendLevel

    # --- 连续偏置系数 ---
    open_bias_long: float
    open_bias_short: float
    size_bias_long: float
    size_bias_short: float
    hold_bias_long: float
    hold_bias_short: float
    exit_bias_long: float
    exit_bias_short: float
    # 与 open_bias 分离：只影响风险类阈值，不放宽开仓 edge
    risk_tolerance_bias_long: float
    risk_tolerance_bias_short: float

    # --- 行为开关 ---
    allow_open_long: bool
    allow_open_short: bool
    allow_add_long: bool
    allow_add_short: bool
    force_exit_long: bool
    force_exit_short: bool

    # --- 区间锚点 ---
    active_leg_id: int | None
    active_leg_type: str
    sub_phase: str
    leg_progress_ratio: float

    # --- 阶段 / 破坏（供生命周期与诊断）---
    regime_strength: str           # WEAK / NORMAL / STRONG / EXTREME
    regime_phase: str                # EARLY / CONTINUATION / ACCELERATION / EXHAUSTION / REVERSAL_RISK
    is_confirmed: bool
    is_trend_breaking: bool

    # --- 专项标记（诊断用，不替代 P1 crash 紧急通道）---
    slow_up_active: bool
    crash_short_active: bool

    # --- 诊断 ---
    source_confidence: float         # [0, 1]，Builder 对本次裁决的整体置信
    reason_codes: list[str]
```

---

## 4. 字段明细表

| 字段 | 类型 | 默认值 | 来源模块 | 允许缺失 | 进日志 | 影响交易 |
|------|------|--------|----------|----------|--------|----------|
| `macro_direction` | `BiasDirection` | `NONE` | 长窗口 EMA / segment 汇总 / 临时 TrendSignal 推导 | 是（缺省 `NONE`） | 是 | 是（alignment 权重 ±1） |
| `leg_direction` | `BiasDirection` | `NONE` | `TrendSegmentEngine` → 回退 `TrendSignal` | 否 | 是 | 是（核心，权重 ±2） |
| `micro_direction` | `BiasDirection` | `NONE` | `TrendSignal` + `sub_phase` | 否 | 是 | 是（权重 ±1） |
| `alignment_score_long` | `int` | `0` | Builder 计算 | 否 | 是 | 是 |
| `alignment_score_short` | `int` | `0` | Builder 计算 | 否 | 是 | 是 |
| `counter_level_long` | `CounterTrendLevel` | `LIGHT` | 由 `alignment_score_long` 映射 | 否 | 是 | 是 |
| `counter_level_short` | `CounterTrendLevel` | `LIGHT` | 由 `alignment_score_short` 映射 | 否 | 是 | 是 |
| `open_bias_long` | `float` | `1.0` | Builder + 配置 | 否 | 是 | 是（阈值） |
| `open_bias_short` | `float` | `1.0` | Builder + 配置 | 否 | 是 | 是 |
| `size_bias_long` | `float` | `1.0` | Builder + 配置 | 否 | 是 | 是（仓位） |
| `size_bias_short` | `float` | `1.0` | Builder + 配置 | 否 | 是 | 是 |
| `hold_bias_long` | `float` | `1.0` | Builder + 配置 | 否 | 是 | 是（最小持仓） |
| `hold_bias_short` | `float` | `1.0` | Builder + 配置 | 否 | 是 | 是 |
| `exit_bias_long` | `float` | `1.0` | Builder + 配置 | 否 | 是 | 是（退出票门槛） |
| `exit_bias_short` | `float` | `1.0` | Builder + 配置 | 否 | 是 | 是 |
| `risk_tolerance_bias_long` | `float` | `1.0` | Builder（独立于 open） | 否 | 是 | 是（`p_risk` 上限） |
| `risk_tolerance_bias_short` | `float` | `1.0` | Builder | 否 | 是 | 是 |
| `allow_open_long` | `bool` | `True` | Builder | 否 | 是 | 是 |
| `allow_open_short` | `bool` | `True` | Builder | 否 | 是 | 是 |
| `allow_add_long` | `bool` | `False` | Builder + segment 阶段 | 否 | 是 | 是 |
| `allow_add_short` | `bool` | `False` | Builder | 否 | 是 | 是 |
| `force_exit_long` | `bool` | `False` | crash / leg_end 等 P1 趋势强退 | 否 | 是 | 是（P1 建议；P0 见 `risk_event`） |
| `force_exit_short` | `bool` | `False` | 同上 | 否 | 是 | 是 |
| `active_leg_id` | `int \| None` | `None` | `SegmentContext` | 是 | 是 | 是（生命周期） |
| `active_leg_type` | `str` | `""` | `SegmentContext` | 是 | 是 | 诊断为主 |
| `sub_phase` | `str` | `""` | `SegmentContext` | 是 | 是 | 是 |
| `leg_progress_ratio` | `float` | `0.0` | `SegmentContext` | 是 | 是 | 是 |
| `regime_strength` | `str` | `NORMAL` | `TrendSignal` | 是 | 是 | 间接 |
| `regime_phase` | `str` | `CONTINUATION` | `TrendSignal` + segment | 是 | 是 | 是 |
| `is_confirmed` | `bool` | `False` | `TrendSignal` + segment | 否 | 是 | 是 |
| `is_trend_breaking` | `bool` | `False` | 多源 OR | 否 | 是 | 是 |
| `slow_up_active` | `bool` | `False` | `SlowTrendContext` | 是 | 是 | 诊断 |
| `crash_short_active` | `bool` | `False` | `CrashContext` | 是 | 是 | 诊断 |
| `source_confidence` | `float` | `0.5` | Builder 按规则计算 | 否 | 是 | 诊断 |
| `reason_codes` | `list[str]` | `[]` | Builder 每步裁决 | 否 | 是 | 可解释性 |

### `source_confidence` 计算规则（Phase 0 冻结）

```text
1.0  if SegmentContext 存在已确认 active_leg
0.7  elif TrendSignal.is_confirmed and leg 未确认
0.5  elif 仅 fallback / MACRO_DERIVED_WEAK
0.3  elif 多源冲突未消解（reason 含 CONFLICT_UNRESOLVED）
```

不得固定写死 `0.5`；冲突时取最低档并写入 `reason_codes`。

---

## 5. `macro_direction` 弱来源约束

在独立 Regime 模块落地前：

1. `macro_direction` 默认允许为 `NONE`；
2. `alignment_score_*` 计算时，若 `macro_direction == NONE`，该层贡献 **0 分**（不扣分也不加分）；
3. `leg_direction` 权重始终高于 `macro_direction`（±2 vs ±1）；
4. 若 macro 由 TrendSignal 长窗口推导，必须在 `reason_codes` 写入 `MACRO_DERIVED_WEAK`。

---

## 5.1 `HARD_BLOCK` 强不变量

当 `counter_level_{side} == HARD_BLOCK` 时，必须同时：

```text
allow_open_{side} = false
open_bias_{side}  = 0
size_bias_{side}  = 0
allow_add_{side}  = false
```

试探仓须走独立配置 `allow_hard_counter_probe`（见偏置公式 §10.1）。

---

## 6. 已废弃别名（禁止在新代码中使用）

| 废弃名 | 冻结名 |
|--------|--------|
| `long_open_bias` | `open_bias_long` |
| `short_open_bias` | `open_bias_short` |
| `long_size_bias` | `size_bias_long` |
| `short_size_bias` | `size_bias_short` |
| `long_hold_bias` | `hold_bias_long` |
| `short_hold_bias` | `hold_bias_short` |
| `long_exit_bias` | `exit_bias_long` |
| `short_exit_bias` | `exit_bias_short` |
| `allow_long_open` | `allow_open_long` |
| `allow_short_open` | `allow_open_short` |
| `allow_long_add` | `allow_add_long` |
| `allow_short_add` | `allow_add_short` |
| `block_counter_long` | 由 `counter_level_long == HARD_BLOCK` 强不变量表达 |
| `block_counter_short` | 由 `counter_level_short == HARD_BLOCK` 强不变量表达 |
| `long_hold_bias_up` / `long_add_bias_up` | `hold_bias_long` / `allow_add_long` |
| `bias_block_counter_*` | `trend_bias.counter_level_*` / `allow_open_*` |
| `counter_trend_level`（单值） | `counter_level_long` + `counter_level_short` |
| `alignment_score`（单值） | `alignment_score_long` + `alignment_score_short` |

---

## 7. 配置开关（与契约配套）

```json
{
  "trend_bias": {
    "enabled": true,
    "decision_scope": "observe",
    "disable_legacy_trend_rules": false
  }
}
```

`decision_scope` 取值：

| 值 | 含义 |
|----|------|
| `observe` | Phase A：只记录，不参与决策 |
| `open_only` | Phase B：接管 `allow_open_*` 与 `open_bias_*` |
| `open_size` | Phase C：额外接管 `size_bias_*` + `RiskBudget` |
| `full` | Phase D：接管加仓 / 持仓 / 退出；旧趋势直接 block 关闭 |

---

## 8. 相关文档

- 冲突裁决：`软件设计师-021-TrendBiasBuilder冲突裁决表.md`
- 偏置公式：`软件设计师-021-偏置数学公式说明.md`
- 消费伪代码：`软件设计师-021-TrendBiasContext字段与伪代码说明.md`
- A/B 验收：`软件设计师-021-A-B回测验收表.md`
