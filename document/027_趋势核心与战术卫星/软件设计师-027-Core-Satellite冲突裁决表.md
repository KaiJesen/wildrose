# Core–Satellite 跨 Slot 冲突裁决表 v1

## 1. 文档说明

| 项 | 内容 |
|----|------|
| 角色 | 软件设计师 |
| 流水号 | 027 |
| 版本 | **v1.0（冻结）** |
| 依据 | `架构师-027-趋势核心与战术卫星组合立项评审报告.md` §4.2、§9 |
| 与 021 关系 | 021 裁决 **单策略内** 趋势模块优先级；本表裁决 **跨 Slot 持仓方向** |
| 默认立场 | **DAD**（Directional Alignment Default） |

---

## 2. DAD 总则

```text
Core 有仓 LONG  → Satellite 仅可 OPEN_LONG / ADD_LONG / FLAT（禁止开空）
Core 有仓 SHORT → Satellite 仅可 OPEN_SHORT / ADD_SHORT / FLAT（禁止开多）
Core FLAT       → Satellite 可双向（受 Satellite 自身规则与账户风险预算约束）

账户层面禁止净多净空并存（同一时刻不允许 Core LONG + Satellite SHORT 或反之）。
```

**例外（须架构师额外书面批准，027 主路径不启用）**：

- Satellite 对冲 ≤ Core 仓位 20%；Core `hold_mode=TREND` 时例外自动关闭。

---

## 3. 裁决优先级

| 优先级 | 条件 | 动作 | reason_code 示例 |
|--------|------|------|------------------|
| **P0** | 账户日 DD ≥ `day_drawdown_stop` | 全 Slot 强制平仓 | `CLOSE_ACCOUNT_DAY_DD` |
| **P0** | `account_circuit_breaker` | 禁止一切开仓 | `BLOCK_ACCOUNT_CIRCUIT` |
| **P1** | Core LONG 且 Sat 欲开 SHORT | **BLOCK** Sat 开空 | `BLOCK_DAD_CORE_LONG` |
| **P1** | Core SHORT 且 Sat 欲开 LONG | **BLOCK** Sat 开多 | `BLOCK_DAD_CORE_SHORT` |
| **P2** | Core `hold_mode=TREND` | Sat 仅同向或 FLAT（强化 P1） | `BLOCK_DAD_TREND_HOLD` |
| **P3** | `leg_type` 结束 / bias `force_exit` | Core 平；Sat 同向减仓 50% | `REDUCE_SAT_LEG_EXIT` |
| **P4** | 账户 `margin_used` 超总上限 | 先缩 Satellite，再缩 Core | `REDUCE_ACCOUNT_MARGIN` |
| **P5** | Sat 单笔盈利 ≥ X ATR | Sat 独立平仓，不影响 Core | `CLOSE_SAT_PROFIT_TARGET` |

**已移除（相对立项 v1.0 草案）**：原 P1/P2「同 bar 禁止或仅减仓」的模糊表述，统一为 **DAD 账户层禁止反向**。

---

## 4. 决策顺序（每 bar）

```text
1. 账户级风险预检（日/周 DD、连亏冷却）
2. Core RuleEngine → Core 动作
3. 应用 DAD 过滤 → Satellite 候选动作
4. Satellite RuleEngine → Satellite 动作（若未被 DAD BLOCK）
5. AccountEquity netting + 总 margin 校验
6. 执行 + 记录 slot_id / slot_pnl
```

---

## 5. 接口字段（Phase 0 冻结）

`AccountEquity` 须暴露：

| 字段 | 类型 | 说明 |
|------|------|------|
| `core_position` | `PositionState` | Core 持仓 |
| `satellite_position` | `PositionState` | Satellite 持仓 |
| `net_side` | `LONG \| SHORT \| FLAT` | 净方向（DAD 下与 Core 同向或 FLAT） |
| `allow_satellite_short` | `bool` | `core` 非 LONG 且 DAD 通过 |
| `allow_satellite_long` | `bool` | `core` 非 SHORT 且 DAD 通过 |
| `total_margin_used` | `float` | 两 Slot 保证金之和 |

---

## 6. 修订记录

| 版本 | 日期 | 说明 |
|------|------|------|
| v1.0 | 2026-06-27 | 吸收架构师评审 DAD 默认立场；替代立项 v1.0 §5.4 草案 |
