# 027 项目结案说明

| 项 | 内容 |
|----|------|
| 结案日期 | 2026-06-27 |
| 触发 | Phase 2a FAIL — `verdict=no_signal`（架构师全景裁定 §7） |
| 保留资产 | `portfolio_slots.py`、`engine_dual.py`、Phase 0 等价证明 |
| prod | **维持 v1.1.0 不变** |
| 研究基线 | v1.1.1 (B0) + v1.1.2 (026 M2) |

## 结论

双 Slot 架构在 Phase 0 验证可行，但 Core 松绑（Phase 1）与 Satellite 接入（Phase 2/2a）均未在 valid 窗产生正边际。017 BP 在此数据窗不可用于独立实仓。

## Phase 2a 最终证据（valid）

| 指标 | 值 |
|------|-----|
| WATCH_SLOW bars | 523 |
| Satellite 开仓（watch 窗内） | **0** |
| 分窗 return | **0.00%**（门禁：> 0% 且 ≥1 笔） |
| Combined vs Core | 0 pp（无增量） |

归因：523 根 `WATCH_SLOW_UPTREND` 中仅约 22 根 `bp_long≥0.60`，与 Core 空仓及 BP 阈值叠加后仍无成交；无法填补 023→027 贯穿的 slow_up 盲区。

## 架构师裁定已执行项

- Phase 2a 范围缩小（`require_core_watch_slow_uptrend` + `max_daily_opens=1`）
- 不启动 BP 重训
- 不降低 Phase 2 valid 门禁
- 未推进 test 探索门
- 双 Slot 基础设施入库；prod 不变
