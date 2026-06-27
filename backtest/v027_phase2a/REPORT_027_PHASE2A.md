# 027 Phase 2a 报告（架构师最后实验）

**Split**: valid
**条件**: Core FLAT ∧ WATCH_SLOW_UPTREND；max_daily_opens=1；仅做多

| 指标 | 值 |
|------|-----|
| Core return | -4.12% |
| Combined return | -4.12% |
| WATCH_SLOW bars | 523 |
| Sat trades (watch window) | 0 |
| **Subwindow return** | **0.00%** |

**Phase 2a gate (分窗 return > 0 且 ≥1 笔)**: FAIL (no_signal)
**项目状态**: 027 结案（架构师裁定 §7）

```bash
python examples/run_v027_phase2a.py --split valid
```
