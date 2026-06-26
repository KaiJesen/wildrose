# wildrose prod v1.1.1

**024 B0 研究基线固化包**（非生产交易版本）。

| 项 | 值 |
|---|---|
| 产品版本 | `v1.1.1` |
| 类型 | **research baseline**（研究基线） |
| 策略 | v024 phase1c TEQ · `c1_pw20` · `w_part=0.35` |
| 模型 | `0065a_leg_align_c1_pw20` |
| 生产交易 | **不变** — 仍使用 [v1.1.0](v1.1.0/README.md)（0062e，9.31%） |

## 为何存在 v1.1.1

025 架构师裁定（2026-06-26）结案：A3a/A3b 无法在不牺牲 return 的前提下闭合 28% coverage 探索线。  
本包固化 **B0**（return 9.01%，coverage 26.7%，teq=3）及跨机复现所需的 checkpoint、K 线切片、TEQ 校准与代码快照。

## 目录结构

```text
prod/v1.1.1/
  README.md
  MANIFEST.json
  config/trading_rule.json
  checkpoint/market_state_best.pt      # c1_pw20（入库）
  calibration/teq_edge_calibration.json
  data/kline/..._end20260625.csv       # 冻结 K 线（入库）
  metrics/
  docs/
  code/
  scripts/
    verify_phase0.sh
    run_backtest.sh
    backtest.py
  pack.sh
```

## OOS test 参考指标（Phase 0 门）

| 指标 | 值 |
|---|---:|
| total_return | 9.01% |
| leg_count_coverage | 26.7% |
| trend_qualified_open | 3 |
| checkpoint hash (prefix) | `82ca51cf637a258c` |

## 跨机复现

在仓库根目录：

```bash
# 冒烟：B0 Phase 0 三门
bash prod/v1.1.1/scripts/verify_phase0.sh

# 完整 025 Phase 0（使用 prod 内嵌 artifact 回退）
python examples/run_v025_phase0.py --skip-train
```

`examples/_v025_common.py` 在 `checkpoints/` 或 `data/cache/` 缺失时，自动回退到 `prod/v1.1.1/` 内路径。

## 重新打包

```bash
bash prod/v1.1.1/pack.sh
```

需要本机已有：`checkpoints/0065a_leg_align_c1_pw20/`、冻结 K 线、`backtest/v024_constrained/teq_edge_calibration.json`。

## 版本关系

```text
v1.1.0 (0062e)  ← 当前生产候选
v1.1.1 (c1_pw20 B0)  ← 024/025 研究基线（025 已结案）
```
