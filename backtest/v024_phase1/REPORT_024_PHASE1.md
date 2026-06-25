# 024 Phase 1 Report

- updated: 2026-06-25

## 0065a-0（参与 BCE）

| 配置 | valid participation_auc | long AUC | gate ≥0.55 |
|------|-------------------------|----------|------------|
| stride=8 默认 | 0.458 | ~0.42 | FAIL |
| stride=1, oversample=30, λ_part=1.5 | **0.611** | **0.722** | **PASS** |

Checkpoint（本地，未入库）: `backtest/v024_phase1_tune/s1_os30_pw15/checkpoint/market_state_best.pt`

## 0065a-1（+ L_12/L_24）

| metric | valid (best ep10) | test |
|--------|-------------------|------|
| participation_auc | **0.633** | 0.794 (train report) |
| participation_auc_long | **0.766** | — |
| hz_direction_acc_24 | 0.443 | — |
| cum_return_ic | 0.088 | -0.019 |

Checkpoint（本地）: `checkpoints/0065a_leg_align_v1/market_state_best.pt`  
Eval JSON: `backtest/v024_phase1/eval_model_participation_v1.json`

## 0065a-2（+ L_leg_dir）

| metric | valid (best ep1) | notes |
|--------|------------------|-------|
| participation_auc | **0.622** | PASS |
| participation_auc_long | **0.745** | |
| hz_direction_acc_24 | **0.586** | vs v1 0.443 ↑ |

Checkpoint（本地）: `checkpoints/0065a_leg_align_v2/market_state_best.pt`

**Phase 1 ablation 完成** → 进入 Phase 2（valid 校准 + `teq_edge_*` 接线，不改全局 `edge`）

## Phase 1 gate

- participation_auc ≥ 0.55: **PASS**（v0 / v1）
- 下一步: 0065a-2（+ L_leg_dir）→ Phase 2 校准

## 复现

```bash
python examples/run_v024_phase1b_0065a1.py
python examples/train_market_state_0065a.py --variant 2 \
  --init-checkpoint checkpoints/0065a_leg_align_v1/market_state_best.pt
```
