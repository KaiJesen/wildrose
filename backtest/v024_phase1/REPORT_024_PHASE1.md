# 024 Phase 1 Report (0065a-0)

- timestamp: `2026-06-25` (initial scaffold + 8-epoch run)
- variant: `0` (participation BCE only)
- checkpoint: `checkpoints/0065a_leg_align_v0/market_state_best.pt`

## Model-track valid (best epoch)

| metric | value | Phase 1 gate |
|--------|-------|--------------|
| participation_auc | 0.4583 | ≥ 0.55 **FAIL** |
| cum_return_ic | -0.0277 (best epoch 6) | vs 0062e 退化 ≤5% **监控** |
| confirmed_leg_flat_edge_p50_long | 待 eval json | 方向正确 |

## Model-track test

| metric | value |
|--------|-------|
| participation_auc | 0.5000 |
| cum_return_ic | 0.0145 |

## 说明

Phase 1 **工程骨架已就绪**（多头 + 损失 + 训练/评估脚本）。当前 8 epoch 初训未过 `participation_auc ≥ 0.55`，主因：

1. ideal 正样本极稀疏（valid long rate ~0.8%）
2. 需加长训练 / 调 `λ_part` / 正负采样策略后再评估

## 复现

```bash
python examples/build_leg_participation_labels.py --split train --split valid --split test
python examples/train_market_state_0065a.py --variant 0 --epochs 8
python examples/eval_model_participation.py \
  --checkpoint checkpoints/0065a_leg_align_v0/market_state_best.pt \
  --output backtest/v024_phase1/eval_model_participation.json
```

## 下一步

- 加长 0065a-0 训练或提高 `participation_weight`
- valid `participation_auc ≥ 0.55` 后进入 0065a-1（+ L_24）
- Phase 2 前勿接线 `teq_edge_*` 至 rules
