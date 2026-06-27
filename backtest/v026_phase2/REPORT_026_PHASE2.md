# 026 Phase 2 — A1 Three-Tier CORAL Labels

- init: `checkpoints/026_phase1_c1d1/market_state_best.pt`
- labels: `data/labels/leg_participation_a1`
- checkpoint: `checkpoints/026_phase2_a1/market_state_best.pt` (`75b2948f3c563d90`)
- recipe: `configs/training_recipe_026_phase2_a1.json`

## vs Phase 1 C1+D1

| metric | Phase1 | A1 CORAL | Δ | gate |
|--------|--------|----------|---|------|
| valid part_auc (tier2) | 0.6969 | 0.6327 | -0.0642 | ≥ 0.62 |
| valid part_auc_tier1 | — | 0.8137 | — | — |
| cum_IC degradation | — | 0.00% | — | ≤ 10% |

**Phase 2 model gate: PASS**

## Routing

- 模型轨达标 → 可重跑 Phase 3 全链路（M3 臂 + valid TEQ 重调）

## Reproduction
```bash
python examples/run_v026_phase2.py
```
