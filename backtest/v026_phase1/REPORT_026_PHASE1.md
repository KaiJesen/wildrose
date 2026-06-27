# 026 Phase 1 — C1 Leg Context + D1 Sampling (ablation)

- init: `checkpoints/026_phase0_c3/market_state_best.pt`
- checkpoint: `checkpoints/026_phase1_c1d1/market_state_best.pt` (`a26eaba612cebe74`)
- recipe: `configs/training_recipe_026_phase1_c1d1.json`

## vs Phase 0 C3

| metric | C3 baseline | C1+D1 | Δ | gate |
|--------|-------------|-------|---|------|
| valid part_auc | 0.6950 | 0.6969 | +0.0019 | ≥ 0.65 |
| cum_IC degradation | — | 0.94% | — | ≤ 10% |

**Stack value: PASS** (maintain ≥0.65 & IC gate)
**Phase 1 ablation: PASS** (no >2pp regression vs C3)

## Routing

- C1+D1 叠加有增益 → 可跳过 Phase 2 主路径，进入 Phase 3 全链路（仍建议 A1 并行准备）

## Reproduction
```bash
python examples/run_v026_phase1.py
```
