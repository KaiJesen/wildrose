# 026 Phase 0 — B0 Reproduction + C3 ParticipationAttn

## B0 gate

| metric | expected | actual | status |
|--------|----------|--------|--------|
| total_return | 9.01% ±0.20pp | 9.01% | **PASS** |
| leg_count_coverage | 26.70% ±1.00pp | 26.67% | **PASS** |
| teq_open | 3 | 3 | **PASS** |
| B0 checkpoint hash | — | `82ca51cf637a258c` | — |

**B0 overall: PASS**

## C3 model track

- init: `prod/v1.1.1/checkpoint/market_state_best.pt`
- C3 checkpoint: `checkpoints/026_phase0_c3/market_state_best.pt` (`0321f86630482186`)
- recipe: `configs/training_recipe_026_phase0_c3.json`

| metric | gate | actual | status |
|--------|------|--------|--------|
| valid participation_auc | ≥ 0.60 | 0.6950 | **PASS** |
| cum_return_ic degradation | ≤ 8.00% | 5.81% | **PASS** |
| baseline cum_return_ic | — | 0.0395 | — |
| valid cum_return_ic | — | 0.0372 | — |

**C3 overall: PASS**

## Phase 0 routing (§4.1.1)

- C3 ≥ 0.65 → 可跳过 Phase 1 主路径（建议 1 个 C1+D1 消融对照臂）

**Phase 0 overall: PASS**

## Reproduction
```bash
python examples/run_v026_phase0.py
```
