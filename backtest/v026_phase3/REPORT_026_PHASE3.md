# 026 Phase 3 — B0 vs M2 Full-Chain A/B

- B0 checkpoint: `checkpoints/0065a_leg_align_c1_pw20/market_state_best.pt` (`82ca51cf637a258c`)
- M2 checkpoint: `checkpoints/026_phase1_c1d1/market_state_best.pt` (`a26eaba612cebe74`)
- M2 config: `backtest/v026_phase3/configs/m2_wp0.30.json` (w_part=0.3)
- TEQ calibration: `backtest/v026_phase3/teq_edge_calibration.json`

## Test metrics

| arm | return | max_dd | trades | teq_open | teq_pnl | coverage | counter_leg |
|-----|--------|--------|--------|----------|---------|----------|-------------|
| b0 | 9.01% | -1.21% | 13 | 3 | 0.80% | 26.67% | 2 |
| m2 | 9.14% | -2.05% | 14 | 3 | -0.10% | 26.67% | 2 |

## Exploration gate (M2 test)

| check | gate | M2 | status |
|-------|------|-----|--------|
| total_return | ≥ 8.84% | 9.14% | **PASS** |
| leg_count_coverage | ≥ 28.00% | 26.67% | **FAIL** |
| teq_pnl | ≥ 0 | -0.10% | **FAIL** |
| max_drawdown | ≥ -2.50% | -2.05% | **PASS** |
| counter_leg | ≤ B0+2 (2+2) | 2 | **PASS** |

**Explore dual gate: FAIL**
**Phase 3 overall: FAIL**

## Reproduction
```bash
python examples/run_v026_ab_phase1c.py
```
