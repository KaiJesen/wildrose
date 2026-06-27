# 026 Phase 3 — B0 vs M3 Full-Chain A/B

- B0 checkpoint: `checkpoints/0065a_leg_align_c1_pw20/market_state_best.pt` (`82ca51cf637a258c`)
- M3 checkpoint: `checkpoints/026_phase2_a1/market_state_best.pt` (`75b2948f3c563d90`)
- M3 config: `backtest/v026_phase3/configs/m3_wp0.30.json` (w_part=0.3)
- TEQ calibration: `backtest/v026_phase3/teq_edge_calibration_m3.json`

## Test metrics

| arm | return | max_dd | trades | teq_open | teq_pnl | coverage | counter_leg |
|-----|--------|--------|--------|----------|---------|----------|-------------|
| b0 | 9.01% | -1.21% | 13 | 3 | 0.80% | 26.67% | 2 |
| m3 | 8.07% | -2.06% | 14 | 3 | -0.10% | 26.67% | 2 |

## Exploration gate (M3 test)

| check | gate | M3 | status |
|-------|------|-----|--------|
| total_return | ≥ 8.84% | 8.07% | **FAIL** |
| leg_count_coverage | ≥ 28.00% | 26.67% | **FAIL** |
| teq_pnl | ≥ 0 | -0.10% | **FAIL** |
| max_drawdown | ≥ -2.50% | -2.06% | **PASS** |
| counter_leg | ≤ B0+2 (2+2) | 2 | **PASS** |

**Explore dual gate: FAIL**
**Phase 3 overall: FAIL**

## Reproduction
```bash
python examples/run_v026_ab_phase1c.py --candidate m3
```
