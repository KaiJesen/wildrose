# 024 Phase 2 Report (TEQ edge wiring)

- teq config: `trading_rule_v024_phase1c_teq_0065a1.json`
- calibration: `backtest/v024_phase2/teq_edge_calibration.json`
- checkpoint A2: `checkpoints/0065a_leg_align_v1/market_state_best.pt`

## Valid split
- A0 teq opens: 3.0
- A2 teq opens: 7.0

- model [valid] part_auc=0.6438053097345133 recall@5%=0.14285714285714285
- model [test] part_auc=0.8013309213076625 recall@5%=0.0

## Test split
## Phase 2 gates (test)
- baseline phase1c teq opens (frozen): **1**
- A0 teq opens (0062e, teq off): **1**
- A2 teq opens (0065a-1 + teq): **4** (ratio vs baseline 4.00x)
- gate teq trigger ≥2x baseline: **PASS** (need ≥2)
- counter_leg_participation_count A2=0 (gate ≤ phase1c+2=5): **PASS**
- teq_open_on_counter_leg_count A2=0 (gate ≤ phase1c+1=4): **PASS**
- test return A0=0.0777 A2=-0.0812
- leg_count_coverage_ratio A2=0.0000
