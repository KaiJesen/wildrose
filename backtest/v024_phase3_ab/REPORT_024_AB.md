# 024 Phase 3 A/B Report (frozen phase1c)

## Test metrics

| arm | return | max_dd | trades | teq_opens | teq_pnl |
|-----|--------|--------|--------|-----------|---------|
| a0_0062e | 7.77% | -1.77% | 10 | 1 | -1.03% |
| a1_0065a0 | -8.34% | -15.30% | 77 | 3 | -1.04% |
| a2_teq | -8.12% | -8.15% | 51 | 4 | -1.76% |

## Exploration gate (A2)
- return ≥ 8.84%: **FAIL** (A2=-8.12%)
- leg_count_coverage ≥ 28.00%: **PASS** (A2=33.33%)
- overall explore line: **FAIL**

## Reproduction
```bash
python examples/run_v024_ab_phase1c.py
```
