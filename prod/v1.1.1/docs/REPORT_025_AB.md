# 025 A/B Report (B0 vs A3a)

## Test metrics

| arm | return | coverage | teq | slow_up | teq_pnl | slow_up_pnl | counter_leg |
|-----|--------|----------|-----|---------|---------|-------------|-------------|
| b0 | 9.01% | 26.67% | 3 | 0 | 0.80% | 0.00% | 2 |
| a3a | 7.25% | 26.67% | 3 | 2 | 0.80% | -1.21% | 2 |

## A3a slow-up gate
- slow_up_incremental_coverage_pp: **0.00%** (suggest ≥ 0.80%)
- slow_up_false_entry_count: 0
- explore return ≥ 8.84%: **FAIL**
- explore coverage ≥ 28.00%: **FAIL**
- explore dual gate: **FAIL**
- A3a continue recommendation: **REVIEW / close A3a**

## Reproduction
```bash
python examples/run_v025_phase0.py
python examples/run_v025_ab_phase1c.py
```
