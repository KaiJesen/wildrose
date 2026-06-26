# 025 A/B Report (B0 vs A3b)

A3b config: `configs/trading_rule_v025_a3b_std_trend.json`

## Test metrics

| arm | return | coverage | teq | trades | teq_pnl | counter_leg |
|-----|--------|----------|-----|--------|---------|-------------|
| b0 | 9.01% | 26.67% | 3 | 13 | 0.80% | 2 |
| a3b | 6.81% | 30.00% | 1 | 12 | -0.23% | 1 |

## A3b std_trend gate
- std_trend_incremental_coverage_pp: **3.33%** (suggest ≥ 0.80%)
- explore return ≥ 8.84%: **FAIL**
- explore coverage ≥ 28.00%: **PASS**
- explore dual gate: **FAIL**

## Reproduction
```bash
python examples/run_v025_phase0.py --skip-train
python examples/tune_v025_std_trend.py
python examples/run_v025_ab_a3b.py
```
