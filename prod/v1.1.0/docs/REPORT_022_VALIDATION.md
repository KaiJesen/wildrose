# 022 Phase 3 Linkage Validation

- split: `test`
- baseline: `configs/trading_rule_v021_full_bias_0062e.json`
- candidate: `configs/trading_rule_v022_trend_quality_0062e.json`

| metric | v021 full_bias | v022 trend_quality |
|--------|----------------|---------------------|
| annualized_return | 0.7003655970836125 | 0.8154927499275995 |
| total_return | 0.08248887257602444 | 0.09312992708799195 |
| max_drawdown | -0.00928903744097188 | -0.008545561025824003 |
| trade_count | 8.0 | 8.0 |
| missed_confirmed_trend_bars | 171.0 | 248.0 |
| trend_upgrade_count | 0.0 | 0.0 |
| avg_bars_held | 14.875 | 14.875 |
| hard_counter_open_count | 0.0 | 0.0 |

## Business底线

- total_return ≥ 70% baseline: **PASS** (9.31% vs 8.25%)
- max_drawdown not worse: **PASS**
- trade_count ≤ 1.5× baseline: **PASS** (8 vs 8)
- overall: **PASS**
