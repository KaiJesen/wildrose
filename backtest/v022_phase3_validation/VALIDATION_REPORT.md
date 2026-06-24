# 022 Phase 3 Linkage Validation

- split: `test`
- baseline: `configs/trading_rule_v021_full_bias_0062e.json`
- candidate: `configs/trading_rule_v022_trend_quality_0062e.json`

| metric | v021 full_bias | v022 trend_quality |
|--------|----------------|---------------------|
| annualized_return | 0.7003655970836125 | 0.12142191909341093 |
| total_return | 0.08248887257602444 | 0.017258360101213954 |
| max_drawdown | -0.00928903744097188 | -0.0337004490908156 |
| trade_count | 8.0 | 11.0 |
| missed_confirmed_trend_bars | 171.0 | 151.0 |
| trend_upgrade_count | 0.0 | 1.0 |
| avg_bars_held | 14.875 | 14.454545454545455 |
| hard_counter_open_count | 0.0 | 0.0 |

## Business底线

- total_return ≥ 70% baseline: **FAIL** (1.73% vs 8.25%)
- max_drawdown not worse: **FAIL**
- trade_count ≤ 1.5× baseline: **PASS** (11 vs 8)
- overall: **FAIL**
