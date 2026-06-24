# 022 Phase 3 Linkage Validation

- split: `valid`
- baseline: `configs/trading_rule_v021_full_bias_0062e.json`
- candidate: `configs/trading_rule_v022_trend_quality_0062e.json`

| metric | v021 full_bias | v022 trend_quality |
|--------|----------------|---------------------|
| annualized_return | 0.2773126717102481 | 0.307343424792506 |
| total_return | 0.037251116959242436 | 0.04085928471795919 |
| max_drawdown | -0.019579783808906237 | -0.019514409906585373 |
| trade_count | 18.0 | 17.0 |
| missed_confirmed_trend_bars | 206.0 | 254.0 |
| trend_upgrade_count | 0.0 | 0.0 |
| avg_bars_held | 7.388888888888889 | 7.647058823529412 |
| hard_counter_open_count | 0.0 | 0.0 |

## Business底线

- total_return ≥ 70% baseline: **PASS** (4.09% vs 3.73%)
- max_drawdown not worse: **PASS**
- trade_count ≤ 1.5× baseline: **PASS** (17 vs 18)
- overall: **PASS**
