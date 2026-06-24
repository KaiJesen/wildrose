# v022 System Integration Test

- timestamp: `2026-06-24 16:07:36`

integration_smoke: PASS
unit_tests: PASS (12 tests)

## Baseline v021

| split | return | max_dd | trades |
|-------|--------|--------|--------|
| valid | 3.73% | -1.96% | 18 |
| test | 8.25% | -0.93% | 8 |

## Candidate comparison

| candidate | valid ret | test ret | business | mod gates (v/t) | score |
|-----------|-----------|----------|----------|-----------------|-------|
| current_p10_ir7 | 4.09% | 9.31% | PASS | 5/4 | 10.68 |
| p10_ir8 | 4.09% | 9.31% | PASS | 5/4 | 10.68 |
| p10_ir7_h3 | 4.09% | 9.31% | PASS | 5/4 | 10.68 |
| p10_ir7_chop | 3.88% | 9.10% | PASS | 5/4 | 10.44 |
| p10_ir7_bias_soft | 4.28% | 9.68% | FAIL | 5/4 | 8.97 |

## Recommended: `current_p10_ir7`

- valid: 4.09% (v021 3.73%), test: 9.31% (v021 8.25%)
- module gates valid/test: 5/6, 4/6 (shared fail: false_confirm_on_range)
- elapsed: 571s

## Integration checklist

- `TradingEngine` wires `TrendSignalProvider`, `TrendSegmentEngine`, `TrendBiasBuilder`
- Config: `configs/trading_rule_v022_trend_quality_0062e.json` (recipe `hybrid_v021_mem_bias_p10_ir7`)
- Linkage validation: `backtest/v022_system_test/linkage_valid|test/VALIDATION_REPORT.md`
- Re-run: `python examples/system_test_v022_integration.py`
