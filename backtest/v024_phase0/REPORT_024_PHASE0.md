# 024 Phase 0 Report

- timestamp: `2026-06-25 02:44:03 UTC`
- frozen 023 report: `backtest/v023_rule_stack_frozen/REPORT_023_RULE_STACK_FROZEN.md`
- labels dir: `data/labels/leg_participation`

## Leg-count alignment (gate < 2%)

| split | label legs | participation legs | deviation | gate |
|-------|------------|--------------------|-----------|------|
| valid | 32 | 32 | 0.00% | PASS |
| test | 30 | 30 | 0.00% | PASS |

## Label summary

### valid

- ideal_participate_long_rate: `0.0084`
- ideal_participate_short_rate: `0.0000`
- confirmed_leg_count: `32`

### test

- ideal_participate_long_rate: `0.0008`
- ideal_participate_short_rate: `0.0008`
- confirmed_leg_count: `30`

## Phase 0 exit

- leg_count_alignment_all_pass: **PASS**

Phase 1 may start only after PASS + frozen 023 report present.
