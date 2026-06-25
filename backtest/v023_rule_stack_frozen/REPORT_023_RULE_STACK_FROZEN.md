# 023 Rule Stack Frozen Report

- timestamp: `2026-06-25 02:43:53 UTC`
- checkpoint: `prod/v0.0.0/checkpoint/market_state_best.pt`

## Config hashes

- phase1c: `configs/trading_rule_v023_phase1c_0062e.json` → `5eb39d1d75820d0e8bdd15da7672e5213435dcba02bb6594bd0ac7fa441ab2e3`
- teq_ceiling: `configs/trading_rule_v023_teq_ceiling_0062e.json` → `55f00bf590e260046ce99fa1ecefcd33b31fdd1db747f93b416f37dd0b3e5bcd`

## phase1c test metrics

| metric | value |
|--------|-------|
| total_return | 7.77% |
| max_drawdown | -1.77% |
| trade_count | 10 |
| trend_qualified_open_count | 1 |
| trend_qualified_pnl | -1.03% |
| leg_count_coverage_ratio | 16.67% |
| counter_leg_participation_count | 3 |

## teq_ceiling test metrics

| metric | value |
|--------|-------|
| total_return | 1.42% |
| max_drawdown | -4.11% |
| trade_count | 15 |
| trend_qualified_open_count | 6 |
| trend_qualified_pnl | -6.20% |
| leg_count_coverage_ratio | 20.00% |
| counter_leg_participation_count | 3 |

## Participation artifacts

- phase1c: `backtest/v023_phase1c/participation_metrics.json`
- teq_ceiling: `backtest/v023_teq_ceiling/participation_metrics.json`
- frozen bundle: `backtest/v023_rule_stack_frozen/participation_metrics.json`

## Reproduction commands

```bash
python examples/run_v023_phase1c.py
python examples/diagnose_v023_teq_ceiling.py
python examples/freeze_v023_rule_stack.py
```

024 references **only** this report (or hash-aligned metrics) for 023 rule-stack numbers.
