# 025 Phase 0 — B0 Reproduction

- checkpoint: `checkpoints/0065a_leg_align_c1_pw20/market_state_best.pt` (`2f9e49454371e2e4`)
- config: `configs/trading_rule_v024_phase1c_teq_0065a_c1_pw20.json` (`23d73fe824acacff`)
- calibration: `backtest/v024_constrained/teq_edge_calibration.json`

## B0 test gate

| metric | expected | actual | status |
|--------|----------|--------|--------|
| total_return | 9.01% ±0.20% | 7.52% | **FAIL** |
| leg_count_coverage | 26.70% ±1.00% | 31.03% | **FAIL** |
| teq_open | 3 | 4 | **FAIL** |

**Phase 0 overall: FAIL**

## Field对照（decisions.csv ↔ eval_participation）

| decisions.csv | eval / metrics |
|---------------|----------------|
| edge_source | 通道归因（teq / slow_up / legacy） |
| channel_threshold_snapshot | τ_slow 或 w_part 快照 |
| participate_score_long | participation 主门 |
| slow_up_edge_long | 慢涨通道 edge |
| trend_qualified_open_count | metrics.json teq 开仓数 |
| leg_count_coverage_ratio | participation_metrics |

## Reproduction
```bash
python examples/run_v025_phase0.py
```
