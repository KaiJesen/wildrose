# 025 Phase 0 — B0 Reproduction

- checkpoint: `checkpoints/0065a_leg_align_c1_pw20/market_state_best.pt` (`82ca51cf637a258c`)
- config: `configs/trading_rule_v024_phase1c_teq_0065a_c1_pw20.json` (`23d73fe824acacff`)
- calibration: `backtest/v024_constrained/teq_edge_calibration.json`
- frozen kline: `data/cache/kline/binance_vision_BTCUSDT_1h_365d_end20260625.csv`

## B0 test gate

| metric | expected | actual | status |
|--------|----------|--------|--------|
| total_return | 9.01% ±0.20% | 9.01% | **PASS** |
| leg_count_coverage | 26.70% ±1.00% | 26.67% | **PASS** |
| teq_open | 3 | 3 | **PASS** |

**Phase 0 overall: PASS**

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
