# 023 Phase 0 Baseline Report

- timestamp: `2026-06-24 11:35:15 UTC`
- checkpoint: `prod/v0.0.0/checkpoint/market_state_best.pt`
- v022 config: `configs/trading_rule_v022_trend_quality_0062e.json`
- v022 config hash: `3da83d117118a7b20434652fc707c51047d6cb2da7cc54eab56e57a541b52864`
- v023 config: `configs/trading_rule_v023_baseline_0062e.json`
- v023 config hash: `f9ffaa60dc104b212a6e57e3332fa27b6f6f91dd422c2e8b13cfc23a8697ade4`

## Reproduction vs v022_trade_points_plot (test)

| metric | v022 reference | v023 baseline | delta |
|--------|----------------|---------------|-------|
| total_return | 9.31% | 9.31% | 0.00% |
| max_drawdown | -0.85% | -0.85% | 0.00% |
| trade_count | 8 | 8 | +0 |
| missed_confirmed_trend_bars | 248 | 248 | +0 |
| leg_coverage_ratio | 17.32% | 17.32% | 0.00% |

## 023 Participation Metrics (§5.3)

### valid

- leg_count: `31.0`
- leg_count_covered: `2.0`
- leg_count_coverage_ratio: `6.45%`
- leg_bar_coverage_ratio: `9.57%`
- leg_pnl_capture_ratio: `18.65%`
- counter_leg_participation_count: `3.0`
- counter_overlap_bar_ratio: `1.74%`
- small_move_leg_count: `18.0`
- leg_loss_coverage_count: `0.0`
- slow_up_watch_to_open_ratio: `0.00%`
- slow_up_false_entry_count: `0.0`

### test

- leg_count: `30.0`
- leg_count_covered: `3.0`
- leg_count_coverage_ratio: `10.00%`
- leg_bar_coverage_ratio: `9.52%`
- leg_pnl_capture_ratio: `-0.75%`
- counter_leg_participation_count: `3.0`
- counter_overlap_bar_ratio: `7.79%`
- small_move_leg_count: `17.0`
- leg_loss_coverage_count: `1.0`
- slow_up_watch_to_open_ratio: `0.00%`
- slow_up_false_entry_count: `0.0`

## Artifacts

- participation metrics: `backtest/v023_baseline/participation_metrics.json`
- overlay plot (test): `backtest/v023_baseline/test/participation_overlay.png`

## Phase 0 exit

Reproduction within 0.1% on key runner metrics → **ready for Phase 1a**.
