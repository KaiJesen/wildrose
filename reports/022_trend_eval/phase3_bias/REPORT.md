# 022 Trend Module Eval Report

## Metadata
```json
{
  "symbol": "BTCUSDT",
  "interval": "1h",
  "start": "2026-04-29 10:00:00+00:00",
  "end": "2026-06-22 23:00:00+00:00",
  "split": "test",
  "data_source": "binance_vision",
  "data_cache_path": "",
  "teacher_label_version": "020_major_legs_v1",
  "teacher_params_sha256": "513b40210a065732256f94717ac57db8d5af4af9ab78b625bfede1d8b24663aa",
  "config_path": "configs/trading_rule_v022_trend_quality_0062e.json",
  "config_sha256": "6fbd9d936557f2ec8117bd510cbbd3fc7dffc5ddfe05443495897dd3eae706e7",
  "git_commit": "bbb020519c0a8df73858aee98005ebb4228474c2",
  "run_timestamp": "2026-06-24T02:57:08.248026+00:00",
  "missing_data_dates": []
}
```

## Metrics
| metric | value | gate |
|--------|-------|------|
| teacher_trend_coverage | 0.5726 | FAIL |
| teacher_coverage_confirmed_only | 0.4915 | — |
| teacher_coverage_confirmed_or_sustained | 0.5726 | — |
| confirmed_precision_vs_teacher | 0.5709 | PASS |
| false_confirm_on_range_teacher | 0.3755 | FAIL |
| choppy_false_confirm_rate | 0.0000 | PASS |
| broken_ratio | 0.6122 | FAIL |
| confirmed_direction_macro_f1 | 0.4656 | — |
| bar_count | 1310.0000 | — |
| teacher_trend_bar_count | 234.0000 | — |
| signal_confirmed_bar_count | 261.0000 | — |
| chop_triggered_bar_count | 655.0000 | — |
| segment_runtime_8745 | 26.9998 | PASS |
| hard_block_long_ratio | 0.1702 | PASS |
| hard_block_short_ratio | 0.1008 | — |
| block_reason_entropy_long | 0.8757 | — |
| block_reason_entropy_short | 0.5784 | — |
| hard_block_symmetry_range_transition_ratio | 2.3448 | — |
| block_long_by_reason | {"CRASH": 15, "SEGMENT": 124, "LEGACY": 84} | — |
| block_short_by_reason | {"COUNTER": 35, "SEGMENT": 97} | — |
| chop_soft_micro_exposure_rate | 0.1654 | — |

