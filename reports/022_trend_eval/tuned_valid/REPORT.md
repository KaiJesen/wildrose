# 022 Trend Module Eval Report

## Metadata
```json
{
  "symbol": "BTCUSDT",
  "interval": "1h",
  "start": "2026-03-05 20:00:00+00:00",
  "end": "2026-04-29 09:00:00+00:00",
  "split": "valid",
  "data_source": "binance_vision",
  "data_cache_path": "",
  "teacher_label_version": "020_major_legs_v1",
  "teacher_params_sha256": "513b40210a065732256f94717ac57db8d5af4af9ab78b625bfede1d8b24663aa",
  "config_path": "configs/trading_rule_v022_trend_quality_0062e.json",
  "config_sha256": "3da83d117118a7b20434652fc707c51047d6cb2da7cc54eab56e57a541b52864",
  "git_commit": "bbb020519c0a8df73858aee98005ebb4228474c2",
  "run_timestamp": "2026-06-24T04:11:58.754037+00:00",
  "missing_data_dates": []
}
```

## Metrics
| metric | value | gate |
|--------|-------|------|
| teacher_trend_coverage | 0.7490 | PASS |
| teacher_coverage_confirmed_only | 0.6337 | — |
| teacher_coverage_confirmed_or_sustained | 0.7490 | — |
| confirmed_precision_vs_teacher | 0.5515 | PASS |
| false_confirm_on_range_teacher | 0.4273 | FAIL |
| choppy_false_confirm_rate | 0.0000 | PASS |
| broken_ratio | 0.4137 | PASS |
| confirmed_direction_macro_f1 | 0.5341 | — |
| bar_count | 1310.0000 | — |
| teacher_trend_bar_count | 243.0000 | — |
| signal_confirmed_bar_count | 330.0000 | — |
| chop_triggered_bar_count | 0.0000 | — |
| segment_runtime_8745 | 16.3313 | PASS |
| hard_block_long_ratio | 0.1076 | PASS |
| hard_block_short_ratio | 0.1931 | — |
| block_reason_entropy_long | 0.8756 | — |
| block_reason_entropy_short | 0.6410 | — |
| hard_block_symmetry_range_transition_ratio | 0.8857 | — |
| block_long_by_reason | {"CRASH": 6, "LEGACY": 70, "SEGMENT": 64, "MACRO": 1} | — |
| block_short_by_reason | {"SEGMENT": 167, "COUNTER": 86} | — |

