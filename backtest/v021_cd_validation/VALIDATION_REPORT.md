# v021 Phase C/D Validation (annualized return)

- checkpoint: `prod/v0.0.0/checkpoint/market_state_best.pt`
- splits: train / valid (full window each)
- June spot-check split: `test`

## Metrics (annualized primary)

| variant | split | ann_return | bench_ann | excess_ann | max_dd | trades | max_pos | hard_counter |
|---------|-------|------------|-----------|------------|--------|--------|---------|--------------|
| v020 | train | -44.94% | -42.84% | -2.11% | -33.48% | 84 | 12.24% | 21 |
| open_size_bias | train | -32.60% | -42.84% | 10.24% | -23.62% | 86 | 16.41% | 0 |
| full_bias | train | -32.60% | -42.84% | 10.24% | -23.62% | 86 | 16.41% | 0 |
| v020 | valid | 108.55% | 32.66% | 75.89% | -1.03% | 13 | 11.65% | 0 |
| open_size_bias | valid | 87.98% | 32.66% | 55.32% | -1.36% | 15 | 17.23% | 0 |
| full_bias | valid | 87.98% | 32.66% | 55.32% | -1.36% | 15 | 17.23% | 0 |

## Acceptance gates

### open_size_bias_train
- PASS max_position_ratio <= 20%
- INFO trend_add_candidate_count=0
- WARN avg_trend_hold_bars 8.9 < v020*0.95 (9.4)

### full_bias_train
- PASS max_position_ratio <= 20%
- INFO trend_add_candidate_count=0
- WARN avg_trend_hold_bars 8.9 < v020*0.95 (9.4)
- PASS legacy_trend_direct_read_count=0
- PASS hard_counter_open_count=0
- PASS bias_reason_codes_coverage=1.0

### open_size_bias_valid
- PASS max_position_ratio <= 20%
- INFO trend_add_candidate_count=0
- PASS avg_trend_hold_bars vs v020

### full_bias_valid
- PASS max_position_ratio <= 20%
- INFO trend_add_candidate_count=0
- PASS avg_trend_hold_bars vs v020
- PASS legacy_trend_direct_read_count=0
- PASS hard_counter_open_count=0
- PASS bias_reason_codes_coverage=1.0


## June decline bias spot check

- window: `2026-06-01 .. 2026-06-18` (432 bars)
- crash bars: 8, long opens: 1, short opens: 2

### Checks
- crash_p1_block_long_present: PASS
- crash_short_boost_or_block_present: PASS
- crash_bars_block_long: PASS

### Top bias reason codes
- `MICRO_ALIGNED`: 406
- `SEGMENT_ADD_ALLOWED`: 388
- `TREND_BREAKING_TIGHTEN`: 285
- `LEG_NONE`: 275
- `MACRO_FROM_REGIME`: 242
- `MACRO_DERIVED_WEAK`: 190
- `LEGACY_DOWNTREND_TIGHTEN_LONG`: 163
- `TREND_SIGNAL_FALLBACK`: 71
- `SEGMENT_LEG_UP`: 59
- `SEGMENT_CONFIRMED_UP_LEG_BLOCK_SHORT`: 59
- `SLOW_UP_BOOST_LONG_SIZE`: 43
- `SEGMENT_LEG_DOWN`: 27

### Crash-window samples
- `2026-06-01 14:00:00+00:00` action=OPEN_SHORT reason=OPEN_SHORT_CRASH allow_L=0 open_bias_S=1.35 leg=FAST_DOWN_LEG/IMPULSE reasons=TREND_SIGNAL_FALLBACK|MICRO_ALIGNED|MACRO_FROM_REGIME|SEGMENT_ADD_ALLOWED|CRASH_P1_BLOCK_LONG|CRASH_P1_BOOST_SHORT_OPEN|LEGACY_DOWNTREND_TIGHTEN_LONG
- `2026-06-01 15:00:00+00:00` action=HOLD reason=HOLD_CRASH_TREND_CONFIRMING allow_L=0 open_bias_S=1.0 leg=FAST_DOWN_LEG/IMPULSE reasons=TREND_SIGNAL_FALLBACK|MICRO_ALIGNED|MACRO_FROM_REGIME|SEGMENT_ADD_ALLOWED|CRASH_P1_BLOCK_LONG|LEGACY_DOWNTREND_TIGHTEN_LONG
- `2026-06-02 14:00:00+00:00` action=HOLD reason=HOLD_CRASH_TREND_CONFIRMING allow_L=0 open_bias_S=0.0 leg=FAST_DOWN_LEG/IMPULSE reasons=SEGMENT_LEG_UP|MICRO_ACCELERATION|MACRO_DERIVED_WEAK|SEGMENT_CONFIRMED_UP_LEG_BLOCK_SHORT|SEGMENT_ADD_ALLOWED|CRASH_P1_BLOCK_LONG
- `2026-06-02 15:00:00+00:00` action=HOLD reason=HOLD_CRASH_TREND_CONFIRMING allow_L=0 open_bias_S=0.0 leg=FAST_DOWN_LEG/IMPULSE reasons=SEGMENT_LEG_UP|MICRO_ACCELERATION|MACRO_DERIVED_WEAK|SEGMENT_CONFIRMED_UP_LEG_BLOCK_SHORT|SEGMENT_ADD_ALLOWED|CRASH_P1_BLOCK_LONG
- `2026-06-02 16:00:00+00:00` action=HOLD reason=HOLD_CRASH_TREND_CONFIRMING allow_L=0 open_bias_S=0.0 leg=FAST_DOWN_LEG/IMPULSE reasons=SEGMENT_LEG_UP|MICRO_ALIGNED|MACRO_DERIVED_WEAK|SEGMENT_CONFIRMED_UP_LEG_BLOCK_SHORT|SEGMENT_ADD_ALLOWED|CRASH_P1_BLOCK_LONG
- `2026-06-04 00:00:00+00:00` action=BLOCK reason=BLOCK_BIAS_SHORT_OPEN allow_L=0 open_bias_S=0.0 leg=FAST_DOWN_LEG/IMPULSE reasons=SEGMENT_LEG_UP|MICRO_ACCELERATION|MACRO_DERIVED_WEAK|SEGMENT_CONFIRMED_UP_LEG_BLOCK_SHORT|SEGMENT_ADD_ALLOWED|CRASH_P1_BLOCK_LONG|CRASH_P1_BOOST_SHORT_OPEN
- `2026-06-04 01:00:00+00:00` action=BLOCK reason=BLOCK_BIAS_SHORT_OPEN allow_L=0 open_bias_S=0.0 leg=FAST_DOWN_LEG/IMPULSE reasons=SEGMENT_LEG_UP|MICRO_ACCELERATION|MACRO_DERIVED_WEAK|SEGMENT_CONFIRMED_UP_LEG_BLOCK_SHORT|SEGMENT_ADD_ALLOWED|CRASH_P1_BLOCK_LONG|CRASH_P1_BOOST_SHORT_OPEN
- `2026-06-18 15:00:00+00:00` action=REDUCE reason=REDUCE_TREND_PROFIT_LOCK allow_L=0 open_bias_S=1.0 leg=FAST_DOWN_LEG/IMPULSE reasons=TREND_SIGNAL_FALLBACK|MICRO_ALIGNED|MACRO_DERIVED_WEAK|SEGMENT_ADD_ALLOWED|CRASH_P1_BLOCK_LONG|LEGACY_DOWNTREND_TIGHTEN_LONG
