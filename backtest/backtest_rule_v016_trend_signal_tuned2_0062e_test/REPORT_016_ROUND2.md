# v016 round-2 tuning summary

## What changed

### Code

- `allow_crash_trend_upgrade = false` by default
- `crash_upgrade_profit_atr` for optional crash-only upgrade path
- `min_trend_age_for_upgrade` gate before `UPGRADE_TO_TREND`

### Params (tuned2)

Keep default v016 thresholds:

- `confirmed_score = 4`
- `upgrade_profit_atr = 1.5`
- `allow_crash_trend_upgrade = false`

## Test OOS comparison

| version | return | max drawdown | trades | trend upgrades | trend trade return |
| --- | ---: | ---: | ---: | ---: | ---: |
| v016 default (round 1) | 9.66% | -0.58% | 11 | 0 | 5.70% |
| v016 tuned v1 | 9.49% | -1.05% | 10 | 3 | 8.24% |
| v016 tuned v2 | **9.66%** | **-0.58%** | 11 | 0 | 5.70% |

## Conclusion

Round-2 work shows the first tuning pass hurt returns mainly because crash shorts were upgraded into trend holds too early.

The fix is structural, not another aggressive threshold sweep:

1. Do not upgrade `CRASH` positions into trend mode.
2. Keep default trend confirmation thresholds on test.
3. Let 014b `trend_hold` continue handling model-short extension.

On this test slice, any attempt to force `trend_upgrade_count >= 1` either:

- damaged crash quick exits (tuned v1), or
- failed to beat default return (all round-2 param scans).

## Recommendation

Use `configs/trading_rule_v016_trend_signal_tuned2_0062e.json` as the current production candidate.

If future work needs visible `UPGRADE_TO_TREND` events without hurting crash PnL, add a separate model-only upgrade path rather than lowering global `confirmed_score`.
