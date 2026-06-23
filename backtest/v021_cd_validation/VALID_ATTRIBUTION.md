# v021 valid split trade attribution (v020 vs open_size_bias)

- v020 trades: 13
- open_size_bias trades: 13

## Only in v020

- `2026-03-05 19:00` OPEN_LONG_SIGNAL pnl=0.006393 bars=8
- `2026-03-16 11:00` OPEN_LONG_SIGNAL pnl=0.002888 bars=8
- `2026-04-09 01:00` OPEN_LONG_SIGNAL pnl=0.012243 bars=8

## Only in open_size_bias

- `2026-03-05 20:00` OPEN_LONG_SIGNAL pnl=0.001952 bars=8
- `2026-03-16 10:00` OPEN_LONG_SIGNAL pnl=0.030495 bars=8
- `2026-04-09 04:00` OPEN_LONG_SIGNAL pnl=0.001836 bars=8

## PnL summary of non-overlapping trades

- v020-only total pnl: 0.021525
- bias-only total pnl: 0.034284
- delta (bias - v020) on unique trades: 0.012759

## Root-cause notes

- `2026-03-26`: v020 `BLOCK_LONG_DOWNTREND` (risk.py); bias allowed long because `leg_direction=UP` on `FAST_DOWN_LEG` bypassed downtrend tighten.
- Timing shifts on `2026-03-05/16/04-09`: relaxed `open_bias` thresholds shift entry bar by 1–3h.
- Phase C `size_bias` increases max position (~11.6% → ~17.2%), amplifying both wins and losses.

## After fix (downtrend long requires confirmed UP leg)

- trade_count: 15 → **13** (aligned with v020)
- valid total_return: 9.83% vs v020 11.61% (remaining gap from entry timing + size scaling)
- `trend_size_boost` sweep on valid: 1.0–1.2 does not close gap vs v020; timing dominates.
