# Prod v1.1.0 巩固报告（架构师 P0）

**Split**: test
**Health vs baseline**: False

## 指标

| 指标 | 当前 | 基线 |
|------|------|------|
| total_return | 0.10091510103731793 | 0.09312992708799195 |
| max_drawdown | -0.008545561025824003 | -0.008545561025824003 |
| trade_count | 10 | 8 |
| risk_rule_violations | 0 | — |

## 未成交归因（FLAT）

- FLAT 占比: **89.8%** (1174/1308 bars)
- WATCH_SLOW_UPTREND: **469** bars，34 段（最长 55 bars）
- slow_up 开仓: **0**（watch→open 比 0.0000）
- HOLD_NO_ENTRY: 693 bars
- TEQ reject bars: 0

### FLAT Top reasons

- `HOLD_NO_ENTRY`: 693
- `WATCH_SLOW_UPTREND`: 469
- `OPEN_SHORT_CRASH`: 5
- `OPEN_LONG_SIGNAL`: 4
- `BLOCK_BIAS_SHORT_OPEN`: 2
- `OPEN_SHORT_SIGNAL`: 1

```bash
bash scripts/prod_health_monitor.sh
```
