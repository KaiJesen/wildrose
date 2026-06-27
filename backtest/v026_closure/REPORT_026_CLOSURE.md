# 026 项目结案报告

| 项 | 内容 |
|----|------|
| 状态 | **结案** — 模型轨部分达标，探索 coverage 膝点未突破 |
| 数据 | BTCUSDT 1h test split（冻结 K 线） |
| 产物目录 | `backtest/v026_closure` |

## 回测与日均交易次数

| 臂 | 总交易 | 日历天数 | **日均交易** | 有交易日均笔数 | 总收益 | 最大回撤 | 事后最优(DP) |
|----|--------|----------|--------------|----------------|--------|----------|--------------|
| B0 | 13 | 55 | **0.236** | 1.182 | 9.01% | -1.21% | 526 |
| M2 | 14 | 55 | **0.255** | 1.077 | 9.14% | -2.05% | 526 |
| M3 | 14 | 55 | **0.255** | 1.077 | 8.07% | -2.06% | 526 |

说明：
- **日均交易** = 成交笔数 ÷ 回测区间日历天数（含无交易日）。
- **有交易日均笔数** = 成交笔数 ÷ 至少有一笔开仓的自然日数。
- **事后最优买卖点** 来自 `trade/tools/optimal_trade_points` 动态规划（hindsight，非实盘信号）。

## 图表

### B0 研究基线 (c1_pw20) (B0)

- 策略 + 最优 overlay：`backtest/v026_closure/b0_test/trades_with_optimal.png`
- 引擎默认买卖点：`backtest/v026_closure/b0_test/trade_points.png`
- 回测目录：`backtest/v026_closure/b0_test`

### M2 C3+C1+D1 (M2)

- 策略 + 最优 overlay：`backtest/v026_closure/m2_test/trades_with_optimal.png`
- 引擎默认买卖点：`backtest/v026_closure/m2_test/trade_points.png`
- 回测目录：`backtest/v026_closure/m2_test`

### M3 +A1 CORAL (M3)

- 策略 + 最优 overlay：`backtest/v026_closure/m3_test/trades_with_optimal.png`
- 引擎默认买卖点：`backtest/v026_closure/m3_test/trade_points.png`
- 回测目录：`backtest/v026_closure/m3_test`

## 结案结论

1. Phase 0~2 模型轨：C3/C1+D1/A1 依次 PASS（part_auc ≥ 0.62）。
2. Phase 3 探索门：M2/M3 test coverage 均 **26.67%**（门禁 28%），M3 return 8.07% 低于 M2。
3. **prod v1.1.0 保持不变**；026 作为研究支线结案。

## 复现
```bash
python examples/run_v026_closure.py
```
