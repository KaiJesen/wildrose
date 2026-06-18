# 0052_market_state_class_weights 多任务市场状态模型训练报告

## 实验依据

- `document/002/架构师-002-模型指标训练指导.md`
- `document/002/软件设计师_002_市场状态模型下一轮优化[训练建议].md`
- `document/003/软件设计师_003_市场状态模型最终训练建议[训练建议].md`

## 阶段 B：类别权重（在 0051 配置基础上）

- 与 0051 相同：阈值 quantile 0.25/0.70，损失权重 0.30/0.50/0.10/0.10
- 额外启用 `--use-class-weights`（direction/risk CE 按 train 频率反比加权）

## 测试集指标（vs 0050）

| 指标 | 0052 | 0050 |
|------|------|------|
| cum_direction_acc | 54.7% | 55.4% |
| direction_macro_f1 | **0.318** | 0.269 |
| return_ic | 0.029 | 0.073 |
| volatility_mae | 0.091 | 0.085 |
| risk_f1 | **0.520** | 0.455 |

## 诊断要点

- 预测方向分布更接近真实（flat 类不再被严重忽略）：pred c0/c1/c2 ≈ 30%/31%/39%
- risk 正样本率 pred 0.391 vs true 0.232，风险头不再坍缩到全负类

## 验收结论

- **decision: accept**（五项中四项达标）
- 未达标：`cum_direction_acc >= 56%`

## 图表

- `01_training_curves.png`
- `02_test_metrics.png`
