# 0050 多任务市场状态模型正式实验报告

## 实验依据

依据 `document/软件设计师_001_多任务市场状态改造实施与训练建议.md` 第一轮正式配置执行。

## 数据与模型

- 数据源: `binance_vision` / `BTCUSDT` / `1h` / `365` 天
- 上下文: `128` bar，预测 horizon: `5`
- `d_model=128`, `n_heads=4`, `trunk_layers=2`
- 多任务头: return / direction(3-class) / volatility / risk(2-class)
- 初始化 encoder: `checkpoints/0050_market_state_embed/stage2_vqvae.pt`

## 标签阈值（仅 train 拟合）

- `direction_threshold=0.00120399`
- `risk_vol_threshold=0.00492658`

## 损失权重

- return=0.4, direction=0.4, volatility=0.15, risk=0.05

## 测试集指标

| 指标 | 正式实验 | Smoke(0049) |
|------|----------|-----------|
| cum_direction_acc | 55.4% | 54.7% |
| direction_acc | 34.6% | 34.6% |
| direction_macro_f1 | 0.269 | 0.283 |
| return_ic | 0.073 | -0.026 |
| return_mae | 0.025979 | 0.040250 |
| volatility_mae | 0.085249 | 0.150485 |
| risk_f1 | 0.455 | 0.453 |
| loss | 0.6810 | 0.7321 |

## 最佳验证集（选模分数）

- composite_score=0.6347
- cum_direction_acc=56.5%
- return_ic=0.034
- direction_macro_f1=0.246
- volatility_mae=0.083297

## 图表

- `01_training_curves.png`：训练/验证损失与多任务指标曲线
- `02_test_metrics.png`：测试集指标（与 smoke 对比）

## 结论与下一步

- 累计方向准确率不低于 smoke 基线。
- 收益 IC 在测试集为正，回归头具备一定预测力。
- 后续可做 CPC 主干 vs VQ 主干同头对比（文档 5.3）。
