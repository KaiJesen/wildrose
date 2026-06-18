# 0051_market_state_tuned 多任务市场状态模型训练报告

## 实验依据

- `document/架构师-002-模型指标训练指导.md`
- `document/软件设计师_002_市场状态模型下一轮优化[训练建议].md`
- `document/软件设计师_003_市场状态模型最终训练建议[训练建议].md`

## 阶段 A：阈值 + 损失权重调优

- `direction_threshold_quantile=0.25`
- `risk_threshold_quantile=0.7`
- return/direction/volatility/risk = 0.3/0.5/0.1/0.1
- epochs=60, class_weights=False

## 数据与模型

- 数据源: `binance_vision` / `BTCUSDT` / `1h` / `365` 天
- 初始化 encoder: `checkpoints/0050_market_state_embed/stage2_vqvae.pt`

## 标签阈值（仅 train 拟合）

- `direction_threshold=0.00081339`
- `risk_vol_threshold=0.00411622`

## Train 类别分布

- direction: `{'c0': 0.36871657754010695, 'c1': 0.25, 'c2': 0.38128342245989305}`
- risk_positive_rate: `0.409`

## 测试集指标

| 指标 | 0051 | 0050 |
|------|------|------|
| cum_direction_acc | 50.0% | 55.4% |
| direction_acc | 34.2% | 34.6% |
| direction_macro_f1 | 0.285 | 0.269 |
| return_ic | 0.043 | 0.073 |
| return_mae | 0.069752 | 0.025979 |
| volatility_mae | 0.098955 | 0.085249 |
| risk_f1 | 0.447 | 0.455 |
| loss | 0.7689 | 0.6810 |

## 最佳验证集

- composite_score=0.4020
- cum_direction_acc=55.8%
- direction_macro_f1=0.321
- return_ic=0.017
- risk_f1=0.402
- volatility_mae=0.097066

## 测试诊断

- direction_pred: `{'direction_pred_c0': 0.554, 'direction_pred_c1': 0.041, 'direction_pred_c2': 0.405}`
- risk_positive_rate_true/pred: 0.232 / 0.019

## 验收结论

- decision: **conditional**
- 未达标项: cum_direction_acc>=56%, direction_macro_f1>=0.30, risk_f1>=0.48

## 图表

- `01_training_curves.png`
- `02_test_metrics.png`

