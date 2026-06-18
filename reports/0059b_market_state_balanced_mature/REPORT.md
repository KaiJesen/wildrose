# 0059b_market_state_balanced_mature 多任务市场状态模型训练报告

## 实验依据

- `document/004/架构师-004-当前训练进度复盘与目标修正.md`
- `document/003/架构师-003-理想模型指标目标指导.md`

## 目标阶段: **balanced_mature**（稳定可用→成熟过渡）

## 本轮训练配置

- `direction_threshold_quantile=0.25`
- `risk_threshold_quantile=0.7`
- return/direction/volatility/risk = 0.35/0.45/0.12/0.08
- cum_direction_weight=0.04
- class_weights=True, balanced_class_weights=True
- init_market_checkpoint=`none`
- score=balanced_0059, epochs=60

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

| 指标 | 0059b_market_state_balanced_mature | 0050 |
|------|------|------|
| cum_direction_acc | 58.1% | 60.8% |
| direction_acc | 34.3% | 31.2% |
| direction_macro_f1 | 0.323 | 0.308 |
| return_ic | 0.048 | 0.054 |
| return_mae | 0.156641 | 0.035691 |
| volatility_mae | 0.111790 | 0.088098 |
| risk_f1 | 0.511 | 0.530 |
| loss | 0.7812 | 0.8144 |

## 最佳验证集

- composite_score=0.3537
- cum_direction_acc=51.0%
- direction_macro_f1=0.303
- return_ic=0.029
- risk_f1=0.493
- volatility_mae=0.109142

## 测试诊断

- direction_pred: `{'direction_pred_c0': 0.522, 'direction_pred_c1': 0.269, 'direction_pred_c2': 0.209}`
- risk_positive_rate_true/pred: 0.232 / 0.393

- risk_precision/recall: 0.261 / 0.442
- direction_recall down/flat/up: 0.512 / 0.289 / 0.198

## 验收结论（稳定可用→成熟过渡）

- target_stage: **balanced_mature**
- decision: **reject**
- gates_passed: 4/8
- blocking_metric: `direction_macro_f1>=0.33`
- 未达标项: direction_macro_f1>=0.33, return_ic>=0.05, volatility_mae<=0.085, risk_f1>=0.52

## 图表

- `01_training_curves.png`
- `02_test_metrics.png`

