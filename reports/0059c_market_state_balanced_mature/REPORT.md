# 0059c_market_state_balanced_mature 多任务市场状态模型训练报告

## 实验依据

- `document/004_训练进度复盘与目标修正/架构师-004-当前训练进度复盘与目标修正.md`
- `document/003_理想模型指标目标/架构师-003-理想模型指标目标指导.md`

## 目标阶段: **balanced_mature**（稳定可用→成熟过渡）

## 本轮训练配置

- `direction_threshold_quantile=0.25`
- `risk_threshold_quantile=0.7`
- return/direction/volatility/risk = 0.35/0.45/0.12/0.08
- cum_direction_weight=0.035
- class_weights=True, balanced_class_weights=True
- init_market_checkpoint=`checkpoints/0058_market_state_usable/market_state_best.pt`
- score=balanced_0059, epochs=25

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

| 指标 | 0059c_market_state_balanced_mature | 0050 |
|------|------|------|
| cum_direction_acc | 56.1% | 60.8% |
| direction_acc | 34.6% | 31.2% |
| direction_macro_f1 | 0.341 | 0.308 |
| return_ic | 0.038 | 0.054 |
| return_mae | 0.023671 | 0.035691 |
| volatility_mae | 0.061748 | 0.088098 |
| risk_f1 | 0.542 | 0.530 |
| loss | 0.7289 | 0.8144 |

## 最佳验证集

- composite_score=0.3971
- cum_direction_acc=58.5%
- direction_macro_f1=0.345
- return_ic=0.004
- risk_f1=0.501
- volatility_mae=0.059322

## 测试诊断

- direction_pred: `{'direction_pred_c0': 0.378, 'direction_pred_c1': 0.318, 'direction_pred_c2': 0.304}`
- risk_positive_rate_true/pred: 0.232 / 0.297

- risk_precision/recall: 0.291 / 0.372
- direction_recall down/flat/up: 0.387 / 0.321 / 0.319

## 验收结论（稳定可用→成熟过渡）

- target_stage: **balanced_mature**
- decision: **conditional**
- gates_passed: 6/8
- blocking_metric: `cum_direction_acc>=58%`
- 未达标项: cum_direction_acc>=58%, return_ic>=0.05

## 图表

- `01_training_curves.png`
- `02_test_metrics.png`

