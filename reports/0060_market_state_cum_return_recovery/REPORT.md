# 0060_market_state_cum_return_recovery 多任务市场状态模型训练报告

## 实验依据

- `document/架构师-005-0059训练复盘与0060目标指导.md`
- `document/架构师-004-当前训练进度复盘与目标修正.md`

## 目标阶段: **cum_return_recovery**（累计方向+收益排序恢复）

## 本轮训练配置

- `direction_threshold_quantile=0.25`
- `risk_threshold_quantile=0.7`
- return/direction/volatility/risk = 0.38/0.42/0.12/0.08
- cum_direction_weight=0.045
- class_weights=True, balanced_class_weights=True
- init_market_checkpoint=`checkpoints/0059c_market_state_balanced_mature/market_state_best.pt`
- score=recovery_0060, epochs=40, lr=6e-05

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

| 指标 | 0060_market_state_cum_return_recovery | 0059c balanced |
|------|------|------|
| cum_direction_acc | 56.8% | 56.1% |
| direction_acc | 35.8% | 34.6% |
| direction_macro_f1 | 0.335 | 0.341 |
| return_ic | 0.027 | 0.038 |
| return_mae | 0.022604 | 0.023671 |
| volatility_mae | 0.055999 | 0.061748 |
| risk_f1 | 0.546 | 0.542 |
| loss | 0.6975 | 0.7289 |

## 最佳验证集

- composite_score=0.4054
- cum_direction_acc=58.5%
- direction_macro_f1=0.352
- return_ic=-0.003
- risk_f1=0.513
- volatility_mae=0.054308

## 测试诊断

- direction_pred: `{'direction_pred_c0': 0.503, 'direction_pred_c1': 0.224, 'direction_pred_c2': 0.273}`
- risk_positive_rate_true/pred: 0.232 / 0.242

- risk_precision/recall: 0.302 / 0.314
- direction_recall down/flat/up: 0.519 / 0.221 / 0.281

## 验收结论（累计方向+收益排序恢复）

- target_stage: **cum_return_recovery**
- decision: **reject**
- gates_passed: 4/9
- blocking_metric: `cum_direction_acc>=58%`
- 未达标项: cum_direction_acc>=58%, direction_macro_f1>=0.34, return_ic>=0.05, direction_pred_down_in_[32%,43%], direction_pred_flat_in_[25%,38%]

## 图表

- `01_training_curves.png`
- `02_test_metrics.png`

