# 0061_market_state_return_direction_branch 多任务市场状态模型训练报告

## 实验依据

- `document/007_0061新结构训练目标/架构师-007-0061新结构训练目标指导.md`
- `document/架构师-005-0059训练复盘与0060目标指导.md`

## 目标阶段: **return_direction_branch**（收益/累计方向分支解耦）

## 本轮训练配置

- `direction_threshold_quantile=0.25`
- `risk_threshold_quantile=0.7`
- return/direction/volatility/risk = 0.35/0.38/0.1/0.07
- cum_direction_weight=0.045
- cum_return_weight=0.15
- cum_direction_head_weight=0.12
- return_consistency_weight=0.03
- use_cum_heads=True, use_horizon_return_head=True, detach_risk_vol_heads=True
- class_weights=True, balanced_class_weights=True
- init_market_checkpoint=`checkpoints/0060_market_state_cum_return_recovery/market_state_best.pt`
- score=branch_0061, epochs=40, lr=6e-05

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

| 指标 | 0061_market_state_return_direction_branch | 0059c balanced |
|------|------|------|
| cum_direction_acc | 46.6% | 56.1% |
| cum_direction_head_acc | 40.5% |
| cum_direction_from_return_acc | 46.6% |
| direction_acc | 39.6% | 34.6% |
| direction_macro_f1 | 0.223 | 0.341 |
| return_ic | 0.039 | 0.038 |
| cum_return_ic | 0.105 |
| return_mae | 0.023639 | 0.023671 |
| cum_return_mae | 0.044202 |
| volatility_mae | 0.037922 | 0.061748 |
| risk_f1 | 0.434 | 0.542 |
| loss | 0.6871 | 0.7289 |

## 最佳验证集

- composite_score=0.3259
- cum_direction_acc=46.6%
- cum_direction_head_acc=40.5%
- cum_direction_from_return_acc=46.6%
- direction_macro_f1=0.223
- return_ic=0.039
- cum_return_ic=0.105
- risk_f1=0.434
- volatility_mae=0.037922
- best_selection_mode=hard_gated

## 测试诊断

- direction_pred: `{'direction_pred_c0': 0.942, 'direction_pred_c1': 0.0, 'direction_pred_c2': 0.058}`
- risk_positive_rate_true/pred: 0.232 / 0.000

- risk_precision/recall: 0.000 / 0.000
- direction_recall down/flat/up: 0.965 / 0.000 / 0.061

## 验收结论（收益/累计方向分支解耦）

- target_stage: **return_direction_branch**
- decision: **reject**
- gates_passed: 1/11
- blocking_metric: `cum_direction_head_acc>=58%`
- 未达标项: cum_direction_head_acc>=58%, cum_direction_from_return_acc>=56%, direction_macro_f1>=0.34, return_ic>=0.04, risk_f1>=0.50, risk_ratio_in_[0.8,1.6], direction_pred_down_in_[30%,48%], direction_pred_flat_in_[20%,40%], direction_pred_up_in_[24%,40%], risk_prediction_collapsed

## 图表

- `01_training_curves.png`
- `02_test_metrics.png`

