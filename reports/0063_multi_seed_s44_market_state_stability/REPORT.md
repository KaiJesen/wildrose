# 0063_multi_seed_s44_market_state_stability 多任务市场状态模型训练报告

## 实验依据

- `document/008_0062累计收益稳定化/架构师-008-0061训练复盘与指导修正.md`
- `document/007_0061新结构训练目标/架构师-007-0061新结构训练目标指导.md`

## 目标阶段: **return_direction_branch**（收益/累计方向分支解耦）

## 本轮训练配置

- `direction_threshold_quantile=0.25`
- `risk_threshold_quantile=0.7`
- return/direction/volatility/risk = 0.45/0.22/0.1/0.09
- cum_direction_weight=0.0
- cum_return_weight=0.14
- cum_direction_head_weight=0.03
- return_consistency_weight=0.02
- use_cum_heads=True, use_horizon_return_head=True, detach_risk_vol_heads=False
- class_weights=True, balanced_class_weights=False
- direction_class_weights=False, risk_class_weights=None
- detach_risk_vol_after_epoch=0
- init_market_checkpoint=`checkpoints/0062c_market_state_cum_return_stabilized/market_state_best.pt`
- score=branch_0062, epochs=30, lr=6e-05

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

| 指标 | 0063_multi_seed_s44_market_state_stability | 0059c balanced |
|------|------|------|
| cum_direction_acc | 59.5% | 56.1% |
| cum_direction_head_acc | 55.4% |
| cum_direction_from_return_acc | 59.5% |
| direction_acc | 35.7% | 34.6% |
| direction_macro_f1 | 0.323 | 0.341 |
| return_ic | 0.008 | 0.038 |
| cum_return_ic | 0.113 |
| return_mae | 0.034933 | 0.023671 |
| cum_return_mae | 0.102275 |
| volatility_mae | 0.036715 | 0.061748 |
| risk_f1 | 0.546 | 0.542 |
| loss | 0.4360 | 0.7289 |

## 最佳验证集

- composite_score=0.3033
- cum_direction_acc=50.3%
- cum_direction_head_acc=53.7%
- cum_direction_from_return_acc=50.3%
- direction_macro_f1=0.341
- return_ic=-0.008
- cum_return_ic=0.042
- risk_f1=0.513
- volatility_mae=0.037708
- best_selection_mode=hard_gated
- no_valid_checkpoint=False

## 验证集分布（最佳 checkpoint）

- direction_pred: `{'direction_pred_c0': 0.395, 'direction_pred_c1': 0.156, 'direction_pred_c2': 0.449}`
- risk_positive_rate_true/pred: 0.336 / 0.327

## 测试诊断

- direction_pred: `{'direction_pred_c0': 0.399, 'direction_pred_c1': 0.168, 'direction_pred_c2': 0.434}`
- risk_positive_rate_true/pred: 0.232 / 0.305

- risk_precision/recall: 0.296 / 0.390
- direction_recall down/flat/up: 0.418 / 0.116 / 0.464

## 坍缩门槛

- valid collapse gates: `{'direction_pred_down<=60%': True, 'direction_pred_flat>=8%': True, 'direction_pred_up>=10%': True, 'risk_positive_rate_pred>=5%': True, 'risk_ratio<=1.8': True}`
- test collapse gates: `{'direction_pred_down<=60%': True, 'direction_pred_flat>=8%': True, 'direction_pred_up>=10%': True, 'risk_positive_rate_pred>=5%': True, 'risk_ratio<=1.8': True}`

## 验收结论（收益/累计方向分支解耦）

- target_stage: **return_direction_branch**
- decision: **accept**
- gates_passed: 9/10
- blocking_metric: `return_ic>=0.035`
- 未达标项: return_ic>=0.035

## 图表

- `01_training_curves.png`
- `02_test_metrics.png`

