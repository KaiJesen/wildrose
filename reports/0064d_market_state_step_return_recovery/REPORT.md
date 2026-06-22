# 0064d_market_state_step_return_recovery 多任务市场状态模型训练报告

## 实验依据

- `document/010/架构师-010-0064收益指标恢复执行指导.md`
- `document/010/架构师-010-0064收益指标恢复执行指导.md`
- `document/009/项目经理-009-双轨验收与基线说明.md`
- `document/008/架构师-008-0061训练复盘与指导修正.md`
- `document/007/架构师-007-0061新结构训练目标指导.md`

## 验收轨道

- acceptance_track: **B**（新结构分支轨）
- 轨道说明: `document/010/架构师-010-0064收益指标恢复执行指导.md`

## 目标阶段: **step_return_recovery**（step return_ic 恢复（0064））

## 本轮训练配置

- `direction_threshold_quantile=0.25`
- `risk_threshold_quantile=0.7`
- return/direction/volatility/risk = 0.42/0.22/0.1/0.09
- cum_direction_weight=0.0
- cum_return_weight=0.14
- cum_direction_head_weight=0.03
- return_consistency_weight=0.005
- return_horizon_weights=[1.2, 1.15, 1.1, 1.0, 1.0]
- use_cum_heads=True, use_horizon_return_head=True, detach_risk_vol_heads=False
- class_weights=True, balanced_class_weights=False
- direction_class_weights=False, risk_class_weights=True
- detach_risk_vol_after_epoch=0
- init_market_checkpoint=`checkpoints/0062e_market_state_return_ic_recovery/market_state_best.pt`
- score=recovery_0064, epochs=30, lr=6e-05

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

| 指标 | 0064d_market_state_step_return_recovery | 0059c balanced |
|------|------|------|
| cum_direction_acc | 51.4% | 56.1% |
| cum_direction_head_acc | 41.2% |
| cum_direction_from_return_acc | 51.4% |
| direction_acc | 38.6% | 34.6% |
| direction_macro_f1 | 0.295 | 0.341 |
| return_ic | 0.009 | 0.038 |
| cum_return_ic | -0.002 |
| return_mae | 0.011013 | 0.023671 |
| cum_return_mae | 0.036411 |
| volatility_mae | 0.018584 | 0.061748 |
| risk_f1 | 0.563 | 0.542 |
| loss | 0.4024 | 0.7289 |

## 最佳验证集

- composite_score=0.2870
- cum_direction_acc=51.4%
- cum_direction_head_acc=41.2%
- cum_direction_from_return_acc=51.4%
- direction_macro_f1=0.295
- return_ic=0.009
- cum_return_ic=-0.002
- risk_f1=0.563
- volatility_mae=0.018584
- best_selection_mode=diagnostic_last
- no_valid_checkpoint=True

## 验证集分布（最佳 checkpoint）

- direction_pred: `{'direction_pred_c0': 0.412, 'direction_pred_c1': 0.0, 'direction_pred_c2': 0.588}`
- risk_positive_rate_true/pred: 0.232 / 0.282

## 测试诊断

- direction_pred: `{'direction_pred_c0': 0.412, 'direction_pred_c1': 0.0, 'direction_pred_c2': 0.588}`
- risk_positive_rate_true/pred: 0.232 / 0.282

- risk_precision/recall: 0.321 / 0.390
- direction_recall down/flat/up: 0.439 / 0.000 / 0.608
- step_cum_return_gap_mae=0.054148
- return_ic_h1..h5: [0.004, 0.013, -0.121, 0.036, 0.05]

## 坍缩门槛

- valid collapse gates: `{'direction_pred_down<=60%': True, 'direction_pred_flat>=8%': False, 'direction_pred_up>=10%': True, 'risk_positive_rate_pred>=5%': True, 'risk_ratio<=1.8': True}`
- test collapse gates: `{'direction_pred_down<=60%': True, 'direction_pred_flat>=8%': False, 'direction_pred_up>=10%': True, 'risk_positive_rate_pred>=5%': True, 'risk_ratio<=1.8': True}`

## 验收结论（step return_ic 恢复（0064））

- target_stage: **step_return_recovery**
- decision: **reject**
- gates_passed: -3/10
- blocking_metric: `return_ic>=0.020`
- 未达标项: no_valid_checkpoint, return_ic>=0.020, cum_return_ic>=0.100, cum_direction_from_return_acc>=58%, direction_macro_f1>=0.320, direction_pred_flat_in_[10%,45%], direction_pred_up_in_[15%,45%], return_ic<0.015, cum_return_ic<0.090, cum_direction_from_return_acc<56%, collapse_gates_failed, direction_pred_flat==0, cum_direction_head_acc<45%

## 图表

- `01_training_curves.png`
- `02_test_metrics.png`

