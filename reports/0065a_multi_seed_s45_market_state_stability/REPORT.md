# 0065a_multi_seed_s45_market_state_stability 多任务市场状态模型训练报告

## 实验依据

- `document/011_0065分支固化/架构师-011-0064训练复盘与下一阶段方向.md`
- `document/009_新结构分支稳定性验证/项目经理-009-双轨验收与基线说明.md`
- `document/008_0062累计收益稳定化/架构师-008-0061训练复盘与指导修正.md`
- `document/007_0061新结构训练目标/架构师-007-0061新结构训练目标指导.md`

## 验收轨道

- acceptance_track: **B**（新结构分支轨）
- 轨道说明: `document/011_0065分支固化/架构师-011-0064训练复盘与下一阶段方向.md`

## 分支类型

- branch_type: **cum_return_candidate**
- known_limitation: `weak_step_return_ic`
- branch_status: `recommended_stable_template`

## 目标阶段: **return_direction_branch**（收益/累计方向分支解耦）

## 本轮训练配置

- `direction_threshold_quantile=0.25`
- `risk_threshold_quantile=0.7`
- return/direction/volatility/risk = 0.45/0.22/0.1/0.09
- cum_direction_weight=0.0
- cum_return_weight=0.14
- cum_direction_head_weight=0.03
- return_consistency_weight=0.02
- return_horizon_weights=uniform(1.0)
- use_cum_heads=True, use_horizon_return_head=True, detach_risk_vol_heads=False
- class_weights=True, balanced_class_weights=False
- direction_class_weights=False, risk_class_weights=True
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

| 指标 | 0065a_multi_seed_s45_market_state_stability | 0059c balanced |
|------|------|------|
| cum_direction_acc | 60.8% | 56.1% |
| cum_direction_head_acc | 59.5% |
| cum_direction_from_return_acc | 60.8% |
| direction_acc | 35.9% | 34.6% |
| direction_macro_f1 | 0.338 | 0.341 |
| return_ic | 0.006 | 0.038 |
| cum_return_ic | 0.137 |
| return_mae | 0.029064 | 0.023671 |
| cum_return_mae | 0.120830 |
| volatility_mae | 0.044423 | 0.061748 |
| risk_f1 | 0.554 | 0.542 |
| loss | 0.4429 | 0.7289 |

## 最佳验证集

- composite_score=0.3055
- cum_direction_acc=50.3%
- cum_direction_head_acc=52.4%
- cum_direction_from_return_acc=50.3%
- direction_macro_f1=0.361
- return_ic=-0.015
- cum_return_ic=0.041
- risk_f1=0.510
- volatility_mae=0.043319
- best_selection_mode=hard_gated
- no_valid_checkpoint=False

## 验证集分布（最佳 checkpoint）

- direction_pred: `{'direction_pred_c0': 0.384, 'direction_pred_c1': 0.2, 'direction_pred_c2': 0.416}`
- risk_positive_rate_true/pred: 0.336 / 0.298

## 测试诊断

- direction_pred: `{'direction_pred_c0': 0.384, 'direction_pred_c1': 0.207, 'direction_pred_c2': 0.409}`
- risk_positive_rate_true/pred: 0.232 / 0.272

- risk_precision/recall: 0.308 / 0.360
- direction_recall down/flat/up: 0.397 / 0.179 / 0.449
- step_cum_return_gap_mae=0.167079
- return_ic_h1..h5: [-0.039, -0.017, -0.015, 0.026, 0.019]

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

