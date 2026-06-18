# 0054_market_state_usable 多任务市场状态模型训练报告

## 实验依据

- `document/003/架构师-003-理想模型指标目标指导.md`
- `document/002/架构师-002-模型指标训练指导.md`
- `document/003/软件设计师_003_市场状态模型最终训练建议[训练建议].md`

## 目标阶段: **usable**

## 本轮变更（0053 = 0052 + focal loss + cum_direction aux + score_v1）

- `direction_threshold_quantile=0.25`
- `risk_threshold_quantile=0.7`
- return/direction/volatility/risk = 0.3/0.5/0.1/0.1
- cum_direction_weight=0.1
- class_weights=True, risk_focal_loss=False
- epochs=60, early_stop_patience=15

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

| 指标 | 0054_market_state_usable | 0050 |
|------|------|------|
| cum_direction_acc | 48.6% | 54.7% |
| direction_acc | 34.9% | 32.2% |
| direction_macro_f1 | 0.346 | 0.318 |
| return_ic | -0.003 | 0.029 |
| return_mae | 0.032847 | 0.040444 |
| volatility_mae | 0.026159 | 0.091253 |
| risk_f1 | 0.558 | 0.520 |
| loss | 0.7853 | 0.7740 |

## 最佳验证集

- composite_score=0.3749
- cum_direction_acc=55.1%
- direction_macro_f1=0.330
- return_ic=-0.022
- risk_f1=0.490
- volatility_mae=0.026170

## 测试诊断

- direction_pred: `{'direction_pred_c0': 0.264, 'direction_pred_c1': 0.331, 'direction_pred_c2': 0.405}`
- risk_positive_rate_true/pred: 0.232 / 0.308

## 验收结论（可用模型 5 项至少 4 项）

- target_stage: **usable**
- decision: **reject**
- gates_passed: 3/5
- blocking_metric: `cum_direction_acc>=56%`
- 未达标项: cum_direction_acc>=56%, return_ic>0

## 图表

- `01_training_curves.png`
- `02_test_metrics.png`

