# 020 TrendLegClassifier 训练报告

**项目**: 软件设计师-020 趋势区间分段框架（Phase E–F Student）  
**数据**: BTCUSDT 1h，365d（`binance_vision_BTCUSDT_1h_365d_end20260623.csv`）  
**Teacher 标签**: `data/labels/trend_leg_v020_teacher/teacher_labels.csv`  
**Checkpoint**: `checkpoints/020_trend_leg_classifier/020a_trend_leg_baseline/best.pt`  
**运行**: `python3 examples/train_trend_leg_classifier.py --epochs 15 --run-name 020a_trend_leg_baseline`

---

## 1. 目标与验收

| 指标 | 文档验收线 | 本次 test | 结论 |
|------|-----------:|----------:|------|
| macro_f1_leg_type | ≥ 0.35 | **0.3347** | 略低于线，接近 |
| f1_confirmed_only | — | 0.2725 | 已确认 bar 上可区分 |
| kappa_vs_teacher | — | 0.2118 | 与 Teacher 一致性中等 |

Student 模型当前**未接入**交易引擎回测（Phase F 融合待做）；本报告仅覆盖 Teacher 蒸馏训练。

---

## 2. Teacher 标签统计

| 项 | 值 |
|----|-----|
| 样本行数 | 8,734 |
| 已确认 leg 占比 | 15.98% |
| 平均 bars_since_leg_start | 10.8 |

**leg_type 分布**（训练集偏斜明显）:

| leg_type | 数量 |
|----------|-----:|
| TRANSITION_LEG | 4,961 |
| RANGE_LEG | 1,858 |
| FAST_DOWN_LEG | 913 |
| FAST_UP_LEG | 810 |
| SLOW_DOWN_LEG | 145 |
| SLOW_UP_LEG | 43 |
| SURGE_LEG | 4 |

**sub_phase**: IMPULSE 占主导（7,941 / 8,734）。

---

## 3. 模型与训练配置

| 参数 | 值 |
|------|-----|
| 架构 | TransformerEncoder，2 层，d_model=128，4 heads |
| context_bars | 128 |
| 特征 | `compute_causal_features`（OHLCV 因果特征） |
| 划分 | train 70% / valid 15% / test 15%（时间序） |
| batch_size | 64 |
| lr | 3e-4，AdamW weight_decay=1e-2 |
| epochs | 15 |
| 损失 | CE(leg_type) + 0.5·CE(sub_phase) + 0.3·Huber(progress) + 0.2·BCE(confirmed) |
| 类别权重 | leg_type 逆频率加权 |

---

## 4. 训练曲线摘要

| Epoch | train_loss | valid_loss | macro_f1 | conf_f1 | kappa |
|------:|-----------:|-----------:|---------:|--------:|------:|
| 1 | 1.045 | 0.951 | 0.291 | 0.295 | 0.179 |
| 8 | 0.701 | 0.977 | 0.296 | 0.287 | 0.176 |
| 11 | 0.666 | 1.001 | **0.302** | **0.307** | **0.192** |
| 15 | 0.555 | 1.109 | 0.297 | 0.255 | 0.177 |

- **最佳 valid score**（macro_f1 + 0.5·f1_confirmed）: **0.4549**（epoch 1 附近早停倾向，后期 valid loss 上升）
- **Test OOS**（加载 best checkpoint）: macro_f1=**0.3347**, f1_confirmed=0.2725, kappa=0.2118

---

## 5. 结论与后续

1. **基线可用**：test macro-F1 0.33，接近文档 0.35 验收线；confirmed-only F1 0.27 说明在已确认趋势 leg 上仍有判别力。
2. **主要瓶颈**：TRANSITION_LEG / RANGE_LEG 占比过高，稀有类（SLOW_UP、SURGE）样本极少；建议增加 focal loss 或重采样。
3. **过拟合迹象**：train loss 持续下降而 valid loss 在 epoch 8 后走高，可缩短 epoch 或加强 dropout。
4. **下一步（Phase F）**：实现 `trading_system/adapters/trend_leg_model.py`，将 Student 概率与 `SegmentContext` 融合后再做 A/B 回测。

详细数值见 `metrics.json`。
