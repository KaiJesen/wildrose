# 017c BestPointSignal 训练报告（major_legs 标签）

**Teacher**: `trade/tools/optimal_trade_points.py` · `mode=major_legs`  
**标签**: `data/labels/best_point_v017_major_legs/`  
**配置**: `configs/best_point_signal_v017_major_legs.json`  
**Checkpoint**: `checkpoints/017_best_point_signal/017c_best_point_major_legs/best.pt`

---

## 1. 与 017b 的差异

| 项 | 017b (dp) | **017c (major_legs)** |
|----|-----------|------------------------|
| 标注模式 | 动态规划最大总收益 | ZigZag 主波段整腿 |
| Teacher 交易数 | ~碎片化多笔 | **869** 笔 |
| 平均持仓 | 较短 | **9.2 bars** |
| 标签覆盖率 | 较低 | **94.7%** |
| avg_net_roi (Teacher) | — | **30.1%** |

`major_legs` 与修正后工具一致：覆盖主要涨跌腿，避免趋势内频繁反手造成的标签碎片。

---

## 2. 训练配置（同 017 流程）

- 特征：`compute_causal_features`，context=96
- 划分：train 70% / valid 15% / test 15%
- 模型：TransformerEncoder BestPointSignalModel
- 损失：entry + hold + exit CE + opportunity Huber
- epochs=12, lr=3e-4, batch=64

---

## 3. 指标

| 指标 | valid (best) | **test OOS** | 017b valid |
|------|-------------:|-------------:|-----------:|
| entry_acc | 0.551 | 0.478 | 0.529 |
| opportunity_ic | **0.384** | **0.416** | 0.283 |
| entry_macro_f1 | — | **0.458** | — |
| best_score | 0.935 | — | 0.968 |

**结论**：`major_legs` 标签下 test **opportunity_ic 0.416**，显著高于 017b 的 0.283；entry 准确率略降，符合标签更宽、机会区更大的预期。模型可用于 observe 模式辅助过滤。

---

## 4. 复现命令

```bash
# 1. 构建标签
python3 examples/build_best_point_labels.py --mode major_legs

# 2. 训练
python3 examples/train_best_point_signal_model.py \
  --labels-file data/labels/best_point_v017_major_legs/BTCUSDT_1h_labels.csv \
  --run-name 017c_best_point_major_legs

# 3. test 评估
python3 examples/evaluate_best_point_signal_model.py \
  --checkpoint checkpoints/017_best_point_signal/017c_best_point_major_legs/best.pt \
  --labels-file data/labels/best_point_v017_major_legs/BTCUSDT_1h_labels.csv
```
