# BERT思想迁移到时序建模：双向注意力、泄漏风险与因果改造

## 问题定义
- 问题：BERT的双向注意力能否直接用于K线预测？
- 结论：不能直接用于实盘可回测预测任务，因为双向注意力会读取未来信息，产生标签泄漏。

## 方案候选
- 方案A：直接使用双向Encoder（BERT风格）做监督学习。
- 方案B：改为因果注意力（Causal Mask）只看历史。
- 方案C：编码器保留双向，但仅用于离线表征学习，再迁移到因果预测头。

## 选型结论
- 选择：优先方案B，必要时结合方案C。
- 选择原因：
  - 回测和实盘场景要求严格因果性。
  - 方案B与交易时信息集一致，评估更可信。
  - 方案C可用于预训练增强，但上线预测仍需因果约束。
- 代价与权衡：
  - 因果注意力的短期效果可能弱于“作弊式”双向模型。
  - 需要更细致的特征工程和多尺度上下文设计弥补性能。

## Python实现
- 输入与特征工程：
  - 每个时刻输入 `f_t=[OHLCV, returns, indicators, calendar]`。
  - 滚动指标只允许使用 `<= t` 的历史窗口。
- Embedding实现要点：
  - `x_t = E_value(f_t) + E_time(t) + E_asset(a) + E_pos(t)`。
  - 推理时只输入历史序列 `1...t`，预测 `t+h`。
- 关键代码片段：

```python
import torch

def causal_mask(seq_len: int, device=None):
    # True 表示被mask（不可见）
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

# x: [B, T, D]
# attn_mask: [T, T]，上三角为True，确保位置i只能看到<=i的信息
attn_mask = causal_mask(T, device=x.device)
```

## 回测设计
- 数据切分：
  - 时间顺序切分：train/valid/test，不允许随机打乱。
  - 使用walk-forward做滚动验证。
- 回测区间与频率：
  - 至少覆盖一个完整牛熊周期（若数据允许）。
  - 频率与任务一致（如5m/1h/1d）。
- 指标与基线：
  - 预测指标：IC、RankIC、方向准确率、MSE/MAE（按任务）。
  - 策略指标：年化收益、夏普、最大回撤、换手、交易成本后收益。
  - 基线：线性模型、XGBoost、LSTM、仅OHLCV特征模型。
- 消融实验：
  - 双向注意力 vs 因果注意力（验证泄漏影响）。
  - 有/无时间嵌入，有/无指标特征。
  - 线性 value embedding vs MLP value embedding。

## 泄漏与稳健性检查
- 潜在泄漏路径：
  - 双向注意力读取未来K线。
  - centered rolling 指标（例如中心窗口平滑）包含未来点。
  - 全局标准化使用了测试期统计量。
- 规避措施：
  - 强制因果mask。
  - 所有指标改为 trailing window。
  - 标准化器只在训练窗口拟合，验证/测试仅transform。
- 稳健性验证：
  - 训练集打乱后性能应显著下降（排查伪相关）。
  - 加入交易成本与滑点后仍保持正向超额。

## 与BERT的关系（学习要点）
- 可借鉴点：
  - 预训练+微调范式。
  - 丰富embedding组合与多头注意力表达能力。
- 需改造点：
  - 预测阶段必须因果化，不可照搬双向注意力。
  - 时序任务优先保证信息集一致性，再追求精度。
