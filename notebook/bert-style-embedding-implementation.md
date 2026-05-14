# BERT 风格 K 线 Embedding 实现笔记

> 关联代码：`transformer_kit/embeddings.py`、`transformer_kit/features.py`、`examples/bert_embedding_demo.py`
> 关联笔记：`notebook/bert-to-time-series-causal-attention.md`（方案选型）

## 问题定义
- 当前要解决的Embedding问题是什么？
  - 把每根 K 线（连续特征 OHLCV/returns/calendar）映射成一个 `d_model` 维向量，作为下游因果 Transformer 的输入。
  - 必须满足：(1) 与 BERT 思想一致的可加性组合；(2) 严格因果，不引入未来信息；(3) 单资产与多资产场景同构可切换。

## 方案候选
- 方案A：BERT 风格可加 embedding（`E_value + E_pos + E_time + E_asset` → LayerNorm → Dropout）。
- 方案B：把每个特征独立 embedding 后 concat，再过线性压缩到 `d_model`。
- 方案C：仅 `E_value`（线性投影）+ sin/cos 位置编码，去掉日历/资产维度，做最朴素的 baseline。

## 选型结论
- 选择：方案A 作为默认实现，方案C 作为消融下界，方案B 保留为后续对比。
- 选择原因：
  - 与上层 BERT 风格的因果 Transformer 接口一致，便于复用预训练范式。
  - 可加性结构允许任意子项独立开关，消融实验成本低。
  - LayerNorm 把多路嵌入对齐到同一尺度，训练稳定。
- 代价与权衡：
  - 强制所有子项同维 `d_model`，比 concat 方案略浪费容量；用 `d_model=128` 起步即可。
  - 单资产时若不显式关闭 `E_asset` 会引入冗余参数，需要 `n_assets=0` 严格控制。

## Python实现
- 输入与特征工程（`transformer_kit/features.py`）：
  - 输入 DataFrame 来自 `market_data`，列名遵循 `market_data.schema`（`time/open/high/low/close/volume`）。
  - `add_log_returns`：补 `log_ret`（首行=0，避免 NaN 污染 z-score）。
  - `add_calendar_features`：补 `minute_of_day`（0..1439）、`dow`（0..6）。
  - `causal_zscore`：trailing 窗口 z-score，`min_periods=window//4` 防早期 NaN。
  - `build_feature_frame`：默认 `value_cols = (open, high, low, close, volume, log_ret)`，输出 `z_*` 前缀的标准化特征。
  - `make_sliding_windows`：把 `[N, F]` 切成 `[B, T, F]`，同时切日历数组保持对齐。
- Embedding实现要点（`transformer_kit/embeddings.py`）：
  - 公式：`x_t = E_value(f_t) + E_pos(t) + E_time(t) + E_asset(a)`，最后 `LayerNorm(eps=1e-12)` → `Dropout`。
  - `E_value`：`value_proj="linear"` 默认；`value_proj="mlp"` 用 `Linear → GELU → Linear` 2 层。
  - `E_pos`：`position_type="learned"` 默认（与 BERT 一致）；`"sincos"` 走 Transformer 原文的不可训练表。
  - `E_time`：`minute_of_day` 与 `dow` 各自 `nn.Embedding`，可独立关闭（`use_time_minute` / `use_time_dow`）。
  - `E_asset`：`n_assets=0` 即禁用；多资产场景传 `n_assets > 0`，`asset_ids` 支持 `[B]`（整个窗口同一资产）或 `[B, T]`。
  - 初始化：所有 `Linear`/`Embedding` 用 `N(0, 0.02)`，`LayerNorm` 权重 1 偏置 0，复刻 BERT 默认。
  - 严格校验：feat_dim/形状/长度/`asset_ids.dtype==long` 全部在 forward 入口断言。
- 关键代码片段（节选自仓库）：

```31:35:transformer_kit/embeddings.py
@dataclass(frozen=True)
class KlineBertEmbeddingConfig:
    """KlineBertEmbedding 的配置。"""

    feat_dim: int
```

```183:201:transformer_kit/embeddings.py
        x = self.value_emb(feats)

        if self.pos_emb is not None:
            pos_ids = torch.arange(t, device=feats.device).unsqueeze(0).expand(b, -1)
            x = x + self.pos_emb(pos_ids)
        else:
            x = x + self.pos_emb_buffer[:t].unsqueeze(0)

        if self.minute_emb is not None:
            if minute_ids is None:
                raise ValueError("minute_ids is required when use_time_minute=True")
            _check_id_shape(minute_ids, b, t, "minute_ids")
            x = x + self.minute_emb(minute_ids)

        if self.dow_emb is not None:
            if dow_ids is None:
                raise ValueError("dow_ids is required when use_time_dow=True")
            _check_id_shape(dow_ids, b, t, "dow_ids")
            x = x + self.dow_emb(dow_ids)
```

- 端到端调用示例（伪代码，详见 `examples/bert_embedding_demo.py`）：

```python
from market_data import get_kline_provider
from transformer_kit.features import build_feature_frame, make_sliding_windows
from transformer_kit.embeddings import KlineBertEmbedding, KlineBertEmbeddingConfig

df = get_kline_provider("akshare_em").fetch_kline("600519", "60m", start, end)
feat_df, feat_cols = build_feature_frame(df, zscore_window=60)
feats, minutes, dows = make_sliding_windows(
    feat_df[feat_cols].to_numpy("float32"),
    feat_df["minute_of_day"].to_numpy("int64"),
    feat_df["dow"].to_numpy("int64"),
    window=64,
)

cfg = KlineBertEmbeddingConfig(feat_dim=feats.shape[-1], d_model=128, max_len=64, n_assets=0)
emb = KlineBertEmbedding(cfg)
x = emb(torch.from_numpy(feats), minute_ids=torch.from_numpy(minutes), dow_ids=torch.from_numpy(dows))
# x.shape == [B, 64, 128]
```

## 回测设计
- 数据切分：
  - 按 `time` 升序切 train / valid / test（如 70/15/15），不打乱。
  - z-score 仅在 train 段拟合 rolling 统计的归一化器；valid/test 仅 transform（本实现的 `causal_zscore` 是逐样本 trailing，已天然满足）。
- 回测区间与频率：
  - 频率随 `--interval` 决定（5m / 60m / 1d）；首版用 1d/60m 验证管线。
  - 至少覆盖一次回撤 + 反弹，便于稳健性观测。
- 指标与基线：
  - 表征质量：embedding 输出在不同市场状态下的 cosine 相似度分布、PCA 可视化。
  - 下游任务（接因果 Transformer + 预测头）：IC、RankIC、方向准确率、MSE/MAE。
  - 基线：去掉 `E_time` 的 baseline；只用 `E_value` 的 baseline（方案C）。
- 消融实验：
  - `value_proj`: linear vs mlp。
  - `position_type`: learned vs sincos。
  - `use_time_minute` / `use_time_dow`：单独 off。
  - `zscore_window`: 30 / 60 / 120。
  - `n_assets`：单资产 vs 联合训练（需要扩到多标的取数）。

## 泄漏与稳健性检查
- 潜在泄漏路径：
  - rolling 指标若用 centered window 会读未来；本实现仅用 `Series.rolling(window=W, min_periods=...)`，pandas 默认右对齐。
  - 全局标准化（如 `(x-mean(all))/std(all)`）会用到测试期统计量；本实现禁用，全部走 trailing z-score。
  - 切窗时若把同一原始时间点切入相邻 batch 的 train/valid 窗口，可能间接泄漏；切分应按时间块而非样本随机抽样。
- 规避措施：
  - 上层注意力强制 causal mask（见 `notebook/bert-to-time-series-causal-attention.md`）。
  - 所有特征变换在 `transformer_kit/features.py` 内统一只走 trailing/shift。
  - 多资产合批时不混入未来上市的资产。
- 稳健性验证：
  - 把训练样本 shuffle 后训练，性能应明显下降；若没有下降，怀疑特征中藏了未来信息。
  - 输出 `x.mean()` / `x.std()` 在不同 batch 上保持稳定（已加入 demo 打印）。
  - 关闭 `LayerNorm` 时损失曲线应明显更不稳，作为 sanity 检查。

## 默认参数速查
- `d_model=128`，单资产 60m K 线、window=64 时 trainable params ≈ 200–300k，单 4090 上几乎零开销。
- `dropout=0.1`：与 BERT 默认一致。小数据下可调到 0.2–0.3。
- `init_std=0.02`、`layer_norm_eps=1e-12`：BERT 风格默认。
- `zscore_window=60`：60 根 K 线作为短期上下文；日内分钟级建议 60–120。

## 后续 TODO
- [ ] 接一个 2-block 因果 Transformer + 1-bar return 回归头跑通端到端。
- [ ] 多资产联合训练验证 `E_asset` 收益（先用 5–10 个相关标的）。
- [ ] 把 `build_feature_frame` 中的 indicator 集合扩展到 RSI/MACD/ATR/Bollinger（保持因果）。
- [ ] 评估 RoPE 替换绝对位置在长序列上的表现（>=512 步）。
