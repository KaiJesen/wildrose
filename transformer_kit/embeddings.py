"""BERT 风格的可加性 embedding，针对 K 线时序设计。

组合方式（与 HuggingFace ``BertEmbeddings`` 思想一致）:

    x_t = E_value(f_t) + E_pos(t) + E_time(t) + E_asset(a)
    x_t = Dropout(LayerNorm(x_t))

其中:
- ``E_value`` 来自连续特征 ``f_t``（OHLCV/returns/indicators），
  用 Linear 或 2 层 MLP 投影到 ``d_model``。
- ``E_pos`` 是绝对位置嵌入，支持 learned / sin-cos 两种。
- ``E_time`` 把日内分钟、星期等日历特征做可学习的离散嵌入。
- ``E_asset`` 在多资产联合训练时使用；单资产场景应禁用（n_assets=0）。

详细设计说明见 ``notebook/bert-style-embedding-implementation.md``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn

ValueProj = Literal["linear", "mlp"]
PositionType = Literal["learned", "sincos"]


@dataclass(frozen=True)
class KlineBertEmbeddingConfig:
    """KlineBertEmbedding 的配置。"""

    feat_dim: int
    """连续特征 f_t 的维度。"""

    d_model: int = 128
    """隐层维度，所有子 embedding 都对齐到这个维度后相加。"""

    max_len: int = 512
    """最大支持的序列长度，position embedding 的词表大小。"""

    n_assets: int = 0
    """资产数量。0 表示禁用 E_asset（单资产场景）。"""

    n_minutes_of_day: int = 1440
    """日内分钟桶数量，常规为 24*60。"""

    n_dows: int = 7
    """星期数。"""

    dropout: float = 0.1
    """embedding 输出 dropout。"""

    value_proj: ValueProj = "linear"
    """value 投影方式: 'linear' 或 2 层 'mlp'。"""

    position_type: PositionType = "learned"
    """位置编码: 'learned'（默认）或 'sincos'（不可训练）。"""

    use_time_minute: bool = True
    """是否启用 minute-of-day embedding。"""

    use_time_dow: bool = True
    """是否启用 day-of-week embedding。"""

    layer_norm_eps: float = 1e-12
    """LayerNorm 的 eps，与 BERT 默认值一致。"""

    init_std: float = 0.02
    """Embedding 权重初始化标准差，与 BERT 默认值一致。"""

    def __post_init__(self) -> None:
        if self.feat_dim <= 0:
            raise ValueError("feat_dim must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.position_type == "sincos" and self.d_model % 2 != 0:
            raise ValueError("d_model must be even for sincos position encoding")
        if self.value_proj not in ("linear", "mlp"):
            raise ValueError(f"value_proj must be 'linear' or 'mlp', got {self.value_proj!r}")
        if self.position_type not in ("learned", "sincos"):
            raise ValueError(
                f"position_type must be 'learned' or 'sincos', got {self.position_type!r}"
            )


def _build_sincos_pe(max_len: int, d_model: int) -> torch.Tensor:
    """构造 Transformer 原文的 sin/cos 位置编码表 [max_len, d_model]。"""
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class KlineBertEmbedding(nn.Module):
    """BERT 风格的 K 线时序 embedding 层。

    Forward 输入 / 输出形状:
        feats         : ``[B, T, feat_dim]``  连续特征张量
        minute_ids    : ``[B, T]``（int64）   日内分钟桶 id，0..n_minutes_of_day-1
        dow_ids       : ``[B, T]``（int64）   星期 id，0..n_dows-1
        asset_ids     : ``[B]`` 或 ``[B, T]``  资产 id（仅在 n_assets>0 时需要）
        return        : ``[B, T, d_model]``   embedding 输出
    """

    def __init__(self, cfg: KlineBertEmbeddingConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        if cfg.value_proj == "linear":
            self.value_emb: nn.Module = nn.Linear(cfg.feat_dim, d)
        else:
            self.value_emb = nn.Sequential(
                nn.Linear(cfg.feat_dim, d),
                nn.GELU(),
                nn.Linear(d, d),
            )

        if cfg.position_type == "learned":
            self.pos_emb: nn.Embedding | None = nn.Embedding(cfg.max_len, d)
            self.register_buffer("pos_emb_buffer", torch.empty(0), persistent=False)
        else:
            self.pos_emb = None
            self.register_buffer(
                "pos_emb_buffer",
                _build_sincos_pe(cfg.max_len, d),
                persistent=False,
            )

        self.asset_emb: nn.Embedding | None = (
            nn.Embedding(cfg.n_assets, d) if cfg.n_assets > 0 else None
        )
        self.minute_emb: nn.Embedding | None = (
            nn.Embedding(cfg.n_minutes_of_day, d) if cfg.use_time_minute else None
        )
        self.dow_emb: nn.Embedding | None = (
            nn.Embedding(cfg.n_dows, d) if cfg.use_time_dow else None
        )

        self.layer_norm = nn.LayerNorm(d, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        std = self.cfg.init_std
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        feats: torch.Tensor,
        *,
        minute_ids: torch.Tensor | None = None,
        dow_ids: torch.Tensor | None = None,
        asset_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if feats.dim() != 3:
            raise ValueError(f"feats must be [B, T, feat_dim], got shape {tuple(feats.shape)}")
        b, t, f = feats.shape
        if f != self.cfg.feat_dim:
            raise ValueError(
                f"feat_dim mismatch: cfg.feat_dim={self.cfg.feat_dim}, got {f}"
            )
        if t > self.cfg.max_len:
            raise ValueError(f"seq length {t} exceeds max_len {self.cfg.max_len}")

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

        if self.asset_emb is not None:
            if asset_ids is None:
                raise ValueError("asset_ids is required when n_assets > 0")
            if asset_ids.dtype != torch.long:
                raise TypeError(
                    f"asset_ids must be torch.long for nn.Embedding, got {asset_ids.dtype}"
                )
            if asset_ids.dim() == 1:
                if asset_ids.shape[0] != b:
                    raise ValueError(
                        f"asset_ids shape {tuple(asset_ids.shape)} incompatible with B={b}"
                    )
                a = self.asset_emb(asset_ids).unsqueeze(1).expand(-1, t, -1)
            elif asset_ids.dim() == 2:
                if asset_ids.shape != (b, t):
                    raise ValueError(
                        f"asset_ids shape {tuple(asset_ids.shape)} does not match feats ({b}, {t})"
                    )
                a = self.asset_emb(asset_ids)
            else:
                raise ValueError(f"asset_ids must be 1D or 2D, got {asset_ids.dim()}D")
            x = x + a

        x = self.layer_norm(x)
        x = self.dropout(x)
        return x

    @torch.no_grad()
    def num_parameters(self, *, trainable_only: bool = True) -> int:
        """统计参数量，便于消融对比。"""
        return sum(
            p.numel()
            for p in self.parameters()
            if (p.requires_grad or not trainable_only)
        )

    def extra_repr(self) -> str:
        return (
            f"feat_dim={self.cfg.feat_dim}, d_model={self.cfg.d_model}, "
            f"max_len={self.cfg.max_len}, n_assets={self.cfg.n_assets}, "
            f"value_proj={self.cfg.value_proj}, position_type={self.cfg.position_type}, "
            f"use_minute={self.cfg.use_time_minute}, use_dow={self.cfg.use_time_dow}"
        )


def _check_id_shape(ids: torch.Tensor, b: int, t: int, name: str) -> None:
    if ids.shape != (b, t):
        raise ValueError(
            f"{name} shape {tuple(ids.shape)} does not match feats batch/time ({b}, {t})"
        )
    if ids.dtype != torch.long:
        raise TypeError(f"{name} must be torch.long for nn.Embedding, got {ids.dtype}")
