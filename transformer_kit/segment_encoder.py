"""变长 K 柱片段编码：MHA + 汇总 token（结束符机制）。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SegmentEncoderConfig:
    feat_dim: int = 5
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_len: int = 32
    activation: str = "gelu"


def _lengths_to_pad_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """``[B, max_len]``，True 表示 padding 位置（需 mask）。"""
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return idx >= lengths.unsqueeze(1)


class SegmentMHAEncoder(nn.Module):
    """变长 K 柱 → 连续向量。

    在序列前插入可学习 **汇总 token**（类似 BERT [CLS] / 结束符聚合），
    经 ``TransformerEncoder`` 自注意力后取该 token 输出作为片段表征。
    """

    def __init__(self, cfg: SegmentEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.input_proj = nn.Linear(cfg.feat_dim, d)
        self.summary_token = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation=cfg.activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.out_norm = nn.LayerNorm(d)
        nn.init.normal_(self.summary_token, std=0.02)

    def forward(
        self,
        bars: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """bars: ``[B, L, F]``，lengths: ``[B]``（有效 bar 数，≤ L）。

        返回 ``[B, d_model]``。
        """
        if bars.dim() != 3:
            raise ValueError(f"bars must be [B,L,F], got {tuple(bars.shape)}")
        b, max_len, f = bars.shape
        if f != self.cfg.feat_dim:
            raise ValueError(f"feat_dim mismatch: cfg={self.cfg.feat_dim}, got {f}")
        if lengths.shape != (b,):
            raise ValueError(f"lengths shape {tuple(lengths.shape)} != ({b},)")
        if (lengths > max_len).any() or (lengths < 1).any():
            raise ValueError("lengths must be in [1, L]")

        x = self.input_proj(bars)
        summary = self.summary_token.expand(b, -1, -1)
        x = torch.cat([summary, x], dim=1)

        bar_pad = _lengths_to_pad_mask(lengths, max_len)
        summary_pad = torch.zeros(b, 1, dtype=torch.bool, device=bars.device)
        pad_mask = torch.cat([summary_pad, bar_pad], dim=1)

        h = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.out_norm(h[:, 0, :])


class SegmentDecoder(nn.Module):
    """片段向量 → 重建归一化 bar 序列 ``[B, L_max, F]``（位置 query + cross-attention）。"""

    def __init__(self, cfg: SegmentEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.pos_queries = nn.Parameter(torch.randn(cfg.max_len, d) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            d, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, cfg.feat_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2:
            raise ValueError(f"z must be [B, d_model], got {tuple(z.shape)}")
        b = z.size(0)
        memory = z.unsqueeze(1)
        q = self.pos_queries.unsqueeze(0).expand(b, -1, -1)
        h, _ = self.cross_attn(q, memory, memory)
        h = self.norm(h + q)
        return self.out_proj(h)


def segment_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """仅在有效长度内计算 MSE。"""
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    b, max_len, f = pred.shape
    mask = _lengths_to_pad_mask(lengths, max_len).unsqueeze(-1)
    diff = (pred - target).pow(2)
    diff = diff.masked_fill(mask, 0.0)
    denom = (lengths.float().sum() * f).clamp(min=1.0)
    return diff.sum() / denom
