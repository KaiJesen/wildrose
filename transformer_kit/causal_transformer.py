"""因果自注意力编码器（用于 embedding 序列）。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def causal_attention_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """上三角为 True（被 mask，不可 attend 未来）。"""
    return torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )


@dataclass(frozen=True)
class CausalTransformerConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    activation: str = "gelu"


class CausalTransformer(nn.Module):
    """标准 ``TransformerEncoder`` + 因果 mask。"""

    def __init__(self, cfg: CausalTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation=cfg.activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, d_model] → 同形状。"""
        if x.dim() != 3:
            raise ValueError(f"expected [B,T,D], got {tuple(x.shape)}")
        t = x.size(1)
        mask = causal_attention_mask(t, x.device)
        return self.encoder(x, mask=mask, is_causal=False)
