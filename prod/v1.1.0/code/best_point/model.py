from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class BestPointModelConfig:
    input_dim: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1


class BestPointSignalModel(nn.Module):
    def __init__(self, cfg: BestPointModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.pool = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.Tanh())
        self.entry_head = nn.Linear(cfg.d_model, 3)
        self.hold_head = nn.Linear(cfg.d_model, 3)
        self.exit_head = nn.Linear(cfg.d_model, 3)
        self.opp_head = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model // 2), nn.GELU(), nn.Linear(cfg.d_model // 2, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.input_proj(x)
        h = self.encoder(h)
        pooled = self.pool(h[:, -1, :])
        return {
            "entry_logits": self.entry_head(pooled),
            "hold_logits": self.hold_head(pooled),
            "exit_logits": self.exit_head(pooled),
            "opportunity_pred": self.opp_head(pooled),
        }

