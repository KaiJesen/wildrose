from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from trend_leg.labels import LEG_TYPES, SUB_PHASES


@dataclass(frozen=True)
class TrendLegModelConfig:
    input_dim: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    num_leg_types: int = len(LEG_TYPES)
    num_sub_phases: int = len(SUB_PHASES)


class TrendLegClassifier(nn.Module):
    def __init__(self, cfg: TrendLegModelConfig) -> None:
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
        self.leg_type_head = nn.Linear(cfg.d_model, cfg.num_leg_types)
        self.sub_phase_head = nn.Linear(cfg.d_model, cfg.num_sub_phases)
        self.leg_progress_head = nn.Sequential(nn.Linear(cfg.d_model, 1), nn.Sigmoid())
        self.leg_confirmed_head = nn.Linear(cfg.d_model, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.input_proj(x)
        h = self.encoder(h)
        pooled = self.pool(h[:, -1, :])
        return {
            "leg_type_logits": self.leg_type_head(pooled),
            "sub_phase_logits": self.sub_phase_head(pooled),
            "leg_progress_pred": self.leg_progress_head(pooled),
            "leg_confirmed_logit": self.leg_confirmed_head(pooled),
        }
