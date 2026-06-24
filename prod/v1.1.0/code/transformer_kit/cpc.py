"""CPC encoder and lightweight direction decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_kit.causal_transformer import CausalTransformer, CausalTransformerConfig


@dataclass(frozen=True)
class CPCEncoderConfig:
    feat_dim: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    max_ctx_len: int = 256


class CPCEncoder(nn.Module):
    """Causal encoder for sequence representation learning."""

    def __init__(self, cfg: CPCEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Sequential(
            nn.LayerNorm(cfg.feat_dim),
            nn.Linear(cfg.feat_dim, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.pos_emb = nn.Embedding(cfg.max_ctx_len, cfg.d_model)
        self.backbone = CausalTransformer(
            CausalTransformerConfig(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                n_layers=cfg.n_layers,
                dim_feedforward=cfg.d_model * 4,
                dropout=cfg.dropout,
            )
        )
        self.out_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,T,F] -> z: [B,T,D]."""
        if x.dim() != 3:
            raise ValueError(f"expected [B,T,F], got {tuple(x.shape)}")
        b, t, _ = x.shape
        if t > self.cfg.max_ctx_len:
            raise ValueError(f"sequence too long {t} > {self.cfg.max_ctx_len}")
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, -1)
        h = self.in_proj(x) + self.pos_emb(pos)
        z = self.backbone(h)
        return self.out_norm(z)


class CPCLossHead(nn.Module):
    """InfoNCE head for k-step future latent prediction."""

    def __init__(self, d_model: int, pred_steps: int = 5, temperature: float = 0.1) -> None:
        super().__init__()
        self.pred_steps = pred_steps
        self.temperature = temperature
        self.predictors = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(pred_steps)])

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute CPC loss from latent sequence z [B,T,D]."""
        b, t, d = z.shape
        if t <= self.pred_steps + 1:
            raise ValueError("sequence too short for CPC")
        z = F.normalize(z, dim=-1)
        losses: list[torch.Tensor] = []
        accs: list[float] = []
        for k in range(1, self.pred_steps + 1):
            ctx = z[:, : t - k, :]
            tgt = z[:, k:, :].detach()
            q = F.normalize(self.predictors[k - 1](ctx), dim=-1)
            n = q.shape[0] * q.shape[1]
            qf = q.reshape(n, d)
            tf = tgt.reshape(n, d)
            logits = (qf @ tf.T) / self.temperature
            labels = torch.arange(n, device=z.device)
            loss_k = F.cross_entropy(logits, labels)
            losses.append(loss_k)
            with torch.no_grad():
                accs.append(float((logits.argmax(dim=1) == labels).float().mean().item()))
        loss = torch.stack(losses).mean()
        return loss, {"cpc_top1": float(sum(accs) / max(1, len(accs)))}


class DirectionDecoder(nn.Module):
    """Lightweight transformer decoder for H-step direction logits."""

    def __init__(
        self,
        encoder: CPCEncoder,
        *,
        pred_horizon: int = 5,
        decoder_layers: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        d = encoder.cfg.d_model
        self.decoder = CausalTransformer(
            CausalTransformerConfig(
                d_model=d,
                n_heads=encoder.cfg.n_heads,
                n_layers=max(1, decoder_layers),
                dim_feedforward=d * 2,
                dropout=encoder.cfg.dropout,
            )
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, pred_horizon),
        )
        self.pred_horizon = pred_horizon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        h = self.decoder(z)
        last = h[:, -1, :]
        return self.head(last)

