"""Vector Quantizer（VQ-VAE 离散码本，含 EMA + dead-code / dominant-code rebalance）。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VectorQuantizerOutput:
    z_q: torch.Tensor
    codes: torch.Tensor
    vq_loss: torch.Tensor
    perplexity: torch.Tensor


def _compute_distances(z: torch.Tensor, embed: torch.Tensor, *, cosine: bool = False) -> torch.Tensor:
    if cosine:
        z_n = F.normalize(z, dim=1)
        e_n = F.normalize(embed, dim=1)
        return 1.0 - z_n @ e_n.T
    return (
        z.pow(2).sum(dim=1, keepdim=True)
        - 2 * z @ embed.T
        + embed.pow(2).sum(dim=1)
    )


def _perplexity_from_codes(codes: torch.Tensor, num_codes: int) -> torch.Tensor:
    one_hot = F.one_hot(codes, num_codes).float()
    avg_probs = one_hot.mean(dim=0)
    return torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))


def code_usage_entropy_loss(codes: torch.Tensor, num_codes: int) -> torch.Tensor:
    """Batch 内码分布熵不足时的惩罚项，范围约 ``[0, 1]``（0=均匀，1=塌缩到单码）。"""
    if codes.numel() == 0:
        return codes.new_zeros(())
    one_hot = F.one_hot(codes.long().clamp(0, num_codes - 1), num_codes).float()
    usage = one_hot.mean(dim=0) + 1e-10
    entropy = -(usage * usage.log()).sum()
    max_ent = math.log(num_codes)
    return 1.0 - entropy / max_ent


def soft_code_usage_entropy_loss(
    z_e: torch.Tensor,
    embed: torch.Tensor,
    *,
    temperature: float = 2.0,
    cosine: bool = False,
) -> torch.Tensor:
    """可微码分布熵惩罚：基于 softmax(-dist/T) 的软分配，梯度可回传到 encoder。"""
    if z_e.numel() == 0:
        return z_e.new_zeros(())
    num_codes = embed.size(0)
    d = _compute_distances(z_e, embed, cosine=cosine)
    probs = F.softmax(-d / max(temperature, 1e-4), dim=1)
    usage = probs.mean(dim=0) + 1e-10
    entropy = -(usage * usage.log()).sum()
    max_ent = math.log(num_codes)
    return 1.0 - entropy / max_ent


def z_variance_spread_loss(z: torch.Tensor, *, target_std: float = 0.5) -> torch.Tensor:
    """鼓励 batch 内 segment 向量在各维上有足够方差，避免 encoder 输出塌缩。"""
    if z.size(0) < 2:
        return z.new_zeros(())
    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-6)
    return F.relu(target_std - std).mean()


@torch.no_grad()
def code_usage_stats(codes: torch.Tensor, num_codes: int) -> dict[str, float]:
    """统计码本使用：active 比例、归一化熵、最大码占比。"""
    if codes.numel() == 0:
        return {"active_ratio": 0.0, "norm_entropy": 0.0, "max_code_frac": 1.0}
    one_hot = F.one_hot(codes.long().clamp(0, num_codes - 1), num_codes).float()
    usage = one_hot.mean(dim=0)
    active = int((usage > 0).sum().item())
    ent = -(usage.clamp(min=1e-10) * usage.clamp(min=1e-10).log()).sum().item()
    norm_ent = ent / math.log(num_codes)
    return {
        "active_ratio": active / num_codes,
        "norm_entropy": float(norm_ent),
        "max_code_frac": float(usage.max().item()),
    }


class VectorQuantizer(nn.Module):
    """最近邻量化 + straight-through（梯度更新码本）。"""

    def __init__(
        self,
        num_codes: int,
        dim: int,
        *,
        beta: float = 0.25,
    ) -> None:
        super().__init__()
        if num_codes < 2:
            raise ValueError("num_codes must be >= 2")
        self.num_codes = num_codes
        self.dim = dim
        self.beta = beta
        self.codebook = nn.Embedding(num_codes, dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z_e: torch.Tensor) -> VectorQuantizerOutput:
        if z_e.dim() != 2:
            raise ValueError(f"z_e must be [B, D], got {tuple(z_e.shape)}")
        d = _compute_distances(z_e, self.codebook.weight)
        codes = d.argmin(dim=1)
        z_q = self.codebook(codes)

        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commit_loss = F.mse_loss(z_q.detach(), z_e)
        vq_loss = codebook_loss + self.beta * commit_loss
        z_st = z_e + (z_q - z_e).detach()
        perplexity = _perplexity_from_codes(codes, self.num_codes)

        return VectorQuantizerOutput(z_q=z_st, codes=codes, vq_loss=vq_loss, perplexity=perplexity)


class VectorQuantizerEMA(nn.Module):
    """EMA 码本更新 + commitment loss（推荐，缓解 code collapse）。"""

    def __init__(
        self,
        num_codes: int,
        dim: int,
        *,
        beta: float = 1.0,
        decay: float = 0.99,
        eps: float = 1e-5,
        training_stochastic_prob: float = 0.12,
        training_stochastic_topk: int = 3,
        gumbel_tau: float = 1.0,
        cosine_distance: bool = True,
    ) -> None:
        super().__init__()
        if num_codes < 2:
            raise ValueError("num_codes must be >= 2")
        self.num_codes = num_codes
        self.dim = dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.training_stochastic_prob = training_stochastic_prob
        self.training_stochastic_topk = training_stochastic_topk
        self.gumbel_tau = gumbel_tau
        self.cosine_distance = cosine_distance

        embed = torch.randn(num_codes, dim) * (1.0 / num_codes)
        self.register_buffer("_embed", embed)
        self.register_buffer("_ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("_ema_w", embed.clone())
        self.register_buffer("_epoch_usage", torch.zeros(num_codes))

    @property
    def weight(self) -> torch.Tensor:
        return self._embed

    def forward(self, z_e: torch.Tensor) -> VectorQuantizerOutput:
        if z_e.dim() != 2:
            raise ValueError(f"z_e must be [B, D], got {tuple(z_e.shape)}")

        d = _compute_distances(z_e, self._embed, cosine=self.cosine_distance)
        codes = d.argmin(dim=1)
        z_q = F.embedding(codes, self._embed)

        if self.training:
            one_hot = F.one_hot(codes, self.num_codes).type_as(z_e)
            cluster_size = one_hot.sum(dim=0)
            embed_sum = one_hot.T @ z_e.detach()

            self._ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
            self._ema_w.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            n = self._ema_cluster_size.sum()
            cluster_size_norm = (
                (self._ema_cluster_size + self.eps)
                / (n + self.num_codes * self.eps)
                * n
            )
            self._embed.copy_(self._ema_w / cluster_size_norm.unsqueeze(1))
            with torch.no_grad():
                self._epoch_usage.add_(F.one_hot(codes, self.num_codes).float().sum(dim=0))

        commit_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = self.beta * commit_loss
        z_st = z_e + (z_q - z_e).detach()
        perplexity = _perplexity_from_codes(codes, self.num_codes)

        return VectorQuantizerOutput(z_q=z_st, codes=codes, vq_loss=vq_loss, perplexity=perplexity)

    @torch.no_grad()
    def _reset_code_indices(
        self,
        mask: torch.Tensor,
        *,
        z_samples: torch.Tensor | None = None,
    ) -> int:
        n = int(mask.sum().item())
        if n == 0:
            return 0
        if z_samples is not None and z_samples.numel() > 0:
            pick = torch.randint(0, z_samples.size(0), (n,), device=z_samples.device)
            new = z_samples[pick].clone()
            new += torch.randn_like(new) * 0.02
        else:
            new = torch.randn(n, self.dim, device=self._embed.device) * 0.02
        self._embed[mask] = new
        self._ema_w[mask] = new.clone()
        target = (self._ema_cluster_size.sum() / self.num_codes).clamp(min=0.5)
        self._ema_cluster_size[mask] = target
        return n

    @torch.no_grad()
    def reset_dead_codes(
        self,
        threshold: float = 0.5,
        *,
        z_samples: torch.Tensor | None = None,
    ) -> int:
        """将长期未使用的码向量重置；若提供 ``z_samples`` 则从编码向量中采样。"""
        dead = self._ema_cluster_size < threshold
        return self._reset_code_indices(dead, z_samples=z_samples)

    @torch.no_grad()
    def max_ema_usage_frac(self) -> float:
        total = self._ema_cluster_size.sum().clamp(min=1e-8)
        return float((self._ema_cluster_size / total).max().item())

    @torch.no_grad()
    def max_epoch_usage_frac(self) -> float:
        total = self._epoch_usage.sum().clamp(min=1e-8)
        if float(total) <= 0:
            return 0.0
        return float((self._epoch_usage / total).max().item())

    @torch.no_grad()
    def reset_epoch_usage(self) -> None:
        self._epoch_usage.zero_()

    @torch.no_grad()
    def rebalance_codes(
        self,
        *,
        dead_threshold: float = 0.1,
        max_usage_frac: float = 0.22,
        z_samples: torch.Tensor | None = None,
    ) -> tuple[int, int]:
        """重置死码 + 占用过高的主导码，缓解 code 0 一家独大。"""
        if float(self._epoch_usage.sum()) > 0:
            usage = self._epoch_usage
            total = usage.sum().clamp(min=1e-8)
            frac = usage / total
            dead = usage <= 0
        else:
            usage = self._ema_cluster_size
            total = usage.sum().clamp(min=1e-8)
            frac = usage / total
            dead = usage < dead_threshold
        n_dead = self._reset_code_indices(dead, z_samples=z_samples)

        if float(self._epoch_usage.sum()) > 0:
            total = self._epoch_usage.sum().clamp(min=1e-8)
            frac = self._epoch_usage / total
        else:
            total = self._ema_cluster_size.sum().clamp(min=1e-8)
            frac = self._ema_cluster_size / total
        dominant = frac > max_usage_frac
        if int(dominant.sum()) > max(1, self.num_codes // 3):
            worst = frac.topk(max(1, self.num_codes // 3)).indices
            dom_mask = torch.zeros_like(dominant)
            dom_mask[worst] = True
            dominant = dom_mask
        n_dom = self._reset_code_indices(dominant, z_samples=z_samples)
        self._epoch_usage.zero_()
        return n_dead, n_dom

    @torch.no_grad()
    def init_from_encoder_outputs(self, z: torch.Tensor, *, n_iter: int = 8) -> None:
        """k-means++ 初始化 + Lloyd 迭代，使码本覆盖 encoder 输出分布。"""
        if z.dim() != 2 or z.size(0) < self.num_codes:
            raise ValueError(f"need z [N>={self.num_codes}, D], got {tuple(z.shape)}")
        n = z.size(0)
        k = self.num_codes
        centers: list[torch.Tensor] = []
        idx0 = torch.randint(0, n, (1,), device=z.device)
        centers.append(z[idx0])
        for _ in range(1, k):
            dist_sq = torch.stack([((z - c) ** 2).sum(dim=1) for c in centers], dim=0).min(dim=0).values
            if float(dist_sq.sum()) <= 1e-12:
                idx = torch.randint(0, n, (1,), device=z.device)
            else:
                probs = dist_sq / dist_sq.sum().clamp(min=1e-12)
                idx = torch.multinomial(probs, 1)
            centers.append(z[idx])
        embed = torch.cat(centers, dim=0)
        for _ in range(max(0, n_iter)):
            d = _compute_distances(z, embed, cosine=self.cosine_distance)
            assign = d.argmin(dim=1)
            for ci in range(k):
                mask = assign == ci
                if bool(mask.any()):
                    embed[ci] = z[mask].mean(dim=0)
                else:
                    embed[ci] = z[torch.randint(0, n, (1,), device=z.device)].squeeze(0)
        embed = embed + torch.randn_like(embed) * 0.01
        self._embed.copy_(embed)
        self._ema_w.copy_(embed)
        self._ema_cluster_size.fill_(float(n) / k)
