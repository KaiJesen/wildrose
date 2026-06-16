"""形态 token 序列 + 因果 Transformer → 预测未来 log_ret。

Embedding 使用 ``AutoSegmentVQEncoder``：第一层因果 MHA 自动切分，再 VQ 编码。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_kit.auto_segment_encoder import AutoSegmentConfig, AutoSegmentVQEncoder
from transformer_kit.causal_transformer import CausalTransformer, CausalTransformerConfig


@dataclass(frozen=True)
class PatternPredictorConfig:
    auto_segment: AutoSegmentConfig
    trunk: CausalTransformerConfig
    pred_horizon: int = 5
    pred_feat_dim: int = 1
    use_pos_emb: bool = True
    pool_mode: str = "attn"  # attn | mean | last
    learnable_scale: bool = False
    use_raw_context: bool = True


class SegmentAttentionPool(nn.Module):
    """可学习 query 对形态 token 做 cross-attention 汇聚（保留末段/关键段信息）。"""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, seg_mask: torch.Tensor) -> torch.Tensor:
        """h ``[B,S,D]``，seg_mask ``[B,S]`` True=有效段。"""
        b = h.size(0)
        q = self.query.expand(b, -1, -1)
        key_pad = ~seg_mask
        out, _ = self.attn(q, h, h, key_padding_mask=key_pad)
        return self.norm(out.squeeze(1))


class KlinePatternPredictor(nn.Module):
    """Stage 3：自动切分 Embedding + 因果 Transformer + 未来 log_ret 预测。"""

    def __init__(self, cfg: PatternPredictorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.auto_segment.d_model
        self.auto_encoder = AutoSegmentVQEncoder(cfg.auto_segment)
        self.trunk = CausalTransformer(cfg.trunk)
        self.pos_emb = nn.Embedding(cfg.auto_segment.max_segments, d) if cfg.use_pos_emb else None
        if cfg.pool_mode == "attn":
            self.segment_pool = SegmentAttentionPool(d, cfg.trunk.n_heads)
        else:
            self.segment_pool = None
        self.raw_context_dim = 17
        self.raw_context_proj = (
            nn.Sequential(
                nn.LayerNorm(self.raw_context_dim),
                nn.Linear(self.raw_context_dim, d),
                nn.GELU(),
                nn.Linear(d, d),
            )
            if cfg.use_raw_context
            else None
        )
        self.future_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, cfg.pred_horizon * cfg.pred_feat_dim),
        )
        self.direction_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1),
        )
        self.out_scale = nn.Parameter(torch.ones(1)) if cfg.learnable_scale else None

    def forward(
        self,
        ctx_bars: torch.Tensor,
        ctx_lengths: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """ctx_bars ``[B,T,F]``，ctx_lengths ``[B]`` 有效 bar 数。"""
        auto_out = self.auto_encoder(ctx_bars, ctx_lengths)
        tokens = auto_out.tokens
        b, s, d = tokens.shape

        seg_mask = torch.arange(s, device=tokens.device).unsqueeze(0) < auto_out.num_segments.unsqueeze(1)
        tokens = tokens * seg_mask.unsqueeze(-1).float()

        if self.pos_emb is not None:
            pos = torch.arange(s, device=tokens.device).unsqueeze(0).expand(b, -1)
            tokens = tokens + self.pos_emb(pos)

        h = self.trunk(tokens)
        if self.cfg.pool_mode == "attn" and self.segment_pool is not None:
            pooled = self.segment_pool(h, seg_mask)
        elif self.cfg.pool_mode == "last":
            last_idx = (auto_out.num_segments - 1).clamp(min=0)
            pooled = h[torch.arange(b, device=h.device), last_idx]
        else:
            seg_mask_f = seg_mask.unsqueeze(-1).float()
            pooled = (h * seg_mask_f).sum(dim=1) / auto_out.num_segments.clamp(min=1).unsqueeze(-1).float()
        if self.raw_context_proj is not None:
            pooled = pooled + self.raw_context_proj(raw_context_features(ctx_bars, ctx_lengths))
        n = self.cfg.pred_horizon
        f = self.cfg.pred_feat_dim
        future_pred = self.future_head(pooled).view(-1, n, f)
        if self.out_scale is not None:
            future_pred = future_pred * self.out_scale

        if not return_aux:
            return future_pred

        aux = {
            "vq_loss": auto_out.vq_loss,
            "perplexity": auto_out.perplexity,
            "break_reg_loss": auto_out.break_reg_loss,
            "num_segments": auto_out.num_segments.float().mean(),
            "cum_direction_logit": self.direction_head(pooled).squeeze(-1),
        }
        return future_pred, aux

    def encode_pattern_codes(
        self,
        ctx_bars: torch.Tensor,
        ctx_lengths: torch.Tensor,
    ) -> torch.Tensor:
        out = self.auto_encoder(ctx_bars, ctx_lengths)
        return out.codes


def _trailing_sum(x: torch.Tensor, lengths: torch.Tensor, window: int) -> torch.Tensor:
    csum = F.pad(x.cumsum(dim=1), (1, 0))
    end = lengths.clamp(max=x.size(1))
    start = (end - window).clamp(min=0)
    rows = torch.arange(x.size(0), device=x.device)
    return csum[rows, end] - csum[rows, start]


def raw_context_features(ctx_bars: torch.Tensor, ctx_lengths: torch.Tensor) -> torch.Tensor:
    """最近收益/波动率/末根形态的轻量旁路特征。"""
    log_ret = ctx_bars[..., 0]
    lengths = ctx_lengths.clamp(min=1, max=ctx_bars.size(1))
    feats: list[torch.Tensor] = []
    for w in (1, 3, 6, 12, 24, 48):
        feats.append(_trailing_sum(log_ret, lengths, w))
    for w in (6, 12, 24):
        denom = lengths.clamp(max=w).float()
        feats.append(_trailing_sum(log_ret.abs(), lengths, w) / denom)
    sq = log_ret.pow(2)
    for w in (6, 12, 24):
        denom = lengths.clamp(max=w).float()
        mean = _trailing_sum(log_ret, lengths, w) / denom
        mean_sq = _trailing_sum(sq, lengths, w) / denom
        feats.append((mean_sq - mean.pow(2)).clamp(min=0).sqrt())
    rows = torch.arange(ctx_bars.size(0), device=ctx_bars.device)
    last = ctx_bars[rows, lengths - 1, :]
    feats.extend(last.unbind(dim=1))
    return torch.stack(feats, dim=1)


def future_prediction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    log_ret_only: bool = True,
) -> torch.Tensor:
    if log_ret_only or pred.shape[-1] == 1:
        pred_r = pred.squeeze(-1) if pred.shape[-1] == 1 else pred[..., 0]
        return F.mse_loss(pred_r, target[..., 0])
    return F.mse_loss(pred, target)


def batch_pearson_corr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Batch 内 flatten 后的 Pearson 相关系数（可微）。"""
    p = pred.reshape(-1)
    t = target.reshape(-1)
    p = p - p.mean()
    t = t - t.mean()
    num = (p * t).sum()
    den = torch.sqrt((p * p).sum() * (t * t).sum()).clamp(min=1e-8)
    return num / den


def cumulative_return_corr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """各样本 H 步 log_ret 累加后的 batch Pearson 相关（信噪比高于逐步 IC）。"""
    pred_r = pred.squeeze(-1) if pred.dim() == 3 else pred
    tgt = target[..., 0] if target.dim() >= 2 else target
    return batch_pearson_corr(pred_r.sum(dim=1), tgt.sum(dim=1))


def batch_pairwise_rank_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Batch 内累计收益成对排序损失（提升截面 IC）。"""
    pred_r = pred.squeeze(-1) if pred.dim() == 3 else pred
    tgt = target[..., 0] if target.dim() >= 2 else target
    pc, tc = pred_r.sum(dim=1), tgt.sum(dim=1)
    n = pc.size(0)
    if n < 3:
        return pc.new_zeros(())
    dp = pc.unsqueeze(1) - pc.unsqueeze(0)
    dt = tc.unsqueeze(1) - tc.unsqueeze(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=pc.device)
    return F.softplus(-dp * dt)[mask].mean()


def cumulative_direction_loss(direction_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """未来 H 步累计收益的涨跌二分类损失。"""
    tgt = target[..., 0] if target.dim() >= 2 else target
    direction = (tgt.sum(dim=1) > 0).float()
    return F.binary_cross_entropy_with_logits(direction_logit, direction)


def stage3_prediction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    direction_logit: torch.Tensor | None = None,
    mse_weight: float = 0.7,
    step_corr_weight: float = 0.25,
    cum_corr_weight: float = 0.35,
    sign_weight: float = 0.15,
    rank_weight: float = 0.0,
    direction_weight: float = 0.0,
    corr_weight: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Stage 3 联合损失：回归排序 + 累计方向分类。"""
    if corr_weight is not None and corr_weight > 0:
        step_corr_weight = corr_weight
    pred_r = pred.squeeze(-1) if pred.dim() == 3 else pred
    tgt = target[..., 0] if target.dim() >= 2 else target
    mse = F.mse_loss(pred_r, tgt)
    step_corr = batch_pearson_corr(pred_r, tgt)
    cum_corr = cumulative_return_corr(pred, target)
    sign_loss = F.softplus(-pred_r * tgt).mean()
    total = (
        mse_weight * mse
        + step_corr_weight * (1.0 - step_corr)
        + cum_corr_weight * (1.0 - cum_corr)
        + sign_weight * sign_loss
    )
    if rank_weight > 0:
        rank_loss = batch_pairwise_rank_loss(pred, target)
        total = total + rank_weight * rank_loss
    else:
        rank_loss = pred_r.new_zeros(())
    if direction_weight > 0 and direction_logit is not None:
        dir_loss = cumulative_direction_loss(direction_logit, target)
        total = total + direction_weight * dir_loss
    else:
        dir_loss = pred_r.new_zeros(())
    return total, {
        "mse": float(mse.detach()),
        "step_corr": float(step_corr.detach()),
        "cum_corr": float(cum_corr.detach()),
        "sign_loss": float(sign_loss.detach()),
        "rank_loss": float(rank_loss.detach()),
        "direction_loss": float(dir_loss.detach()),
    }
