"""形态 token 序列 + 因果 Transformer → 预测未来 log_ret。

Embedding 使用 ``AutoSegmentVQEncoder``：第一层因果 MHA 自动切分，再 VQ 编码。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_kit.auto_segment_encoder import AutoSegmentConfig, AutoSegmentVQEncoder
from transformer_kit.leg_context import LegContextFusion
from transformer_kit.causal_transformer import CausalTransformer, CausalTransformerConfig
from transformer_kit.segment_features import BAR_SHAPE_DIM, LOG_RET_COL
from transformer_kit.trend_features import DEFAULT_TREND_WINDOWS, trend_col_index


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
    use_horizon_head: bool = False
    use_market_state_head: bool = False
    direction_classes: int = 3
    risk_classes: int = 2
    use_cum_heads: bool = False
    use_horizon_return_head: bool = False
    detach_risk_vol_heads: bool = False
    return_direction_hidden_mult: float = 1.0
    use_participation_heads: bool = False
    use_participation_attn: bool = False
    use_leg_context: bool = False
    use_coral_participation: bool = False
    participation_tiers: int = 3
    leg_align_horizons: tuple[int, ...] = ()


@dataclass
class MarketStateOutput:
    return_pred: torch.Tensor
    direction_logits: torch.Tensor
    volatility_pred: torch.Tensor
    risk_logits: torch.Tensor
    aux: dict[str, torch.Tensor]
    cum_return_pred: torch.Tensor | None = None
    cum_direction_logit: torch.Tensor | None = None
    participation_logit_long: torch.Tensor | None = None
    participation_logit_short: torch.Tensor | None = None
    hz_return_pred: dict[int, torch.Tensor] | None = None


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


class HorizonAwarePredictionHead(nn.Module):
    """每个未来 lead 用独立 query cross-attend token，减少多步预测被平滑成滞后曲线。"""

    def __init__(self, d_model: int, n_heads: int, horizon: int, feat_dim: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.feat_dim = feat_dim
        self.horizon_query = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, feat_dim),
        )

    def forward(self, pooled: torch.Tensor, tokens: torch.Tensor, seg_mask: torch.Tensor) -> torch.Tensor:
        b = pooled.size(0)
        q = self.horizon_query.expand(b, -1, -1) + pooled.unsqueeze(1)
        h, _ = self.cross_attn(q, tokens, tokens, key_padding_mask=~seg_mask)
        h = self.norm(h + q)
        return self.out(h)


class MarketStateHead(nn.Module):
    """Multi-task market-state prediction head."""

    def __init__(
        self,
        d_model: int,
        horizon: int,
        *,
        n_heads: int,
        direction_classes: int = 3,
        risk_classes: int = 2,
        use_cum_heads: bool = False,
        use_horizon_return_head: bool = False,
        detach_risk_vol_heads: bool = False,
        return_direction_hidden_mult: float = 1.0,
        use_participation_heads: bool = False,
        use_participation_attn: bool = False,
        use_leg_context: bool = False,
        use_coral_participation: bool = False,
        participation_tiers: int = 3,
        leg_align_horizons: tuple[int, ...] = (12, 24, 48),
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.direction_classes = direction_classes
        self.risk_classes = risk_classes
        self.use_cum_heads = use_cum_heads
        self.use_horizon_return_head = use_horizon_return_head
        self.detach_risk_vol_heads = detach_risk_vol_heads
        self.use_participation_heads = use_participation_heads
        self.use_participation_attn = use_participation_attn
        self.use_leg_context = use_leg_context
        self.use_coral_participation = use_coral_participation
        self.leg_align_horizons = tuple(leg_align_horizons)
        coral_thresholds = max(0, int(participation_tiers) - 1) if use_coral_participation else 0
        hidden = max(8, int(d_model * return_direction_hidden_mult))
        self.return_head = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, horizon),
        )
        self.horizon_return_head = (
            HorizonReturnHead(d_model=d_model, n_heads=n_heads, horizon=horizon)
            if use_horizon_return_head
            else None
        )
        self.direction_state_head = DirectionStateHead(
            d_model=d_model,
            horizon=horizon,
            direction_classes=direction_classes,
            hidden_mult=return_direction_hidden_mult,
        )
        self.volatility_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, horizon),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, horizon * risk_classes),
        )
        part_hidden = max(8, d_model // 2)
        use_mlp_part = use_participation_heads and not use_participation_attn
        self.participation_attn_long = (
            ParticipationAttn(d_model=d_model, n_heads=n_heads, coral_thresholds=coral_thresholds)
            if use_participation_heads and use_participation_attn
            else None
        )
        self.participation_attn_short = (
            ParticipationAttn(d_model=d_model, n_heads=n_heads, coral_thresholds=coral_thresholds)
            if use_participation_heads and use_participation_attn
            else None
        )
        self.leg_context_fusion = (
            LegContextFusion(d_model)
            if use_participation_heads and use_participation_attn and use_leg_context
            else None
        )
        self.participation_logit_long = (
            nn.Sequential(
                nn.Linear(d_model, part_hidden),
                nn.GELU(),
                nn.Linear(part_hidden, 1),
            )
            if use_mlp_part
            else None
        )
        self.participation_logit_short = (
            nn.Sequential(
                nn.Linear(d_model, part_hidden),
                nn.GELU(),
                nn.Linear(part_hidden, 1),
            )
            if use_mlp_part
            else None
        )
        self.hz_return_heads = (
            nn.ModuleDict(
                {
                    str(h): nn.Sequential(
                        nn.Linear(d_model, part_hidden),
                        nn.GELU(),
                        nn.Linear(part_hidden, 1),
                    )
                    for h in self.leg_align_horizons
                }
            )
            if self.leg_align_horizons
            else None
        )

    def forward(
        self,
        pooled: torch.Tensor,
        tokens: torch.Tensor | None = None,
        seg_mask: torch.Tensor | None = None,
        leg_context: dict[str, torch.Tensor] | None = None,
    ) -> MarketStateOutput:
        b = pooled.size(0)
        cum_return_pred: torch.Tensor | None = None
        if self.horizon_return_head is not None:
            if tokens is None or seg_mask is None:
                raise ValueError("tokens/seg_mask required when use_horizon_return_head=True")
            ret, cum_return_pred = self.horizon_return_head(pooled, tokens, seg_mask)
        else:
            ret = self.return_head(pooled)
            if self.use_cum_heads:
                cum_return_pred = ret.sum(dim=1)
        direction, cum_direction_logit = self.direction_state_head(pooled)
        if not self.use_cum_heads:
            cum_direction_logit = None
        risk_vol_pooled = pooled.detach() if self.detach_risk_vol_heads else pooled
        volatility = self.volatility_head(risk_vol_pooled)
        risk = self.risk_head(risk_vol_pooled).view(b, self.horizon, self.risk_classes)
        participation_logit_long = None
        participation_logit_short = None
        hz_return_pred: dict[int, torch.Tensor] | None = None
        leg_ctx_vec = None
        if self.leg_context_fusion is not None and leg_context is not None:
            leg_ctx_vec = self.leg_context_fusion(leg_context)
        if self.participation_attn_long is not None:
            if tokens is None or seg_mask is None:
                raise ValueError("tokens/seg_mask required when use_participation_attn=True")
            participation_logit_long = self.participation_attn_long(
                pooled, tokens, seg_mask, leg_ctx_vec=leg_ctx_vec
            )
            participation_logit_short = self.participation_attn_short(
                pooled, tokens, seg_mask, leg_ctx_vec=leg_ctx_vec
            )
        elif self.participation_logit_long is not None:
            participation_logit_long = self.participation_logit_long(pooled).squeeze(-1)
            participation_logit_short = self.participation_logit_short(pooled).squeeze(-1)
        if self.hz_return_heads is not None:
            hz_return_pred = {int(h): head(pooled).squeeze(-1) for h, head in self.hz_return_heads.items()}
        return MarketStateOutput(
            return_pred=ret,
            direction_logits=direction,
            volatility_pred=volatility,
            risk_logits=risk,
            aux={},
            cum_return_pred=cum_return_pred if self.use_cum_heads else None,
            cum_direction_logit=cum_direction_logit,
            participation_logit_long=participation_logit_long,
            participation_logit_short=participation_logit_short,
            hz_return_pred=hz_return_pred,
        )


class ParticipationAttn(nn.Module):
    """C3 cross-attention participation head (scalar BCE or CORAL thresholds per side).

    Mirrors ``HorizonReturnHead`` attention pattern: learned query attends over
  segment tokens. ``seg_mask`` True marks valid segments; padding uses
  ``key_padding_mask=~seg_mask`` (026 C1 clarification).
    """

    def __init__(self, d_model: int, n_heads: int, *, coral_thresholds: int = 0) -> None:
        super().__init__()
        self.coral_thresholds = int(coral_thresholds)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        hidden = max(8, d_model // 2)
        out_dim = max(1, self.coral_thresholds) if self.coral_thresholds > 0 else 1
        self.out = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(
        self,
        pooled: torch.Tensor,
        tokens: torch.Tensor,
        seg_mask: torch.Tensor,
        *,
        leg_ctx_vec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = pooled.size(0)
        q = self.query.expand(b, -1, -1) + pooled.unsqueeze(1)
        if leg_ctx_vec is not None:
            q = q + leg_ctx_vec.unsqueeze(1)
        h, _ = self.cross_attn(q, tokens, tokens, key_padding_mask=~seg_mask)
        h = self.norm(h + q)
        out = self.out(h)
        if out.size(-1) == 1:
            return out.reshape(b)
        return out.reshape(b, -1)


class HorizonReturnHead(nn.Module):
    """Horizon-aware return head with explicit cumulative return output."""

    def __init__(self, d_model: int, n_heads: int, horizon: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.query = nn.Parameter(torch.randn(1, horizon, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.step_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.cum_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        pooled: torch.Tensor,
        tokens: torch.Tensor,
        seg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = pooled.size(0)
        q = self.query.expand(b, -1, -1) + pooled.unsqueeze(1)
        h, _ = self.cross_attn(q, tokens, tokens, key_padding_mask=~seg_mask)
        h = self.norm(h + q)
        ret = self.step_out(h).squeeze(-1)
        cum_ctx = h.mean(dim=1) + pooled
        cum_ret = self.cum_out(cum_ctx).squeeze(-1)
        return ret, cum_ret


class DirectionStateHead(nn.Module):
    """Step-wise direction logits + cumulative direction logit."""

    def __init__(
        self,
        d_model: int,
        horizon: int,
        *,
        direction_classes: int = 3,
        hidden_mult: float = 1.0,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.direction_classes = direction_classes
        hidden = max(8, int(d_model * hidden_mult))
        self.step_head = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, horizon * direction_classes),
        )
        self.cum_head = nn.Sequential(
            nn.Linear(d_model, max(4, hidden // 2)),
            nn.GELU(),
            nn.Linear(max(4, hidden // 2), 1),
        )

    def forward(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = pooled.size(0)
        step = self.step_head(pooled).view(b, self.horizon, self.direction_classes)
        cum = self.cum_head(pooled).squeeze(-1)
        return step, cum


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
        self.raw_context_dim = raw_context_feature_dim(cfg.auto_segment.feat_dim)
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
        self.horizon_head = (
            HorizonAwarePredictionHead(d, cfg.trunk.n_heads, cfg.pred_horizon, cfg.pred_feat_dim)
            if cfg.use_horizon_head
            else None
        )
        self.market_state_head = (
            MarketStateHead(
                d,
                cfg.pred_horizon,
                n_heads=cfg.trunk.n_heads,
                direction_classes=cfg.direction_classes,
                risk_classes=cfg.risk_classes,
                use_cum_heads=cfg.use_cum_heads,
                use_horizon_return_head=cfg.use_horizon_return_head,
                detach_risk_vol_heads=cfg.detach_risk_vol_heads,
                return_direction_hidden_mult=cfg.return_direction_hidden_mult,
                use_participation_heads=cfg.use_participation_heads,
                use_participation_attn=cfg.use_participation_attn,
                use_leg_context=cfg.use_leg_context,
                use_coral_participation=cfg.use_coral_participation,
                participation_tiers=cfg.participation_tiers,
                leg_align_horizons=cfg.leg_align_horizons,
            )
            if cfg.use_market_state_head
            else None
        )
        self.direction_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1),
        )
        self.code_supervision_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1),
        )
        # Linear-chain CRF params for step-wise direction sequence decode.
        self.direction_seq_start = nn.Parameter(torch.zeros(2))
        self.direction_seq_end = nn.Parameter(torch.zeros(2))
        self.direction_seq_trans = nn.Parameter(torch.zeros(2, 2))
        self.out_scale = nn.Parameter(torch.ones(1)) if cfg.learnable_scale else None

    def forward(
        self,
        ctx_bars: torch.Tensor,
        ctx_lengths: torch.Tensor,
        *,
        leg_context: dict[str, torch.Tensor] | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | MarketStateOutput | tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
        last_idx = (auto_out.num_segments - 1).clamp(min=0)
        last_seg_token = h[torch.arange(b, device=h.device), last_idx]

        aux = {
            "vq_loss": auto_out.vq_loss,
            "perplexity": auto_out.perplexity,
            "break_reg_loss": auto_out.break_reg_loss,
            "num_segments": auto_out.num_segments.float().mean(),
            "cum_direction_logit": self.direction_head(pooled).squeeze(-1),
            "code_supervision_logit": self.code_supervision_head(last_seg_token).squeeze(-1),
            "break_logits": auto_out.break_logits,
            "segment_codes": auto_out.codes,
            "ctx_num_segments": auto_out.num_segments,
        }
        if self.cfg.use_market_state_head and self.market_state_head is not None:
            mso = self.market_state_head(pooled, h, seg_mask, leg_context=leg_context)
            if self.out_scale is not None:
                mso.return_pred = mso.return_pred * self.out_scale
                if mso.cum_return_pred is not None:
                    mso.cum_return_pred = mso.cum_return_pred * self.out_scale
            mso.aux = aux
            return mso

        n = self.cfg.pred_horizon
        f = self.cfg.pred_feat_dim
        if self.horizon_head is not None:
            future_pred = self.horizon_head(pooled, h, seg_mask)
        else:
            future_pred = self.future_head(pooled).view(-1, n, f)
        if self.out_scale is not None:
            future_pred = future_pred * self.out_scale

        if not return_aux:
            return future_pred
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


def _trailing_window(x: torch.Tensor, lengths: torch.Tensor, window: int) -> torch.Tensor:
    """取每个样本末尾 window 个点，按时间顺序返回 ``[B, window]``。"""
    lengths = lengths.clamp(min=1, max=x.size(1))
    start = (lengths - window).clamp(min=0)
    pos = start.unsqueeze(1) + torch.arange(window, device=x.device).unsqueeze(0)
    pos = pos.clamp(max=x.size(1) - 1)
    return x.gather(1, pos)


def raw_context_feature_dim(input_feat_dim: int = BAR_SHAPE_DIM) -> int:
    """旁路上下文特征维度（随是否启用趋势特征扩展）。"""
    base = 17
    if input_feat_dim > BAR_SHAPE_DIM:
        # 各窗口 trailing strength/r2 + 短期末值 + 多尺度方向一致性
        return base + 3 + 3 + 3 + 1
    return base


def raw_context_features(ctx_bars: torch.Tensor, ctx_lengths: torch.Tensor) -> torch.Tensor:
    """最近收益/波动率/末根形态，及可选多尺度趋势摘要。"""
    log_ret = ctx_bars[..., LOG_RET_COL]
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
    feats.extend(last[..., :BAR_SHAPE_DIM].unbind(dim=1))

    if ctx_bars.size(-1) > BAR_SHAPE_DIM:
        trend_windows = DEFAULT_TREND_WINDOWS
        for w in trend_windows:
            idx = trend_col_index(w, "strength", bar_shape_dim=BAR_SHAPE_DIM)
            denom = lengths.clamp(max=w).float()
            feats.append(_trailing_sum(ctx_bars[..., idx], lengths, w) / denom)
        for w in trend_windows:
            idx = trend_col_index(w, "r2", bar_shape_dim=BAR_SHAPE_DIM)
            denom = lengths.clamp(max=w).float()
            feats.append(_trailing_sum(ctx_bars[..., idx], lengths, w) / denom)
        for metric in ("slope", "resid_std", "strength"):
            idx = trend_col_index(trend_windows[0], metric, bar_shape_dim=BAR_SHAPE_DIM)
            feats.append(last[..., idx])
        slopes = [
            last[..., trend_col_index(w, "slope", bar_shape_dim=BAR_SHAPE_DIM)]
            for w in trend_windows
        ]
        signs = torch.stack(slopes, dim=1).sign()
        feats.append((signs == signs[:, :1]).float().mean(dim=1))

    return torch.stack(feats, dim=1)


def denorm_predicted_log_ret(
    pred: torch.Tensor,
    log_ret_mean: torch.Tensor,
    log_ret_std: torch.Tensor,
) -> torch.Tensor:
    """将模型输出的 z-score log_ret 还原到原始收益尺度（支持逐步 mean/std）。"""
    pred_r = _pred_log_ret(pred)
    if log_ret_mean.shape == pred_r.shape:
        return pred_r * log_ret_std + log_ret_mean
    if log_ret_mean.dim() == 1 and log_ret_mean.shape[-1] == pred_r.shape[-1]:
        return pred_r * log_ret_std.unsqueeze(0) + log_ret_mean.unsqueeze(0)
    return pred_r * log_ret_std.unsqueeze(-1) + log_ret_mean.unsqueeze(-1)


def _target_tensor_from_raw(raw: torch.Tensor) -> torch.Tensor:
    return raw.unsqueeze(-1) if raw.dim() == 2 else raw


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


def _pred_log_ret(pred: torch.Tensor) -> torch.Tensor:
    if pred.dim() == 3:
        return pred[..., 0]
    return pred


def batch_pearson_corr(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batch 内 flatten 后的 Pearson 相关系数（可微）。"""
    p = pred.reshape(-1)
    t = target.reshape(-1)
    if weight is None:
        p = p - p.mean()
        t = t - t.mean()
        num = (p * t).sum()
        den = torch.sqrt((p * p).sum() * (t * t).sum()).clamp(min=1e-8)
        return num / den
    w = weight.reshape(-1).type_as(p).clamp(min=0)
    w = w / w.sum().clamp(min=1e-8)
    p = p - (w * p).sum()
    t = t - (w * t).sum()
    num = (w * p * t).sum()
    den = torch.sqrt((w * p * p).sum() * (w * t * t).sum()).clamp(min=1e-8)
    return num / den


def cumulative_return_corr(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """各样本 H 步 log_ret 累加后的 batch Pearson 相关（信噪比高于逐步 IC）。"""
    pred_r = _pred_log_ret(pred)
    tgt = target[..., 0] if target.dim() >= 2 else target
    return batch_pearson_corr(pred_r.sum(dim=1), tgt.sum(dim=1), sample_weight)


def realized_volatility_score(target: torch.Tensor) -> torch.Tensor:
    """未来窗口内的已实现波动强度；训练时只用于样本重要性。"""
    tgt = target[..., 0] if target.dim() >= 2 else target
    return tgt.abs().mean(dim=1)


def volatility_focus_weights(
    target: torch.Tensor,
    *,
    focus_weight: float = 0.0,
    top_frac: float = 0.3,
) -> torch.Tensor | None:
    """高波动样本给更大权重，平缓样本保持权重 1。"""
    if focus_weight <= 0:
        return None
    score = realized_volatility_score(target)
    if score.numel() < 3:
        return torch.ones_like(score)
    k = max(1, int(score.numel() * top_frac))
    threshold = score.detach().topk(k, largest=True).values[-1]
    high = (score >= threshold).type_as(score)
    return 1.0 + focus_weight * high


def move_focus_weights(
    target: torch.Tensor,
    *,
    focus_weight: float = 0.0,
    scale: float = 3.0,
) -> torch.Tensor | None:
    """按未来累计 |log_ret| 连续加权，大行情样本权重更高。"""
    if focus_weight <= 0:
        return None
    tgt = target[..., 0] if target.dim() >= 2 else target
    score = tgt.abs().sum(dim=1)
    if score.numel() < 2:
        return torch.ones_like(score)
    norm = score / score.median().clamp(min=1e-6)
    return 1.0 + focus_weight * norm.clamp(max=scale)


def break_focus_weights(
    break_logits: torch.Tensor,
    ctx_lengths: torch.Tensor,
    *,
    focus_weight: float = 0.0,
    tail_window: int = 16,
) -> torch.Tensor | None:
    """按上下文末尾附近的最大切分概率加权（结构转折点）。"""
    if focus_weight <= 0:
        return None
    probs = torch.sigmoid(break_logits)
    b, t = probs.shape
    end = ctx_lengths.clamp(min=1, max=t).long()
    start = (end - tail_window).clamp(min=0)
    idx = torch.arange(t, device=probs.device).unsqueeze(0)
    valid = (idx >= start.unsqueeze(1)) & (idx < end.unsqueeze(1))
    max_prob = probs.masked_fill(~valid, 0.0).max(dim=1).values
    return 1.0 + focus_weight * max_prob


def combine_sample_weights(*parts: torch.Tensor | None) -> torch.Tensor | None:
    out: torch.Tensor | None = None
    for w in parts:
        if w is None:
            continue
        out = w if out is None else out * w
    return out


def build_stage3_sample_weights(
    target: torch.Tensor,
    *,
    ctx_lengths: torch.Tensor | None = None,
    break_logits: torch.Tensor | None = None,
    vol_focus_weight: float = 0.0,
    vol_focus_top_frac: float = 0.3,
    move_focus_weight: float = 0.0,
    move_focus_scale: float = 3.0,
    break_focus_weight: float = 0.0,
    break_focus_tail: int = 16,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """合并 Stage3 多样本重要性权重。"""
    result = combine_sample_weights(
        sample_weight,
        volatility_focus_weights(
            target, focus_weight=vol_focus_weight, top_frac=vol_focus_top_frac,
        ),
        move_focus_weights(
            target, focus_weight=move_focus_weight, scale=move_focus_scale,
        ),
        break_focus_weights(
            break_logits, ctx_lengths, focus_weight=break_focus_weight, tail_window=break_focus_tail,
        )
        if break_logits is not None and ctx_lengths is not None
        else None,
    )
    if result is not None:
        # Keep weights on a stable scale to avoid overpowering base losses.
        result = result.detach()
        result = result / result.mean().clamp(min=1e-6)
        result = result.clamp(min=0.25, max=3.0)
    return result


def _weighted_sample_mean(values: torch.Tensor, sample_weight: torch.Tensor | None) -> torch.Tensor:
    if sample_weight is None:
        return values.mean()
    w = sample_weight.type_as(values)
    return (values * w).sum() / w.sum().clamp(min=1e-8)


def batch_pairwise_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batch 内累计收益成对排序损失（提升截面 IC）。"""
    pred_r = _pred_log_ret(pred)
    tgt = target[..., 0] if target.dim() >= 2 else target
    pc, tc = pred_r.sum(dim=1), tgt.sum(dim=1)
    n = pc.size(0)
    if n < 3:
        return pc.new_zeros(())
    dp = pc.unsqueeze(1) - pc.unsqueeze(0)
    dt = tc.unsqueeze(1) - tc.unsqueeze(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=pc.device)
    loss = F.softplus(-dp * dt)
    if sample_weight is None:
        return loss[mask].mean()
    pair_w = sample_weight.unsqueeze(1) * sample_weight.unsqueeze(0)
    return (loss[mask] * pair_w[mask]).sum() / pair_w[mask].sum().clamp(min=1e-8)


def cumulative_direction_loss(
    direction_logit: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """未来 H 步累计收益的涨跌二分类损失。"""
    tgt = target[..., 0] if target.dim() >= 2 else target
    direction = (tgt.sum(dim=1) > 0).float()
    return F.binary_cross_entropy_with_logits(direction_logit, direction, weight=sample_weight)


def future_shape_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """监督 body_ratio / upper_wick / lower_wick 等 K 线形态维度。"""
    if pred.dim() != 3 or pred.size(-1) <= 1:
        return pred.new_zeros(())
    dims = min(pred.size(-1), target.size(-1))
    diff = (pred[..., 1:dims] - target[..., 1:dims]).pow(2).mean(dim=(1, 2))
    return _weighted_sample_mean(diff, sample_weight)


def cumulative_path_shape_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """匹配未来累计收益路径的相对形状，弱化绝对幅度差异。"""
    pred_r = _pred_log_ret(pred)
    tgt = target[..., 0] if target.dim() >= 2 else target
    pred_path = pred_r.cumsum(dim=1)
    tgt_path = tgt.cumsum(dim=1)
    pred_norm = (pred_path - pred_path.mean(dim=1, keepdim=True)) / pred_path.std(dim=1, keepdim=True).clamp(min=1e-6)
    tgt_norm = (tgt_path - tgt_path.mean(dim=1, keepdim=True)) / tgt_path.std(dim=1, keepdim=True).clamp(min=1e-6)
    loss = (pred_norm - tgt_norm).pow(2).mean(dim=1)
    return _weighted_sample_mean(loss, sample_weight)


def cumulative_magnitude_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """累计 log_ret 的 MSE，直接监督预测幅度。"""
    pred_r = _pred_log_ret(pred)
    tgt = target[..., 0] if target.dim() >= 2 else target
    loss = (pred_r.sum(dim=1) - tgt.sum(dim=1)).pow(2)
    return _weighted_sample_mean(loss, sample_weight)


def relative_price_magnitude_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-4,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """价格变动相对误差损失：|pred_chg - tgt_chg| / |tgt_chg|。"""
    pred_r = _pred_log_ret(pred)
    tgt = target[..., 0] if target.dim() >= 2 else target
    pred_cum = pred_r.sum(dim=1)
    tgt_cum = tgt.sum(dim=1)
    pred_chg = pred_cum.exp() - 1.0
    tgt_chg = tgt_cum.exp() - 1.0
    rel_err = (pred_chg - tgt_chg).abs() / tgt_chg.abs().clamp(min=eps)
    loss = F.smooth_l1_loss(rel_err, torch.zeros_like(rel_err), reduction="none")
    return _weighted_sample_mean(loss, sample_weight)


def anti_lag_mimicry_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    ctx_bars: torch.Tensor,
    ctx_lengths: torch.Tensor,
    *,
    margin: float = 0.05,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """惩罚“预测路径更像刚过去走势，而不是未来走势”的滞后跟随行为。"""
    pred_r = _pred_log_ret(pred)
    tgt = target[..., 0] if target.dim() >= 2 else target
    horizon = pred_r.size(1)
    pred_path = pred_r.cumsum(dim=1)
    future_path = tgt.cumsum(dim=1)
    past_path = _trailing_window(ctx_bars[..., 0], ctx_lengths, horizon).cumsum(dim=1)
    step_weight = sample_weight.unsqueeze(1).expand_as(pred_path) if sample_weight is not None else None
    future_corr = batch_pearson_corr(pred_path, future_path, step_weight)
    past_corr = batch_pearson_corr(pred_path, past_path, step_weight)
    pred_norm = (pred_path - pred_path.mean(dim=1, keepdim=True)) / pred_path.std(dim=1, keepdim=True).clamp(min=1e-6)
    past_norm = (past_path - past_path.mean(dim=1, keepdim=True)) / past_path.std(dim=1, keepdim=True).clamp(min=1e-6)
    mimicry = _weighted_sample_mean((pred_norm - past_norm).pow(2).mean(dim=1).neg(), sample_weight)
    return F.relu(past_corr - future_corr + margin) + 0.1 * F.relu(mimicry + margin)


def stage3_prediction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    ctx_bars: torch.Tensor | None = None,
    ctx_lengths: torch.Tensor | None = None,
    break_logits: torch.Tensor | None = None,
    direction_logit: torch.Tensor | None = None,
    mse_weight: float = 0.7,
    step_corr_weight: float = 0.25,
    cum_corr_weight: float = 0.35,
    sign_weight: float = 0.15,
    rank_weight: float = 0.0,
    direction_weight: float = 0.0,
    shape_weight: float = 0.0,
    path_shape_weight: float = 0.0,
    cum_magnitude_weight: float = 0.0,
    relative_magnitude_weight: float = 0.0,
    raw_mse_weight: float = 0.0,
    vol_focus_weight: float = 0.0,
    vol_focus_top_frac: float = 0.3,
    move_focus_weight: float = 0.0,
    move_focus_scale: float = 3.0,
    break_focus_weight: float = 0.0,
    break_focus_tail: int = 16,
    code_supervision_weight: float = 0.0,
    code_supervision_logit: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
    anti_lag_weight: float = 0.0,
    anti_lag_margin: float = 0.05,
    corr_weight: float | None = None,
    target_raw_log_ret: torch.Tensor | None = None,
    log_ret_mean: torch.Tensor | None = None,
    log_ret_std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Stage 3 联合损失：回归排序 + 累计方向分类。"""
    if corr_weight is not None and corr_weight > 0:
        step_corr_weight = corr_weight
    pred_r = _pred_log_ret(pred)
    tgt = target[..., 0] if target.dim() >= 2 else target
    if target_raw_log_ret is not None and log_ret_mean is not None and log_ret_std is not None:
        pred_mag = denorm_predicted_log_ret(pred, log_ret_mean, log_ret_std)
        tgt_mag = target_raw_log_ret
        mag_target = _target_tensor_from_raw(tgt_mag)
    else:
        pred_mag = pred_r
        tgt_mag = tgt
        mag_target = target
    sample_weight = build_stage3_sample_weights(
        target,
        ctx_lengths=ctx_lengths,
        break_logits=break_logits,
        vol_focus_weight=vol_focus_weight,
        vol_focus_top_frac=vol_focus_top_frac,
        move_focus_weight=move_focus_weight,
        move_focus_scale=move_focus_scale,
        break_focus_weight=break_focus_weight,
        break_focus_tail=break_focus_tail,
        sample_weight=sample_weight,
    )
    mse = _weighted_sample_mean((pred_r - tgt).pow(2).mean(dim=1), sample_weight)
    if raw_mse_weight > 0 and target_raw_log_ret is not None and log_ret_mean is not None and log_ret_std is not None:
        pred_raw = denorm_predicted_log_ret(pred, log_ret_mean, log_ret_std)
        raw_mse = _weighted_sample_mean((pred_raw - target_raw_log_ret).pow(2).mean(dim=1), sample_weight)
    else:
        raw_mse = pred_r.new_zeros(())
    step_weight = sample_weight.unsqueeze(1).expand_as(pred_r) if sample_weight is not None else None
    step_corr = batch_pearson_corr(pred_r, tgt, step_weight)
    cum_corr = cumulative_return_corr(pred, target, sample_weight)
    sign_loss = _weighted_sample_mean(F.softplus(-pred_r * tgt).mean(dim=1), sample_weight)
    total = (
        mse_weight * mse
        + raw_mse_weight * raw_mse
        + step_corr_weight * (1.0 - step_corr)
        + cum_corr_weight * (1.0 - cum_corr)
        + sign_weight * sign_loss
    )
    if rank_weight > 0:
        rank_loss = batch_pairwise_rank_loss(pred, target, sample_weight)
        total = total + rank_weight * rank_loss
    else:
        rank_loss = pred_r.new_zeros(())
    if direction_weight > 0 and direction_logit is not None:
        dir_loss = cumulative_direction_loss(direction_logit, target, sample_weight)
        total = total + direction_weight * dir_loss
    else:
        dir_loss = pred_r.new_zeros(())
    if code_supervision_weight > 0 and code_supervision_logit is not None:
        code_sup_loss = cumulative_direction_loss(code_supervision_logit, target, sample_weight)
        total = total + code_supervision_weight * code_sup_loss
    else:
        code_sup_loss = pred_r.new_zeros(())
    if shape_weight > 0:
        shape = future_shape_loss(pred, target, sample_weight)
        total = total + shape_weight * shape
    else:
        shape = pred_r.new_zeros(())
    if path_shape_weight > 0:
        path_shape = cumulative_path_shape_loss(pred, target, sample_weight)
        total = total + path_shape_weight * path_shape
    else:
        path_shape = pred_r.new_zeros(())
    if cum_magnitude_weight > 0:
        cum_mag = cumulative_magnitude_mse_loss(
            pred_mag.unsqueeze(-1) if pred_mag.dim() == 2 else pred_mag,
            mag_target,
            sample_weight,
        )
        total = total + cum_magnitude_weight * cum_mag
    else:
        cum_mag = pred_r.new_zeros(())
    if relative_magnitude_weight > 0:
        rel_mag = relative_price_magnitude_loss(
            pred_mag.unsqueeze(-1) if pred_mag.dim() == 2 else pred_mag,
            mag_target,
            sample_weight=sample_weight,
        )
        total = total + relative_magnitude_weight * rel_mag
    else:
        rel_mag = pred_r.new_zeros(())
    if anti_lag_weight > 0 and ctx_bars is not None and ctx_lengths is not None:
        anti_lag = anti_lag_mimicry_loss(
            pred,
            target,
            ctx_bars,
            ctx_lengths,
            margin=anti_lag_margin,
            sample_weight=sample_weight,
        )
        total = total + anti_lag_weight * anti_lag
    else:
        anti_lag = pred_r.new_zeros(())
    return total, {
        "mse": float(mse.detach()),
        "raw_mse": float(raw_mse.detach()),
        "step_corr": float(step_corr.detach()),
        "cum_corr": float(cum_corr.detach()),
        "sign_loss": float(sign_loss.detach()),
        "rank_loss": float(rank_loss.detach()),
        "direction_loss": float(dir_loss.detach()),
        "code_supervision_loss": float(code_sup_loss.detach()),
        "shape_loss": float(shape.detach()),
        "path_shape_loss": float(path_shape.detach()),
        "cum_magnitude_loss": float(cum_mag.detach()),
        "relative_magnitude_loss": float(rel_mag.detach()),
        "anti_lag_loss": float(anti_lag.detach()),
    }
