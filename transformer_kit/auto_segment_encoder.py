"""自动切分 Embedding：因果 MHA 预测中断 → 变长片段 → VQ-VAE。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_kit.causal_transformer import causal_attention_mask
from transformer_kit.segment_encoder import SegmentDecoder, SegmentEncoderConfig, SegmentMHAEncoder
from transformer_kit.vector_quantizer import VectorQuantizer, VectorQuantizerEMA, VectorQuantizerOutput


@dataclass(frozen=True)
class AutoSegmentConfig:
    feat_dim: int = 5
    d_model: int = 256
    n_heads: int = 4
    segment_mha_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_ctx_len: int = 128
    max_seg_len: int = 32
    max_segments: int = 16
    min_seg_len: int = 4
    num_codes: int = 16
    vq_beta: float = 1.0
    vq_use_ema: bool = True
    vq_ema_decay: float = 0.99
    break_threshold: float = 0.5
    gumbel_tau: float = 0.7
    break_sparsity_weight: float = 0.01
    min_seg_penalty_weight: float = 0.05
    break_vol_weight: float = 0.12
    break_vol_window: int = 12
    break_vol_top_frac: float = 0.12


@dataclass
class AutoSegmentOutput:
    """自动切分 + VQ 编码输出。"""

    tokens: torch.Tensor
    """``[B, S, d_model]`` 形态 token 序列（S ≤ max_segments）。"""
    num_segments: torch.Tensor
    """``[B]`` 每个样本的有效段数。"""
    break_logits: torch.Tensor
    """``[B, T]`` 在 bar ``t`` 之后插入中断的 logit。"""
    break_hard: torch.Tensor
    """``[B, T]`` bool，是否在 bar ``t`` 后切分。"""
    seg_bars: torch.Tensor
    """``[B, S, L, F]`` 切分后的 padded 片段。"""
    seg_lengths: torch.Tensor
    """``[B, S]`` 每段有效长度。"""
    vq_loss: torch.Tensor
    perplexity: torch.Tensor
    codes: torch.Tensor
    break_reg_loss: torch.Tensor
    z_vec: torch.Tensor
    """``[N_active, d_model]`` 进入 VQ 前的 segment 向量（仅有效段）。"""


def _lengths_to_pad_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return idx >= lengths.unsqueeze(1)


def enforce_break_constraints(
    break_logits: torch.Tensor,
    lengths: torch.Tensor,
    *,
    min_seg_len: int,
    max_seg_len: int,
    max_segments: int,
    threshold: float = 0.5,
) -> torch.Tensor:
    """将 break logits 转为满足长度/段数约束的 hard mask ``[B,T]``。"""
    b, t = break_logits.shape
    probs = torch.sigmoid(break_logits)
    out = torch.zeros(b, t, dtype=torch.bool, device=break_logits.device)
    for i in range(b):
        ln = int(lengths[i].item())
        if ln <= 0:
            continue
        cand = probs[i, :ln] > threshold
        cuts: list[int] = []
        start = 0
        for pos in range(ln - 1):
            seg_len = pos - start + 1
            must_cut = seg_len >= max_seg_len
            want_cut = bool(cand[pos].item()) and seg_len >= min_seg_len
            if must_cut or want_cut:
                cuts.append(pos)
                start = pos + 1
                if len(cuts) >= max_segments - 1:
                    break
        for c in cuts:
            out[i, c] = True
    return out


def extract_segments(
    bars: torch.Tensor,
    lengths: torch.Tensor,
    break_after: torch.Tensor,
    *,
    max_segments: int,
    max_seg_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """由中断 mask 切分 bar 序列。

    返回 ``seg_bars [B,S,L,F]``, ``seg_lengths [B,S]``, ``num_segments [B]``。
    """
    b, t, f = bars.shape
    s = max_segments
    seg_bars = torch.zeros(b, s, max_seg_len, f, device=bars.device, dtype=bars.dtype)
    seg_lengths = torch.zeros(b, s, dtype=torch.long, device=bars.device)
    num_segments = torch.zeros(b, dtype=torch.long, device=bars.device)

    for i in range(b):
        ln = int(lengths[i].item())
        cuts = torch.nonzero(break_after[i, :ln], as_tuple=False).view(-1).tolist()
        bounds = [0] + [c + 1 for c in cuts] + [ln]
        bounds = bounds[: max_segments + 1]
        if bounds[-1] != ln:
            bounds[-1] = ln
        n_seg = len(bounds) - 1
        num_segments[i] = n_seg
        for j in range(n_seg):
            a, end = bounds[j], bounds[j + 1]
            seg_len = end - a
            if seg_len <= 0:
                continue
            use = min(seg_len, max_seg_len)
            seg_bars[i, j, :use] = bars[i, a : a + use]
            seg_lengths[i, j] = use
    return seg_bars, seg_lengths, num_segments


def _segment_encode_masked(
    segment_encoder: SegmentMHAEncoder,
    flat_bars: torch.Tensor,
    flat_lengths: torch.Tensor,
    flat_active: torch.Tensor,
) -> torch.Tensor:
    """仅对 active 段编码；padding 槽位填零向量。"""
    n = flat_bars.size(0)
    d = segment_encoder.out_norm.normalized_shape[0]
    z = flat_bars.new_zeros(n, d)
    if flat_active.any():
        z[flat_active] = segment_encoder(flat_bars[flat_active], flat_lengths[flat_active])
    return z


def break_regularization_loss(
    break_logits: torch.Tensor,
    lengths: torch.Tensor,
    seg_lengths: torch.Tensor,
    num_segments: torch.Tensor,
    *,
    min_seg_len: int,
    sparsity_weight: float,
    min_seg_penalty_weight: float,
) -> torch.Tensor:
    """中断稀疏 + 过短片段惩罚（无标签自监督）。"""
    mask = _lengths_to_pad_mask(lengths, break_logits.size(1))
    probs = torch.sigmoid(break_logits).masked_fill(mask, 0.0)
    sparsity = probs.sum() / lengths.float().sum().clamp(min=1.0)

    b, s = seg_lengths.shape
    seg_mask = torch.arange(s, device=seg_lengths.device).unsqueeze(0) < num_segments.unsqueeze(1)
    short = F.relu(min_seg_len - seg_lengths.float()) * seg_mask.float()
    short_pen = short.sum() / num_segments.float().sum().clamp(min=1.0)

    return sparsity_weight * sparsity + min_seg_penalty_weight * short_pen


def trailing_vol(log_ret: torch.Tensor, window: int) -> torch.Tensor:
    """因果 trailing 波动率 ``[B,T]``。"""
    b, t = log_ret.shape
    vol = log_ret.new_zeros(b, t)
    for i in range(t):
        s = max(0, i - window + 1)
        vol[:, i] = log_ret[:, s : i + 1].std(dim=1, unbiased=False)
    return vol


def break_volatility_pseudo_loss(
    break_logits: torch.Tensor,
    bars: torch.Tensor,
    lengths: torch.Tensor,
    *,
    vol_window: int = 12,
    top_frac: float = 0.12,
) -> torch.Tensor:
    """波动率跃迁伪标签 → 监督 break 头。"""
    log_ret = bars[..., 0]
    b, t = log_ret.shape
    if t < 3:
        return break_logits.new_zeros(())
    vol = trailing_vol(log_ret, vol_window)
    chg = (vol[:, 1:] - vol[:, :-1]).abs()
    logits = break_logits[:, : t - 1]
    targets = torch.zeros_like(chg)
    for i in range(b):
        ln = min(int(lengths[i].item()) - 1, t - 1)
        if ln < 2:
            continue
        k = max(1, int(ln * top_frac))
        _, idx = chg[i, :ln].topk(k, largest=True)
        targets[i, idx] = 1.0
    pad = _lengths_to_pad_mask((lengths - 1).clamp(min=0), t - 1)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = loss.masked_fill(pad, 0.0)
    return loss.sum() / (~pad).sum().clamp(min=1.0)


class BarSequenceSegmentingMHA(nn.Module):
    """Embedding 第一层：因果 MHA 扫描 K 线序列，预测在何处插入中断 token。"""

    def __init__(self, cfg: AutoSegmentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.input_proj = nn.Linear(cfg.feat_dim, d)
        self.break_token = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.break_head = nn.Linear(d, 1)
        nn.init.normal_(self.break_token, std=0.02)

    def forward(
        self,
        bars: torch.Tensor,
        lengths: torch.Tensor,
        *,
        hard_breaks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """bars ``[B,T,F]`` → hidden ``[B,T,d]``, break_logits ``[B,T]``, break_hard ``[B,T]``。"""
        if bars.dim() != 3:
            raise ValueError(f"bars must be [B,T,F], got {tuple(bars.shape)}")
        b, t, _ = bars.shape
        x = self.input_proj(bars)
        pad = _lengths_to_pad_mask(lengths, t)
        mask = causal_attention_mask(t, bars.device)
        h = self.encoder(x, mask=mask, src_key_padding_mask=pad)
        break_logits = self.break_head(h).squeeze(-1)

        if hard_breaks is None:
            if self.training:
                u = torch.rand_like(break_logits)
                g = -torch.log(-torch.log(u.clamp(1e-6, 1 - 1e-6)))
                soft = torch.sigmoid((break_logits + g) / self.cfg.gumbel_tau)
                soft = soft.masked_fill(pad, 0.0)
                hard = enforce_break_constraints(
                    break_logits.detach() + (soft - soft.detach()) * 10.0,
                    lengths,
                    min_seg_len=self.cfg.min_seg_len,
                    max_seg_len=self.cfg.max_seg_len,
                    max_segments=self.cfg.max_segments,
                    threshold=0.5,
                )
            else:
                hard = enforce_break_constraints(
                    break_logits,
                    lengths,
                    min_seg_len=self.cfg.min_seg_len,
                    max_seg_len=self.cfg.max_seg_len,
                    max_segments=self.cfg.max_segments,
                    threshold=self.cfg.break_threshold,
                )
        else:
            hard = hard_breaks
        return h, break_logits, hard


class AutoSegmentVQEncoder(nn.Module):
    """自动切分 + 各段 SegmentMHA + VQ（完整 Embedding 层）。"""

    def __init__(self, cfg: AutoSegmentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.segmenting_mha = BarSequenceSegmentingMHA(cfg)
        seg_cfg = SegmentEncoderConfig(
            feat_dim=cfg.feat_dim,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.segment_mha_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            max_len=cfg.max_seg_len,
        )
        self.segment_encoder = SegmentMHAEncoder(seg_cfg)
        if cfg.vq_use_ema:
            self.vq: VectorQuantizer | VectorQuantizerEMA = VectorQuantizerEMA(
                cfg.num_codes, cfg.d_model, beta=cfg.vq_beta, decay=cfg.vq_ema_decay
            )
        else:
            self.vq = VectorQuantizer(cfg.num_codes, cfg.d_model, beta=cfg.vq_beta)
        self.token_proj = nn.Linear(cfg.d_model, cfg.d_model)

    def encode(
        self,
        bars: torch.Tensor,
        lengths: torch.Tensor,
    ) -> AutoSegmentOutput:
        """ctx bars ``[B,T,F]`` → 形态 token 序列。"""
        _, break_logits, break_hard = self.segmenting_mha(bars, lengths)
        seg_bars, seg_lengths, num_segments = extract_segments(
            bars,
            lengths,
            break_hard,
            max_segments=self.cfg.max_segments,
            max_seg_len=self.cfg.max_seg_len,
        )

        b, s, max_len, f = seg_bars.shape
        flat_bars = seg_bars.reshape(b * s, max_len, f)
        flat_lengths = seg_lengths.reshape(b * s)
        active = torch.arange(b * s, device=bars.device).reshape(b, s)
        active = active < num_segments.unsqueeze(1)
        flat_active = active.reshape(-1)

        z = _segment_encode_masked(self.segment_encoder, flat_bars, flat_lengths, flat_active)
        vq_out = self.vq(z)
        tokens_flat = self.token_proj(vq_out.z_q)

        tokens = tokens_flat.view(b, s, -1)
        codes = vq_out.codes.view(b, s)
        tokens = tokens * active.unsqueeze(-1).float()

        vq_loss = vq_out.vq_loss
        if flat_active.any():
            vq_loss = vq_out.vq_loss  # batch-level; active segments dominate

        br = break_regularization_loss(
            break_logits,
            lengths,
            seg_lengths,
            num_segments,
            min_seg_len=self.cfg.min_seg_len,
            sparsity_weight=self.cfg.break_sparsity_weight,
            min_seg_penalty_weight=self.cfg.min_seg_penalty_weight,
        )
        if self.cfg.break_vol_weight > 0:
            br = br + self.cfg.break_vol_weight * break_volatility_pseudo_loss(
                break_logits,
                bars,
                lengths,
                vol_window=self.cfg.break_vol_window,
                top_frac=self.cfg.break_vol_top_frac,
            )

        return AutoSegmentOutput(
            tokens=tokens,
            num_segments=num_segments,
            break_logits=break_logits,
            break_hard=break_hard,
            seg_bars=seg_bars,
            seg_lengths=seg_lengths,
            vq_loss=vq_loss,
            perplexity=vq_out.perplexity,
            codes=codes,
            break_reg_loss=br,
            z_vec=z,
        )

    def forward(self, bars: torch.Tensor, lengths: torch.Tensor) -> AutoSegmentOutput:
        return self.encode(bars, lengths)


@dataclass
class AutoSegmentVQVAEOutput:
    recon_segments: torch.Tensor
    auto_out: AutoSegmentOutput
    recon_loss: torch.Tensor
    total_loss: torch.Tensor


class AutoSegmentVQVAE(nn.Module):
    """Stage 2：自动切分 + 各段 VQ 重建。"""

    def __init__(self, cfg: AutoSegmentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.auto_encoder = AutoSegmentVQEncoder(cfg)
        dec_cfg = SegmentEncoderConfig(
            feat_dim=cfg.feat_dim,
            d_model=cfg.d_model,
            max_len=cfg.max_seg_len,
            n_heads=cfg.n_heads,
        )
        self.decoder = SegmentDecoder(dec_cfg)

    def forward(self, bars: torch.Tensor, lengths: torch.Tensor) -> AutoSegmentVQVAEOutput:
        out = self.auto_encoder(bars, lengths)
        b, s, max_len, f = out.seg_bars.shape
        flat_bars = out.seg_bars.reshape(b * s, max_len, f)
        flat_lengths = out.seg_lengths.reshape(b * s)

        active = torch.arange(s, device=bars.device).unsqueeze(0) < out.num_segments.unsqueeze(1)
        flat_active = active.reshape(-1)
        z = _segment_encode_masked(self.auto_encoder.segment_encoder, flat_bars, flat_lengths, flat_active)
        vq_out = self.auto_encoder.vq(z)
        recon = self.decoder(vq_out.z_q).view(b, s, max_len, f)

        active = torch.arange(s, device=bars.device).unsqueeze(0) < out.num_segments.unsqueeze(1)
        active_flat = active.reshape(-1)
        mask = torch.zeros(b * s, max_len, 1, device=bars.device, dtype=torch.bool)
        for idx in range(b * s):
            if not active_flat[idx]:
                mask[idx, :, 0] = True
            else:
                ln = int(flat_lengths[idx].item())
                mask[idx, ln:, 0] = True
        diff = (recon.view(b * s, max_len, f) - flat_bars).pow(2)
        diff = diff.masked_fill(mask, 0.0)
        denom = (flat_lengths.float().sum() * f).clamp(min=1.0)
        recon_loss = diff.sum() / denom

        total = recon_loss + out.vq_loss + out.break_reg_loss
        return AutoSegmentVQVAEOutput(
            recon_segments=recon,
            auto_out=out,
            recon_loss=recon_loss,
            total_loss=total,
        )
