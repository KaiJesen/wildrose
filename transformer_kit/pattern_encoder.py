"""形态编码层（兼容入口 → 自动切分 + VQ）。"""

from __future__ import annotations

from dataclasses import dataclass

from transformer_kit.auto_segment_encoder import (
    AutoSegmentConfig,
    AutoSegmentOutput,
    AutoSegmentVQEncoder,
    AutoSegmentVQVAE,
    AutoSegmentVQVAEOutput,
)

# 旧名映射，便于训练脚本过渡
PatternEncoderConfig = AutoSegmentConfig
PatternEncoder = AutoSegmentVQEncoder
PatternVQVAE = AutoSegmentVQVAE


@dataclass
class PatternVQVAEOutput:
    recon: object
    vq_out: object
    recon_loss: object
    total_loss: object


def pattern_config_from_args(args) -> AutoSegmentConfig:
    """从 argparse 命名空间构建 ``AutoSegmentConfig``。"""
    return AutoSegmentConfig(
        feat_dim=5,
        d_model=args.d_model,
        n_heads=args.n_heads,
        segment_mha_layers=getattr(args, "encoder_layers", 2),
        max_ctx_len=getattr(args, "context_bars", 128),
        max_seg_len=args.max_seg_len,
        max_segments=getattr(args, "max_segments", 16),
        min_seg_len=args.min_seg_len,
        num_codes=args.num_codes,
        vq_beta=args.vq_beta,
        break_vol_weight=getattr(args, "break_vol_weight", 0.12),
        break_vol_window=getattr(args, "break_vol_window", 12),
        break_vol_top_frac=getattr(args, "break_vol_top_frac", 0.12),
    )


__all__ = [
    "AutoSegmentConfig",
    "AutoSegmentOutput",
    "AutoSegmentVQEncoder",
    "AutoSegmentVQVAE",
    "AutoSegmentVQVAEOutput",
    "PatternEncoderConfig",
    "PatternEncoder",
    "PatternVQVAE",
    "PatternVQVAEOutput",
    "pattern_config_from_args",
]
