"""Transformer 建模工具：自动切分 MHA + VQ + 预测。"""

from __future__ import annotations

__all__ = [
    "AutoSegmentConfig",
    "AutoSegmentVQEncoder",
    "AutoSegmentVQVAE",
    "BarSequenceSegmentingMHA",
    "KlinePatternPredictor",
    "PatternPredictorConfig",
    "CausalTransformer",
    "CausalTransformerConfig",
    "WarmupCosineAnnealingWarmRestarts",
    "build_adamw_with_warmup_cosine_restarts",
]

_LAZY = {
    "AutoSegmentConfig": "transformer_kit.auto_segment_encoder",
    "AutoSegmentVQEncoder": "transformer_kit.auto_segment_encoder",
    "AutoSegmentVQVAE": "transformer_kit.auto_segment_encoder",
    "BarSequenceSegmentingMHA": "transformer_kit.auto_segment_encoder",
    "KlinePatternPredictor": "transformer_kit.pattern_model",
    "PatternPredictorConfig": "transformer_kit.pattern_model",
    "CausalTransformer": "transformer_kit.causal_transformer",
    "CausalTransformerConfig": "transformer_kit.causal_transformer",
    "WarmupCosineAnnealingWarmRestarts": "transformer_kit.schedulers",
    "build_adamw_with_warmup_cosine_restarts": "transformer_kit.schedulers",
}


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module 'transformer_kit' has no attribute {name!r}")
    import importlib

    mod = importlib.import_module(_LAZY[name])
    return getattr(mod, name)
