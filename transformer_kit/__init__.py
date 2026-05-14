"""Transformer 建模工具：面向 K 线时序的 embedding 与组件。

依赖：torch（可选）。本包不在导入时强制要求 torch，按需子模块再 import。
"""

from __future__ import annotations

__all__ = [
    "KlineBertEmbedding",
    "KlineBertEmbeddingConfig",
]


def __getattr__(name: str):
    # 懒加载：未装 torch 时仅 `import transformer_kit` 不会炸
    if name in {"KlineBertEmbedding", "KlineBertEmbeddingConfig"}:
        from transformer_kit.embeddings import (
            KlineBertEmbedding,
            KlineBertEmbeddingConfig,
        )

        return {"KlineBertEmbedding": KlineBertEmbedding, "KlineBertEmbeddingConfig": KlineBertEmbeddingConfig}[name]
    raise AttributeError(f"module 'transformer_kit' has no attribute {name!r}")
