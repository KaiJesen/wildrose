"""检查点保存/加载。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def load_auto_encoder(module: nn.Module, ckpt_path: str | Path) -> None:
    """加载 ``AutoSegmentVQEncoder`` / ``AutoSegmentVQVAE`` 权重。"""
    ckpt = load_checkpoint(ckpt_path)
    if "model" in ckpt:
        full = ckpt["model"]
        if hasattr(module, "auto_encoder"):
            prefix = "auto_encoder."
            state = {k[len(prefix) :]: v for k, v in full.items() if k.startswith(prefix)}
            if state:
                module.auto_encoder.load_state_dict(state)
                return
        module.load_state_dict(full, strict=False)
        return
    if "auto_encoder" in ckpt:
        module.load_state_dict(ckpt["auto_encoder"])
        return
    raise KeyError(f"no auto_encoder weights in {ckpt_path}")


# 兼容旧接口
load_segment_encoder = load_auto_encoder
load_pattern_encoder = load_auto_encoder
