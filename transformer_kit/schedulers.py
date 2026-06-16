"""学习率调度：线性 warmup + 余弦退火重启（模拟退火式）。"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineAnnealingWarmRestarts(LRScheduler):
    """线性 warmup 后接 ``CosineAnnealingWarmRestarts`` 风格的分段余弦衰减。

    每个周期内学习率从 ``eta_max`` 按余弦降至 ``eta_min``，周期结束重启（模拟退火）。
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        warmup_steps: int,
        t0: int,
        t_mult: int = 1,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if t0 < 1:
            raise ValueError("t0 must be >= 1")
        self.warmup_steps = warmup_steps
        self.t0 = t0
        self.t_mult = max(1, t_mult)
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def _cycle_length(self, restart_idx: int) -> int:
        return self.t0 * (self.t_mult ** restart_idx)

    def _restart_index(self, step: int) -> tuple[int, int]:
        """返回 (第几次重启, 重启后经过的步数)。"""
        if step <= self.warmup_steps:
            return 0, step
        s = step - self.warmup_steps
        idx = 0
        offset = s
        length = self._cycle_length(idx)
        while offset >= length:
            offset -= length
            idx += 1
            length = self._cycle_length(idx)
        return idx, offset

    def get_lr(self) -> list[float]:
        step = self.last_epoch + 1
        if step <= self.warmup_steps and self.warmup_steps > 0:
            scale = step / self.warmup_steps
            return [base * scale for base in self.base_lrs]

        restart_idx, pos_in_cycle = self._restart_index(step)
        cycle_len = self._cycle_length(restart_idx)
        progress = pos_in_cycle / max(1, cycle_len)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.eta_min + (base - self.eta_min) * cosine for base in self.base_lrs
        ]


def build_adamw_with_warmup_cosine_restarts(
    params: Sequence[torch.nn.Parameter] | list[dict],
    *,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    t0: int = 500,
    t_mult: int = 2,
    eta_min: float = 1e-6,
) -> tuple[torch.optim.AdamW, WarmupCosineAnnealingWarmRestarts]:
    """创建 AdamW + warmup 余弦退火重启调度器。

    ``params`` 可为参数列表，或带 ``lr`` 的 param groups。
    """
    if params and isinstance(params[0], dict):
        opt = torch.optim.AdamW(params, weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = WarmupCosineAnnealingWarmRestarts(
        opt,
        warmup_steps=warmup_steps,
        t0=t0,
        t_mult=t_mult,
        eta_min=eta_min,
    )
    return opt, sched
