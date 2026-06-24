from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LossWeights:
    entry_weight: float = 1.0
    hold_weight: float = 0.6
    exit_weight: float = 0.7
    opportunity_weight: float = 0.4
    ignore_index: int = -100


def compute_loss(batch: dict, out: dict, lw: LossWeights) -> tuple[torch.Tensor, dict[str, float]]:
    ce = nn.CrossEntropyLoss(ignore_index=lw.ignore_index)
    huber = nn.SmoothL1Loss()
    l_entry = ce(out["entry_logits"], batch["entry"])
    l_hold = ce(out["hold_logits"], batch["hold"])
    l_exit = ce(out["exit_logits"], batch["exit"])
    l_opp = huber(out["opportunity_pred"], batch["opp"])
    total = lw.entry_weight * l_entry + lw.hold_weight * l_hold + lw.exit_weight * l_exit + lw.opportunity_weight * l_opp
    return total, {
        "entry_ce": float(l_entry.detach().cpu()),
        "hold_ce": float(l_hold.detach().cpu()),
        "exit_ce": float(l_exit.detach().cpu()),
        "opp_huber": float(l_opp.detach().cpu()),
    }


@torch.no_grad()
def evaluate_epoch(model, loader, device: str, lw: LossWeights) -> dict[str, float]:
    model.eval()
    losses = []
    correct = 0
    total = 0
    opp_pred = []
    opp_true = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["x"])
        loss, _ = compute_loss(batch, out, lw)
        losses.append(float(loss.detach().cpu()))
        pred = out["entry_logits"].argmax(dim=-1)
        mask = batch["entry"] != lw.ignore_index
        correct += int(((pred == batch["entry"]) & mask).sum().item())
        total += int(mask.sum().item())
        opp_pred.append(out["opportunity_pred"].detach().cpu().flatten())
        opp_true.append(batch["opp"].detach().cpu().flatten())
    ic = 0.0
    if opp_pred:
        p = torch.cat(opp_pred).numpy()
        y = torch.cat(opp_true).numpy()
        if len(p) > 2 and y.std() > 1e-12:
            import numpy as np

            ic = float(np.corrcoef(p, y)[0, 1])
    return {
        "loss": float(sum(losses) / max(1, len(losses))),
        "entry_acc": float(correct / max(1, total)),
        "opportunity_ic": ic,
    }


def train_epoch(model, loader, optimizer, device: str, lw: LossWeights, grad_clip: float = 1.0) -> dict[str, float]:
    model.train()
    losses = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["x"])
        loss, _ = compute_loss(batch, out, lw)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {"loss": float(sum(losses) / max(1, len(losses)))}

