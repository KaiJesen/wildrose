from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class LossWeights:
    leg_type_weight: float = 1.0
    sub_phase_weight: float = 0.5
    progress_weight: float = 0.3
    confirmed_weight: float = 0.2


def _class_weights(counts: np.ndarray) -> torch.Tensor:
    inv = 1.0 / np.clip(counts.astype(np.float64), 1.0, None)
    w = inv / inv.mean()
    return torch.tensor(w, dtype=torch.float32)


def compute_loss(
    batch: dict,
    out: dict,
    lw: LossWeights,
    *,
    leg_type_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    ce = nn.CrossEntropyLoss()
    ce_leg = nn.CrossEntropyLoss(weight=leg_type_weights)
    bce = nn.BCEWithLogitsLoss()
    huber = nn.SmoothL1Loss()
    conf_w = batch["teacher_conf"].squeeze(-1)
    l_type = ce_leg(out["leg_type_logits"], batch["leg_type"])
    l_sub = ce(out["sub_phase_logits"], batch["sub_phase"])
    l_prog = huber(out["leg_progress_pred"], batch["leg_progress"])
    l_conf = bce(out["leg_confirmed_logit"], batch["is_confirmed"])
    total = (
        lw.leg_type_weight * l_type
        + lw.sub_phase_weight * l_sub
        + lw.progress_weight * l_prog
        + lw.confirmed_weight * l_conf
    )
    total = total * conf_w.mean()
    return total, {
        "leg_type_ce": float(l_type.detach().cpu()),
        "sub_phase_ce": float(l_sub.detach().cpu()),
        "progress_huber": float(l_prog.detach().cpu()),
        "confirmed_bce": float(l_conf.detach().cpu()),
    }


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = np.unique(np.concatenate([y_true, y_pred]))
    scores = []
    for c in classes:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return float(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)


def _cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    n = len(y_true)
    if n == 0:
        return 0.0
    conf = np.zeros((len(labels), len(labels)), dtype=np.float64)
    label_to_i = {int(l): i for i, l in enumerate(labels)}
    for t, p in zip(y_true, y_pred):
        conf[label_to_i[int(t)], label_to_i[int(p)]] += 1.0
    conf /= n
    po = np.trace(conf)
    pe = (conf.sum(axis=0) * conf.sum(axis=1)).sum()
    return float((po - pe) / (1.0 - pe)) if pe < 1.0 else 0.0


@torch.no_grad()
def evaluate_epoch(
    model,
    loader,
    device: str,
    lw: LossWeights,
    *,
    leg_type_weights: torch.Tensor | None = None,
) -> dict[str, float]:
    model.eval()
    losses = []
    leg_true, leg_pred = [], []
    conf_true, conf_pred = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["x"])
        loss, _ = compute_loss(batch, out, lw, leg_type_weights=leg_type_weights)
        losses.append(float(loss.detach().cpu()))
        leg_true.append(batch["leg_type"].detach().cpu().numpy())
        leg_pred.append(out["leg_type_logits"].argmax(dim=-1).detach().cpu().numpy())
        conf_true.append(batch["is_confirmed"].detach().cpu().numpy().astype(int).ravel())
        conf_pred.append((torch.sigmoid(out["leg_confirmed_logit"]) > 0.5).detach().cpu().numpy().astype(int).ravel())
    y_true = np.concatenate(leg_true)
    y_pred = np.concatenate(leg_pred)
    macro_f1 = _macro_f1(y_true, y_pred)
    kappa = _cohen_kappa(y_true, y_pred)
    c_true = np.concatenate(conf_true)
    c_pred = np.concatenate(conf_pred)
    conf_f1 = _binary_f1(c_true, c_pred)
    confirmed_mask = c_true == 1
    confirmed_f1 = 0.0
    if confirmed_mask.any():
        confirmed_f1 = _macro_f1(y_true[confirmed_mask], y_pred[confirmed_mask])
    return {
        "loss": float(sum(losses) / max(1, len(losses))),
        "macro_f1_leg_type": macro_f1,
        "kappa_vs_teacher": kappa,
        "confirmed_f1": conf_f1,
        "f1_confirmed_only": confirmed_f1,
    }


def train_epoch(
    model,
    loader,
    optimizer,
    device: str,
    lw: LossWeights,
    *,
    leg_type_weights: torch.Tensor | None = None,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    model.train()
    losses = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["x"])
        loss, _ = compute_loss(batch, out, lw, leg_type_weights=leg_type_weights)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {"loss": float(sum(losses) / max(1, len(losses)))}
