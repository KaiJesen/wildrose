#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_EX = Path(__file__).resolve().parent
_ROOT = _EX.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))

from _train_common import add_data_args, apply_real_data_defaults, fetch_ohlcv_df
from best_point.dataset import BestPointDataset, time_split_indices
from best_point.evaluation import confusion_matrix, macro_f1_from_confusion
from best_point.features import compute_causal_features
from best_point.model import BestPointModelConfig, BestPointSignalModel
from best_point.training import LossWeights, evaluate_epoch


def _load_label_file(path: str):
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    return pd.read_parquet(p)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate BestPointSignalModel (v017)")
    add_data_args(p)
    p.add_argument("--labels-file", default="data/labels/best_point_v017/BTCUSDT_1h_labels.parquet")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--context-bars", type=int, default=96)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


@torch.no_grad()
def eval_entry_f1(model, loader, device: str, ignore_index: int) -> float:
    model.eval()
    yt, yp = [], []
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        out = model(b["x"])
        pred = out["entry_logits"].argmax(dim=-1)
        m = b["entry"] != ignore_index
        yt.append(b["entry"][m].detach().cpu().numpy())
        yp.append(pred[m].detach().cpu().numpy())
    if not yt:
        return 0.0
    y_true = np.concatenate(yt)
    y_pred = np.concatenate(yp)
    cm = confusion_matrix(y_true, y_pred, n_classes=3)
    return macro_f1_from_confusion(cm)


def main() -> int:
    args = parse_args()
    apply_real_data_defaults(args)
    device = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location=device)
    feature_columns = ck["feature_columns"]
    mean = np.asarray(ck["feature_mean"], dtype=np.float32)
    std = np.asarray(ck["feature_std"], dtype=np.float32)
    context_bars = int(ck.get("context_bars", args.context_bars))

    df = fetch_ohlcv_df(args).reset_index(drop=True)
    labels = _load_label_file(args.labels_file).reset_index(drop=True)
    m = min(len(df), len(labels))
    feat = compute_causal_features(df.iloc[:m]).loc[:, feature_columns]
    labels = labels.iloc[:m].reset_index(drop=True)

    split = time_split_indices(len(feat))
    test_ds = BestPointDataset(
        feat,
        labels,
        context_bars=context_bars,
        start_idx=split.valid_end,
        end_idx=len(feat),
        feature_mean=mean,
        feature_std=std,
    )
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    cfg = BestPointModelConfig(**ck["config"])
    model = BestPointSignalModel(cfg).to(device)
    model.load_state_dict(ck["model"])
    lw = LossWeights()
    metrics = evaluate_epoch(model, test_loader, str(device), lw)
    metrics["entry_macro_f1"] = eval_entry_f1(model, test_loader, str(device), lw.ignore_index)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

