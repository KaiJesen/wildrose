#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from best_point.features import compute_causal_features
from trend_leg.dataset import TrendLegDataset, time_split_indices
from trend_leg.labels import LEG_TYPES
from trend_leg.model import TrendLegClassifier, TrendLegModelConfig
from trend_leg.training import LossWeights, evaluate_epoch, train_epoch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TrendLegClassifier Student (v020)")
    add_data_args(p)
    p.add_argument("--labels-file", default="data/labels/trend_leg_v020_teacher/teacher_labels.csv")
    p.add_argument("--context-bars", type=int, default=128)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--checkpoint-dir", default="checkpoints/020_trend_leg_classifier")
    p.add_argument("--report-dir", default="reports/020_trend_leg_classifier")
    p.add_argument("--run-name", default="020a_trend_leg_baseline")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    apply_real_data_defaults(args)
    device = torch.device(args.device)

    df = fetch_ohlcv_df(args).reset_index(drop=True)
    labels = pd.read_csv(args.labels_file).reset_index(drop=True)
    m = min(len(labels), len(df))
    df = df.iloc[:m].reset_index(drop=True)
    labels = labels.iloc[:m].reset_index(drop=True)

    feat = compute_causal_features(df)
    split = time_split_indices(len(feat))
    train_ds = TrendLegDataset(feat, labels, context_bars=args.context_bars, start_idx=0, end_idx=split.train_end)
    valid_ds = TrendLegDataset(
        feat,
        labels,
        context_bars=args.context_bars,
        start_idx=split.train_end,
        end_idx=split.valid_end,
        feature_mean=train_ds.mean,
        feature_std=train_ds.std,
    )
    test_ds = TrendLegDataset(
        feat,
        labels,
        context_bars=args.context_bars,
        start_idx=split.valid_end,
        end_idx=len(feat),
        feature_mean=train_ds.mean,
        feature_std=train_ds.std,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    import numpy as np

    counts = np.bincount(train_ds.leg_type[train_ds.start : train_ds.end], minlength=len(LEG_TYPES))
    leg_weights = torch.tensor(1.0 / np.clip(counts.astype(np.float64), 1.0, None), dtype=torch.float32)
    leg_weights = (leg_weights / leg_weights.mean()).to(device)

    model = TrendLegClassifier(TrendLegModelConfig(input_dim=feat.shape[1])).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    lw = LossWeights()
    best = None
    best_state = None
    history = []
    for ep in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, optim, str(device), lw, leg_type_weights=leg_weights)
        va = evaluate_epoch(model, valid_loader, str(device), lw, leg_type_weights=leg_weights)
        score = va["macro_f1_leg_type"] + 0.5 * va["f1_confirmed_only"]
        history.append({"epoch": ep, "train": tr, "valid": va, "score": score})
        print(
            f"epoch {ep}: train_loss={tr['loss']:.4f} valid_loss={va['loss']:.4f} "
            f"macro_f1={va['macro_f1_leg_type']:.4f} conf_f1={va['f1_confirmed_only']:.4f} kappa={va['kappa_vs_teacher']:.4f}"
        )
        if best is None or score > best:
            best = score
            best_state = {
                "model": model.state_dict(),
                "feature_columns": feat.columns.tolist(),
                "feature_mean": train_ds.mean.tolist(),
                "feature_std": train_ds.std.tolist(),
                "config": TrendLegModelConfig(input_dim=feat.shape[1]).__dict__,
                "context_bars": args.context_bars,
                "leg_types": LEG_TYPES,
            }

    te_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    if best_state is not None:
        model.load_state_dict(best_state["model"])
    te = evaluate_epoch(model, test_loader, str(device), lw, leg_type_weights=leg_weights)
    print(f"test: macro_f1={te['macro_f1_leg_type']:.4f} conf_f1={te['f1_confirmed_only']:.4f}")

    ckpt_dir = Path(args.checkpoint_dir) / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best.pt"
    torch.save(best_state, ckpt_path)

    report_dir = Path(args.report_dir) / args.run_name
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"best_valid_score": best, "test": te, "history": history, "label_counts_train": counts.tolist()}
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (report_dir / "REPORT_TRAIN.md").write_text(
        f"# TrendLegClassifier Training Report\n\n"
        f"- checkpoint: `{ckpt_path}`\n"
        f"- best valid score: {best:.4f}\n"
        f"- test macro_f1: {te['macro_f1_leg_type']:.4f}\n"
        f"- test f1_confirmed_only: {te['f1_confirmed_only']:.4f}\n"
        f"- test kappa: {te['kappa_vs_teacher']:.4f}\n",
        encoding="utf-8",
    )
    print(f"saved checkpoint: {ckpt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
