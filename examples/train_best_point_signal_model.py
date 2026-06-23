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
from best_point.dataset import BestPointDataset, time_split_indices
from best_point.features import compute_causal_features
from best_point.model import BestPointModelConfig, BestPointSignalModel
from best_point.report import write_report
from best_point.training import LossWeights, evaluate_epoch, train_epoch
from market_data.schema import COL_TIME


def _load_label_file(path: str):
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    return pd.read_parquet(p)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BestPointSignalModel (v017)")
    add_data_args(p)
    p.add_argument("--labels-file", default="data/labels/best_point_v017/BTCUSDT_1h_labels.parquet")
    p.add_argument("--context-bars", type=int, default=96)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--checkpoint-dir", default="checkpoints/017_best_point_signal")
    p.add_argument("--report-dir", default="reports/017_best_point_signal")
    p.add_argument("--run-name", default="017b_best_point_signal_baseline")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    apply_real_data_defaults(args)
    device = torch.device(args.device)

    df = fetch_ohlcv_df(args).reset_index(drop=True)
    labels = _load_label_file(args.labels_file).reset_index(drop=True)
    if len(labels) != len(df):
        m = min(len(labels), len(df))
        df = df.iloc[:m].reset_index(drop=True)
        labels = labels.iloc[:m].reset_index(drop=True)

    feat = compute_causal_features(df)
    split = time_split_indices(len(feat))
    train_ds = BestPointDataset(feat, labels, context_bars=args.context_bars, start_idx=0, end_idx=split.train_end)
    valid_ds = BestPointDataset(
        feat,
        labels,
        context_bars=args.context_bars,
        start_idx=split.train_end,
        end_idx=split.valid_end,
        feature_mean=train_ds.mean,
        feature_std=train_ds.std,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = BestPointSignalModel(BestPointModelConfig(input_dim=feat.shape[1])).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    lw = LossWeights()
    best = None
    best_state = None
    for ep in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, optim, str(device), lw)
        va = evaluate_epoch(model, valid_loader, str(device), lw)
        score = va["entry_acc"] + max(0.0, va["opportunity_ic"])
        if best is None or score > best:
            best = score
            best_state = {
                "model": model.state_dict(),
                "feature_columns": feat.columns.tolist(),
                "feature_mean": train_ds.mean.tolist(),
                "feature_std": train_ds.std.tolist(),
                "config": BestPointModelConfig(input_dim=feat.shape[1]).__dict__,
                "context_bars": args.context_bars,
            }
        print(f"epoch {ep}: train_loss={tr['loss']:.4f} valid_loss={va['loss']:.4f} entry_acc={va['entry_acc']:.4f} ic={va['opportunity_ic']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state["model"])

    ckpt_dir = Path(args.checkpoint_dir) / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "best.pt"
    torch.save(best_state, ckpt)

    metrics = evaluate_epoch(model, valid_loader, str(device), lw)
    metrics.update({"best_score": float(best), "run_name": args.run_name, "checkpoint": str(ckpt)})
    report_dir = Path(args.report_dir) / args.run_name
    write_report(report_dir, metrics, title="017 BestPointSignal Training Report")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

