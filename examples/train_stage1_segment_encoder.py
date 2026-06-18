#!/usr/bin/env python3
"""Stage 1：自动切分 MHA + 各段 VQ-VAE 联合预训练。

  python examples/train_stage1_segment_encoder.py --synthetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_EX = Path(__file__).resolve().parent
_ROOT = _EX.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))

from _train_common import add_data_args, add_feature_args, add_segment_args, add_train_args, add_vq_args, apply_real_data_defaults, fetch_ohlcv_df, prepare_bar_series_from_args
from transformer_kit.auto_segment_encoder import AutoSegmentConfig, AutoSegmentVQVAE
from transformer_kit.pattern_encoder import pattern_config_from_args
from transformer_kit.schedulers import build_adamw_with_warmup_cosine_restarts
from transformer_kit.segment_dataset import BarWindowDataset
from transformer_kit.train_utils import save_checkpoint
from transformer_kit.training import evaluate_auto_vqvae, init_vq_codebook_from_loader, train_auto_vqvae_epoch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1: auto-segment MHA + VQ-VAE")
    add_data_args(p)
    add_feature_args(p)
    add_train_args(p)
    add_segment_args(p)
    add_vq_args(p)
    p.add_argument("--context-bars", type=int, default=128)
    p.add_argument("--max-segments", type=int, default=16)
    p.set_defaults(epochs=40)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    apply_real_data_defaults(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print("[1/4] load data")
    bundle = prepare_bar_series_from_args(fetch_ohlcv_df(args), args)

    train_ds = BarWindowDataset(
        bundle.bars, bundle.train_idx,
        window=args.context_bars, samples_per_epoch=args.samples_per_epoch, seed=args.seed,
    )
    valid_ds = BarWindowDataset(
        bundle.bars, bundle.valid_idx,
        window=args.context_bars, samples_per_epoch=max(300, args.samples_per_epoch // 10),
        seed=args.seed + 1,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

    print("[2/4] build AutoSegmentVQVAE (MHA break + segment VQ)")
    cfg = pattern_config_from_args(args)
    model = AutoSegmentVQVAE(cfg).to(device)

    opt, sched = build_adamw_with_warmup_cosine_restarts(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps, t0=args.cosine_t0, t_mult=args.cosine_t_mult, eta_min=args.eta_min,
    )

    print("[3/4] train")
    ckpt_dir = Path(args.checkpoint_dir)
    init_vq_codebook_from_loader(model, train_loader, device)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr = train_auto_vqvae_epoch(model, train_loader, opt, sched, device, grad_clip=args.grad_clip)
        va = evaluate_auto_vqvae(model, valid_loader, device)
        mark = ""
        if va["loss"] < best:
            best = va["loss"]
            path = save_checkpoint(
                ckpt_dir / "stage1_auto_segment_vqvae.pt",
                {"stage": 1, "model": model.state_dict(), "config": cfg.__dict__},
            )
            mark = f" *saved {path}"
        print(
            f"  epoch {epoch:03d}/{args.epochs} train={tr.loss:.6f} valid={va['loss']:.6f} "
            f"ppl={va['perplexity']:.1f} segs={tr.extras.get('avg_segments', 0):.1f} lr={tr.lr:.2e}{mark}"
        )

    print("[4/4] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
