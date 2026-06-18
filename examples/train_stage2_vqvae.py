#!/usr/bin/env python3
"""Stage 2：在 Stage 1 基础上继续微调 VQ 码本（可选）。

  python examples/train_stage1_segment_encoder.py --synthetic
  python examples/train_stage2_vqvae.py --synthetic --init-checkpoint checkpoints/pattern/stage1_auto_segment_vqvae.pt
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
from transformer_kit.auto_segment_encoder import AutoSegmentVQVAE
from transformer_kit.pattern_encoder import pattern_config_from_args
from transformer_kit.schedulers import build_adamw_with_warmup_cosine_restarts
from transformer_kit.segment_dataset import BarWindowDataset
from transformer_kit.train_utils import load_checkpoint, save_checkpoint
from transformer_kit.training import evaluate_auto_vqvae, train_auto_vqvae_epoch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2: fine-tune auto-segment VQ-VAE")
    add_data_args(p)
    add_feature_args(p)
    add_train_args(p)
    add_segment_args(p)
    add_vq_args(p)
    p.add_argument("--context-bars", type=int, default=128)
    p.add_argument("--max-segments", type=int, default=16)
    p.add_argument("--init-checkpoint", default="checkpoints/pattern/stage1_auto_segment_vqvae.pt")
    p.add_argument("--freeze-segmenting", action="store_true", help="冻结切分 MHA，只训 segment VQ")
    p.set_defaults(epochs=25)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    apply_real_data_defaults(args)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    bundle = prepare_bar_series_from_args(fetch_ohlcv_df(args), args)
    train_loader = DataLoader(
        BarWindowDataset(bundle.bars, bundle.train_idx, window=args.context_bars,
                         samples_per_epoch=args.samples_per_epoch, seed=args.seed),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    valid_loader = DataLoader(
        BarWindowDataset(bundle.bars, bundle.valid_idx, window=args.context_bars,
                         samples_per_epoch=300, seed=args.seed + 1),
        batch_size=args.batch_size, shuffle=False,
    )

    cfg = pattern_config_from_args(args)
    model = AutoSegmentVQVAE(cfg).to(device)
    init_path = Path(args.init_checkpoint)
    if init_path.is_file():
        ckpt = load_checkpoint(init_path)
        model.load_state_dict(ckpt["model"])
        print(f"  loaded {init_path}")

    if args.freeze_segmenting:
        for p in model.auto_encoder.segmenting_mha.parameters():
            p.requires_grad = False

    opt, sched = build_adamw_with_warmup_cosine_restarts(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps, t0=args.cosine_t0, t_mult=args.cosine_t_mult, eta_min=args.eta_min,
    )

    ckpt_dir = Path(args.checkpoint_dir)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr = train_auto_vqvae_epoch(model, train_loader, opt, sched, device, grad_clip=args.grad_clip)
        va = evaluate_auto_vqvae(model, valid_loader, device)
        mark = ""
        if va["loss"] < best:
            best = va["loss"]
            path = save_checkpoint(
                ckpt_dir / "stage2_vqvae.pt",
                {"stage": 2, "model": model.state_dict(), "config": cfg.__dict__},
            )
            mark = f" *saved {path}"
        print(f"  epoch {epoch:03d} train={tr.loss:.6f} valid={va['loss']:.6f} ppl={va['perplexity']:.1f}{mark}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
