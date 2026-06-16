#!/usr/bin/env python3
"""Stage 3：自动切分形态 token + 因果 Transformer → 预测未来 log_ret。"""

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

from _train_common import add_data_args, add_segment_args, add_stage3_loss_args, add_train_args, add_vq_args, fetch_ohlcv_df
from transformer_kit.causal_transformer import CausalTransformerConfig
from transformer_kit.pattern_encoder import pattern_config_from_args
from transformer_kit.pattern_model import KlinePatternPredictor, PatternPredictorConfig
from transformer_kit.schedulers import build_adamw_with_warmup_cosine_restarts
from transformer_kit.segment_dataset import PatternSequenceDataset, build_sequence_sample_indices, prepare_bar_series
from transformer_kit.train_utils import load_auto_encoder, save_checkpoint
from transformer_kit.training import evaluate_stage3, train_stage3_epoch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 3: auto-segment tokens + predictor")
    add_data_args(p)
    add_train_args(p)
    add_segment_args(p)
    add_vq_args(p)
    add_stage3_loss_args(p)
    p.add_argument("--context-bars", type=int, default=128)
    p.add_argument("--max-segments", type=int, default=16)
    p.add_argument("--pred-horizon", type=int, default=5)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--trunk-layers", type=int, default=2)
    p.add_argument("--aux-vq-weight", type=float, default=0.1)
    p.add_argument("--aux-break-weight", type=float, default=0.05)
    p.add_argument("--init-checkpoint", default="checkpoints/pattern/stage2_vqvae.pt")
    p.add_argument("--encoder-lr-scale", type=float, default=0.1)
    p.set_defaults(epochs=40)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if len(sys.argv) == 1:
        args.synthetic = True

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    bundle = prepare_bar_series(fetch_ohlcv_df(args))

    def split(idx):
        return build_sequence_sample_indices(
            bundle.bars.shape[0], context_bars=args.context_bars, pred_horizon=args.pred_horizon,
            stride=args.stride, index_min=int(idx.min()), index_max=int(idx.max()),
        )

    train_loader = DataLoader(PatternSequenceDataset(bundle.bars, split(bundle.train_idx)),
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(PatternSequenceDataset(bundle.bars, split(bundle.valid_idx)),
                              batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(PatternSequenceDataset(bundle.bars, split(bundle.test_idx)),
                             batch_size=args.batch_size, shuffle=False)

    auto_cfg = pattern_config_from_args(args)
    model = KlinePatternPredictor(PatternPredictorConfig(
        auto_segment=auto_cfg,
        trunk=CausalTransformerConfig(d_model=args.d_model, n_heads=args.n_heads, n_layers=args.trunk_layers),
        pred_horizon=args.pred_horizon,
        pred_feat_dim=1,
        pool_mode=args.pool_mode,
        learnable_scale=not args.no_learnable_scale,
    )).to(device)

    init_path = Path(args.init_checkpoint)
    if not init_path.is_file():
        init_path = Path(args.checkpoint_dir) / "stage1_auto_segment_vqvae.pt"
    if init_path.is_file():
        print(f"  load auto encoder from {init_path}")
        load_auto_encoder(model.auto_encoder, init_path)

    enc_params = list(model.auto_encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    trunk_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt, sched = build_adamw_with_warmup_cosine_restarts(
        [{"params": enc_params, "lr": args.lr * args.encoder_lr_scale}, {"params": trunk_params, "lr": args.lr}],
        lr=args.lr, weight_decay=args.weight_decay, warmup_steps=args.warmup_steps,
        t0=args.cosine_t0, t_mult=args.cosine_t_mult, eta_min=args.eta_min,
    )

    ckpt_dir = Path(args.checkpoint_dir)
    best = float("inf")
    corr_kw = {} if args.corr_weight <= 0 else {"corr_weight": args.corr_weight}
    s3kw = dict(
        aux_vq_weight=args.aux_vq_weight,
        aux_break_weight=args.aux_break_weight,
        mse_weight=args.mse_weight,
        step_corr_weight=args.step_corr_weight,
        cum_corr_weight=args.cum_corr_weight,
        sign_weight=args.sign_weight,
        rank_weight=args.rank_weight,
        direction_weight=args.direction_weight,
        use_ic_loss=not args.no_ic_loss,
        **corr_kw,
    )
    for epoch in range(1, args.epochs + 1):
        tr = train_stage3_epoch(model, train_loader, opt, sched, device, grad_clip=args.grad_clip, **s3kw)
        va = evaluate_stage3(model, valid_loader, device, **s3kw)
        mark = ""
        if va["loss"] < best:
            best = va["loss"]
            save_checkpoint(ckpt_dir / "stage3_predictor.pt", {"stage": 3, "model": model.state_dict()})
            mark = " *saved"
        extra = f" ic={va['ic']:.3f}" if "ic" in va else ""
        if "direction_head_acc" in va:
            extra += f" head_dir={va['direction_head_acc']:.1%}"
        print(f"  epoch {epoch:03d} train={tr.loss:.6f} valid={va['loss']:.6f}{extra}{mark}")

    te = evaluate_stage3(model, test_loader, device, **s3kw)
    print(f"  test loss={te['loss']:.6f}", end="")
    if "ic" in te:
        print(
            f" ic={te['ic']:.3f} dir={te.get('direction_acc', 0):.1%} "
            f"head_dir={te.get('direction_head_acc', 0):.1%}"
        )
    else:
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
