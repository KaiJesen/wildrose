#!/usr/bin/env python3
"""024 Phase 1: train 0065a leg-align market-state model (ablation 0/1/2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

_EX = Path(__file__).resolve().parent
_ROOT = _EX.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))

from _train_common import (
    add_data_args,
    add_feature_args,
    add_segment_args,
    add_train_args,
    add_vq_args,
    apply_real_data_defaults,
    fetch_ohlcv_df,
    prepare_bar_series_from_args,
)
from transformer_kit.causal_transformer import CausalTransformerConfig
from transformer_kit.labels import estimate_market_state_thresholds
from transformer_kit.leg_align_dataset import LegParticipationSequenceDataset, load_label_dataframe
from transformer_kit.pattern_encoder import pattern_config_from_args
from transformer_kit.pattern_model import KlinePatternPredictor, PatternPredictorConfig
from transformer_kit.schedulers import build_adamw_with_warmup_cosine_restarts
from transformer_kit.segment_dataset import build_sequence_sample_indices
from transformer_kit.train_utils import load_checkpoint, save_checkpoint
from transformer_kit.training import evaluate_leg_align_market_state, train_leg_align_market_state_epoch

PROD_CKPT = "prod/v0.0.0/checkpoint/market_state_best.pt"
LABELS_DIR = "data/labels/leg_participation"

ABLATION = {
    "0": {
        "participation_weight": 1.5,
        "hz_12_weight": 0.0,
        "hz_24_weight": 0.0,
        "hz_48_weight": 0.0,
        "leg_dir_weight": 0.0,
        "leg_align_horizons": (),
    },
    "1": {
        "participation_weight": 0.25,
        "hz_12_weight": 0.10,
        "hz_24_weight": 0.15,
        "hz_48_weight": 0.0,
        "leg_dir_weight": 0.0,
        "leg_align_horizons": (12, 24),
    },
    "2": {
        "participation_weight": 0.25,
        "hz_12_weight": 0.10,
        "hz_24_weight": 0.15,
        "hz_48_weight": 0.0,
        "leg_dir_weight": 0.10,
        "leg_align_horizons": (12, 24),
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 024 0065a leg-align model")
    add_data_args(p)
    add_feature_args(p)
    add_train_args(p)
    add_segment_args(p)
    add_vq_args(p)
    p.add_argument("--pred-horizon", type=int, default=5)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--trunk-layers", type=int, default=2)
    p.add_argument("--variant", choices=["0", "1", "2"], default="0")
    p.add_argument("--init-checkpoint", default=PROD_CKPT)
    p.add_argument("--labels-dir", default=LABELS_DIR)
    p.add_argument("--baseline-cum-return-ic", type=float, default=0.0, help="0062e valid cum_return_ic for drift gate")
    p.add_argument("--early-stop-patience", type=int, default=8)
    p.add_argument("--report-dir", default="")
    p.add_argument("--participation-weight", type=float, default=-1.0, help="override ablation λ_part; <0 keeps variant default")
    p.add_argument("--positive-oversample", type=float, default=30.0, help="train sampler weight multiplier for ideal_participate=1")
    p.add_argument("--freeze-encoder", action="store_true", help="train participation heads only")
    p.set_defaults(
        epochs=12,
        batch_size=64,
        d_model=128,
        n_heads=4,
        encoder_layers=2,
        lr=4e-5,
        encoder_lr_scale=0.0,
        checkpoint_dir="checkpoints/0065a_leg_align_v0",
        use_cum_heads=True,
        use_horizon_return_head=True,
        source="binance_vision",
        symbol="BTCUSDT",
        interval="1h",
        days=365,
    )
    return p.parse_args()


def _split_idx(bundle, split: str) -> np.ndarray:
    if split == "train":
        return bundle.train_idx
    if split == "valid":
        return bundle.valid_idx
    return bundle.test_idx


def _build_samples(bundle, idx: np.ndarray, args) -> list:
    return build_sequence_sample_indices(
        bundle.bars.shape[0],
        context_bars=args.context_bars,
        pred_horizon=args.pred_horizon,
        stride=args.stride,
        index_min=int(idx.min()),
        index_max=int(idx.max()),
    )


def _count_ideal_samples(samples, label_df: pd.DataFrame) -> int:
    by_bar = label_df.set_index("bar_idx")
    count = 0
    for spec in samples:
        anchor = spec.context_end - 1
        if anchor not in by_bar.index:
            continue
        row = by_bar.loc[anchor]
        if int(row["ideal_participate_long"]) == 1 or int(row["ideal_participate_short"]) == 1:
            count += 1
    return count


def _sample_weights(samples, label_df: pd.DataFrame, *, oversample: float) -> list[float]:
    by_bar = label_df.set_index("bar_idx")
    weights: list[float] = []
    for spec in samples:
        anchor = spec.context_end - 1
        w = 1.0
        if anchor in by_bar.index:
            row = by_bar.loc[anchor]
            if int(row["ideal_participate_long"]) == 1 or int(row["ideal_participate_short"]) == 1:
                w = oversample
        weights.append(w)
    return weights


def _make_loader(bundle, samples, label_df, args, thr, *, shuffle: bool, oversample: float = 1.0) -> DataLoader:
    ab = ABLATION[args.variant]
    ds = LegParticipationSequenceDataset(
        bundle.bars,
        samples,
        bundle.raw_log_ret,
        label_df,
        zscore_window=bundle.zscore_window,
        direction_threshold=thr.direction_threshold,
        risk_vol_threshold=thr.risk_vol_threshold,
        leg_align_horizons=ab["leg_align_horizons"],
    )
    sampler = None
    if shuffle and oversample > 1.0:
        weights = _sample_weights(samples, label_df, oversample=oversample)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, drop_last=shuffle and sampler is None, sampler=sampler)


def _collect_future_windows(raw_log_ret: np.ndarray, samples) -> np.ndarray:
    rows = [raw_log_ret[s.context_end : s.future_end].astype(np.float32) for s in samples]
    return np.stack(rows, axis=0)


def main() -> int:
    args = parse_args()
    apply_real_data_defaults(args)
    ab = ABLATION[args.variant]
    if not args.report_dir:
        args.report_dir = f"reports/0065a_leg_align_v{args.variant}"
    if args.checkpoint_dir == "checkpoints/0065a_leg_align_v0" and args.variant != "0":
        args.checkpoint_dir = f"checkpoints/0065a_leg_align_v{args.variant}"

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    labels_dir = Path(args.labels_dir)
    for split in ("train", "valid", "test"):
        if not (labels_dir / f"leg_participation_{split}.csv").is_file():
            raise FileNotFoundError(f"missing label file for split={split} under {labels_dir}")

    df = fetch_ohlcv_df(args)
    bundle = prepare_bar_series_from_args(df, args)
    train_samples = _build_samples(bundle, _split_idx(bundle, "train"), args)
    valid_samples = _build_samples(bundle, _split_idx(bundle, "valid"), args)
    test_samples = _build_samples(bundle, _split_idx(bundle, "test"), args)

    thr = estimate_market_state_thresholds(
        _collect_future_windows(bundle.raw_log_ret, train_samples),
        direction_quantile=0.25,
        risk_quantile=0.70,
    )
    train_labels = load_label_dataframe(str(labels_dir / "leg_participation_train.csv"))
    valid_labels = load_label_dataframe(str(labels_dir / "leg_participation_valid.csv"))
    test_labels = load_label_dataframe(str(labels_dir / "leg_participation_test.csv"))

    part_w = ab["participation_weight"] if args.participation_weight < 0 else args.participation_weight
    train_pos = _count_ideal_samples(train_samples, train_labels)
    valid_pos = _count_ideal_samples(valid_samples, valid_labels)
    print(
        f"stride={args.stride} λ_part={part_w} oversample={args.positive_oversample} "
        f"train_ideal_samples={train_pos} valid_ideal_samples={valid_pos}"
    )

    train_loader = _make_loader(
        bundle, train_samples, train_labels, args, thr,
        shuffle=True, oversample=args.positive_oversample,
    )
    valid_loader = _make_loader(bundle, valid_samples, valid_labels, args, thr, shuffle=False)
    test_loader = _make_loader(bundle, test_samples, test_labels, args, thr, shuffle=False)

    auto_cfg = pattern_config_from_args(args)
    model = KlinePatternPredictor(
        PatternPredictorConfig(
            auto_segment=auto_cfg,
            trunk=CausalTransformerConfig(d_model=args.d_model, n_heads=args.n_heads, n_layers=args.trunk_layers),
            pred_horizon=args.pred_horizon,
            pred_feat_dim=1,
            pool_mode="attn",
            learnable_scale=True,
            use_horizon_head=False,
            use_market_state_head=True,
            use_cum_heads=True,
            use_horizon_return_head=True,
            use_participation_heads=True,
            leg_align_horizons=ab["leg_align_horizons"],
        )
    ).to(device)

    init_ckpt = Path(args.init_checkpoint)
    if init_ckpt.is_file():
        ck = load_checkpoint(init_ckpt, map_location=device)
        model.load_state_dict(ck["model"], strict=False)
        print(f"loaded init checkpoint: {init_ckpt}")
        if args.baseline_cum_return_ic <= 0 and isinstance(ck.get("args"), dict):
            pass

    enc_params = list(model.auto_encoder.parameters())
    trunk_params = list(model.trunk.parameters())
    enc_ids = {id(p) for p in enc_params} | {id(p) for p in trunk_params}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    if args.freeze_encoder:
        for p in enc_params:
            p.requires_grad = False
        for p in trunk_params:
            p.requires_grad = False
        opt, sched = build_adamw_with_warmup_cosine_restarts(
            [{"params": head_params, "lr": args.lr}],
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            t0=args.cosine_t0,
            t_mult=args.cosine_t_mult,
            eta_min=args.eta_min,
        )
        print("encoder+trunk frozen; training participation/aux heads only")
    else:
        opt, sched = build_adamw_with_warmup_cosine_restarts(
            [{"params": enc_params, "lr": args.lr * args.encoder_lr_scale}, {"params": head_params, "lr": args.lr}],
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            t0=args.cosine_t0,
            t_mult=args.cosine_t_mult,
            eta_min=args.eta_min,
        )

    ckpt_dir = Path(args.checkpoint_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    best_score = -1e9
    best_epoch = -1
    stale = 0
    history: list[dict] = []
    baseline_ic = float(args.baseline_cum_return_ic)

    for epoch in range(1, args.epochs + 1):
        tr = train_leg_align_market_state_epoch(
            model,
            train_loader,
            opt,
            sched,
            device,
            grad_clip=args.grad_clip,
            participation_weight=part_w,
            hz_12_weight=ab["hz_12_weight"],
            hz_24_weight=ab["hz_24_weight"],
            hz_48_weight=ab["hz_48_weight"],
            leg_dir_weight=ab["leg_dir_weight"],
        )
        valid_m = evaluate_leg_align_market_state(
            model,
            valid_loader,
            device,
            participation_weight=part_w,
            hz_12_weight=ab["hz_12_weight"],
            hz_24_weight=ab["hz_24_weight"],
            hz_48_weight=ab["hz_48_weight"],
            leg_dir_weight=ab["leg_dir_weight"],
        )
        row = {"epoch": epoch, "train_loss": tr.loss, **{f"valid_{k}": v for k, v in valid_m.items()}}
        history.append(row)
        score = valid_m["participation_auc"]
        ic_ok = baseline_ic <= 0 or valid_m["cum_return_ic"] >= baseline_ic * 0.95
        print(
            f"epoch {epoch:02d} loss={tr.loss:.4f} "
            f"part_auc={valid_m['participation_auc']:.4f} "
            f"part_auc_long={valid_m.get('participation_auc_long', 0):.4f} "
            f"hz24_acc={valid_m.get('hz_direction_acc_24', 0):.4f} "
            f"cum_ic={valid_m['cum_return_ic']:.4f} "
            f"score={score:.4f} ic_gate={'PASS' if ic_ok else 'FAIL'}"
        )
        if score > best_score and ic_ok:
            best_score = score
            best_epoch = epoch
            stale = 0
            save_checkpoint(
                ckpt_dir / "market_state_best.pt",
                {"model": model.state_dict(), "args": vars(args), "metrics": valid_m},
            )
        else:
            stale += 1
        if stale >= args.early_stop_patience:
            print(f"early stop at epoch {epoch}")
            break

    best_path = ckpt_dir / "market_state_best.pt"
    if best_path.is_file():
        ck = load_checkpoint(best_path, map_location=device)
        model.load_state_dict(ck["model"], strict=False)
    test_m = evaluate_leg_align_market_state(model, test_loader, device)

    out = {
        "variant": args.variant,
        "best_epoch": best_epoch,
        "valid_best_score": best_score,
        "ablation": {**ab, "participation_weight": part_w},
        "tuning": {
            "stride": args.stride,
            "positive_oversample": args.positive_oversample,
            "freeze_encoder": args.freeze_encoder,
            "train_ideal_samples": train_pos,
            "valid_ideal_samples": valid_pos,
        },
        "history": history,
        "test_metrics": test_m,
        "checkpoint": str(best_path.relative_to(_ROOT)),
    }
    (report_dir / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved checkpoint: {best_path}")
    print(f"saved report: {report_dir / 'metrics.json'}")
    print(
        f"test participation_auc={test_m.get('participation_auc', 0):.4f} "
        f"cum_return_ic={test_m.get('cum_return_ic', 0):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
