#!/usr/bin/env python3
"""026 Phase 1: C1 leg context + D1 sampling on top of Phase 0 C3 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

_EX = Path(__file__).resolve().parent
_ROOT = _EX.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))

from train_v026_phase0_c3 import (
    _composite_score,
    _read_recipe,
    _repo_rel,
    parse_args as _base_parse_args,
)
from train_market_state_0065a import (
    ABLATION,
    _build_samples,
    _collect_future_windows,
    _count_ideal_samples,
    _resolve_constraint_args,
    _split_idx,
)
from _train_common import apply_real_data_defaults, fetch_ohlcv_df, prepare_bar_series_from_args
from transformer_kit.causal_transformer import CausalTransformerConfig
from transformer_kit.labels import estimate_market_state_thresholds
from transformer_kit.leg_align_dataset import LegParticipationSequenceDataset, load_label_dataframe
from transformer_kit.leg_context import LEG_CONTEXT_VERSION
from transformer_kit.pattern_encoder import pattern_config_from_args
from transformer_kit.pattern_model import KlinePatternPredictor, PatternPredictorConfig
from transformer_kit.schedulers import build_adamw_with_warmup_cosine_restarts
from transformer_kit.train_utils import load_checkpoint, save_checkpoint
from transformer_kit.training import (
    collect_leg_align_head_params,
    evaluate_leg_align_market_state,
    freeze_legacy_market_state_heads,
    train_leg_align_market_state_epoch,
)

RECIPE = _ROOT / "configs/training_recipe_026_phase1_c1d1.json"
INIT_CKPT = _ROOT / "checkpoints/026_phase0_c3/market_state_best.pt"


def _make_loader(bundle, samples, label_df, args, thr, *, shuffle: bool) -> DataLoader:
    ab = ABLATION["0"]
    ds = LegParticipationSequenceDataset(
        bundle.bars,
        samples,
        bundle.raw_log_ret,
        label_df,
        zscore_window=bundle.zscore_window,
        direction_threshold=thr.direction_threshold,
        risk_vol_threshold=thr.risk_vol_threshold,
        leg_align_horizons=ab["leg_align_horizons"],
        use_leg_context=True,
        d1_sampling=True,
    )
    sampler = None
    if shuffle:
        weights = [
            ds._label_by_bar.get(spec.context_end - 1, {}).get("sample_weight", 0.5)
            for spec in samples
        ]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, drop_last=False, sampler=sampler)


def main() -> int:
    if "--recipe" not in sys.argv:
        sys.argv.extend(["--recipe", str(RECIPE)])
    args = _base_parse_args()
    apply_real_data_defaults(args)
    recipe = _read_recipe(Path(args.recipe))
    td = recipe["train_defaults"]

    args.stride = int(td["stride"]) if args.stride < 0 else args.stride
    args.early_stop_patience = int(td["early_stop_patience"]) if args.early_stop_patience < 0 else args.early_stop_patience
    args.cum_ic_min_ratio = float(td["cum_ic_min_ratio"]) if args.cum_ic_min_ratio < 0 else args.cum_ic_min_ratio
    args.part_auc_floor = float(td.get("part_auc_floor", 0.58)) if args.part_auc_floor < 0 else args.part_auc_floor
    args.composite_part_weight = float(td.get("composite_part_weight", 0.6)) if args.composite_part_weight < 0 else args.composite_part_weight
    args.composite_ic_weight = float(td.get("composite_ic_weight", 0.4)) if args.composite_ic_weight < 0 else args.composite_ic_weight
    if not args.init_checkpoint:
        args.init_checkpoint = recipe.get("init_checkpoint", str(INIT_CKPT.relative_to(_ROOT)))
    if not args.labels_dir:
        args.labels_dir = recipe.get("labels_dir", "data/labels/leg_participation")
    args.checkpoint_dir = recipe.get("checkpoint_dir", "checkpoints/026_phase1_c1d1")
    if not args.report_dir:
        args.report_dir = recipe.get("report_dir", "reports/026_phase1_c1d1")
    if args.participation_weight < 0:
        args.participation_weight = float(td["participation_weight"])
    if not args.auto_baseline_ic and td.get("auto_baseline_ic"):
        args.auto_baseline_ic = True

    args.variant = "0"
    ab = ABLATION["0"]
    for name, default in (
        ("freeze_legacy_heads", True),
        ("freeze_encoder", True),
        ("base_loss_scale", 0.0),
        ("drift_weight", 0.0),
    ):
        if not hasattr(args, name):
            setattr(args, name, default)
    cargs = _resolve_constraint_args(args)
    cargs["freeze_legacy_heads"] = True
    cargs["freeze_encoder"] = True

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

    part_w = args.participation_weight
    print(
        f"026 Phase1 C1+D1 λ_part={part_w} lr={args.lr} d1_sampling=True "
        f"leg_context={LEG_CONTEXT_VERSION} cum_ic_min_ratio={args.cum_ic_min_ratio}"
    )

    train_loader = _make_loader(bundle, train_samples, train_labels, args, thr, shuffle=True)
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
            use_participation_attn=True,
            use_leg_context=True,
            leg_align_horizons=ab["leg_align_horizons"],
        )
    ).to(device)

    init_ckpt = Path(args.init_checkpoint)
    if not init_ckpt.is_file():
        raise FileNotFoundError(f"missing init checkpoint: {init_ckpt}")
    ck = load_checkpoint(init_ckpt, map_location=device)
    model.load_state_dict(ck["model"], strict=False)
    print(f"loaded init checkpoint: {init_ckpt}")

    freeze_legacy_market_state_heads(model)
    for p in model.auto_encoder.parameters():
        p.requires_grad = False
    for p in model.trunk.parameters():
        p.requires_grad = False

    train_params = collect_leg_align_head_params(model)
    print(f"training {len(train_params)} C3+C1 tensors")
    opt, sched = build_adamw_with_warmup_cosine_restarts(
        [{"params": train_params, "lr": args.lr}],
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        t0=args.cosine_t0,
        t_mult=args.cosine_t_mult,
        eta_min=args.eta_min,
    )

    init_metrics = evaluate_leg_align_market_state(model, valid_loader, device)
    baseline_ic = float(init_metrics.get("cum_return_ic", 0.0))
    print(
        f"init valid baseline cum_return_ic={baseline_ic:.4f} "
        f"part_auc={init_metrics.get('participation_auc', 0):.4f}"
    )

    ckpt_dir = Path(args.checkpoint_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    best_score = -1e9
    best_epoch = -1
    stale = 0
    history: list[dict] = []
    ic_floor = baseline_ic * args.cum_ic_min_ratio if baseline_ic > 0 else -1e9

    for epoch in range(1, args.epochs + 1):
        tr = train_leg_align_market_state_epoch(
            model,
            train_loader,
            opt,
            sched,
            device,
            grad_clip=args.grad_clip,
            participation_weight=part_w,
            base_loss_scale=0.0,
            drift_weight=0.0,
            teacher=None,
            encoder_aux_loss=False,
        )
        valid_m = evaluate_leg_align_market_state(model, valid_loader, device, participation_weight=part_w)
        history.append({"epoch": epoch, "train_loss": tr.loss, **{f"valid_{k}": v for k, v in valid_m.items()}})
        part_score = float(valid_m["participation_auc"])
        cum_ic = float(valid_m["cum_return_ic"])
        ic_ok = baseline_ic <= 0 or cum_ic >= ic_floor
        part_ok = part_score >= args.part_auc_floor
        score = _composite_score(
            part_score,
            cum_ic,
            baseline_ic,
            part_w=args.composite_part_weight,
            ic_w=args.composite_ic_weight,
        )
        ic_deg = 0.0 if baseline_ic <= 0 else max(0.0, (baseline_ic - cum_ic) / baseline_ic)
        print(
            f"epoch {epoch:02d} loss={tr.loss:.4f} part_auc={part_score:.4f} "
            f"cum_ic={cum_ic:.4f} ic_deg={ic_deg*100:.1f}% score={score:.4f} "
            f"ic_gate={'PASS' if ic_ok else 'FAIL'} part_floor={'PASS' if part_ok else 'FAIL'}"
        )
        if score > best_score and ic_ok and part_ok:
            best_score = score
            best_epoch = epoch
            stale = 0
            save_checkpoint(
                ckpt_dir / "market_state_best.pt",
                {
                    "model": model.state_dict(),
                    "args": {
                        **vars(args),
                        "use_participation_attn": True,
                        "use_leg_context": True,
                        "leg_context_version": LEG_CONTEXT_VERSION,
                        "experiment": "026_phase1_c1d1",
                    },
                    "metrics": valid_m,
                    "constraint": cargs,
                    "baseline_cum_return_ic": baseline_ic,
                },
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
    valid_m = evaluate_leg_align_market_state(model, valid_loader, device)
    test_m = evaluate_leg_align_market_state(model, test_loader, device)
    ic_deg_valid = 0.0 if baseline_ic <= 0 else max(0.0, (baseline_ic - valid_m["cum_return_ic"]) / baseline_ic)

    out = {
        "experiment": "026_phase1_c1d1",
        "best_epoch": best_epoch,
        "valid_best_score": best_score,
        "valid_participation_auc": valid_m.get("participation_auc"),
        "valid_cum_return_ic": valid_m.get("cum_return_ic"),
        "valid_ic_degradation": ic_deg_valid,
        "baseline_cum_return_ic": baseline_ic,
        "leg_context_version": LEG_CONTEXT_VERSION,
        "init_checkpoint": _repo_rel(init_ckpt),
        "tuning": {"lr": args.lr, "participation_weight": part_w, "d1_sampling": True},
        "history": history,
        "test_metrics": test_m,
        "checkpoint": _repo_rel(best_path),
    }
    (report_dir / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"valid part_auc={valid_m.get('participation_auc', 0):.4f} test part_auc={test_m.get('participation_auc', 0):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
