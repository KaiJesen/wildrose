#!/usr/bin/env python3
"""026 Phase 0: train C3 ParticipationAttn head (trunk + legacy heads frozen)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_EX = Path(__file__).resolve().parent
_ROOT = _EX.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))

from train_market_state_0065a import (
    ABLATION,
    _build_samples,
    _collect_future_windows,
    _count_ideal_samples,
    _make_loader,
    _resolve_constraint_args,
    _split_idx,
)
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
from transformer_kit.leg_align_dataset import load_label_dataframe
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

RECIPE = _ROOT / "configs/training_recipe_026_phase0_c3.json"
INIT_CKPT = _ROOT / "prod/v1.1.1/checkpoint/market_state_best.pt"


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_ROOT.resolve()))
    except ValueError:
        return str(path)


def _read_recipe(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _composite_score(
    part_auc: float,
    cum_ic: float,
    baseline_ic: float,
    *,
    part_w: float,
    ic_w: float,
) -> float:
    ic_ratio = (cum_ic / baseline_ic) if baseline_ic > 0 else 1.0
    ic_ratio = min(1.0, max(0.0, ic_ratio))
    return part_w * part_auc + ic_w * ic_ratio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="026 Phase 0 C3 ParticipationAttn training")
    add_data_args(p)
    add_feature_args(p)
    add_train_args(p)
    add_segment_args(p)
    add_vq_args(p)
    p.add_argument("--recipe", default=str(RECIPE))
    p.add_argument("--pred-horizon", type=int, default=5)
    p.add_argument("--stride", type=int, default=-1)
    p.add_argument("--trunk-layers", type=int, default=2)
    p.add_argument("--init-checkpoint", default="")
    p.add_argument("--labels-dir", default="")
    p.add_argument("--report-dir", default="")
    p.add_argument("--participation-weight", type=float, default=-1.0)
    p.add_argument("--positive-oversample", type=float, default=-1.0)
    p.add_argument("--early-stop-patience", type=int, default=-1)
    p.add_argument("--cum-ic-min-ratio", type=float, default=-1.0)
    p.add_argument("--part-auc-floor", type=float, default=-1.0)
    p.add_argument("--composite-part-weight", type=float, default=-1.0)
    p.add_argument("--composite-ic-weight", type=float, default=-1.0)
    p.add_argument("--constraint-profile", choices=["none", "constrained", "soft"], default="constrained")
    p.add_argument("--auto-baseline-ic", action="store_true")
    p.set_defaults(
        epochs=15,
        batch_size=32,
        d_model=128,
        n_heads=4,
        encoder_layers=2,
        lr=3e-4,
        use_cum_heads=True,
        use_horizon_return_head=True,
        source="binance_vision",
        symbol="BTCUSDT",
        interval="1h",
        days=365,
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    apply_real_data_defaults(args)
    recipe = _read_recipe(Path(args.recipe))
    td = recipe["train_defaults"]

    if args.stride < 0:
        args.stride = int(td["stride"])
    if args.positive_oversample < 0:
        args.positive_oversample = float(td["positive_oversample"])
    if args.early_stop_patience < 0:
        args.early_stop_patience = int(td["early_stop_patience"])
    if args.cum_ic_min_ratio < 0:
        args.cum_ic_min_ratio = float(td["cum_ic_min_ratio"])
    if args.part_auc_floor < 0:
        args.part_auc_floor = float(td.get("part_auc_floor", 0.58))
    if args.composite_part_weight < 0:
        args.composite_part_weight = float(td.get("composite_part_weight", 0.6))
    if args.composite_ic_weight < 0:
        args.composite_ic_weight = float(td.get("composite_ic_weight", 0.4))
    if not args.init_checkpoint:
        args.init_checkpoint = recipe.get("init_checkpoint", str(INIT_CKPT.relative_to(_ROOT)))
    if not args.labels_dir:
        args.labels_dir = recipe.get("labels_dir", "data/labels/leg_participation")
    args.checkpoint_dir = recipe.get("checkpoint_dir", "checkpoints/026_phase0_c3")
    if not args.report_dir:
        args.report_dir = recipe.get("report_dir", "reports/026_phase0_c3")
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
    train_pos = _count_ideal_samples(train_samples, train_labels)
    valid_pos = _count_ideal_samples(valid_samples, valid_labels)
    print(
        f"026 C3 stride={args.stride} λ_part={part_w} lr={args.lr} "
        f"oversample={args.positive_oversample} cum_ic_min_ratio={args.cum_ic_min_ratio} "
        f"part_floor={args.part_auc_floor} train_ideal={train_pos} valid_ideal={valid_pos}"
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
            use_participation_attn=True,
            leg_align_horizons=ab["leg_align_horizons"],
        )
    ).to(device)

    init_ckpt = Path(args.init_checkpoint)
    if not init_ckpt.is_file():
        raise FileNotFoundError(f"missing init checkpoint: {init_ckpt}")
    ck = load_checkpoint(init_ckpt, map_location=device)
    model.load_state_dict(ck["model"], strict=False)
    print(f"loaded init checkpoint: {init_ckpt}")

    n_frozen = freeze_legacy_market_state_heads(model)
    for p in model.auto_encoder.parameters():
        p.requires_grad = False
    for p in model.trunk.parameters():
        p.requires_grad = False
    print(f"frozen legacy heads ({n_frozen} tensors); trunk+encoder frozen")

    train_params = collect_leg_align_head_params(model)
    if not train_params:
        raise RuntimeError("no trainable ParticipationAttn parameters")
    print(f"training {len(train_params)} C3 tensors")
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
    baseline_dir_acc = float(init_metrics.get("direction_acc", 0.0))
    print(f"init valid baseline cum_return_ic={baseline_ic:.4f} part_auc={init_metrics.get('participation_auc', 0):.4f}")

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
            hz_12_weight=0.0,
            hz_24_weight=0.0,
            hz_48_weight=0.0,
            leg_dir_weight=0.0,
            base_loss_scale=0.0,
            drift_weight=0.0,
            teacher=None,
            encoder_aux_loss=False,
        )
        valid_m = evaluate_leg_align_market_state(
            model,
            valid_loader,
            device,
            participation_weight=part_w,
        )
        row = {"epoch": epoch, "train_loss": tr.loss, **{f"valid_{k}": v for k, v in valid_m.items()}}
        history.append(row)
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
            f"epoch {epoch:02d} loss={tr.loss:.4f} "
            f"part_auc={part_score:.4f} cum_ic={cum_ic:.4f} "
            f"ic_deg={ic_deg*100:.1f}% score={score:.4f} "
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
                    "args": {**vars(args), "use_participation_attn": True, "experiment": "026_phase0_c3"},
                    "metrics": valid_m,
                    "constraint": cargs,
                    "baseline_cum_return_ic": baseline_ic,
                    "baseline_direction_acc": baseline_dir_acc,
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
    else:
        print("warning: no checkpoint saved; using last epoch weights")
        save_checkpoint(
            best_path,
            {
                "model": model.state_dict(),
                "args": {**vars(args), "use_participation_attn": True, "experiment": "026_phase0_c3"},
                "metrics": valid_m,
                "constraint": cargs,
                "baseline_cum_return_ic": baseline_ic,
            },
        )

    test_m = evaluate_leg_align_market_state(model, test_loader, device)
    valid_m = evaluate_leg_align_market_state(model, valid_loader, device)
    ic_deg_valid = 0.0 if baseline_ic <= 0 else max(0.0, (baseline_ic - valid_m["cum_return_ic"]) / baseline_ic)

    out = {
        "experiment": "026_phase0_c3",
        "best_epoch": best_epoch,
        "valid_best_score": best_score,
        "valid_participation_auc": valid_m.get("participation_auc"),
        "valid_cum_return_ic": valid_m.get("cum_return_ic"),
        "valid_ic_degradation": ic_deg_valid,
        "baseline_cum_return_ic": baseline_ic,
        "constraint": cargs,
        "recipe": str(Path(args.recipe)),
        "init_checkpoint": _repo_rel(init_ckpt),
        "tuning": {
            "lr": args.lr,
            "participation_weight": part_w,
            "cum_ic_min_ratio": args.cum_ic_min_ratio,
            "part_auc_floor": args.part_auc_floor,
            "composite_part_weight": args.composite_part_weight,
            "composite_ic_weight": args.composite_ic_weight,
        },
        "history": history,
        "test_metrics": test_m,
        "checkpoint": _repo_rel(best_path),
    }
    (report_dir / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved checkpoint: {best_path}")
    print(
        f"valid part_auc={valid_m.get('participation_auc', 0):.4f} "
        f"ic_deg={ic_deg_valid*100:.1f}% test part_auc={test_m.get('participation_auc', 0):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
