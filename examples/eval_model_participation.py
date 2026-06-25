#!/usr/bin/env python3
"""024 model-track KPI evaluation (labels + optional checkpoint inference)."""

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

from _train_common import (
    add_data_args,
    add_feature_args,
    add_segment_args,
    add_vq_args,
    apply_real_data_defaults,
    fetch_ohlcv_df,
    prepare_bar_series_from_args,
)
from trading_system.leg_participation_labels import count_confirmed_legs_from_bars, label_summary
from trading_system.participation import compute_participation_metrics
from transformer_kit.causal_transformer import CausalTransformerConfig
from transformer_kit.labels import estimate_market_state_thresholds
from transformer_kit.leg_align_dataset import LegParticipationSequenceDataset, load_label_dataframe
from transformer_kit.pattern_encoder import pattern_config_from_args
from transformer_kit.pattern_model import KlinePatternPredictor, PatternPredictorConfig
from transformer_kit.segment_dataset import build_sequence_sample_indices
from transformer_kit.train_utils import load_checkpoint
from transformer_kit.training import evaluate_leg_align_market_state

MODEL_METRIC_KEYS = (
    "participation_auc",
    "participation_auc_long",
    "participation_auc_short",
    "confirmed_leg_flat_edge_p50_long",
    "confirmed_leg_flat_edge_p50_short",
    "leg_entry_recall_at_k",
    "hz_direction_acc_24",
    "cum_return_ic",
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _leg_count_deviation(label_legs: int, participation_legs: int) -> float:
    denom = max(1, participation_legs)
    return abs(label_legs - participation_legs) / denom


def _split_idx(bundle, split: str):
    if split == "train":
        return bundle.train_idx
    if split == "valid":
        return bundle.valid_idx
    return bundle.test_idx


def _build_model_from_checkpoint(checkpoint: Path, device: torch.device) -> KlinePatternPredictor:
    ck = load_checkpoint(checkpoint, map_location=device)
    ck_args = ck.get("args", {}) if isinstance(ck.get("args"), dict) else {}
    ns = argparse.Namespace(**{k: v for k, v in ck_args.items()})
    if not hasattr(ns, "d_model"):
        ns.d_model = 128
    if not hasattr(ns, "n_heads"):
        ns.n_heads = 4
    if not hasattr(ns, "trunk_layers"):
        ns.trunk_layers = 2
    if not hasattr(ns, "context_bars"):
        ns.context_bars = 128
    if not hasattr(ns, "pred_horizon"):
        ns.pred_horizon = 5
    if not hasattr(ns, "max_seg_len"):
        ns.max_seg_len = 32
    if not hasattr(ns, "max_segments"):
        ns.max_segments = 16
    if not hasattr(ns, "min_seg_len"):
        ns.min_seg_len = 4
    if not hasattr(ns, "num_codes"):
        ns.num_codes = 16
    if not hasattr(ns, "vq_beta"):
        ns.vq_beta = 1.0
    if not hasattr(ns, "vq_inverse_freq_ema"):
        ns.vq_inverse_freq_ema = False
    if not hasattr(ns, "trend_features"):
        ns.trend_features = True
    if not hasattr(ns, "trend_windows"):
        ns.trend_windows = [20, 60, 120]
    if not hasattr(ns, "variant"):
        ns.variant = "0"
    horizons = () if str(ns.variant) == "0" else (12, 24)
    auto_cfg = pattern_config_from_args(ns)
    model = KlinePatternPredictor(
        PatternPredictorConfig(
            auto_segment=auto_cfg,
            trunk=CausalTransformerConfig(d_model=ns.d_model, n_heads=ns.n_heads, n_layers=ns.trunk_layers),
            pred_horizon=ns.pred_horizon,
            pred_feat_dim=1,
            pool_mode="attn",
            learnable_scale=True,
            use_horizon_head=False,
            use_market_state_head=True,
            use_cum_heads=True,
            use_horizon_return_head=True,
            use_participation_heads=True,
            leg_align_horizons=horizons,
        )
    ).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    return model


def _leg_entry_recall_at_k(
    scores: np.ndarray,
    ideal: np.ndarray,
    *,
    k_frac: float = 0.05,
) -> float:
    ideal_idx = np.where(ideal >= 0.5)[0]
    if ideal_idx.size == 0:
        return 0.0
    k = max(1, int(len(scores) * k_frac))
    top = np.argsort(scores)[-k:]
    hit = len(set(top.tolist()) & set(ideal_idx.tolist()))
    return float(hit / ideal_idx.size)


def eval_split(
    *,
    split: str,
    labels_dir: Path,
    backtest_dir: Path | None,
    model_metrics: dict | None = None,
) -> dict:
    labels_path = labels_dir / f"leg_participation_{split}.csv"
    labels = _read_csv(labels_path)
    label_stats = label_summary(labels)
    label_leg_count = count_confirmed_legs_from_bars(labels)

    participation_leg_count = None
    participation_metrics: dict = {}
    leg_count_deviation = None
    leg_count_gate_pass = None

    if backtest_dir is not None and (backtest_dir / "decisions.csv").is_file():
        decisions = _read_csv(backtest_dir / "decisions.csv")
        trades = _read_csv(backtest_dir / "trades.csv") if (backtest_dir / "trades.csv").is_file() else pd.DataFrame()
        part = compute_participation_metrics(decisions, trades)
        participation_metrics = part.to_dict()
        participation_leg_count = int(part.leg_count)
        leg_count_deviation = _leg_count_deviation(label_leg_count, participation_leg_count)
        leg_count_gate_pass = leg_count_deviation < 0.02

    if model_metrics is None:
        model_metrics = {k: None for k in MODEL_METRIC_KEYS}
        model_metrics["phase"] = "0_label_only"

    return {
        "split": split,
        "label_summary": label_stats,
        "label_confirmed_leg_count": float(label_leg_count),
        "participation_leg_count": participation_leg_count,
        "leg_count_deviation_ratio": leg_count_deviation,
        "leg_count_alignment_gate_pass": leg_count_gate_pass,
        "participation_metrics": participation_metrics,
        "model_metrics": model_metrics,
        "labels_path": str(labels_path.relative_to(_ROOT)),
        "backtest_dir": str(backtest_dir.relative_to(_ROOT)) if backtest_dir else None,
    }


def eval_checkpoint_split(
    *,
    split: str,
    model: KlinePatternPredictor,
    bundle,
    label_df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    idx = _split_idx(bundle, split)
    samples = build_sequence_sample_indices(
        bundle.bars.shape[0],
        context_bars=args.context_bars,
        pred_horizon=args.pred_horizon,
        stride=args.stride,
        index_min=int(idx.min()),
        index_max=int(idx.max()),
    )
    train_idx = _split_idx(bundle, "train")
    train_samples = build_sequence_sample_indices(
        bundle.bars.shape[0],
        context_bars=args.context_bars,
        pred_horizon=args.pred_horizon,
        stride=args.stride,
        index_min=int(train_idx.min()),
        index_max=int(train_idx.max()),
    )
    windows = np.stack(
        [bundle.raw_log_ret[s.context_end : s.future_end].astype(np.float32) for s in train_samples],
        axis=0,
    )
    thr = estimate_market_state_thresholds(windows, direction_quantile=0.25, risk_quantile=0.70)
    horizons = () if str(getattr(args, "variant", "0")) == "0" else (12, 24)
    loader = DataLoader(
        LegParticipationSequenceDataset(
            bundle.bars,
            samples,
            bundle.raw_log_ret,
            label_df,
            zscore_window=bundle.zscore_window,
            direction_threshold=thr.direction_threshold,
            risk_vol_threshold=thr.risk_vol_threshold,
            leg_align_horizons=horizons,
        ),
        batch_size=64,
        shuffle=False,
    )
    metrics = evaluate_leg_align_market_state(model, loader, device)
    # leg_entry_recall_at_k from per-sample scores
    long_scores: list[float] = []
    long_ideal: list[float] = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["ctx_bars"].to(device), batch["ctx_lengths"].to(device))
            if out.participation_logit_long is None:
                continue
            pl = torch.sigmoid(out.participation_logit_long).cpu().numpy()
            confirmed = batch["is_leg_confirmed"].numpy() >= 0.5
            up = batch["align_direction_up"].numpy() >= 0.5
            ideal = batch["ideal_participate_long"].numpy()
            mask = confirmed & up
            long_scores.extend(pl[mask].tolist())
            long_ideal.extend(ideal[mask].tolist())
    recall_k = _leg_entry_recall_at_k(np.asarray(long_scores), np.asarray(long_ideal)) if long_scores else 0.0
    return {
        "participation_auc": metrics.get("participation_auc"),
        "participation_auc_long": metrics.get("participation_auc_long"),
        "participation_auc_short": metrics.get("participation_auc_short"),
        "confirmed_leg_flat_edge_p50_long": metrics.get("confirmed_leg_flat_edge_p50_long"),
        "confirmed_leg_flat_edge_p50_short": metrics.get("confirmed_leg_flat_edge_p50_short"),
        "leg_entry_recall_at_k": recall_k,
        "hz_direction_acc_24": metrics.get("hz_direction_acc_24"),
        "cum_return_ic": metrics.get("cum_return_ic"),
        "phase": "1_checkpoint",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate 024 model participation KPIs")
    add_data_args(p)
    add_feature_args(p)
    add_segment_args(p)
    add_vq_args(p)
    p.add_argument("--labels-dir", default="data/labels/leg_participation")
    p.add_argument("--split", action="append", default=[], choices=["train", "valid", "test"])
    p.add_argument("--backtest-dir", action="append", default=[])
    p.add_argument("--phase1c-backtest-root", default="backtest/v023_phase1c")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--pred-horizon", type=int, default=5)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--trunk-layers", type=int, default=2)
    p.add_argument("--variant", default="0")
    p.add_argument("--output", default="backtest/v024_phase0/eval_model_participation.json")
    p.add_argument("--device", default="cpu")
    p.set_defaults(source="binance_vision", symbol="BTCUSDT", interval="1h", days=365)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    apply_real_data_defaults(args)
    splits = args.split or ["valid", "test"]
    labels_dir = Path(args.labels_dir).resolve()
    phase = 1 if args.checkpoint else 0
    out: dict = {"splits": {}, "phase": phase, "checkpoint": args.checkpoint or None}

    model = None
    bundle = None
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    if args.checkpoint:
        ckpt = Path(args.checkpoint).resolve()
        df = fetch_ohlcv_df(args)
        bundle = prepare_bar_series_from_args(df, args)
        model = _build_model_from_checkpoint(ckpt, device)

    bt_dirs = [Path(d).resolve() for d in args.backtest_dir]
    phase1c_root = Path(args.phase1c_backtest_root).resolve()

    all_gate_pass = True
    for i, split in enumerate(splits):
        if i < len(bt_dirs):
            bt_dir = bt_dirs[i]
        else:
            bt_dir = (phase1c_root / split).resolve()
        model_metrics = None
        if model is not None and bundle is not None:
            label_df = load_label_dataframe(str(labels_dir / f"leg_participation_{split}.csv"))
            model_metrics = eval_checkpoint_split(
                split=split,
                model=model,
                bundle=bundle,
                label_df=label_df,
                args=args,
                device=device,
            )
            print(
                f"[{split}] part_auc={model_metrics.get('participation_auc', 0):.4f} "
                f"cum_ic={model_metrics.get('cum_return_ic', 0):.4f}"
            )
        result = eval_split(
            split=split,
            labels_dir=labels_dir,
            backtest_dir=bt_dir,
            model_metrics=model_metrics,
        )
        out["splits"][split] = result
        gate = result.get("leg_count_alignment_gate_pass")
        if gate is False:
            all_gate_pass = False
        if model_metrics is None:
            dev = result.get("leg_count_deviation_ratio")
            dev_s = f"{dev:.4f}" if dev is not None else "n/a"
            gate_s = "PASS" if gate else "FAIL" if gate is not None else "SKIP"
            print(
                f"[{split}] label_legs={int(result['label_confirmed_leg_count'])} "
                f"part_legs={result['participation_leg_count']} "
                f"deviation={dev_s} gate={gate_s}"
            )

    out["leg_count_alignment_all_pass"] = all_gate_pass
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {out_path}")
    return 0 if all_gate_pass or phase == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
