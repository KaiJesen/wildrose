#!/usr/bin/env python3
"""Plot candlestick with model prediction signals under x-axis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

_EX = Path(__file__).resolve().parent
_ROOT = _EX.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))

from _train_common import add_data_args, add_feature_args, add_segment_args, add_vq_args, apply_real_data_defaults, fetch_ohlcv_df, prepare_bar_series_from_args
from market_data.schema import COL_CLOSE, COL_HIGH, COL_LOW, COL_OPEN, COL_TIME
from transformer_kit.causal_transformer import CausalTransformerConfig
from transformer_kit.labels import estimate_market_state_thresholds
from transformer_kit.pattern_encoder import pattern_config_from_args
from transformer_kit.pattern_model import KlinePatternPredictor, MarketStateOutput, PatternPredictorConfig
from transformer_kit.segment_dataset import build_sequence_sample_indices
from transformer_kit.train_utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot K-line with market-state prediction signals")
    add_data_args(p)
    add_feature_args(p)
    add_segment_args(p)
    add_vq_args(p)
    p.add_argument("--checkpoint", default="checkpoints/0062c_market_state_cum_return_stabilized/market_state_best.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--pred-horizon", type=int, default=5)
    p.add_argument("--trunk-layers", type=int, default=2)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--split", choices=["train", "valid", "test"], default="test")
    p.add_argument("--window-bars", type=int, default=160)
    p.add_argument("--start-index", type=int, default=-1, help="absolute bar index; -1 auto-select from split center")
    p.add_argument("--direction-threshold-quantile", type=float, default=0.25)
    p.add_argument("--risk-threshold-quantile", type=float, default=0.70)
    p.add_argument("--output-dir", default="reports/kline_signal_demo")
    p.add_argument("--output-name", default="kline_with_market_state_signals.png")
    p.add_argument("--json-name", default="kline_with_market_state_signals.json")
    p.add_argument("--dpi", type=int, default=160)
    return p.parse_args()


def _merge_ckpt_args(cli: argparse.Namespace, ckpt_args: dict) -> argparse.Namespace:
    merged = vars(cli).copy()
    keep = {
        "d_model",
        "n_heads",
        "encoder_layers",
        "context_bars",
        "max_seg_len",
        "max_segments",
        "min_seg_len",
        "num_codes",
        "vq_beta",
        "vq_inverse_freq_ema",
        "pred_horizon",
        "trunk_layers",
        "trend_features",
        "trend_windows",
        "use_cum_heads",
        "use_horizon_return_head",
        "detach_risk_vol_heads",
        "return_direction_hidden_mult",
        "direction_classes",
        "risk_classes",
        "direction_threshold_quantile",
        "risk_threshold_quantile",
    }
    for k, v in ckpt_args.items():
        if k in keep:
            merged[k] = v
    return SimpleNamespace(**merged)


def _split_idx(bundle, split: str) -> np.ndarray:
    if split == "train":
        return bundle.train_idx
    if split == "valid":
        return bundle.valid_idx
    return bundle.test_idx


def _build_model(args: argparse.Namespace, ckpt_model: dict, device: torch.device) -> KlinePatternPredictor:
    auto_cfg = pattern_config_from_args(args)
    model = KlinePatternPredictor(
        PatternPredictorConfig(
            auto_segment=auto_cfg,
            trunk=CausalTransformerConfig(
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.trunk_layers,
            ),
            pred_horizon=args.pred_horizon,
            pred_feat_dim=1,
            pool_mode="attn",
            learnable_scale=True,
            use_horizon_head=False,
            use_market_state_head=True,
            direction_classes=getattr(args, "direction_classes", 3),
            risk_classes=getattr(args, "risk_classes", 2),
            use_cum_heads=getattr(args, "use_cum_heads", True),
            use_horizon_return_head=getattr(args, "use_horizon_return_head", True),
            detach_risk_vol_heads=getattr(args, "detach_risk_vol_heads", False),
            return_direction_hidden_mult=getattr(args, "return_direction_hidden_mult", 1.0),
        )
    ).to(device)
    model.load_state_dict(ckpt_model, strict=False)
    model.eval()
    return model


def _draw_candles(ax, times, open_, high, low, close) -> None:
    x = mdates.date2num(times)
    width = 0.03 if len(x) < 2 else min(0.03, (x[1] - x[0]) * 0.65)
    for xi, o, h, l, c in zip(x, open_, high, low, close):
        up = c >= o
        color = "#2ca02c" if up else "#d62728"
        ax.vlines(xi, l, h, color=color, linewidth=1.0, alpha=0.85)
        bottom = min(o, c)
        height = max(abs(c - o), max(float(np.mean(close)) * 1e-5, 1e-8))
        ax.add_patch(
            Rectangle(
                (xi - width / 2, bottom),
                width,
                height,
                facecolor=color,
                edgecolor=color,
                alpha=0.8,
            )
        )


def _direction_3class(step_ret: float, threshold: float) -> int:
    if step_ret > threshold:
        return 2
    if step_ret < -threshold:
        return 0
    return 1


def _choose_window_start(
    split_idx: np.ndarray,
    *,
    context_bars: int,
    bars_len: int,
    window_bars: int,
    start_index: int,
) -> int:
    if start_index >= 0:
        start = start_index
    else:
        center = int(split_idx[len(split_idx) // 2])
        start = center - window_bars // 2
    start = max(context_bars + 1, start)
    end = min(bars_len - 1, start + window_bars - 1)
    start = max(context_bars + 1, end - window_bars + 1)
    return start


@torch.no_grad()
def _predict_window(
    model: KlinePatternPredictor,
    bars: np.ndarray,
    raw_log_ret: np.ndarray,
    *,
    start: int,
    end: int,
    context_bars: int,
    direction_threshold: float,
    risk_vol_threshold: float,
    device: torch.device,
) -> dict[str, list]:
    times_idx: list[int] = []
    pred_dir: list[int] = []
    true_dir: list[int] = []
    correct_bin: list[int] = []
    pred_risk: list[int] = []
    true_risk: list[int] = []
    pred_up_prob: list[float] = []
    pred_risk_pos_prob: list[float] = []
    for target_idx in range(start, end + 1):
        anchor = target_idx
        ctx = torch.from_numpy(bars[anchor - context_bars : anchor].astype(np.float32)).unsqueeze(0).to(device)
        ctx_len = torch.tensor([context_bars], dtype=torch.long, device=device)
        out = model(ctx, ctx_len)
        if not isinstance(out, MarketStateOutput):
            raise RuntimeError("model output is not MarketStateOutput")
        dir_prob = torch.softmax(out.direction_logits[0, 0], dim=-1)
        risk_prob = torch.softmax(out.risk_logits[0, 0], dim=-1)
        p_dir = int(dir_prob.argmax().item())
        p_risk = int(risk_prob.argmax().item())
        ret1 = float(raw_log_ret[target_idx])
        t_dir = _direction_3class(ret1, direction_threshold)
        t_risk = 1 if abs(ret1) >= risk_vol_threshold else 0
        p_bin = 1 if p_dir == 2 else 0
        t_bin = 1 if ret1 > 0 else 0
        times_idx.append(target_idx)
        pred_dir.append(p_dir)
        true_dir.append(t_dir)
        correct_bin.append(1 if p_bin == t_bin else 0)
        pred_risk.append(p_risk)
        true_risk.append(t_risk)
        pred_up_prob.append(float(dir_prob[2].item()))
        pred_risk_pos_prob.append(float(risk_prob[1].item()))
    return {
        "bar_index": times_idx,
        "pred_dir": pred_dir,
        "true_dir": true_dir,
        "correct_bin": correct_bin,
        "pred_risk": pred_risk,
        "true_risk": true_risk,
        "pred_up_prob": pred_up_prob,
        "pred_risk_pos_prob": pred_risk_pos_prob,
    }


def _plot_signals(df, sig: dict[str, list], *, start: int, end: int, out_png: Path, dpi: int) -> dict[str, float]:
    view = df.iloc[start : end + 1]
    times = view[COL_TIME].to_numpy()
    open_ = view[COL_OPEN].to_numpy(dtype=np.float64)
    high = view[COL_HIGH].to_numpy(dtype=np.float64)
    low = view[COL_LOW].to_numpy(dtype=np.float64)
    close = view[COL_CLOSE].to_numpy(dtype=np.float64)

    idx_to_pos = {idx: i for i, idx in enumerate(range(start, end + 1))}
    x = [times[idx_to_pos[i]] for i in sig["bar_index"] if i in idx_to_pos]
    pdir = np.array([sig["pred_dir"][k] for k, i in enumerate(sig["bar_index"]) if i in idx_to_pos], dtype=np.int64)
    tdir = np.array([sig["true_dir"][k] for k, i in enumerate(sig["bar_index"]) if i in idx_to_pos], dtype=np.int64)
    corr = np.array([sig["correct_bin"][k] for k, i in enumerate(sig["bar_index"]) if i in idx_to_pos], dtype=np.int64)
    prisk = np.array([sig["pred_risk"][k] for k, i in enumerate(sig["bar_index"]) if i in idx_to_pos], dtype=np.int64)
    trisk = np.array([sig["true_risk"][k] for k, i in enumerate(sig["bar_index"]) if i in idx_to_pos], dtype=np.int64)
    up_prob = np.array([sig["pred_up_prob"][k] for k, i in enumerate(sig["bar_index"]) if i in idx_to_pos], dtype=np.float64)
    risk_prob = np.array([sig["pred_risk_pos_prob"][k] for k, i in enumerate(sig["bar_index"]) if i in idx_to_pos], dtype=np.float64)

    fig, axes = plt.subplots(
        6,
        1,
        figsize=(18, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [5, 1, 1, 1, 1, 1.6], "hspace": 0.05},
    )
    ax_k, ax_dir_pred, ax_dir_true, ax_hit, ax_risk, ax_prob = axes
    _draw_candles(ax_k, times, open_, high, low, close)
    ax_k.set_ylabel("Price")
    ax_k.grid(True, alpha=0.2)

    y_dir_pred = np.zeros(len(x), dtype=np.float64)
    for cls, marker, color, label in [
        (2, "^", "#2ca02c", "Pred Up"),
        (0, "v", "#d62728", "Pred Down"),
        (1, "o", "#7f7f7f", "Pred Flat"),
    ]:
        m = pdir == cls
        if m.any():
            ax_dir_pred.scatter(np.array(x)[m], y_dir_pred[m], s=70, marker=marker, c=color, label=label)
    ax_dir_pred.set_yticks([])
    ax_dir_pred.set_ylabel("PredDir")
    ax_dir_pred.grid(True, alpha=0.15)
    ax_dir_pred.legend(loc="upper left", ncol=3, fontsize=8)

    y_dir_true = np.zeros(len(x), dtype=np.float64)
    for cls, marker, color, label in [
        (2, "^", "#2ca02c", "True Up"),
        (0, "v", "#d62728", "True Down"),
        (1, "o", "#7f7f7f", "True Flat"),
    ]:
        m = tdir == cls
        if m.any():
            ax_dir_true.scatter(np.array(x)[m], y_dir_true[m], s=60, marker=marker, c=color, label=label, alpha=0.9)
    ax_dir_true.set_yticks([])
    ax_dir_true.set_ylabel("TrueDir")
    ax_dir_true.grid(True, alpha=0.15)
    ax_dir_true.legend(loc="upper left", ncol=3, fontsize=8)

    y_hit = np.zeros(len(x), dtype=np.float64)
    for ok, marker, color, label in [(1, "o", "#2ca02c", "Correct"), (0, "x", "#d62728", "Wrong")]:
        m = corr == ok
        if m.any():
            ax_hit.scatter(np.array(x)[m], y_hit[m], s=60, marker=marker, c=color, label=label)
    ax_hit.set_yticks([])
    ax_hit.set_ylabel("Hit")
    ax_hit.grid(True, alpha=0.15)
    ax_hit.legend(loc="upper left", ncol=2, fontsize=8)

    y_risk = np.zeros(len(x), dtype=np.float64)
    for rv, marker, color, label in [(1, "s", "#ff7f0e", "Pred Risk=1"), (0, "s", "#1f77b4", "Pred Risk=0")]:
        m = prisk == rv
        if m.any():
            ax_risk.scatter(np.array(x)[m], y_risk[m] + 0.12, s=55, marker=marker, c=color, label=label)
    for rv, marker, color, label in [(1, "x", "#c75d00", "True Risk=1"), (0, "x", "#004c99", "True Risk=0")]:
        m = trisk == rv
        if m.any():
            ax_risk.scatter(np.array(x)[m], y_risk[m] - 0.12, s=45, marker=marker, c=color, label=label, alpha=0.8)
    ax_risk.plot(x, np.zeros(len(x)), color="#999999", linewidth=0.8, alpha=0.45)
    ax_risk.set_yticks([])
    ax_risk.set_ylabel("Risk")
    ax_risk.grid(True, alpha=0.15)
    ax_risk.legend(loc="upper left", ncol=4, fontsize=8)

    ax_prob.plot(x, up_prob, color="#2ca02c", linewidth=1.3, label="P(dir=up)")
    ax_prob.plot(x, risk_prob, color="#ff7f0e", linewidth=1.3, label="P(risk=1)")
    ax_prob.axhline(0.5, color="#888888", linestyle="--", linewidth=0.9, alpha=0.7)
    ax_prob.set_ylim(-0.02, 1.02)
    ax_prob.set_ylabel("Prob")
    ax_prob.grid(True, alpha=0.2)
    ax_prob.legend(loc="upper left", ncol=2, fontsize=8)
    ax_prob.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

    hit_rate = float(corr.mean()) if corr.size else 0.0
    risk_match = float((prisk == trisk).mean()) if prisk.size else 0.0
    risk_pred_pos = float(prisk.mean()) if prisk.size else 0.0
    risk_true_pos = float(trisk.mean()) if trisk.size else 0.0
    fig.suptitle(
        f"K-line + Prediction Signals  |  hit={hit_rate:.1%}  risk_match={risk_match:.1%}  "
        f"risk_pred/true={risk_pred_pos:.2f}/{risk_true_pos:.2f}  "
        f"P(up)={float(up_prob.mean()):.2f}  P(risk=1)={float(risk_prob.mean()):.2f}",
        y=0.995,
    )
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {
        "hit_rate": hit_rate,
        "risk_match_rate": risk_match,
        "risk_pred_positive_rate": risk_pred_pos,
        "risk_true_positive_rate": risk_true_pos,
        "direction_pred_flat_rate": float((pdir == 1).mean()) if pdir.size else 0.0,
        "direction_true_flat_rate": float((tdir == 1).mean()) if tdir.size else 0.0,
        "pred_up_prob_mean": float(up_prob.mean()) if up_prob.size else 0.0,
        "pred_up_prob_std": float(up_prob.std()) if up_prob.size else 0.0,
        "pred_risk_pos_prob_mean": float(risk_prob.mean()) if risk_prob.size else 0.0,
        "pred_risk_pos_prob_std": float(risk_prob.std()) if risk_prob.size else 0.0,
        "num_points": int(len(x)),
    }


def main() -> int:
    args = parse_args()
    apply_real_data_defaults(args)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    merged = _merge_ckpt_args(args, ckpt_args if isinstance(ckpt_args, dict) else {})
    df = fetch_ohlcv_df(merged)
    bundle = prepare_bar_series_from_args(df, merged)

    train_samples = build_sequence_sample_indices(
        bundle.bars.shape[0],
        context_bars=merged.context_bars,
        pred_horizon=merged.pred_horizon,
        stride=merged.stride,
        index_min=int(bundle.train_idx.min()),
        index_max=int(bundle.train_idx.max()),
    )
    train_future = np.stack([bundle.raw_log_ret[s.context_end : s.future_end].astype(np.float32) for s in train_samples], axis=0)
    thr = estimate_market_state_thresholds(
        train_future,
        direction_quantile=merged.direction_threshold_quantile,
        risk_quantile=merged.risk_threshold_quantile,
    )

    model = _build_model(merged, ckpt["model"], device)
    split_idx = _split_idx(bundle, merged.split)
    start = _choose_window_start(
        split_idx,
        context_bars=merged.context_bars,
        bars_len=bundle.bars.shape[0],
        window_bars=merged.window_bars,
        start_index=merged.start_index,
    )
    end = min(bundle.bars.shape[0] - 1, start + merged.window_bars - 1)

    sig = _predict_window(
        model,
        bundle.bars,
        bundle.raw_log_ret,
        start=start,
        end=end,
        context_bars=merged.context_bars,
        direction_threshold=float(thr.direction_threshold),
        risk_vol_threshold=float(thr.risk_vol_threshold),
        device=device,
    )

    png_path = out_dir / merged.output_name
    metric = _plot_signals(df, sig, start=start, end=end, out_png=png_path, dpi=merged.dpi)
    payload = {
        "checkpoint": str(merged.checkpoint),
        "split": merged.split,
        "start_index": int(start),
        "end_index": int(end),
        "start_time": str(df.iloc[start][COL_TIME]),
        "end_time": str(df.iloc[end][COL_TIME]),
        "direction_threshold": float(thr.direction_threshold),
        "risk_vol_threshold": float(thr.risk_vol_threshold),
        "metrics": metric,
        "signals": sig,
    }
    json_path = out_dir / merged.json_name
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved chart: {png_path}")
    print(f"saved json: {json_path}")
    print(
        f"window={start}-{end} hit={metric['hit_rate']:.1%} "
        f"risk_match={metric['risk_match_rate']:.1%} points={metric['num_points']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

