#!/usr/bin/env python3
"""Icon-style demo for market-state prediction outputs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

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
from transformer_kit.causal_transformer import CausalTransformerConfig
from transformer_kit.labels import estimate_market_state_thresholds
from transformer_kit.pattern_encoder import pattern_config_from_args
from transformer_kit.pattern_model import KlinePatternPredictor, MarketStateOutput, PatternPredictorConfig
from transformer_kit.segment_dataset import PatternSequenceDataset, SequenceSampleIndex, build_sequence_sample_indices
from transformer_kit.train_utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Icon-based market-state prediction demo")
    add_data_args(p)
    add_feature_args(p)
    add_segment_args(p)
    add_vq_args(p)
    p.add_argument("--checkpoint", default="checkpoints/0062c_market_state_cum_return_stabilized/market_state_best.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--pred-horizon", type=int, default=5)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--trunk-layers", type=int, default=2)
    p.add_argument("--direction-threshold-quantile", type=float, default=0.25)
    p.add_argument("--risk-threshold-quantile", type=float, default=0.70)
    p.add_argument("--split", choices=["train", "valid", "test"], default="test")
    p.add_argument("--num-samples", type=int, default=12)
    p.add_argument("--output-dir", default="reports/icon_demo")
    p.add_argument("--output-name", default="market_state_icon_demo.png")
    p.add_argument("--json-name", default="market_state_icon_demo.json")
    p.add_argument("--dpi", type=int, default=160)
    return p.parse_args()


def _merge_ckpt_args(cli: argparse.Namespace, ckpt_args: dict) -> argparse.Namespace:
    merged = vars(cli).copy()
    for k, v in ckpt_args.items():
        if k in {
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
        }:
            merged[k] = v
    return SimpleNamespace(**merged)


def _split_samples(
    bundle,
    *,
    context_bars: int,
    pred_horizon: int,
    stride: int,
) -> tuple[list[SequenceSampleIndex], list[SequenceSampleIndex], list[SequenceSampleIndex]]:
    def split(idx: np.ndarray) -> list[SequenceSampleIndex]:
        return build_sequence_sample_indices(
            bundle.bars.shape[0],
            context_bars=context_bars,
            pred_horizon=pred_horizon,
            stride=stride,
            index_min=int(idx.min()),
            index_max=int(idx.max()),
        )

    return split(bundle.train_idx), split(bundle.valid_idx), split(bundle.test_idx)


def _collect_future_windows(raw_log_ret: np.ndarray, samples: list[SequenceSampleIndex]) -> np.ndarray:
    rows = [raw_log_ret[s.context_end : s.future_end].astype(np.float32) for s in samples]
    return np.stack(rows, axis=0)


def _pick_samples(samples: list[SequenceSampleIndex], num_samples: int, seed: int) -> list[SequenceSampleIndex]:
    if num_samples >= len(samples):
        return samples
    rng = np.random.default_rng(seed)
    picked = sorted(rng.choice(len(samples), size=num_samples, replace=False).tolist())
    return [samples[i] for i in picked]


def _time_col(df) -> str | None:
    for name in ("time", "open_time", "timestamp", "datetime"):
        if name in df.columns:
            return name
    return None


def _fmt_time(df, idx: int) -> str:
    col = _time_col(df)
    if col is None:
        return str(idx)
    return str(df.iloc[idx][col])


def _dir_icon(cls: int) -> str:
    if cls == 2:
        return "▲"
    if cls == 0:
        return "▼"
    return "■"


def _risk_icon(v: int) -> str:
    return "⚠" if int(v) == 1 else "◇"


@dataclass
class DemoRow:
    anchor_time: str
    true_direction_icons: str
    pred_direction_icons: str
    step_match_icons: str
    true_risk_icons: str
    pred_risk_icons: str
    true_cum_return: float
    pred_cum_return: float
    true_cum_dir_icon: str
    pred_cum_dir_icon: str


def _build_rows(df, samples: list[SequenceSampleIndex], ds: PatternSequenceDataset, model: KlinePatternPredictor, device: torch.device) -> list[DemoRow]:
    rows: list[DemoRow] = []
    model.eval()
    with torch.no_grad():
        for i, spec in enumerate(samples):
            item = ds[i]
            ctx = item["ctx_bars"].unsqueeze(0).to(device)
            ctx_len = item["ctx_lengths"].unsqueeze(0).to(device)
            out = model(ctx, ctx_len)
            if not isinstance(out, MarketStateOutput):
                raise RuntimeError("expected MarketStateOutput")
            true_dir = item["target_direction"].numpy().astype(int)
            pred_dir = out.direction_logits.argmax(dim=-1).squeeze(0).cpu().numpy().astype(int)
            true_risk = item["target_risk"].numpy().astype(int)
            pred_risk = out.risk_logits.argmax(dim=-1).squeeze(0).cpu().numpy().astype(int)
            true_ret = item["target_return"].numpy().astype(np.float32)
            pred_ret = out.return_pred.squeeze(0).cpu().numpy().astype(np.float32)
            pred_cum_ret = (
                float(out.cum_return_pred.squeeze(0).item())
                if out.cum_return_pred is not None
                else float(pred_ret.sum())
            )
            true_cum_ret = float(true_ret.sum())
            if out.cum_direction_logit is not None:
                pred_cum_dir = 1 if float(out.cum_direction_logit.squeeze(0).item()) > 0 else 0
            else:
                pred_cum_dir = 1 if float(pred_ret.sum()) > 0 else 0
            true_cum_dir = 1 if true_cum_ret > 0 else 0
            rows.append(
                DemoRow(
                    anchor_time=_fmt_time(df, spec.context_end - 1),
                    true_direction_icons=" ".join(_dir_icon(int(x)) for x in true_dir),
                    pred_direction_icons=" ".join(_dir_icon(int(x)) for x in pred_dir),
                    step_match_icons=" ".join("✓" if int(a) == int(b) else "✗" for a, b in zip(true_dir, pred_dir)),
                    true_risk_icons=" ".join(_risk_icon(int(x)) for x in true_risk),
                    pred_risk_icons=" ".join(_risk_icon(int(x)) for x in pred_risk),
                    true_cum_return=true_cum_ret,
                    pred_cum_return=pred_cum_ret,
                    true_cum_dir_icon="↑" if true_cum_dir == 1 else "↓",
                    pred_cum_dir_icon="↑" if pred_cum_dir == 1 else "↓",
                )
            )
    return rows


def _render_icon_board(rows: list[DemoRow], out_path: Path, *, title: str, dpi: int) -> None:
    if not rows:
        raise ValueError("no demo rows")
    n = len(rows)
    fig_h = max(6.0, 1.0 + 0.62 * n)
    fig, ax = plt.subplots(figsize=(18, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n + 2)
    ax.axis("off")

    headers = [
        ("Anchor Time", 0.01),
        ("True Direction(H)", 0.19),
        ("Pred Direction(H)", 0.39),
        ("Step Match", 0.58),
        ("True Risk(H)", 0.69),
        ("Pred Risk(H)", 0.78),
        ("Cum Return T/P", 0.87),
        ("Cum Dir T/P", 0.96),
    ]
    for text, x in headers:
        ax.text(x, n + 1.3, text, fontsize=10, fontweight="bold", ha="center", va="center")

    for i, row in enumerate(rows):
        y = n - i + 0.2
        bg = "#f7f7f7" if i % 2 == 0 else "#ffffff"
        ax.add_patch(plt.Rectangle((0.0, y - 0.35), 1.0, 0.7, color=bg, ec="none"))
        ax.text(0.01, y, row.anchor_time, fontsize=9, ha="left", va="center")
        ax.text(0.19, y, row.true_direction_icons, fontsize=11, ha="center", va="center")
        ax.text(0.39, y, row.pred_direction_icons, fontsize=11, ha="center", va="center")
        ax.text(0.58, y, row.step_match_icons, fontsize=11, ha="center", va="center")
        ax.text(0.69, y, row.true_risk_icons, fontsize=11, ha="center", va="center")
        ax.text(0.78, y, row.pred_risk_icons, fontsize=11, ha="center", va="center")
        ax.text(0.87, y, f"{row.true_cum_return:+.4f}/{row.pred_cum_return:+.4f}", fontsize=9, ha="center", va="center")
        ax.text(0.96, y, f"{row.true_cum_dir_icon} {row.pred_cum_dir_icon}", fontsize=11, ha="center", va="center")

    ax.text(0.01, n + 1.75, title, fontsize=13, fontweight="bold", ha="left", va="center")
    ax.text(
        0.01,
        0.35,
        "Legend: Direction ▲ up | ■ flat | ▼ down; Risk ⚠ high-risk | ◇ normal; Match ✓/✗",
        fontsize=9,
        ha="left",
        va="center",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    apply_real_data_defaults(args)
    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    merged_args = _merge_ckpt_args(args, ckpt_args if isinstance(ckpt_args, dict) else {})

    np.random.seed(merged_args.seed)
    torch.manual_seed(merged_args.seed)
    out_dir = Path(merged_args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = fetch_ohlcv_df(merged_args)
    bundle = prepare_bar_series_from_args(df, merged_args)
    train_samples, valid_samples, test_samples = _split_samples(
        bundle,
        context_bars=merged_args.context_bars,
        pred_horizon=merged_args.pred_horizon,
        stride=merged_args.stride,
    )
    thr = estimate_market_state_thresholds(
        _collect_future_windows(bundle.raw_log_ret, train_samples),
        direction_quantile=merged_args.direction_threshold_quantile,
        risk_quantile=merged_args.risk_threshold_quantile,
    )

    split_map = {"train": train_samples, "valid": valid_samples, "test": test_samples}
    selected_specs = _pick_samples(split_map[merged_args.split], merged_args.num_samples, merged_args.seed)
    ds = PatternSequenceDataset(
        bundle.bars,
        selected_specs,
        bundle.raw_log_ret,
        zscore_window=bundle.zscore_window,
        return_market_state_targets=True,
        direction_threshold=thr.direction_threshold,
        risk_vol_threshold=thr.risk_vol_threshold,
    )

    auto_cfg = pattern_config_from_args(merged_args)
    model = KlinePatternPredictor(
        PatternPredictorConfig(
            auto_segment=auto_cfg,
            trunk=CausalTransformerConfig(
                d_model=merged_args.d_model,
                n_heads=merged_args.n_heads,
                n_layers=merged_args.trunk_layers,
            ),
            pred_horizon=merged_args.pred_horizon,
            pred_feat_dim=1,
            pool_mode="attn",
            learnable_scale=True,
            use_horizon_head=False,
            use_market_state_head=True,
            direction_classes=getattr(merged_args, "direction_classes", 3),
            risk_classes=getattr(merged_args, "risk_classes", 2),
            use_cum_heads=getattr(merged_args, "use_cum_heads", True),
            use_horizon_return_head=getattr(merged_args, "use_horizon_return_head", True),
            detach_risk_vol_heads=getattr(merged_args, "detach_risk_vol_heads", False),
            return_direction_hidden_mult=getattr(merged_args, "return_direction_hidden_mult", 1.0),
        )
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)

    rows = _build_rows(df, selected_specs, ds, model, device)
    png_path = out_dir / merged_args.output_name
    title = (
        f"Market-State Icon Demo  |  split={merged_args.split}  samples={len(rows)}  "
        f"symbol={merged_args.symbol} {merged_args.interval}"
    )
    _render_icon_board(rows, png_path, title=title, dpi=merged_args.dpi)

    payload = {
        "checkpoint": str(ckpt_path),
        "split": merged_args.split,
        "symbol": merged_args.symbol,
        "interval": merged_args.interval,
        "num_samples": len(rows),
        "direction_threshold": float(thr.direction_threshold),
        "risk_vol_threshold": float(thr.risk_vol_threshold),
        "rows": [asdict(r) for r in rows],
    }
    json_path = out_dir / merged_args.json_name
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved icon board: {png_path}")
    print(f"saved details: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

