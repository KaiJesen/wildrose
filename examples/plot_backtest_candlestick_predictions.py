#!/usr/bin/env python3
"""滚动回测可视化：OHLC 蜡烛图 + 模型预测路径叠加。"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader

_EX = Path(__file__).resolve().parent
_ROOT = _EX.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))

from _train_common import add_data_args, add_segment_args, add_stage3_loss_args, add_vq_args, fetch_ohlcv_df
from market_data.schema import COL_CLOSE, COL_HIGH, COL_LOW, COL_OPEN, COL_TIME
from transformer_kit.causal_transformer import CausalTransformerConfig
from transformer_kit.pattern_encoder import pattern_config_from_args
from transformer_kit.pattern_model import KlinePatternPredictor, PatternPredictorConfig
from transformer_kit.segment_features import build_bar_shape_frame
from transformer_kit.segment_dataset import (
    PatternSequenceDataset,
    build_sequence_sample_indices,
    prepare_bar_series,
)
from transformer_kit.train_utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot rolling model predictions on BTC candlesticks")
    add_data_args(p)
    add_segment_args(p)
    add_vq_args(p)
    add_stage3_loss_args(p)
    p.add_argument("--pred-horizon", type=int, default=5)
    p.add_argument("--pred-feat-dim", type=int, default=1, help="预测未来特征维度；4 时可绘制预测蜡烛")
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--trunk-layers", type=int, default=2)
    p.add_argument("--checkpoint", default="checkpoints/real_btc_high_vol_top20_focus/stage3_predictor_best_combo.pt")
    p.add_argument("--output-dir", default="reports/shape_candlestick_comparison")
    p.add_argument("--chart-before", type=int, default=96)
    p.add_argument("--chart-after", type=int, default=96)
    p.add_argument("--predict-every", type=int, default=1)
    p.add_argument("--path-plot-every", type=int, default=4)
    p.add_argument("--display-lead", type=int, default=-1, help="主预测线使用第 N 步 ahead；默认 pred_horizon")
    p.add_argument("--calibration", choices=("none", "affine", "residual"), default="residual")
    p.add_argument("--calibration-ridge", type=float, default=1e-3)
    p.add_argument("--fusion", choices=("none", "mlp"), default="mlp")
    p.add_argument("--fusion-hidden", type=int, default=16)
    p.add_argument("--fusion-epochs", type=int, default=300)
    p.add_argument("--fusion-lr", type=float, default=1e-3)
    p.add_argument("--fusion-weight-decay", type=float, default=1e-4)
    p.add_argument("--fusion-augment-prob", type=float, default=0.3)
    p.add_argument("--fusion-augment-strength", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dpi", type=int, default=150)
    p.set_defaults(source="binance_vision", symbol="BTCUSDT", interval="1h", days=365)
    return p.parse_args()


@dataclass(frozen=True)
class ReturnCalibrator:
    """Normalized log_ret 后处理校准器，仅用 train/valid 因果特征拟合。"""

    mode: str
    scale: np.ndarray
    bias: np.ndarray
    coef: np.ndarray | None = None

    def apply(self, pred_norm: np.ndarray, ctx_feats: np.ndarray) -> np.ndarray:
        if self.mode == "none":
            return pred_norm
        if self.mode == "affine" or self.coef is None:
            return pred_norm * self.scale + self.bias
        h = pred_norm.shape[0]
        out = np.empty_like(pred_norm, dtype=np.float32)
        pred_cum = np.cumsum(pred_norm)
        for t in range(h):
            x = np.concatenate([[1.0, pred_norm[t], pred_cum[t]], ctx_feats]).astype(np.float64)
            out[t] = float(x @ self.coef[t])
        return out


@dataclass(frozen=True)
class ShapeCalibrator:
    """Normalized body/upper/lower 后处理校准器。"""

    scale: np.ndarray
    bias: np.ndarray

    def apply(self, pred_shape: np.ndarray) -> np.ndarray:
        if pred_shape.shape[-1] == 0:
            return pred_shape
        return pred_shape * self.scale + self.bias


class LeadFusionNet(nn.Module):
    """5 个 lead 预测收益输入，输出融合后的下一根收益预测。"""

    def __init__(self, horizon: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(horizon, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class LeadFusionModel:
    model: LeadFusionNet
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float
    train_mse: float
    valid_mse: float

    @torch.no_grad()
    def predict_return(self, x: np.ndarray, device: torch.device) -> float:
        self.model.eval()
        xz = (x.astype(np.float32) - self.x_mean) / self.x_std
        xt = torch.from_numpy(xz).unsqueeze(0).to(device)
        yz = float(self.model(xt).cpu().item())
        return yz * self.y_std + self.y_mean


def trailing_log_ret_stats(close: np.ndarray, end_exclusive: int, *, window: int = 120) -> tuple[float, float]:
    """用 anchor 之前的原始 close 计算因果 log_ret 均值/标准差。"""
    log_ret = np.diff(np.log(np.clip(close, 1e-12, None)), prepend=np.log(max(close[0], 1e-12)))
    end = max(1, end_exclusive)
    start = max(0, end - window)
    seg = log_ret[start:end]
    return float(seg.mean()), float(seg.std() + 1e-8)


def trailing_context_features(bars: np.ndarray, anchor: int) -> np.ndarray:
    """只用 anchor 前的 normalized log_ret 构造去滞后校准特征。"""
    r = bars[:anchor, 0]
    feats = []
    for w in (1, 3, 6, 12, 24):
        seg = r[max(0, anchor - w) : anchor]
        feats.append(float(seg[-1] if w == 1 and seg.size else seg.mean() if seg.size else 0.0))
        feats.append(float(seg.sum() if seg.size else 0.0))
    return np.asarray(feats, dtype=np.float32)


@torch.no_grad()
def collect_return_calibration(
    model: KlinePatternPredictor,
    bars: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> ReturnCalibrator:
    """复用训练+验证样本拟合 normalized log_ret 的逐步校准。"""
    idx = np.concatenate([train_idx, valid_idx])
    samples = build_sequence_sample_indices(
        bars.shape[0],
        context_bars=args.context_bars,
        pred_horizon=args.pred_horizon,
        stride=args.stride,
        index_min=int(idx.min()),
        index_max=int(idx.max()),
    )
    loader = DataLoader(PatternSequenceDataset(bars, samples), batch_size=64, shuffle=False)
    preds, targets = [], []
    for batch in loader:
        pred = model(batch["ctx_bars"].to(device), batch["ctx_lengths"].to(device))
        if pred.dim() == 3:
            pred = pred[..., 0]
        preds.append(pred.cpu().numpy())
        targets.append(batch["future_bars"][..., 0].numpy())
    p = np.concatenate(preds, axis=0)
    y = np.concatenate(targets, axis=0)
    if args.calibration == "none":
        return ReturnCalibrator("none", np.ones(p.shape[1], dtype=np.float32), np.zeros(p.shape[1], dtype=np.float32))

    scale = np.ones(p.shape[1], dtype=np.float32)
    bias = np.zeros(p.shape[1], dtype=np.float32)
    for t in range(p.shape[1]):
        pt, yt = p[:, t], y[:, t]
        var = float(pt.var())
        if var > 1e-10:
            cov = float(((pt - pt.mean()) * (yt - yt.mean())).mean())
            scale[t] = float(np.clip(cov / (var + 1e-10), -5.0, 5.0))
            bias[t] = float(yt.mean() - scale[t] * pt.mean())
    if args.calibration == "affine":
        return ReturnCalibrator("affine", scale, bias)

    ctx = np.stack([trailing_context_features(bars, s.context_end) for s in samples], axis=0).astype(np.float64)
    coef = np.zeros((p.shape[1], 3 + ctx.shape[1]), dtype=np.float64)
    ridge = float(args.calibration_ridge)
    for t in range(p.shape[1]):
        x = np.column_stack(
            [
                np.ones(p.shape[0], dtype=np.float64),
                p[:, t].astype(np.float64),
                np.cumsum(p, axis=1)[:, t].astype(np.float64),
                ctx,
            ]
        )
        reg = np.eye(x.shape[1], dtype=np.float64) * ridge
        reg[0, 0] = 0.0
        coef[t] = np.linalg.solve(x.T @ x + reg, x.T @ y[:, t].astype(np.float64))
    return ReturnCalibrator("residual", scale, bias, coef.astype(np.float32))


@torch.no_grad()
def collect_shape_calibration(
    model: KlinePatternPredictor,
    bars: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> ShapeCalibrator | None:
    """用 train+valid 拟合 body/upper/lower 的逐 horizon affine 校准。"""
    if args.pred_feat_dim < 4:
        return None
    idx = np.concatenate([train_idx, valid_idx])
    samples = build_sequence_sample_indices(
        bars.shape[0],
        context_bars=args.context_bars,
        pred_horizon=args.pred_horizon,
        stride=args.stride,
        index_min=int(idx.min()),
        index_max=int(idx.max()),
    )
    loader = DataLoader(PatternSequenceDataset(bars, samples), batch_size=64, shuffle=False)
    preds, targets = [], []
    for batch in loader:
        pred = model(batch["ctx_bars"].to(device), batch["ctx_lengths"].to(device))
        if pred.dim() != 3 or pred.size(-1) < 4:
            return None
        preds.append(pred[..., 1:4].cpu().numpy())
        targets.append(batch["future_bars"][..., 1:4].numpy())
    p = np.concatenate(preds, axis=0)
    y = np.concatenate(targets, axis=0)
    scale = np.ones(p.shape[1:], dtype=np.float32)
    bias = np.zeros(p.shape[1:], dtype=np.float32)
    for t in range(p.shape[1]):
        for j in range(p.shape[2]):
            pt, yt = p[:, t, j], y[:, t, j]
            var = float(pt.var())
            if var > 1e-10:
                cov = float(((pt - pt.mean()) * (yt - yt.mean())).mean())
                scale[t, j] = float(np.clip(cov / (var + 1e-10), -5.0, 5.0))
                bias[t, j] = float(yt.mean() - scale[t, j] * pt.mean())
    return ShapeCalibrator(scale=scale, bias=bias)


def choose_anchor(bars: np.ndarray, test_idx: np.ndarray, args: argparse.Namespace) -> int:
    """选择测试集中未来窗口波动较大的一处，便于展示预测效果。"""
    samples = build_sequence_sample_indices(
        bars.shape[0],
        context_bars=args.context_bars,
        pred_horizon=args.pred_horizon,
        stride=args.stride,
        index_min=int(test_idx.min()),
        index_max=int(test_idx.max()),
    )
    best = max(samples, key=lambda s: float(np.abs(bars[s.context_end : s.future_end, 0]).mean()))
    return best.context_end


def draw_candles(
    ax,
    times,
    open_,
    high,
    low,
    close,
    *,
    alpha: float = 0.75,
    width_scale: float = 0.68,
    up_color: str = "#2ca02c",
    down_color: str = "#d62728",
    label: str | None = None,
) -> None:
    x = mdates.date2num(times)
    width = 0.03 if len(x) < 2 else min(0.03, (x[1] - x[0]) * width_scale)
    for i, (xi, o, h, l, c) in enumerate(zip(x, open_, high, low, close)):
        up = c >= o
        color = up_color if up else down_color
        ax.vlines(xi, l, h, color=color, linewidth=1.0, alpha=alpha)
        bottom = min(o, c)
        height = max(abs(c - o), max(close.mean() * 1e-5, 1e-8))
        ax.add_patch(
            Rectangle(
                (xi - width / 2, bottom),
                width,
                height,
                facecolor=color,
                edgecolor=color,
                alpha=alpha,
                label=label if i == 0 else None,
            )
        )


def trailing_feature_stats(raw_features: np.ndarray, anchor: int, *, window: int = 120) -> tuple[np.ndarray, np.ndarray]:
    end = max(1, anchor)
    start = max(0, end - window)
    seg = raw_features[start:end]
    return seg.mean(axis=0), seg.std(axis=0) + 1e-8


def trailing_range_pct(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, anchor: int) -> float:
    end = max(1, anchor)
    start = max(0, end - 120)
    base = np.clip(close[start:end], 1e-12, None)
    pct = (high[start:end] - low[start:end]) / base
    return float(np.nanmedian(np.clip(pct, 1e-5, None))) if pct.size else 0.001


def shape_to_ohlc(
    prev_close: float,
    raw_shape: np.ndarray,
    *,
    fallback_range_pct: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """由 [log_ret, body_ratio, upper_wick, lower_wick] 递推重建预测 OHLC。"""
    out_o, out_h, out_l, out_c = [], [], [], []
    last_close = float(prev_close)
    for r, body_ratio, upper, lower in raw_shape:
        o = last_close
        c = float(max(o * np.exp(float(r)), 1e-12))
        body_abs_ratio = float(np.clip(abs(body_ratio), 0.06, 0.94))
        upper = float(np.clip(upper, 0.02, 0.92))
        lower = float(np.clip(lower, 0.02, 0.92))
        total = body_abs_ratio + upper + lower
        body_abs_ratio, upper, lower = body_abs_ratio / total, upper / total, lower / total
        body_abs = abs(c - o)
        range_abs = max(body_abs / max(body_abs_ratio, 1e-6), o * fallback_range_pct)
        h = max(o, c) + upper * range_abs
        l = max(1e-12, min(o, c) - lower * range_abs)
        out_o.append(o)
        out_h.append(h)
        out_l.append(l)
        out_c.append(c)
        last_close = c
    return (
        np.asarray(out_o, dtype=np.float64),
        np.asarray(out_h, dtype=np.float64),
        np.asarray(out_l, dtype=np.float64),
        np.asarray(out_c, dtype=np.float64),
    )


@torch.no_grad()
def predict_shape_candles(
    model: KlinePatternPredictor,
    bars: np.ndarray,
    raw_features: np.ndarray,
    df,
    anchor: int,
    calibrator: ReturnCalibrator,
    shape_calibrator: ShapeCalibrator | None,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """使用模型预测的 log_ret/body/upper/lower 重建 anchor 后 H 根蜡烛。"""
    if args.pred_feat_dim < 4:
        return None
    ctx = torch.from_numpy(bars[anchor - args.context_bars : anchor].astype(np.float32)).unsqueeze(0).to(device)
    lengths = torch.tensor([args.context_bars], dtype=torch.long, device=device)
    pred = model(ctx, lengths)
    if pred.dim() != 3 or pred.size(-1) < 4:
        return None
    pred_np = pred[0, :, :4].cpu().numpy()
    pred_np[:, 0] = calibrator.apply(pred_np[:, 0], trailing_context_features(bars, anchor))
    if shape_calibrator is not None:
        pred_np[:, 1:4] = shape_calibrator.apply(pred_np[:, 1:4])
    mean, std = trailing_feature_stats(raw_features[:, :4], anchor)
    raw_shape = pred_np * std + mean
    open_arr = df[COL_OPEN].to_numpy(dtype=np.float64)
    high_arr = df[COL_HIGH].to_numpy(dtype=np.float64)
    low_arr = df[COL_LOW].to_numpy(dtype=np.float64)
    close_arr = df[COL_CLOSE].to_numpy(dtype=np.float64)
    return shape_to_ohlc(
        close_arr[anchor - 1],
        raw_shape,
        fallback_range_pct=trailing_range_pct(open_arr, high_arr, low_arr, close_arr, anchor),
    )


@torch.no_grad()
def rolling_predictions(
    model: KlinePatternPredictor,
    bars: np.ndarray,
    close: np.ndarray,
    chart_start: int,
    chart_end: int,
    calibrator: ReturnCalibrator,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[int, list[tuple[int, float]]], list[tuple[int, np.ndarray]]]:
    """逐步滚动输入上下文，返回每个未来 bar 的预测 close 集合。"""
    pred_by_bar: dict[int, list[tuple[int, float]]] = {}
    paths: list[tuple[int, np.ndarray]] = []
    first_anchor = max(args.context_bars, chart_start + 8)
    last_anchor = min(chart_end - args.pred_horizon, bars.shape[0] - args.pred_horizon)
    for anchor in range(first_anchor, last_anchor + 1, args.predict_every):
        ctx = torch.from_numpy(bars[anchor - args.context_bars : anchor].astype(np.float32)).unsqueeze(0).to(device)
        lengths = torch.tensor([args.context_bars], dtype=torch.long, device=device)
        pred = model(ctx, lengths)
        pred_norm = pred[0, :, 0].cpu().numpy() if pred.dim() == 3 else pred[0].cpu().numpy()
        pred_norm = calibrator.apply(pred_norm, trailing_context_features(bars, anchor))
        mean, std = trailing_log_ret_stats(close, anchor)
        pred_log_ret = pred_norm * std + mean
        pred_close = close[anchor - 1] * np.exp(np.cumsum(pred_log_ret))
        path = np.concatenate([[close[anchor - 1]], pred_close])
        paths.append((anchor, path))
        for k, value in enumerate(pred_close):
            pred_by_bar.setdefault(anchor + k, []).append((k + 1, float(value)))
    return pred_by_bar, paths


@torch.no_grad()
def collect_prediction_dict(
    model: KlinePatternPredictor,
    bars: np.ndarray,
    close: np.ndarray,
    anchor_start: int,
    anchor_end: int,
    calibrator: ReturnCalibrator,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[int, list[tuple[int, float]]]:
    """按 1 根 K 线步长批量滚动，收集每根目标 K 线的 lead1..leadH 预测 close。"""
    pred_by_bar: dict[int, list[tuple[int, float]]] = {}
    first_anchor = max(args.context_bars, anchor_start)
    last_anchor = min(anchor_end, bars.shape[0] - args.pred_horizon)
    if last_anchor < first_anchor:
        return pred_by_bar
    anchors = np.arange(first_anchor, last_anchor + 1, dtype=np.int64)
    for start in range(0, anchors.size, args.batch_size):
        chunk = anchors[start : start + args.batch_size]
        ctx_np = np.stack([bars[a - args.context_bars : a].astype(np.float32) for a in chunk], axis=0)
        ctx = torch.from_numpy(ctx_np).to(device)
        lengths = torch.full((len(chunk),), args.context_bars, dtype=torch.long, device=device)
        pred = model(ctx, lengths)
        pred_np = pred[..., 0].cpu().numpy() if pred.dim() == 3 else pred.cpu().numpy()
        for row, anchor in enumerate(chunk.tolist()):
            pred_norm = calibrator.apply(pred_np[row], trailing_context_features(bars, anchor))
            mean, std = trailing_log_ret_stats(close, anchor)
            pred_log_ret = pred_norm * std + mean
            pred_close = close[anchor - 1] * np.exp(np.cumsum(pred_log_ret))
            for k, value in enumerate(pred_close):
                pred_by_bar.setdefault(anchor + k, []).append((k + 1, float(value)))
    return pred_by_bar


def build_fusion_rows(
    pred_by_bar: dict[int, list[tuple[int, float]]],
    close: np.ndarray,
    *,
    horizon: int,
    index_min: int,
    index_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """每根 K 线用 5 个 lead close 预测构造 MLP 输入，标签为真实单根 log_ret。"""
    xs: list[np.ndarray] = []
    ys: list[float] = []
    indices: list[int] = []
    for idx in sorted(pred_by_bar):
        if idx <= 0 or idx < index_min or idx > index_max:
            continue
        by_lead = {lead: value for lead, value in pred_by_bar[idx]}
        if any(lead not in by_lead for lead in range(1, horizon + 1)):
            continue
        base = max(float(close[idx - 1]), 1e-12)
        x = np.array(
            [np.log(max(float(by_lead[lead]), 1e-12) / base) for lead in range(1, horizon + 1)],
            dtype=np.float32,
        )
        y = float(np.log(max(float(close[idx]), 1e-12) / base))
        xs.append(x)
        ys.append(y)
        indices.append(idx)
    if not xs:
        return (
            np.zeros((0, horizon), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    return np.stack(xs, axis=0), np.asarray(ys, dtype=np.float32), np.asarray(indices, dtype=np.int64)


def train_lead_fusion_mlp(
    model: KlinePatternPredictor,
    bars: np.ndarray,
    close: np.ndarray,
    bundle,
    calibrator: ReturnCalibrator,
    args: argparse.Namespace,
    device: torch.device,
) -> LeadFusionModel | None:
    """用 train split 训练、valid split 早停的 5-input MLP 融合器。"""
    if args.fusion != "mlp":
        return None
    h = args.pred_horizon
    train_pred = collect_prediction_dict(
        model,
        bars,
        close,
        int(bundle.train_idx.min()) - h + 1,
        int(bundle.train_idx.max()),
        calibrator,
        args,
        device,
    )
    valid_pred = collect_prediction_dict(
        model,
        bars,
        close,
        int(bundle.valid_idx.min()) - h + 1,
        int(bundle.valid_idx.max()),
        calibrator,
        args,
        device,
    )
    x_train, y_train, _ = build_fusion_rows(
        train_pred,
        close,
        horizon=h,
        index_min=int(bundle.train_idx.min()),
        index_max=int(bundle.train_idx.max()),
    )
    x_valid, y_valid, _ = build_fusion_rows(
        valid_pred,
        close,
        horizon=h,
        index_min=int(bundle.valid_idx.min()),
        index_max=int(bundle.valid_idx.max()),
    )
    if x_train.shape[0] < 64 or x_valid.shape[0] < 16:
        return None

    x_mean = x_train.mean(axis=0).astype(np.float32)
    x_std = (x_train.std(axis=0) + 1e-6).astype(np.float32)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std() + 1e-6)
    x_raw = torch.from_numpy(x_train).to(device)
    y_raw = torch.from_numpy(y_train).to(device)
    xt = torch.from_numpy((x_train - x_mean) / x_std).to(device)
    yt = torch.from_numpy((y_train - y_mean) / y_std).to(device)
    xv = torch.from_numpy((x_valid - x_mean) / x_std).to(device)
    yv = torch.from_numpy((y_valid - y_mean) / y_std).to(device)
    x_mean_t = torch.from_numpy(x_mean).to(device)
    x_std_t = torch.from_numpy(x_std).to(device)

    fusion = LeadFusionNet(h, args.fusion_hidden).to(device)
    opt = torch.optim.AdamW(fusion.parameters(), lr=args.fusion_lr, weight_decay=args.fusion_weight_decay)
    best_loss = float("inf")
    best_state = None
    rng = np.random.default_rng(args.seed)
    batch_size = min(256, xt.size(0))
    for _ in range(args.fusion_epochs):
        fusion.train()
        order = rng.permutation(xt.size(0))
        for lo in range(0, xt.size(0), batch_size):
            idx = torch.from_numpy(order[lo : lo + batch_size]).to(device)
            xb = xt[idx]
            if args.fusion_augment_prob > 0:
                raw = x_raw[idx]
                truth = y_raw[idx].unsqueeze(1)
                # 以真实值为中心，随机叠加预测偏差，训练 MLP 从带噪 lead 输入中还原真实未来收益。
                alpha = torch.rand((raw.size(0), 1), device=device) * args.fusion_augment_strength
                aug_raw = truth + alpha * (raw - truth)
                use_aug = torch.rand((raw.size(0), 1), device=device) < args.fusion_augment_prob
                xb = torch.where(use_aug, (aug_raw - x_mean_t) / x_std_t, xb)
            pred = fusion(xb)
            loss = torch.mean((pred - yt[idx]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        fusion.eval()
        with torch.no_grad():
            valid_loss = torch.mean((fusion(xv) - yv) ** 2).item()
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in fusion.state_dict().items()}
    if best_state is not None:
        fusion.load_state_dict(best_state)
    fusion.eval()
    with torch.no_grad():
        train_mse = float(torch.mean(((fusion(xt) * y_std + y_mean) - (yt * y_std + y_mean)) ** 2).cpu().item())
        valid_mse = float(torch.mean(((fusion(xv) * y_std + y_mean) - (yv * y_std + y_mean)) ** 2).cpu().item())
    return LeadFusionModel(fusion, x_mean, x_std, y_mean, y_std, train_mse, valid_mse)


def fused_prediction_series(
    pred_by_bar: dict[int, list[tuple[int, float]]],
    close: np.ndarray,
    fusion: LeadFusionModel,
    *,
    horizon: int,
    index_min: int,
    index_max: int,
    device: torch.device,
) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    x_feat, _, indices = build_fusion_rows(
        pred_by_bar,
        close,
        horizon=horizon,
        index_min=index_min,
        index_max=index_max,
    )
    for row, idx in enumerate(indices.tolist()):
        pred_ret = fusion.predict_return(x_feat[row], device)
        ys.append(float(close[idx - 1] * np.exp(pred_ret)))
        xs.append(idx)
    return xs, ys


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = fetch_ohlcv_df(args)
    bundle = prepare_bar_series(df)
    cfg = pattern_config_from_args(args)
    model = KlinePatternPredictor(
        PatternPredictorConfig(
            auto_segment=cfg,
            trunk=CausalTransformerConfig(d_model=args.d_model, n_heads=args.n_heads, n_layers=args.trunk_layers),
            pred_horizon=args.pred_horizon,
            pred_feat_dim=args.pred_feat_dim,
            pool_mode=args.pool_mode,
            learnable_scale=not args.no_learnable_scale,
            use_horizon_head=args.horizon_head,
        )
    ).to(device)
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    display_lead = args.pred_horizon if args.display_lead < 1 else min(args.display_lead, args.pred_horizon)
    close_arr = df[COL_CLOSE].to_numpy(dtype=np.float64)
    calibrator = collect_return_calibration(model, bundle.bars, bundle.train_idx, bundle.valid_idx, args, device)
    shape_calibrator = collect_shape_calibration(model, bundle.bars, bundle.train_idx, bundle.valid_idx, args, device)
    fusion = train_lead_fusion_mlp(model, bundle.bars, close_arr, bundle, calibrator, args, device)

    anchor = choose_anchor(bundle.bars, bundle.test_idx, args)
    chart_start = max(0, anchor - args.chart_before)
    chart_end = min(len(df), anchor + args.chart_after)
    pred_by_bar, paths = rolling_predictions(
        model,
        bundle.bars,
        close_arr,
        chart_start,
        chart_end,
        calibrator,
        args,
        device,
    )
    chart_pred_by_bar = collect_prediction_dict(
        model,
        bundle.bars,
        close_arr,
        chart_start - args.pred_horizon + 1,
        chart_end - 1,
        calibrator,
        args,
        device,
    )
    feat_df, _ = build_bar_shape_frame(df)
    pred_shape_ohlc = predict_shape_candles(
        model,
        bundle.bars,
        feat_df.to_numpy(dtype=np.float32),
        df,
        anchor,
        calibrator,
        shape_calibrator,
        args,
        device,
    )

    view = df.iloc[chart_start:chart_end].reset_index(drop=True)
    x_times = view[COL_TIME].to_numpy()
    full_times = df[COL_TIME].to_numpy()
    fig, (ax_actual, ax_pred) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, sharey=True)
    draw_candles(
        ax_actual,
        x_times,
        view[COL_OPEN].to_numpy(dtype=np.float64),
        view[COL_HIGH].to_numpy(dtype=np.float64),
        view[COL_LOW].to_numpy(dtype=np.float64),
        view[COL_CLOSE].to_numpy(dtype=np.float64),
        label="actual candles",
    )
    ax_actual.plot(view[COL_TIME], view[COL_CLOSE], color="black", alpha=0.35, linewidth=1.0, label="actual close")

    pred_close_for_ylim: list[float] = []
    if pred_shape_ohlc is not None:
        po, ph, pl, pc = pred_shape_ohlc
        pred_times = full_times[anchor : anchor + args.pred_horizon]
        draw_candles(
            ax_pred,
            pred_times,
            po,
            ph,
            pl,
            pc,
            alpha=0.72,
            width_scale=0.52,
            up_color="#1f77b4",
            down_color="#ff7f0e",
            label="predicted candles",
        )
        ax_pred.plot(pred_times, pc, color="#1f77b4", linewidth=1.3, alpha=0.75, label="predicted candle close")
        pred_close_for_ylim.extend(pc.tolist())

    # 多条滚动预测路径。
    for i, (anchor_idx, path) in enumerate(paths):
        if i % max(1, args.path_plot_every) != 0:
            continue
        xs = full_times[anchor_idx - 1 : anchor_idx + args.pred_horizon]
        label = "rolling 5-bar forecasts" if i == 0 else None
        ax_pred.plot(xs, path, color="#ff7f0e", alpha=0.18, linewidth=1.0, label=label)
        pred_close_for_ylim.extend(path.tolist())

    # 同一未来 bar 被多个 anchor 预测时，区分 lead；混合 lead 的均线容易产生视觉滞后。
    all_x, all_y = [], []
    lead_x, lead_y = [], []
    for idx in sorted(pred_by_bar):
        if chart_start <= idx < chart_end:
            vals = pred_by_bar[idx]
            all_x.append(full_times[idx])
            all_y.append(np.mean([v for _, v in vals]))
            fixed = [v for lead, v in vals if lead == display_lead]
            if fixed:
                lead_x.append(full_times[idx])
                lead_y.append(float(np.mean(fixed)))
    ax_pred.plot(
        all_x,
        all_y,
        color="#7f7f7f",
        linestyle="--",
        linewidth=1.2,
        alpha=0.65,
        label="mixed-lead mean (diagnostic)",
    )
    ax_pred.plot(lead_x, lead_y, color="#1f77b4", linewidth=2.2, label=f"fixed lead={display_lead} predicted close")
    pred_close_for_ylim.extend([float(v) for v in all_y])
    pred_close_for_ylim.extend([float(v) for v in lead_y])
    fused_x_idx: list[int] = []
    fused_y: list[float] = []
    if fusion is not None:
        fused_x_idx, fused_y = fused_prediction_series(
            chart_pred_by_bar,
            close_arr,
            fusion,
            horizon=args.pred_horizon,
            index_min=chart_start,
            index_max=chart_end - 1,
            device=device,
        )
        fused_x = [full_times[idx] for idx in fused_x_idx]
        ax_pred.plot(fused_x, fused_y, color="#d62728", linewidth=2.1, label="MLP fused 5 leads")
        pred_close_for_ylim.extend([float(v) for v in fused_y])
    context = df.iloc[chart_start:anchor]
    ax_pred.plot(
        context[COL_TIME],
        context[COL_CLOSE],
        color="black",
        alpha=0.22,
        linewidth=1.0,
        label="context close",
    )
    for ax in (ax_actual, ax_pred):
        ax.axvline(df[COL_TIME].iloc[anchor], color="#9467bd", linestyle="--", linewidth=1.3, label="selected anchor")
        ax.grid(True, alpha=0.25)

    actual_close = close_arr
    comparable = []
    comparable_all = []
    for idx, vals in pred_by_bar.items():
        if not (chart_start <= idx < chart_end):
            continue
        comparable_all.append((idx, float(np.mean([v for _, v in vals])), actual_close[idx]))
        fixed = [v for lead, v in vals if lead == display_lead]
        if fixed:
            comparable.append((idx, float(np.mean(fixed)), actual_close[idx]))
    mse = float(np.mean([(p - y) ** 2 for _, p, y in comparable])) if comparable else float("nan")
    mae = float(np.mean([abs(p - y) for _, p, y in comparable])) if comparable else float("nan")
    dir_acc = float(
        np.mean(
            [
                np.sign(p - actual_close[idx - display_lead]) == np.sign(y - actual_close[idx - display_lead])
                for idx, p, y in comparable
                if idx >= display_lead
            ]
        )
    ) if comparable else float("nan")
    all_mae = float(np.mean([abs(p - y) for _, p, y in comparable_all])) if comparable_all else float("nan")
    fused_comparable = [
        (idx, pred, actual_close[idx]) for idx, pred in zip(fused_x_idx, fused_y) if chart_start <= idx < chart_end
    ]
    fused_mse = float(np.mean([(p - y) ** 2 for _, p, y in fused_comparable])) if fused_comparable else float("nan")
    fused_mae = float(np.mean([abs(p - y) for _, p, y in fused_comparable])) if fused_comparable else float("nan")
    fused_dir_acc = float(
        np.mean(
            [
                np.sign(p - actual_close[idx - 1]) == np.sign(y - actual_close[idx - 1])
                for idx, p, y in fused_comparable
                if idx > 0
            ]
        )
    ) if fused_comparable else float("nan")

    y_values = list(view[COL_HIGH].to_numpy(dtype=np.float64)) + list(view[COL_LOW].to_numpy(dtype=np.float64)) + pred_close_for_ylim
    if pred_shape_ohlc is not None:
        _, ph, pl, _ = pred_shape_ohlc
        y_values.extend(ph.tolist())
        y_values.extend(pl.tolist())
    if y_values:
        ymin, ymax = float(np.nanmin(y_values)), float(np.nanmax(y_values))
        pad = max((ymax - ymin) * 0.08, 1.0)
        ax_actual.set_ylim(ymin - pad, ymax + pad)

    fig.suptitle(
        f"BTCUSDT 1h actual vs predicted candles | anchor={df[COL_TIME].iloc[anchor]} | "
        f"lead{display_lead} MAE={mae:.2f}, dir_acc={dir_acc:.1%}",
        y=0.99,
    )
    ax_actual.set_title("Actual Candlestick")
    ax_pred.set_title("Predicted Candlestick")
    ax_actual.set_ylabel("Price")
    ax_pred.set_ylabel("Price")
    ax_pred.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax_actual.legend(loc="best")
    ax_pred.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_path = out_dir / "btc_rolling_candlestick_predictions.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    metrics_path = out_dir / "metrics.txt"
    metrics_path.write_text(
        "\n".join(
            [
                f"anchor_time={df[COL_TIME].iloc[anchor]}",
                f"chart_start={df[COL_TIME].iloc[chart_start]}",
                f"chart_end={df[COL_TIME].iloc[chart_end - 1]}",
                f"num_rolling_predictions={len(paths)}",
                f"calibration={args.calibration}",
                f"display_lead={display_lead}",
                f"mixed_lead_mae={all_mae:.6f}",
                f"fixed_lead_mae={mae:.6f}",
                f"fixed_lead_mse={mse:.6f}",
                f"fixed_lead_direction_acc={dir_acc:.6f}",
                f"fusion={args.fusion}",
                f"fusion_augment_prob={args.fusion_augment_prob:.6f}",
                f"fusion_augment_strength={args.fusion_augment_strength:.6f}",
                f"fusion_train_mse={fusion.train_mse:.8f}" if fusion is not None else "fusion_train_mse=nan",
                f"fusion_valid_mse={fusion.valid_mse:.8f}" if fusion is not None else "fusion_valid_mse=nan",
                f"fusion_mae={fused_mae:.6f}",
                f"fusion_mse={fused_mse:.6f}",
                f"fusion_direction_acc={fused_dir_acc:.6f}",
                f"shape_candles={'yes' if pred_shape_ohlc is not None else 'no'}",
                f"shape_calibration={'yes' if shape_calibrator is not None else 'no'}",
                f"checkpoint={args.checkpoint}",
            ]
        ),
        encoding="utf-8",
    )
    print(f"saved chart: {out_path}")
    print(f"saved metrics: {metrics_path}")
    print(
        f"lead{display_lead} MAE={mae:.4f} MSE={mse:.4f} dir_acc={dir_acc:.1%} "
        f"MLP_MAE={fused_mae:.4f} MLP_dir={fused_dir_acc:.1%} "
        f"mixed_MAE={all_mae:.4f} predictions={len(paths)} calibration={args.calibration}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
