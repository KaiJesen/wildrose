#!/usr/bin/env python3
"""基于模型多步预测信号的 BTC 模拟交易回测。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

_EX = Path(__file__).resolve().parent
_ROOT = _EX.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))

from _train_common import add_data_args, add_segment_args, add_stage3_loss_args, add_vq_args, fetch_ohlcv_df
from market_data.schema import COL_CLOSE, COL_HIGH, COL_LOW, COL_TIME
from plot_backtest_candlestick_predictions import (
    collect_return_calibration,
    trailing_context_features,
    trailing_log_ret_stats,
)
from transformer_kit.causal_transformer import CausalTransformerConfig
from transformer_kit.pattern_encoder import pattern_config_from_args
from transformer_kit.pattern_model import KlinePatternPredictor, PatternPredictorConfig
from transformer_kit.segment_dataset import prepare_bar_series
from transformer_kit.train_utils import load_checkpoint

LEAD_WEIGHTS = np.array([0.45, 0.25, 0.15, 0.10, 0.05], dtype=np.float64)


@dataclass
class TradeRecord:
    entry_idx: int
    exit_idx: int
    side: int
    entry_price: float
    exit_price: float
    pnl_pct: float
    exit_reason: str


@dataclass
class BacktestState:
    position: float = 0.0
    entry_idx: int = -1
    entry_price: float = 0.0
    bars_held: int = 0
    cooldown_until: int = -1
    loss_streak: int = 0
    equity: float = 1.0
    trades: list[TradeRecord] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    positions: list[float] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulated BTC trading backtest from model signals")
    add_data_args(p)
    add_segment_args(p)
    add_vq_args(p)
    add_stage3_loss_args(p)
    p.add_argument("--pred-horizon", type=int, default=5)
    p.add_argument("--pred-feat-dim", type=int, default=4)
    p.add_argument("--stride", type=int, default=1, help="信号滚动步长；回测建议 1")
    p.add_argument("--trunk-layers", type=int, default=2)
    p.add_argument(
        "--checkpoint",
        default="checkpoints/anti_lag_horizon_prediction/stage3_predictor_best_combo.pt",
    )
    p.add_argument("--output-dir", default="reports/btc_sim_backtest")
    p.add_argument("--calibration", choices=("none", "affine", "residual"), default="residual")
    p.add_argument("--calibration-ridge", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dpi", type=int, default=150)
    # 交易规则参数
    p.add_argument("--accel-lambda", type=float, default=0.4, help="拐点项权重")
    p.add_argument("--theta-scale", type=float, default=0.15, help="开仓阈值 = scale * rolling_std(S)")
    p.add_argument("--theta-window", type=int, default=200)
    p.add_argument("--target-vol", type=float, default=0.01, help="目标单 bar 波动率")
    p.add_argument("--vol-window", type=int, default=24)
    p.add_argument("--fee-bps", type=float, default=4.0)
    p.add_argument("--slippage-bps", type=float, default=2.0)
    p.add_argument("--stop-atr", type=float, default=1.2)
    p.add_argument("--take-atr", type=float, default=2.0)
    p.add_argument("--atr-period", type=int, default=14)
    p.add_argument("--loss-streak-limit", type=int, default=3)
    p.add_argument("--cooldown-bars", type=int, default=12)
    p.set_defaults(source="binance_vision", symbol="BTCUSDT", interval="1h", days=365)
    return p.parse_args()


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = np.empty_like(tr)
    atr[:period] = tr[:period].mean()
    alpha = 1.0 / period
    for i in range(period, len(tr)):
        atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1]
    return atr


def build_signal(
    pred_log_ret: np.ndarray,
    *,
    accel_lambda: float,
) -> tuple[float, float, float]:
    """由 H 步预测 log_ret 构造 S/A/F 分数。"""
    h = min(len(pred_log_ret), len(LEAD_WEIGHTS))
    r = pred_log_ret[:h]
    w = LEAD_WEIGHTS[:h]
    w = w / w.sum()
    s_score = float(np.dot(w, r))
    if h >= 3:
        accel = float((r[0] - r[1]) + 0.5 * (r[1] - r[2]))
    elif h >= 2:
        accel = float(r[0] - r[1])
    else:
        accel = 0.0
    f_score = s_score + accel_lambda * accel
    return s_score, accel, f_score


def rolling_std(values: list[float], window: int) -> float:
    if len(values) < 2:
        return 0.001
    seg = np.asarray(values[-window:], dtype=np.float64)
    return float(max(seg.std(), 1e-6))


@torch.no_grad()
def collect_signals(
    model: KlinePatternPredictor,
    bars: np.ndarray,
    close: np.ndarray,
    indices: np.ndarray,
    calibrator,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[int, dict[str, float]]:
    """为每个 anchor 收集预测收益与方向 logit。"""
    out: dict[int, dict[str, float]] = {}
    h = args.pred_horizon
    for start in range(0, len(indices), args.batch_size):
        chunk = indices[start : start + args.batch_size]
        ctx_np = np.stack([bars[a - args.context_bars : a].astype(np.float32) for a in chunk], axis=0)
        ctx = torch.from_numpy(ctx_np).to(device)
        lengths = torch.full((len(chunk),), args.context_bars, dtype=torch.long, device=device)
        pred, aux = model(ctx, lengths, return_aux=True)
        pred_np = pred[..., 0].cpu().numpy() if pred.dim() == 3 else pred.cpu().numpy()
        dir_logit = aux["cum_direction_logit"].cpu().numpy()
        for row, anchor in enumerate(chunk.tolist()):
            pred_norm = calibrator.apply(pred_np[row], trailing_context_features(bars, anchor))
            mean, std = trailing_log_ret_stats(close, anchor)
            pred_log_ret = pred_norm * std + mean
            s_score, accel, f_score = build_signal(pred_log_ret, accel_lambda=args.accel_lambda)
            out[anchor] = {
                "s_score": s_score,
                "accel": accel,
                "f_score": f_score,
                "dir_logit": float(dir_logit[row]),
                **{f"r{k+1}": float(pred_log_ret[k]) for k in range(min(h, len(pred_log_ret)))},
            }
    return out


def desired_side(f_score: float, dir_logit: float, theta: float) -> int:
    if f_score > theta and dir_logit > 0:
        return 1
    if f_score < -theta and dir_logit < 0:
        return -1
    return 0


def transaction_cost(abs_delta_pos: float, fee_bps: float, slippage_bps: float) -> float:
    return abs_delta_pos * (fee_bps + slippage_bps) * 1e-4


def run_backtest(
    df,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    log_ret: np.ndarray,
    atr: np.ndarray,
    trade_indices: np.ndarray,
    signals: dict[int, dict[str, float]],
    args: argparse.Namespace,
) -> BacktestState:
    state = BacktestState()
    s_history: list[float] = []
    times = df[COL_TIME].to_numpy()

    for i in trade_indices:
        if i <= 0 or i >= len(close) - 1:
            continue
        sig = signals.get(int(i))
        if sig is None:
            state.equity_curve.append(state.equity)
            state.positions.append(state.position)
            continue

        s_history.append(sig["s_score"])
        theta = args.theta_scale * rolling_std(s_history, args.theta_window)
        vol_hat = rolling_std(log_ret[: i + 1].tolist(), args.vol_window)
        target_w = float(np.clip(args.target_vol / vol_hat, 0.0, 1.0))
        target_side = desired_side(sig["f_score"], sig["dir_logit"], theta)

        # 先结算上一根持仓收益（在 bar i 收盘后决定，赚取 i->i+1 收益）
        bar_ret = float(log_ret[i + 1])
        state.equity *= float(np.exp(state.position * bar_ret))

        exit_reason = ""
        new_side = target_side
        if state.position != 0.0:
            state.bars_held += 1
            signed_move = (close[i] - state.entry_price) / max(state.entry_price, 1e-12)
            if state.position > 0:
                hit_stop = signed_move <= -args.stop_atr * atr[i] / max(close[i], 1e-12)
                hit_take = signed_move >= args.take_atr * atr[i] / max(close[i], 1e-12)
            else:
                hit_stop = signed_move >= args.stop_atr * atr[i] / max(close[i], 1e-12)
                hit_take = signed_move <= -args.take_atr * atr[i] / max(close[i], 1e-12)

            if hit_stop:
                new_side = 0
                exit_reason = "stop"
            elif hit_take:
                new_side = 0
                exit_reason = "take_profit"
            elif state.bars_held >= 1 and target_side != int(np.sign(state.position)):
                new_side = target_side
                exit_reason = "reverse_or_flat"

        if state.cooldown_until >= i and new_side != 0:
            new_side = 0

        new_pos = float(new_side) * target_w if new_side != 0 else 0.0
        delta = abs(new_pos - state.position)
        if delta > 1e-8:
            state.equity *= 1.0 - transaction_cost(delta, args.fee_bps, args.slippage_bps)

        if state.position != 0.0 and new_pos == 0.0 and exit_reason:
            side = 1 if state.position > 0 else -1
            pnl_pct = side * (close[i] - state.entry_price) / max(state.entry_price, 1e-12)
            state.trades.append(
                TradeRecord(
                    entry_idx=state.entry_idx,
                    exit_idx=int(i),
                    side=side,
                    entry_price=state.entry_price,
                    exit_price=float(close[i]),
                    pnl_pct=float(pnl_pct),
                    exit_reason=exit_reason,
                )
            )
            if pnl_pct < 0:
                state.loss_streak += 1
            else:
                state.loss_streak = 0
            if state.loss_streak >= args.loss_streak_limit:
                state.cooldown_until = int(i) + args.cooldown_bars
                state.loss_streak = 0

        if new_pos != 0.0 and state.position == 0.0:
            state.entry_idx = int(i)
            state.entry_price = float(close[i])
            state.bars_held = 0
        elif new_pos == 0.0:
            state.entry_idx = -1
            state.entry_price = 0.0
            state.bars_held = 0

        state.position = new_pos
        state.equity_curve.append(state.equity)
        state.positions.append(state.position)

    return state


def buy_and_hold_equity(log_ret: np.ndarray, indices: np.ndarray) -> np.ndarray:
    eq = [1.0]
    for i in indices:
        if i <= 0 or i >= len(log_ret) - 1:
            continue
        eq.append(eq[-1] * float(np.exp(log_ret[i + 1])))
    return np.asarray(eq, dtype=np.float64)


def summarize(state: BacktestState, bh_equity: np.ndarray) -> dict[str, float]:
    eq = np.asarray(state.equity_curve, dtype=np.float64)
    if eq.size < 2:
        return {"strategy_return": 0.0, "buy_hold_return": 0.0}
    rets = np.diff(np.log(np.clip(eq, 1e-12, None)))
    sharpe = float(rets.mean() / (rets.std() + 1e-8) * np.sqrt(24 * 365))
    peak = np.maximum.accumulate(eq)
    max_dd = float(((eq - peak) / np.clip(peak, 1e-12, None)).min())
    wins = [t for t in state.trades if t.pnl_pct > 0]
    return {
        "strategy_return": float(eq[-1] - 1.0),
        "buy_hold_return": float(bh_equity[-1] - 1.0) if bh_equity.size else 0.0,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "num_trades": float(len(state.trades)),
        "win_rate": float(len(wins) / max(1, len(state.trades))),
        "avg_trade_pnl_pct": float(np.mean([t.pnl_pct for t in state.trades])) if state.trades else 0.0,
        "final_equity": float(eq[-1]),
    }


def save_trades(path: Path, df, trades: list[TradeRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["entry_time", "exit_time", "side", "entry_price", "exit_price", "pnl_pct", "exit_reason"]
        )
        for t in trades:
            w.writerow(
                [
                    df[COL_TIME].iloc[t.entry_idx],
                    df[COL_TIME].iloc[t.exit_idx],
                    "long" if t.side > 0 else "short",
                    f"{t.entry_price:.4f}",
                    f"{t.exit_price:.4f}",
                    f"{t.pnl_pct:.6f}",
                    t.exit_reason,
                ]
            )


def plot_equity(path: Path, strategy_eq: np.ndarray, bh_eq: np.ndarray, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(strategy_eq, label="strategy", linewidth=2.0)
    ax.plot(bh_eq, label="buy & hold", linewidth=1.5, alpha=0.8)
    ax.set_title("BTCUSDT 1h simulated backtest")
    ax.set_xlabel("bar")
    ax.set_ylabel("equity")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    df = fetch_ohlcv_df(args)
    bundle = prepare_bar_series(df)
    close = df[COL_CLOSE].to_numpy(dtype=np.float64)
    high = df[COL_HIGH].to_numpy(dtype=np.float64)
    low = df[COL_LOW].to_numpy(dtype=np.float64)
    log_ret = np.diff(np.log(np.clip(close, 1e-12, None)), prepend=np.log(max(close[0], 1e-12)))
    atr = compute_atr(high, low, close, args.atr_period)

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
    model.load_state_dict(load_checkpoint(args.checkpoint, map_location=device)["model"], strict=False)
    model.eval()

    calibrator = collect_return_calibration(
        model, bundle.bars, bundle.train_idx, bundle.valid_idx, args, device
    )

    test_start = int(bundle.test_idx.min())
    test_end = int(bundle.test_idx.max())
    anchors = np.arange(
        max(args.context_bars, test_start),
        min(test_end, len(close) - args.pred_horizon - 1) + 1,
        args.stride,
        dtype=np.int64,
    )
    print(f"collecting signals for {anchors.size} anchors on test split...")
    signals = collect_signals(model, bundle.bars, close, anchors, calibrator, args, device)

    state = run_backtest(
        df, close, high, low, log_ret, atr, anchors, signals, args
    )
    bh_eq = buy_and_hold_equity(log_ret, anchors)
    metrics = summarize(state, bh_eq)

    save_trades(out_dir / "trades.csv", df, state.trades)
    plot_equity(out_dir / "equity_curve.png", np.asarray(state.equity_curve), bh_eq, args.dpi)
    (out_dir / "metrics.json").write_text(
        json.dumps({"metrics": metrics, "config": vars(args)}, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "metrics.txt").write_text(
        "\n".join(
            [
                f"checkpoint={args.checkpoint}",
                f"test_bars={anchors.size}",
                f"strategy_return={metrics['strategy_return']:.4%}",
                f"buy_hold_return={metrics['buy_hold_return']:.4%}",
                f"sharpe={metrics['sharpe']:.3f}",
                f"max_drawdown={metrics['max_drawdown']:.4%}",
                f"num_trades={int(metrics['num_trades'])}",
                f"win_rate={metrics['win_rate']:.2%}",
                f"avg_trade_pnl_pct={metrics['avg_trade_pnl_pct']:.4%}",
            ]
        ),
        encoding="utf-8",
    )

    print(f"saved: {out_dir / 'equity_curve.png'}")
    print(f"saved: {out_dir / 'trades.csv'}")
    print(f"saved: {out_dir / 'metrics.txt'}")
    print(
        f"strategy={metrics['strategy_return']:.2%} buy_hold={metrics['buy_hold_return']:.2%} "
        f"sharpe={metrics['sharpe']:.2f} max_dd={metrics['max_drawdown']:.2%} "
        f"trades={int(metrics['num_trades'])} win={metrics['win_rate']:.1%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
