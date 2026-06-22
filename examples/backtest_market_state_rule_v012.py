#!/usr/bin/env python3
"""Backtest BTC perpetual rules from document v012."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass, field
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
from market_data.schema import COL_CLOSE, COL_HIGH, COL_LOW, COL_OPEN, COL_TIME
from transformer_kit.causal_transformer import CausalTransformerConfig
from transformer_kit.pattern_encoder import pattern_config_from_args
from transformer_kit.pattern_model import KlinePatternPredictor, MarketStateOutput, PatternPredictorConfig
from transformer_kit.train_utils import load_checkpoint


@dataclass
class SignalPoint:
    idx: int
    p_up: float
    p_down: float
    p_flat: float
    p_risk: float
    pred_ret: list[float]
    pred_cum_ret_5: float

    @property
    def edge(self) -> float:
        return self.p_up - self.p_down

    @property
    def conf(self) -> float:
        return abs(self.edge)


@dataclass
class TradeRecord:
    entry_idx: int
    exit_idx: int
    side: int
    entry_price: float
    exit_price: float
    entry_notional: float
    pnl_pct_unlev: float
    pnl_pct_equity: float
    exit_reason: str
    hold_bars: int


@dataclass
class PositionState:
    side: int = 0
    notional: float = 0.0
    entry_idx: int = -1
    entry_price: float = 0.0
    entry_notional: float = 0.0
    bars_held: int = 0
    continue_fail_count: int = 0
    peak_unlev_pnl: float = 0.0
    tp1_done: bool = False
    tp2_done: bool = False


@dataclass
class BacktestState:
    equity: float = 1.0
    equity_curve: list[float] = field(default_factory=list)
    position_curve: list[float] = field(default_factory=list)
    trades: list[TradeRecord] = field(default_factory=list)
    cooldown_until: int = -1
    loss_streak: int = 0
    day_start_equity: float = 1.0
    week_start_equity: float = 1.0
    day_open_block: bool = False
    weekly_observe_mode: bool = False
    position: PositionState = field(default_factory=PositionState)
    blocked_open_count: int = 0
    warnings: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest market-state trading rule v012")
    add_data_args(p)
    add_feature_args(p)
    add_segment_args(p)
    add_vq_args(p)
    p.add_argument("--checkpoint", default="checkpoints/0065a_multi_seed_s45_market_state_stability/market_state_best.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--pred-horizon", type=int, default=5)
    p.add_argument("--trunk-layers", type=int, default=2)
    p.add_argument("--split", choices=["train", "valid", "test"], default="test")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--output-dir", default="backtest/backtest_rule_v012")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--fee-bps", type=float, default=4.0)
    p.add_argument("--slippage-bps", type=float, default=2.0)
    p.add_argument("--atr-period", type=int, default=14)
    p.add_argument("--risk-budget", type=float, default=0.004, help="per-trade risk budget")
    p.add_argument("--max-direction-exposure", type=float, default=1.6)
    p.add_argument("--max-leverage", type=float, default=3.0)
    p.add_argument("--risk-ok-threshold", type=float, default=0.38)
    p.add_argument("--risk-exit-threshold", type=float, default=0.48)
    p.add_argument("--open-edge-threshold", type=float, default=0.08)
    p.add_argument("--open-prob-threshold", type=float, default=0.42)
    p.add_argument("--open-flat-max", type=float, default=0.34)
    p.add_argument("--long-continue-edge-min", type=float, default=-0.03)
    p.add_argument("--short-continue-edge-max", type=float, default=0.03)
    p.add_argument("--reverse-edge-threshold", type=float, default=0.05)
    p.add_argument("--max-hold-bars", type=int, default=6)
    p.add_argument("--stop-atr-mult", type=float, default=1.2)
    p.add_argument("--tp1-atr-mult", type=float, default=1.0)
    p.add_argument("--tp2-atr-mult", type=float, default=2.0)
    p.add_argument("--trail-atr-mult", type=float, default=0.8)
    p.add_argument("--day-dd-stop", type=float, default=0.02)
    p.add_argument("--week-dd-observe", type=float, default=0.05)
    p.add_argument("--observe-size-scale", type=float, default=0.3)
    p.add_argument("--loss-streak-limit", type=int, default=3)
    p.add_argument("--cooldown-bars", type=int, default=12)
    p.set_defaults(source="binance_vision", symbol="BTCUSDT", interval="1h", days=365)
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


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = np.empty_like(tr)
    atr[:period] = tr[:period].mean()
    alpha = 1.0 / max(1, period)
    for i in range(period, len(tr)):
        atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1]
    return atr


def _build_model(args: argparse.Namespace, state_dict: dict, device: torch.device) -> KlinePatternPredictor:
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
            direction_classes=getattr(args, "direction_classes", 3),
            risk_classes=getattr(args, "risk_classes", 2),
            use_cum_heads=getattr(args, "use_cum_heads", True),
            use_horizon_return_head=getattr(args, "use_horizon_return_head", True),
            detach_risk_vol_heads=getattr(args, "detach_risk_vol_heads", False),
            return_direction_hidden_mult=getattr(args, "return_direction_hidden_mult", 1.0),
        )
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


@torch.no_grad()
def collect_signals(
    model: KlinePatternPredictor,
    bars: np.ndarray,
    *,
    context_bars: int,
    pred_horizon: int,
    anchors: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[int, SignalPoint]:
    sigs: dict[int, SignalPoint] = {}
    for st in range(0, len(anchors), batch_size):
        chunk = anchors[st : st + batch_size]
        ctx_np = np.stack([bars[a - context_bars : a].astype(np.float32) for a in chunk], axis=0)
        ctx = torch.from_numpy(ctx_np).to(device)
        ctx_len = torch.full((len(chunk),), context_bars, dtype=torch.long, device=device)
        out = model(ctx, ctx_len)
        if not isinstance(out, MarketStateOutput):
            raise RuntimeError("model must return MarketStateOutput")
        dir_prob = torch.softmax(out.direction_logits[:, 0, :], dim=-1).cpu().numpy()
        risk_prob = torch.softmax(out.risk_logits[:, 0, :], dim=-1).cpu().numpy()
        step_ret = out.return_pred.cpu().numpy()
        if out.cum_return_pred is not None:
            cum_ret = out.cum_return_pred.cpu().numpy()
        else:
            cum_ret = step_ret[:, :pred_horizon].sum(axis=1)
        for r, a in enumerate(chunk.tolist()):
            pred = [float(x) for x in step_ret[r][:pred_horizon]]
            if len(pred) < pred_horizon:
                pred.extend([0.0] * (pred_horizon - len(pred)))
            sigs[a] = SignalPoint(
                idx=int(a),
                p_up=float(dir_prob[r, 2]),
                p_down=float(dir_prob[r, 0]),
                p_flat=float(dir_prob[r, 1]),
                p_risk=float(risk_prob[r, 1]),
                pred_ret=pred,
                pred_cum_ret_5=float(cum_ret[r]),
            )
    return sigs


def _open_long(sig: SignalPoint, args: argparse.Namespace) -> bool:
    return (
        sig.edge >= args.open_edge_threshold
        and sig.p_up >= args.open_prob_threshold
        and sig.p_flat <= args.open_flat_max
        and sig.pred_cum_ret_5 > 0
        and sig.p_risk <= args.risk_ok_threshold
    )


def _open_short(sig: SignalPoint, args: argparse.Namespace) -> bool:
    return (
        sig.edge <= -args.open_edge_threshold
        and sig.p_down >= args.open_prob_threshold
        and sig.p_flat <= args.open_flat_max
        and sig.pred_cum_ret_5 < 0
        and sig.p_risk <= args.risk_ok_threshold
    )


def _continue_ok(side: int, sig: SignalPoint, args: argparse.Namespace) -> bool:
    if side > 0:
        return sig.edge > args.long_continue_edge_min and sig.pred_cum_ret_5 >= 0
    return sig.edge < args.short_continue_edge_max and sig.pred_cum_ret_5 <= 0


def _reversal_exit(side: int, sig: SignalPoint, args: argparse.Namespace) -> bool:
    if side > 0:
        return sig.edge <= -args.reverse_edge_threshold
    return sig.edge >= args.reverse_edge_threshold


def _target_notional(sig: SignalPoint, atr_pct: float, args: argparse.Namespace, observe_mode: bool) -> float:
    conf_norm = min(sig.conf / 0.20, 1.0)
    risk_penalty = np.clip((0.48 - sig.p_risk) / 0.20, 0.0, 1.0)
    pos_scale = 0.35 + 0.65 * conf_norm * risk_penalty
    stop_dist_pct = max(args.stop_atr_mult * atr_pct, 1e-4)
    base_notional = args.risk_budget / stop_dist_pct
    cap = min(args.max_direction_exposure, args.max_leverage)
    out = min(base_notional * pos_scale, cap)
    if observe_mode:
        out *= args.observe_size_scale
    return float(max(0.0, out))


def _transaction_cost(delta: float, fee_bps: float, slippage_bps: float) -> float:
    return max(0.0, delta) * (fee_bps + slippage_bps) * 1e-4


def _sync_time_gates(state: BacktestState, ts, args: argparse.Namespace) -> None:
    last = getattr(_sync_time_gates, "_last_ts", None)
    if last is None:
        state.day_start_equity = state.equity
        state.week_start_equity = state.equity
        _sync_time_gates._last_ts = ts
    else:
        if ts.date() != last.date():
            state.day_start_equity = state.equity
            state.day_open_block = False
        if ts.isocalendar().week != last.isocalendar().week or ts.year != last.year:
            state.week_start_equity = state.equity
            state.weekly_observe_mode = False
        _sync_time_gates._last_ts = ts
    day_dd = (state.equity - state.day_start_equity) / max(1e-12, state.day_start_equity)
    week_dd = (state.equity - state.week_start_equity) / max(1e-12, state.week_start_equity)
    if day_dd <= -args.day_dd_stop:
        state.day_open_block = True
    if week_dd <= -args.week_dd_observe:
        state.weekly_observe_mode = True


def run_backtest(
    df,
    open_px: np.ndarray,
    atr: np.ndarray,
    signals: dict[int, SignalPoint],
    anchors: np.ndarray,
    args: argparse.Namespace,
) -> BacktestState:
    st = BacktestState()
    for i in anchors.tolist():
        if i <= 1 or i >= len(open_px) - 1:
            continue
        ts = df[COL_TIME].iloc[i]
        _sync_time_gates(st, ts, args)
        sig = signals[int(i)]
        pos = st.position

        # mark-to-market open(i-1) -> open(i)
        prev_open, curr_open = float(open_px[i - 1]), float(open_px[i])
        bar_ret = (curr_open - prev_open) / max(prev_open, 1e-12)
        st.equity *= (1.0 + pos.side * pos.notional * bar_ret)
        st.equity = max(st.equity, 1e-9)

        if pos.side != 0:
            pos.bars_held += 1
            unlev_pnl = pos.side * (curr_open - pos.entry_price) / max(pos.entry_price, 1e-12)
            pos.peak_unlev_pnl = max(pos.peak_unlev_pnl, unlev_pnl)
            atr_pct = float(atr[i] / max(curr_open, 1e-12))
            stop_pct = args.stop_atr_mult * atr_pct
            tp1_pct = args.tp1_atr_mult * atr_pct
            tp2_pct = args.tp2_atr_mult * atr_pct
            trail_pct = args.trail_atr_mult * atr_pct

            target_side = pos.side
            target_notional = pos.notional
            exit_reason = ""

            if _reversal_exit(pos.side, sig, args):
                target_side, target_notional, exit_reason = 0, 0.0, "reversal"
            elif sig.p_risk >= args.risk_exit_threshold:
                target_side, target_notional, exit_reason = 0, 0.0, "risk_exit"
            elif pos.bars_held > args.max_hold_bars:
                target_side, target_notional, exit_reason = 0, 0.0, "time_exit"
            elif unlev_pnl <= -stop_pct:
                target_side, target_notional, exit_reason = 0, 0.0, "stop_loss"
            elif (pos.peak_unlev_pnl > 0) and (pos.peak_unlev_pnl - unlev_pnl >= trail_pct):
                target_side, target_notional, exit_reason = 0, 0.0, "trail_stop"

            if exit_reason == "":
                if (not pos.tp1_done) and unlev_pnl >= tp1_pct:
                    target_notional = min(target_notional, pos.entry_notional * 0.60)
                    pos.tp1_done = True
                if (not pos.tp2_done) and unlev_pnl >= tp2_pct:
                    target_notional = min(target_notional, pos.entry_notional * 0.20)
                    pos.tp2_done = True
                if _continue_ok(pos.side, sig, args):
                    pos.continue_fail_count = 0
                else:
                    pos.continue_fail_count += 1
                    if pos.continue_fail_count == 1:
                        target_notional = min(target_notional, pos.entry_notional * 0.50)
                    elif pos.continue_fail_count >= 2:
                        target_side, target_notional, exit_reason = 0, 0.0, "continue_fail_2bars"

            old_signed = pos.side * pos.notional
            new_signed = target_side * target_notional
            delta = abs(new_signed - old_signed)
            if delta > 0:
                st.equity *= (1.0 - _transaction_cost(delta, args.fee_bps, args.slippage_bps))
                st.equity = max(st.equity, 1e-9)
            if pos.side != 0 and target_side == 0:
                pnl_unlev = pos.side * (curr_open - pos.entry_price) / max(pos.entry_price, 1e-12)
                pnl_eq = pnl_unlev * pos.entry_notional
                st.trades.append(
                    TradeRecord(
                        entry_idx=pos.entry_idx,
                        exit_idx=i,
                        side=pos.side,
                        entry_price=pos.entry_price,
                        exit_price=curr_open,
                        entry_notional=pos.entry_notional,
                        pnl_pct_unlev=float(pnl_unlev),
                        pnl_pct_equity=float(pnl_eq),
                        exit_reason=exit_reason if exit_reason else "manual",
                        hold_bars=pos.bars_held,
                    )
                )
                if pnl_eq < 0:
                    st.loss_streak += 1
                else:
                    st.loss_streak = 0
                if st.loss_streak >= args.loss_streak_limit:
                    st.cooldown_until = i + args.cooldown_bars
                    st.loss_streak = 0
                st.position = PositionState()
            else:
                pos.side = target_side
                pos.notional = target_notional

        # flat -> open
        if st.position.side == 0:
            if st.day_open_block or i <= st.cooldown_until:
                st.blocked_open_count += 1
            else:
                desired = 1 if _open_long(sig, args) else (-1 if _open_short(sig, args) else 0)
                if desired != 0:
                    atr_pct = float(atr[i] / max(curr_open, 1e-12))
                    notional = _target_notional(sig, atr_pct, args, st.weekly_observe_mode)
                    if notional > 0:
                        st.equity *= (1.0 - _transaction_cost(notional, args.fee_bps, args.slippage_bps))
                        st.equity = max(st.equity, 1e-9)
                        st.position = PositionState(
                            side=desired,
                            notional=notional,
                            entry_idx=i,
                            entry_price=curr_open,
                            entry_notional=notional,
                            bars_held=0,
                            continue_fail_count=0,
                            peak_unlev_pnl=0.0,
                        )

        st.equity_curve.append(st.equity)
        st.position_curve.append(st.position.side * st.position.notional)
    return st


def summarize(state: BacktestState, buy_hold: np.ndarray) -> dict[str, float]:
    eq = np.asarray(state.equity_curve, dtype=np.float64)
    if eq.size < 2:
        return {}
    log_rets = np.diff(np.log(np.clip(eq, 1e-12, None)))
    sharpe = float(log_rets.mean() / (log_rets.std() + 1e-8) * math.sqrt(24 * 365))
    peak = np.maximum.accumulate(eq)
    max_dd = float(((eq - peak) / np.clip(peak, 1e-12, None)).min())
    bh_ret = float(buy_hold[-1] - 1.0) if buy_hold.size else 0.0
    wins = [t for t in state.trades if t.pnl_pct_equity > 0]
    losses = [t for t in state.trades if t.pnl_pct_equity < 0]
    gross_p = float(sum(t.pnl_pct_equity for t in wins))
    gross_l = float(-sum(t.pnl_pct_equity for t in losses))
    pf = float(gross_p / gross_l) if gross_l > 1e-12 else float("inf")
    return {
        "strategy_return": float(eq[-1] - 1.0),
        "buy_hold_return": bh_ret,
        "excess_return": float((eq[-1] - 1.0) - bh_ret),
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "num_trades": float(len(state.trades)),
        "win_rate": float(len(wins) / max(1, len(state.trades))),
        "profit_factor": pf,
        "avg_trade_pnl_equity": float(np.mean([t.pnl_pct_equity for t in state.trades])) if state.trades else 0.0,
        "blocked_open_count": float(state.blocked_open_count),
        "final_equity": float(eq[-1]),
    }


def save_trades(path: Path, df, trades: list[TradeRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "entry_time",
                "exit_time",
                "side",
                "entry_price",
                "exit_price",
                "entry_notional",
                "pnl_pct_unlev",
                "pnl_pct_equity",
                "hold_bars",
                "exit_reason",
            ]
        )
        for t in trades:
            w.writerow(
                [
                    df[COL_TIME].iloc[t.entry_idx],
                    df[COL_TIME].iloc[t.exit_idx],
                    "long" if t.side > 0 else "short",
                    f"{t.entry_price:.4f}",
                    f"{t.exit_price:.4f}",
                    f"{t.entry_notional:.6f}",
                    f"{t.pnl_pct_unlev:.6f}",
                    f"{t.pnl_pct_equity:.6f}",
                    t.hold_bars,
                    t.exit_reason,
                ]
            )


def plot_equity(path: Path, strategy_eq: np.ndarray, buy_hold: np.ndarray, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(strategy_eq, label="strategy", linewidth=2.0)
    ax.plot(buy_hold, label="buy&hold", linewidth=1.4, alpha=0.8)
    ax.set_title("Rule v012 backtest equity")
    ax.set_xlabel("bar")
    ax.set_ylabel("equity")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def buy_and_hold_open_to_open(open_px: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    eq = [1.0]
    for i in anchors.tolist():
        if i <= 1 or i >= len(open_px) - 1:
            continue
        ret = (float(open_px[i]) - float(open_px[i - 1])) / max(float(open_px[i - 1]), 1e-12)
        eq.append(eq[-1] * (1.0 + ret))
    return np.asarray(eq, dtype=np.float64)


def main() -> int:
    args = parse_args()
    apply_real_data_defaults(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    merged = _merge_ckpt_args(args, ckpt.get("args", {}))
    df = fetch_ohlcv_df(merged)
    bundle = prepare_bar_series_from_args(df, merged)
    open_px = df[COL_OPEN].to_numpy(dtype=np.float64)
    high = df[COL_HIGH].to_numpy(dtype=np.float64)
    low = df[COL_LOW].to_numpy(dtype=np.float64)
    close = df[COL_CLOSE].to_numpy(dtype=np.float64)
    atr = compute_atr(high, low, close, merged.atr_period)

    model = _build_model(merged, ckpt["model"], device)
    idx = _split_idx(bundle, merged.split)
    start = max(int(idx.min()), merged.context_bars + 1)
    end = min(int(idx.max()), len(open_px) - merged.pred_horizon - 2)
    anchors = np.arange(start, end + 1, merged.stride, dtype=np.int64)
    signals = collect_signals(
        model,
        bundle.bars,
        context_bars=merged.context_bars,
        pred_horizon=merged.pred_horizon,
        anchors=anchors,
        batch_size=merged.batch_size,
        device=device,
    )
    state = run_backtest(df, open_px, atr, signals, anchors, merged)
    bh = buy_and_hold_open_to_open(open_px, anchors)
    metrics = summarize(state, bh)

    save_trades(out_dir / "trades.csv", df, state.trades)
    plot_equity(out_dir / "equity_curve.png", np.asarray(state.equity_curve, dtype=np.float64), bh, merged.dpi)

    payload = {
        "metrics": metrics,
        "config": vars(merged),
        "num_anchors": int(len(anchors)),
        "warnings": state.warnings,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "metrics.txt").write_text(
        "\n".join(
            [
                f"checkpoint={merged.checkpoint}",
                f"split={merged.split}",
                f"anchors={len(anchors)}",
                f"strategy_return={metrics.get('strategy_return', 0.0):.4%}",
                f"buy_hold_return={metrics.get('buy_hold_return', 0.0):.4%}",
                f"excess_return={metrics.get('excess_return', 0.0):.4%}",
                f"sharpe={metrics.get('sharpe', 0.0):.3f}",
                f"max_drawdown={metrics.get('max_drawdown', 0.0):.4%}",
                f"num_trades={int(metrics.get('num_trades', 0.0))}",
                f"win_rate={metrics.get('win_rate', 0.0):.2%}",
                f"profit_factor={metrics.get('profit_factor', 0.0):.3f}",
                f"blocked_open_count={int(metrics.get('blocked_open_count', 0.0))}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved: {out_dir / 'metrics.txt'}")
    print(f"saved: {out_dir / 'metrics.json'}")
    print(f"saved: {out_dir / 'equity_curve.png'}")
    print(f"saved: {out_dir / 'trades.csv'}")
    print(
        f"strategy={metrics.get('strategy_return', 0.0):.2%} "
        f"max_dd={metrics.get('max_drawdown', 0.0):.2%} "
        f"sharpe={metrics.get('sharpe', 0.0):.2f} "
        f"trades={int(metrics.get('num_trades', 0.0))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

