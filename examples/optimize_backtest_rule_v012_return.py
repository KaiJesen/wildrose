#!/usr/bin/env python3
"""Random-search optimizer for v012 backtest return."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_EX = Path(__file__).resolve().parent
_ROOT = _EX.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))

from _train_common import apply_real_data_defaults, fetch_ohlcv_df, prepare_bar_series_from_args
from backtest_market_state_rule_v012 import (
    _build_model,
    _merge_ckpt_args,
    _split_idx,
    _sync_time_gates,
    buy_and_hold_open_to_open,
    collect_signals,
    compute_atr,
    parse_args as parse_backtest_args,
    run_backtest,
    summarize,
)
from market_data.schema import COL_CLOSE, COL_HIGH, COL_LOW, COL_OPEN
from transformer_kit.train_utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimize v012 backtest parameters for return")
    p.add_argument("--checkpoint", default="checkpoints/0062e_market_state_return_ic_recovery/market_state_best.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--split", choices=["train", "valid", "test"], default="test")
    p.add_argument("--trials", type=int, default=2500)
    p.add_argument("--min-trades", type=int, default=20)
    p.add_argument("--max-dd-cap", type=float, default=0.12, help="reject if max drawdown exceeds cap")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="backtest/backtest_rule_v012_0062e_opt_return")
    return p.parse_args()


def _sample_param(rng: random.Random) -> dict[str, float | int]:
    # Focused around feasible region found by first-stage grid.
    return {
        "open_edge_threshold": rng.uniform(0.02, 0.12),
        "open_prob_threshold": rng.uniform(0.26, 0.40),
        "open_flat_max": rng.uniform(0.36, 0.55),
        "risk_ok_threshold": rng.uniform(0.38, 0.60),
        "risk_exit_threshold": rng.uniform(0.44, 0.65),
        "long_continue_edge_min": rng.uniform(-0.08, 0.02),
        "short_continue_edge_max": rng.uniform(-0.02, 0.08),
        "reverse_edge_threshold": rng.uniform(0.03, 0.10),
        "max_hold_bars": rng.randint(4, 10),
        "risk_budget": rng.uniform(0.001, 0.008),
        "stop_atr_mult": rng.uniform(0.8, 2.0),
        "tp1_atr_mult": rng.uniform(0.6, 1.8),
        "tp2_atr_mult": rng.uniform(1.2, 3.5),
        "trail_atr_mult": rng.uniform(0.4, 1.5),
    }


def _objective(metrics: dict[str, float], *, min_trades: int, max_dd_cap: float) -> float:
    trades = int(metrics.get("num_trades", 0))
    max_dd = float(metrics.get("max_drawdown", 0.0))
    ret = float(metrics.get("strategy_return", 0.0))
    sharpe = float(metrics.get("sharpe", 0.0))
    pf = float(metrics.get("profit_factor", 0.0))
    if trades < min_trades:
        return -10.0 + ret
    if max_dd < -max_dd_cap:
        return -5.0 + ret
    if not np.isfinite(pf):
        pf = 3.0
    # Return-centric objective with moderate stability regularization.
    return 2.0 * ret + 0.20 * sharpe + 0.08 * pf + 0.6 * max_dd


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        base = parse_backtest_args()
    finally:
        sys.argv = old_argv
    apply_real_data_defaults(base)
    base.checkpoint = args.checkpoint
    base.split = args.split
    base.device = args.device

    ckpt = load_checkpoint(base.checkpoint, map_location=device)
    merged = _merge_ckpt_args(base, ckpt.get("args", {}))
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
    buy_hold = buy_and_hold_open_to_open(open_px, anchors)

    best_row: dict | None = None
    top_rows: list[dict] = []
    for t in range(1, args.trials + 1):
        sample = _sample_param(rng)
        run_args = SimpleNamespace(**vars(merged))
        for k, v in sample.items():
            setattr(run_args, k, v)
        # Ensure logical order.
        run_args.risk_exit_threshold = max(float(run_args.risk_exit_threshold), float(run_args.risk_ok_threshold) + 0.02)
        run_args.tp2_atr_mult = max(float(run_args.tp2_atr_mult), float(run_args.tp1_atr_mult) + 0.25)
        if hasattr(_sync_time_gates, "_last_ts"):
            delattr(_sync_time_gates, "_last_ts")
        state = run_backtest(df, open_px, atr, signals, anchors, run_args)
        m = summarize(state, buy_hold)
        obj = _objective(m, min_trades=args.min_trades, max_dd_cap=args.max_dd_cap)
        row = {**sample, **m, "objective": obj}
        if best_row is None or obj > best_row["objective"]:
            best_row = row
        top_rows.append(row)
        if t % 200 == 0 or t == args.trials:
            print(f"progress {t}/{args.trials} best_obj={best_row['objective']:.4f} best_ret={best_row['strategy_return']:.2%}")

    assert best_row is not None
    top_rows = sorted(top_rows, key=lambda x: x["objective"], reverse=True)[:30]
    payload = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "trials": args.trials,
        "min_trades": args.min_trades,
        "max_dd_cap": args.max_dd_cap,
        "num_anchors": int(len(anchors)),
        "best": best_row,
        "top30": top_rows,
    }
    (out_dir / "opt_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"checkpoint={args.checkpoint}",
        f"split={args.split}",
        f"trials={args.trials}",
        f"num_anchors={len(anchors)}",
        f"min_trades={args.min_trades}",
        f"max_dd_cap={args.max_dd_cap:.2%}",
        "",
        "[best]",
    ]
    for k in (
        "open_edge_threshold",
        "open_prob_threshold",
        "open_flat_max",
        "risk_ok_threshold",
        "risk_exit_threshold",
        "long_continue_edge_min",
        "short_continue_edge_max",
        "reverse_edge_threshold",
        "max_hold_bars",
        "risk_budget",
        "stop_atr_mult",
        "tp1_atr_mult",
        "tp2_atr_mult",
        "trail_atr_mult",
        "strategy_return",
        "buy_hold_return",
        "excess_return",
        "max_drawdown",
        "sharpe",
        "num_trades",
        "win_rate",
        "profit_factor",
        "objective",
    ):
        v = best_row[k]
        if isinstance(v, float):
            lines.append(f"{k}={v:.6f}")
        else:
            lines.append(f"{k}={v}")
    lines.append("")
    lines.append("[top10]")
    for i, r in enumerate(top_rows[:10], start=1):
        lines.append(
            f"{i:02d}) ret={r['strategy_return']:.2%} mdd={r['max_drawdown']:.2%} sharpe={r['sharpe']:.2f} "
            f"trades={int(r['num_trades'])} win={r['win_rate']:.1%} pf={r['profit_factor']:.3f} obj={r['objective']:.4f} "
            f"(edge={r['open_edge_threshold']:.3f}, prob={r['open_prob_threshold']:.3f}, flat={r['open_flat_max']:.3f}, "
            f"risk_ok={r['risk_ok_threshold']:.3f}, risk_exit={r['risk_exit_threshold']:.3f}, "
            f"stop={r['stop_atr_mult']:.2f}, tp1={r['tp1_atr_mult']:.2f}, tp2={r['tp2_atr_mult']:.2f}, trail={r['trail_atr_mult']:.2f}, "
            f"hold={int(r['max_hold_bars'])}, rb={r['risk_budget']:.4f})"
        )
    (out_dir / "opt_results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"saved: {out_dir / 'opt_results.txt'}")
    print(f"saved: {out_dir / 'opt_results.json'}")
    print(
        f"best return={best_row['strategy_return']:.2%} "
        f"max_dd={best_row['max_drawdown']:.2%} "
        f"trades={int(best_row['num_trades'])} "
        f"objective={best_row['objective']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

