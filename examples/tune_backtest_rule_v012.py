#!/usr/bin/env python3
"""Grid-search threshold tuning for backtest_rule_v012."""

from __future__ import annotations

import argparse
import json
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
    p = argparse.ArgumentParser(description="Tune threshold parameters for v012 rule")
    p.add_argument("--checkpoint", default="checkpoints/0062e_market_state_return_ic_recovery/market_state_best.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default="backtest/backtest_rule_v012_0062e_tuning")
    p.add_argument("--split", choices=["train", "valid", "test"], default="test")
    p.add_argument("--grid-edge", default="0.02,0.04,0.06,0.08")
    p.add_argument("--grid-prob", default="0.30,0.34,0.38,0.42")
    p.add_argument("--grid-flat", default="0.34,0.38,0.42,0.46")
    p.add_argument("--grid-risk-ok", default="0.38,0.42,0.46,0.50")
    p.add_argument("--grid-risk-exit", default="0.48,0.52,0.56")
    p.add_argument("--min-trades", type=int, default=5)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _parse_grid(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _score(metrics: dict[str, float]) -> float:
    # Balance return vs drawdown; penalize too-few trades.
    trades = metrics.get("num_trades", 0.0)
    trade_penalty = 0.0 if trades >= 5 else (5 - trades) * 0.02
    return (
        1.5 * metrics.get("strategy_return", 0.0)
        + 0.25 * metrics.get("sharpe", 0.0)
        + 0.5 * metrics.get("profit_factor", 0.0 if np.isfinite(metrics.get("profit_factor", 0.0)) else 3.0)
        + 2.0 * metrics.get("max_drawdown", 0.0)  # max_drawdown is negative
        - trade_penalty
    )


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reuse baseline backtest args for all non-grid parameters.
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        base_args = parse_backtest_args()
    finally:
        sys.argv = old_argv
    apply_real_data_defaults(base_args)
    base_args.checkpoint = args.checkpoint
    base_args.split = args.split
    base_args.device = args.device

    ckpt = load_checkpoint(base_args.checkpoint, map_location=device)
    merged = _merge_ckpt_args(base_args, ckpt.get("args", {}))
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

    edge_grid = _parse_grid(args.grid_edge)
    prob_grid = _parse_grid(args.grid_prob)
    flat_grid = _parse_grid(args.grid_flat)
    risk_ok_grid = _parse_grid(args.grid_risk_ok)
    risk_exit_grid = _parse_grid(args.grid_risk_exit)

    rows: list[dict] = []
    total = len(edge_grid) * len(prob_grid) * len(flat_grid) * len(risk_ok_grid) * len(risk_exit_grid)
    done = 0
    for edge in edge_grid:
        for prob in prob_grid:
            for flat in flat_grid:
                for risk_ok in risk_ok_grid:
                    for risk_exit in risk_exit_grid:
                        done += 1
                        run_args = SimpleNamespace(**vars(merged))
                        run_args.open_edge_threshold = edge
                        run_args.open_prob_threshold = prob
                        run_args.open_flat_max = flat
                        run_args.risk_ok_threshold = risk_ok
                        run_args.risk_exit_threshold = risk_exit
                        # Reset time gate function static memory between runs.
                        if hasattr(_sync_time_gates, "_last_ts"):
                            delattr(_sync_time_gates, "_last_ts")
                        state = run_backtest(df, open_px, atr, signals, anchors, run_args)
                        m = summarize(state, buy_hold)
                        row = {
                            "open_edge_threshold": edge,
                            "open_prob_threshold": prob,
                            "open_flat_max": flat,
                            "risk_ok_threshold": risk_ok,
                            "risk_exit_threshold": risk_exit,
                            **m,
                        }
                        row["score"] = _score(m)
                        rows.append(row)
                        if done % 50 == 0 or done == total:
                            print(f"grid progress: {done}/{total}")

    # Sort and keep full + filtered views.
    rows_sorted = sorted(rows, key=lambda x: x["score"], reverse=True)
    viable = [r for r in rows_sorted if int(r.get("num_trades", 0)) >= args.min_trades]
    best = viable[0] if viable else rows_sorted[0]
    top_rows = (viable if viable else rows_sorted)[: args.top_k]

    payload = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "num_anchors": int(len(anchors)),
        "grid_size": int(total),
        "min_trades_filter": int(args.min_trades),
        "best": best,
        "top": top_rows,
    }
    (out_dir / "tuning_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"checkpoint={args.checkpoint}",
        f"split={args.split}",
        f"num_anchors={len(anchors)}",
        f"grid_size={total}",
        f"min_trades_filter={args.min_trades}",
        "",
        "[best]",
    ]
    for k in (
        "open_edge_threshold",
        "open_prob_threshold",
        "open_flat_max",
        "risk_ok_threshold",
        "risk_exit_threshold",
        "strategy_return",
        "buy_hold_return",
        "excess_return",
        "sharpe",
        "max_drawdown",
        "num_trades",
        "win_rate",
        "profit_factor",
        "score",
    ):
        v = best.get(k)
        if isinstance(v, float):
            lines.append(f"{k}={v:.6f}")
        else:
            lines.append(f"{k}={v}")
    lines.append("")
    lines.append("[top]")
    for i, r in enumerate(top_rows, start=1):
        lines.append(
            f"{i:02d}) edge={r['open_edge_threshold']:.3f} prob={r['open_prob_threshold']:.3f} "
            f"flat={r['open_flat_max']:.3f} risk_ok={r['risk_ok_threshold']:.3f} risk_exit={r['risk_exit_threshold']:.3f} "
            f"ret={r['strategy_return']:.2%} mdd={r['max_drawdown']:.2%} trades={int(r['num_trades'])} "
            f"win={r['win_rate']:.1%} pf={r['profit_factor']:.3f} score={r['score']:.4f}"
        )
    (out_dir / "tuning_results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"saved: {out_dir / 'tuning_results.txt'}")
    print(f"saved: {out_dir / 'tuning_results.json'}")
    print(
        f"best ret={best.get('strategy_return', 0.0):.2%} "
        f"mdd={best.get('max_drawdown', 0.0):.2%} "
        f"trades={int(best.get('num_trades', 0))} "
        f"score={best.get('score', 0.0):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

