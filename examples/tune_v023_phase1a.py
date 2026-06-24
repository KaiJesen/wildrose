#!/usr/bin/env python3
"""Tune v023 Phase 1a slow-up probe params on valid split."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np

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
from trading_system.adapters.market_state_model import ModelSignalProvider
from trading_system.backtest.runner import BacktestResult, run_backtest
from trading_system.config import load_config
from trading_system.signal import TradingSignal


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune v023 phase1a slow-up probe")
    add_data_args(p)
    add_feature_args(p)
    add_segment_args(p)
    add_vq_args(p)
    p.add_argument("--checkpoint", default="prod/v0.0.0/checkpoint/market_state_best.pt")
    p.add_argument("--base-config", default="configs/trading_rule_v023_phase1a_0062e.json")
    p.add_argument("--split", choices=["train", "valid", "test"], default="valid")
    p.add_argument("--trunk-layers", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default="backtest/v023_phase1a_tune")
    p.add_argument("--tuned-config-out", default="configs/trading_rule_v023_phase1a1_0062e.json")
    p.add_argument("--baseline-return", type=float, default=0.0409)
    p.add_argument("--top-k", type=int, default=10)
    p.set_defaults(source="binance_vision", symbol="BTCUSDT", interval="1h", days=365)
    return p.parse_args()


def _split_idx(bundle, split: str):
    if split == "train":
        return bundle.train_idx
    if split == "valid":
        return bundle.valid_idx
    return bundle.test_idx


def _score(metrics: dict[str, float], *, baseline_return: float) -> float:
    total_return = float(metrics.get("total_return", 0.0))
    max_drawdown = float(metrics.get("max_drawdown", 0.0))
    slow_ret = float(metrics.get("slow_up_trade_total_return", 0.0))
    slow_opens = float(metrics.get("slow_up_open_count", 0.0))
    leg_cov = float(metrics.get("leg_coverage_ratio", 0.0))
    long_cap = float(metrics.get("long_trend_capture_ratio", 0.0))
    trades = int(metrics.get("trade_count", 0))

    score = (
        20.0 * total_return
        + 5.0 * max_drawdown
        + 2.0 * slow_ret
        + 0.4 * leg_cov
        + 0.3 * long_cap
    )
    if slow_opens < 1:
        score -= 1.0
    if total_return < baseline_return * 0.5:
        score -= 0.5
    if max_drawdown < -0.06:
        score -= 0.8
    if leg_cov < 0.12:
        score -= 0.3
    if trades > 22:
        score -= (trades - 22) * 0.04
    return score


class _CachedSignalProvider:
    def __init__(self, atr: np.ndarray, signals: list[TradingSignal]) -> None:
        self.atr = atr
        self._signals = signals

    def signal_at(self, idx: int) -> TradingSignal:
        return self._signals[idx]


def _run_with_cfg(df, provider, start_idx, end_idx, cfg) -> dict[str, float]:
    with tempfile.TemporaryDirectory() as tmp:
        result: BacktestResult = run_backtest(
            df, signal_provider=provider, start_idx=start_idx, end_idx=end_idx, cfg=cfg, out_dir=Path(tmp)
        )
    return result.metrics


def main() -> int:
    args = parse_args()
    apply_real_data_defaults(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(args.base_config)
    df = fetch_ohlcv_df(args)
    bundle = prepare_bar_series_from_args(df, args)
    idx = _split_idx(bundle, args.split)
    start_idx = max(int(idx.min()), args.context_bars + 1)
    end_idx = min(int(idx.max()), len(df) - 2)

    model_provider = ModelSignalProvider.from_checkpoint(
        checkpoint=args.checkpoint,
        bars=bundle.bars,
        df=df,
        context_bars=args.context_bars,
        d_model=args.d_model,
        n_heads=args.n_heads,
        trunk_layers=args.trunk_layers,
        trend_features=args.trend_features,
        trend_windows=tuple(args.trend_windows),
        max_seg_len=args.max_seg_len,
        max_segments=args.max_segments,
        min_seg_len=args.min_seg_len,
        num_codes=args.num_codes,
        vq_beta=args.vq_beta,
        vq_inverse_freq_ema=args.vq_inverse_freq_ema,
        cfg=base_cfg,
        device=args.device,
    )
    signals: list[TradingSignal | None] = [None] * end_idx
    for i in range(start_idx, end_idx):
        signals[i] = model_provider.signal_at(i)
    cached = _CachedSignalProvider(model_provider.atr, signals)  # type: ignore[arg-type]

    grid = {
        "allow_trend_upgrade": [False, True],
        "watch_min_bars": [10, 12],
        "position_ratio": [0.025, 0.03],
        "watch_probe_min_cum_ret": [-0.05, 0.0],
        "upgrade_profit_atr": [1.5, 2.0],
    }
    keys = list(grid.keys())
    results: list[dict] = []

    for combo in product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        sp = replace(
            base_cfg.slow_up_position,
            allow_trend_upgrade=params["allow_trend_upgrade"],
            watch_min_bars=params["watch_min_bars"],
            position_ratio=params["position_ratio"],
            watch_probe_min_cum_ret=params["watch_probe_min_cum_ret"],
            upgrade_profit_atr=params["upgrade_profit_atr"],
        )
        cfg = replace(base_cfg, slow_up_position=sp)
        metrics = _run_with_cfg(df, cached, start_idx, end_idx, cfg)
        row = {**params, **{k: float(metrics[k]) for k in (
            "total_return", "max_drawdown", "trade_count", "slow_up_open_count",
            "slow_up_trade_total_return", "leg_coverage_ratio", "long_trend_capture_ratio",
        )}}
        row["score"] = _score(metrics, baseline_return=args.baseline_return)
        results.append(row)
        print(
            f"score={row['score']:.3f} ret={row['total_return']:.4f} "
            f"slow_opens={int(row['slow_up_open_count'])} params={params}"
        )

    results.sort(key=lambda r: r["score"], reverse=True)
    (out_dir / "tuning_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    best = results[0]
    tuned_sp = replace(
        base_cfg.slow_up_position,
        **{k: best[k] for k in keys},
    )
    tuned_cfg = replace(base_cfg, slow_up_position=tuned_sp)
    payload = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    payload["slow_up_position"] = {**payload["slow_up_position"], **{k: best[k] for k in keys}}
    payload["_023_meta"] = {
        "phase": "1a1",
        "recipe": "slow_up_watch_probe_tuned",
        "source_phase1a": args.base_config,
        "tune_split": args.split,
        "tune_score": best["score"],
    }
    tuned_path = Path(args.tuned_config_out)
    tuned_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# v023 Phase 1a tune", "", f"split: {args.split}", "", "## top results", ""]
    for r in results[: args.top_k]:
        lines.append(
            f"- score={r['score']:.3f} ret={r['total_return']:.4f} dd={r['max_drawdown']:.4f} "
            f"slow_opens={int(r['slow_up_open_count'])} slow_ret={r['slow_up_trade_total_return']:.4f} { {k: r[k] for k in keys} }"
        )
    lines.append(f"\nBest config: `{tuned_path}`")
    (out_dir / "TUNING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nBest score={best['score']:.3f} -> {tuned_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
