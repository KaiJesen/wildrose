#!/usr/bin/env python3
"""Grid-search v018 lifecycle and trend-hold parameters."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import replace
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
from trading_system.config import TradingSystemConfig, load_config
from trading_system.signal import TradingSignal


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune v018 lifecycle parameters")
    add_data_args(p)
    add_feature_args(p)
    add_segment_args(p)
    add_vq_args(p)
    p.add_argument("--checkpoint", default="prod/v0.0.0/checkpoint/market_state_best.pt")
    p.add_argument("--base-config", default="configs/trading_rule_v018_lifecycle_0062e.json")
    p.add_argument("--split", choices=["train", "valid", "test"], default="valid")
    p.add_argument("--trunk-layers", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default="backtest/018_lifecycle_tuning_valid")
    p.add_argument("--tuned-config-out", default="configs/trading_rule_v018_lifecycle_tuned_0062e.json")
    p.add_argument("--grid-crash-upgrade-profit-atr", default="0.4,0.6,0.8")
    p.add_argument("--grid-upgrade-profit-atr", default="1.0,1.2,1.5")
    p.add_argument("--grid-min-trend-hold-bars", default="4,6")
    p.add_argument("--grid-exit-confirm-votes", default="3,4")
    p.add_argument("--grid-runner-profit-atr", default="4.0,5.0,6.0")
    p.add_argument("--grid-exhaustion-reduce-scale", default="0.75,0.85,0.95")
    p.add_argument("--grid-max-trend-hold-bars", default="36,48")
    p.add_argument("--baseline-return", type=float, default=0.1023)
    p.add_argument("--baseline-mdd", type=float, default=-0.0053)
    p.add_argument("--trade-count-cap", type=int, default=16)
    p.add_argument("--top-k", type=int, default=15)
    p.set_defaults(source="binance_vision", symbol="BTCUSDT", interval="1h", days=365)
    return p.parse_args()


def _parse_float_grid(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_int_grid(text: str) -> list[int]:
    return [int(float(x.strip())) for x in text.split(",") if x.strip()]


def _split_idx(bundle, split: str):
    if split == "train":
        return bundle.train_idx
    if split == "valid":
        return bundle.valid_idx
    return bundle.test_idx


def _score(metrics: dict[str, float], *, baseline_return: float, baseline_mdd: float, trade_count_cap: int) -> float:
    total_return = float(metrics.get("total_return", 0.0))
    max_drawdown = float(metrics.get("max_drawdown", 0.0))
    trades = int(metrics.get("trade_count", 0.0))
    avg_bars = float(metrics.get("avg_bars_held", 0.0))
    trend_upgrades = float(metrics.get("trend_upgrade_count", 0.0))
    trend_hold = float(metrics.get("avg_trend_hold_bars", 0.0))
    short_capture = float(metrics.get("short_trend_capture_ratio", 0.0))
    missed = float(metrics.get("missed_confirmed_trend_bars", 0.0))
    trend_ret = float(metrics.get("trend_trade_total_return", 0.0))
    pf = float(metrics.get("profit_factor", 0.0))
    if not np.isfinite(pf):
        pf = 20.0
    pf = min(pf, 20.0)

    score = (
        12.0 * total_return
        + 3.0 * float(metrics.get("excess_return", 0.0))
        + 4.0 * max_drawdown
        + 0.05 * pf
        + 0.08 * min(avg_bars, 20.0)
        + 0.12 * min(trend_upgrades, 5.0)
        + 0.02 * min(trend_hold, 30.0)
        + 0.25 * short_capture
        + 0.8 * trend_ret
        - 0.00015 * missed
    )
    if total_return < baseline_return * 0.8:
        score -= 0.35
    if max_drawdown < baseline_mdd - 0.02:
        score -= 0.25
    if avg_bars < 5.0:
        score -= 0.15
    if trend_upgrades < 1.0:
        score -= 0.10
    if trades > trade_count_cap:
        score -= (trades - trade_count_cap) * 0.04
    return score


class _CachedSignalProvider:
    def __init__(self, atr: np.ndarray, signals: list[TradingSignal]) -> None:
        self.atr = atr
        self._signals = signals

    def signal_at(self, idx: int) -> TradingSignal:
        return self._signals[idx]


def _collect_signals(provider: ModelSignalProvider, start_idx: int, end_idx: int) -> list[TradingSignal | None]:
    out: list[TradingSignal | None] = [None] * end_idx
    for i in range(start_idx, end_idx):
        out[i] = provider.signal_at(i)
    return out


def _run_with_cfg(
    df,
    provider: _CachedSignalProvider,
    start_idx: int,
    end_idx: int,
    cfg: TradingSystemConfig,
) -> dict[str, float]:
    with tempfile.TemporaryDirectory() as tmp:
        result: BacktestResult = run_backtest(
            df,
            signal_provider=provider,
            start_idx=start_idx,
            end_idx=end_idx,
            cfg=cfg,
            out_dir=Path(tmp),
        )
    return result.metrics


def _cfg_to_json_dict(cfg: TradingSystemConfig) -> dict:
    return {
        "base": cfg.base.__dict__,
        "rule": cfg.rule.__dict__,
        "risk": cfg.risk.__dict__,
        "trend": cfg.trend.__dict__,
        "protection": cfg.protection.__dict__,
        "trend_hold": cfg.trend_hold.__dict__,
        "trend_signal": cfg.trend_signal.__dict__,
        "trend_position": cfg.trend_position.__dict__,
        "trend_lifecycle": cfg.trend_lifecycle.__dict__,
        "crash": cfg.crash.__dict__,
        "crash_short": cfg.crash_short.__dict__,
        "sizing": {
            **cfg.sizing.__dict__,
            "weak_range": list(cfg.sizing.weak_range),
            "medium_range": list(cfg.sizing.medium_range),
            "strong_range": list(cfg.sizing.strong_range),
            "very_strong_range": list(cfg.sizing.very_strong_range),
        },
        "execution": cfg.execution.__dict__,
    }


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
    print(f"collecting signals for bars [{start_idx},{end_idx}) ...")
    raw_signals = _collect_signals(model_provider, start_idx, end_idx)
    cached = _CachedSignalProvider(model_provider.atr, raw_signals)  # type: ignore[arg-type]

    crash_upgrade_grid = _parse_float_grid(args.grid_crash_upgrade_profit_atr)
    upgrade_grid = _parse_float_grid(args.grid_upgrade_profit_atr)
    min_hold_grid = _parse_int_grid(args.grid_min_trend_hold_bars)
    exit_votes_grid = _parse_int_grid(args.grid_exit_confirm_votes)
    runner_grid = _parse_float_grid(args.grid_runner_profit_atr)
    exhaustion_grid = _parse_float_grid(args.grid_exhaustion_reduce_scale)
    max_hold_grid = _parse_int_grid(args.grid_max_trend_hold_bars)

    rows: list[dict] = []
    done = 0
    total = (
        len(crash_upgrade_grid)
        * len(upgrade_grid)
        * len(min_hold_grid)
        * len(exit_votes_grid)
        * len(runner_grid)
        * len(exhaustion_grid)
        * len(max_hold_grid)
    )
    for crash_up in crash_upgrade_grid:
        for upgrade in upgrade_grid:
            for min_hold in min_hold_grid:
                for exit_votes in exit_votes_grid:
                    for runner_atr in runner_grid:
                        for exhaustion_scale in exhaustion_grid:
                            for max_hold in max_hold_grid:
                                done += 1
                                cfg = replace(
                                    base_cfg,
                                    trend_position=replace(
                                        base_cfg.trend_position,
                                        crash_upgrade_profit_atr=crash_up,
                                        upgrade_profit_atr=upgrade,
                                        max_trend_hold_bars=max_hold,
                                        exhaustion_reduce_scale=exhaustion_scale,
                                    ),
                                    trend_lifecycle=replace(
                                        base_cfg.trend_lifecycle,
                                        min_trend_hold_bars=min_hold,
                                        exit_confirm_votes=exit_votes,
                                        runner_profit_atr=runner_atr,
                                    ),
                                )
                                metrics = _run_with_cfg(df, cached, start_idx, end_idx, cfg)
                                row = {
                                    "crash_upgrade_profit_atr": crash_up,
                                    "upgrade_profit_atr": upgrade,
                                    "min_trend_hold_bars": min_hold,
                                    "exit_confirm_votes": exit_votes,
                                    "runner_profit_atr": runner_atr,
                                    "exhaustion_reduce_scale": exhaustion_scale,
                                    "max_trend_hold_bars": max_hold,
                                    "metrics": metrics,
                                    "score": _score(
                                        metrics,
                                        baseline_return=args.baseline_return,
                                        baseline_mdd=args.baseline_mdd,
                                        trade_count_cap=args.trade_count_cap,
                                    ),
                                }
                                rows.append(row)
                                if done % 50 == 0 or done == total:
                                    print(f"  progress {done}/{total}")

    rows.sort(key=lambda r: r["score"], reverse=True)
    top = rows[: args.top_k]
    best = rows[0]
    best_cfg = replace(
        base_cfg,
        trend_position=replace(
            base_cfg.trend_position,
            crash_upgrade_profit_atr=best["crash_upgrade_profit_atr"],
            upgrade_profit_atr=best["upgrade_profit_atr"],
            max_trend_hold_bars=best["max_trend_hold_bars"],
            exhaustion_reduce_scale=best["exhaustion_reduce_scale"],
        ),
        trend_lifecycle=replace(
            base_cfg.trend_lifecycle,
            min_trend_hold_bars=best["min_trend_hold_bars"],
            exit_confirm_votes=best["exit_confirm_votes"],
            runner_profit_atr=best["runner_profit_atr"],
        ),
    )

    results_path = out_dir / "tuning_results.json"
    payload = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "base_config": args.base_config,
        "best": {k: v for k, v in best.items() if k != "metrics"},
        "best_metrics": best["metrics"],
        "top": [{k: v for k, v in r.items() if k != "metrics"} | {"metrics": r["metrics"]} for r in top],
    }
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    tuned_path = Path(args.tuned_config_out)
    tuned_dict = _cfg_to_json_dict(best_cfg)
    tuned_dict["_tuning_meta"] = {
        "source": "tune_backtest_trading_system_v018.py",
        "checkpoint": args.checkpoint,
        "tune_split": args.split,
        "score": best["score"],
        "valid_metrics": best["metrics"],
    }
    tuned_path.write_text(json.dumps(tuned_dict, indent=2), encoding="utf-8")
    print(f"saved tuning results: {results_path}")
    print(f"saved tuned config: {tuned_path}")
    m = best["metrics"]
    print(
        f"best score={best['score']:.4f} return={m['total_return']:.2%} "
        f"mdd={m['max_drawdown']:.2%} trades={int(m['trade_count'])} "
        f"avg_bars={m['avg_bars_held']:.1f} trend_up={m['trend_upgrade_count']:.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
