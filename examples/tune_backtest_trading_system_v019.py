#!/usr/bin/env python3
"""Grid-search v019 slow-uptrend parameters on top of v018 tuned2 base."""

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
    p = argparse.ArgumentParser(description="Tune v019 slow uptrend parameters")
    add_data_args(p)
    add_feature_args(p)
    add_segment_args(p)
    add_vq_args(p)
    p.add_argument("--checkpoint", default="prod/v0.0.0/checkpoint/market_state_best.pt")
    p.add_argument("--base-config", default="configs/trading_rule_v019_slow_uptrend_0062e.json")
    p.add_argument("--split", choices=["train", "valid", "test"], default="valid")
    p.add_argument("--trunk-layers", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default="backtest/019_slow_uptrend_tuning_valid")
    p.add_argument("--tuned-config-out", default="configs/trading_rule_v019_slow_uptrend_tuned_0062e.json")
    p.add_argument("--grid-stable-score", default="7,8")
    p.add_argument("--grid-stable-slope-48", default="2.5,3.0,3.5")
    p.add_argument("--grid-stable-slope-24", default="1.8,2.2")
    p.add_argument("--grid-stable-persist-fast", default="0.75,0.80")
    p.add_argument("--grid-upgrade-profit-atr", default="1.0,1.2,1.5")
    p.add_argument("--grid-stable-position-ratio", default="0.04,0.06")
    p.add_argument("--grid-exit-votes", default="3,4")
    p.add_argument("--baseline-return", type=float, default=0.1105)
    p.add_argument("--baseline-missed-slow", type=float, default=400.0)
    p.add_argument("--trade-count-cap", type=int, default=14)
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


def _score(
    metrics: dict[str, float],
    *,
    baseline_return: float,
    baseline_missed_slow: float,
    trade_count_cap: int,
) -> float:
    total_return = float(metrics.get("total_return", 0.0))
    max_drawdown = float(metrics.get("max_drawdown", 0.0))
    trades = int(metrics.get("trade_count", 0.0))
    slow_ret = float(metrics.get("slow_up_trade_total_return", 0.0))
    slow_hold = float(metrics.get("avg_slow_up_hold_bars", 0.0))
    long_capture = float(metrics.get("long_trend_capture_ratio", 0.0))
    missed_slow = float(metrics.get("missed_slow_uptrend_bars", 0.0))
    slow_opens = float(metrics.get("slow_up_open_count", 0.0))
    pf = float(metrics.get("profit_factor", 0.0))
    if not np.isfinite(pf):
        pf = 20.0
    pf = min(pf, 20.0)

    score = (
        14.0 * total_return
        + 3.0 * float(metrics.get("excess_return", 0.0))
        + 4.0 * max_drawdown
        + 0.04 * pf
        + 1.2 * slow_ret
        + 0.03 * min(slow_hold, 30.0)
        + 0.35 * long_capture
        + 0.0003 * max(0.0, baseline_missed_slow - missed_slow)
    )
    if total_return < baseline_return * 0.8:
        score -= 0.40
    if slow_ret <= 0 and slow_opens >= 1:
        score -= 0.25
    if slow_opens < 1:
        score -= 0.15
    if slow_hold < 8 and slow_opens >= 1:
        score -= 0.10
    if trades > trade_count_cap:
        score -= (trades - trade_count_cap) * 0.05
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


def _run_with_cfg(df, provider, start_idx, end_idx, cfg) -> dict[str, float]:
    with tempfile.TemporaryDirectory() as tmp:
        result: BacktestResult = run_backtest(
            df, signal_provider=provider, start_idx=start_idx, end_idx=end_idx, cfg=cfg, out_dir=Path(tmp)
        )
    return result.metrics


def _cfg_to_json_dict(cfg: TradingSystemConfig) -> dict:
    return json.loads(json.dumps({
        "base": cfg.base.__dict__,
        "rule": cfg.rule.__dict__,
        "risk": cfg.risk.__dict__,
        "trend": cfg.trend.__dict__,
        "protection": cfg.protection.__dict__,
        "trend_hold": cfg.trend_hold.__dict__,
        "trend_signal": cfg.trend_signal.__dict__,
        "trend_position": cfg.trend_position.__dict__,
        "trend_lifecycle": cfg.trend_lifecycle.__dict__,
        "slow_uptrend": cfg.slow_uptrend.__dict__,
        "slow_up_position": cfg.slow_up_position.__dict__,
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
    }))


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
    print(f"collecting signals [{start_idx},{end_idx}) ...")
    cached = _CachedSignalProvider(model_provider.atr, _collect_signals(model_provider, start_idx, end_idx))  # type: ignore[arg-type]

    stable_score_grid = _parse_int_grid(args.grid_stable_score)
    slope48_grid = _parse_float_grid(args.grid_stable_slope_48)
    slope24_grid = _parse_float_grid(args.grid_stable_slope_24)
    persist_grid = _parse_float_grid(args.grid_stable_persist_fast)
    upgrade_grid = _parse_float_grid(args.grid_upgrade_profit_atr)
    pos_grid = _parse_float_grid(args.grid_stable_position_ratio)
    exit_grid = _parse_int_grid(args.grid_exit_votes)

    rows: list[dict] = []
    done = 0
    total = len(stable_score_grid) * len(slope48_grid) * len(slope24_grid) * len(persist_grid) * len(upgrade_grid) * len(pos_grid) * len(exit_grid)
    for stable_score in stable_score_grid:
        for slope48 in slope48_grid:
            for slope24 in slope24_grid:
                for persist in persist_grid:
                    for upgrade in upgrade_grid:
                        for pos_ratio in pos_grid:
                            for exit_votes in exit_grid:
                                done += 1
                                cfg = replace(
                                    base_cfg,
                                    slow_uptrend=replace(
                                        base_cfg.slow_uptrend,
                                        stable_score=stable_score,
                                        stable_slope_48_atr_min=slope48,
                                        stable_slope_24_atr_min=slope24,
                                        stable_persistence_fast_min=persist,
                                    ),
                                    slow_up_position=replace(
                                        base_cfg.slow_up_position,
                                        upgrade_profit_atr=upgrade,
                                        stable_position_ratio=pos_ratio,
                                        position_ratio=min(pos_ratio, base_cfg.slow_up_position.position_ratio),
                                        exit_votes=exit_votes,
                                    ),
                                )
                                metrics = _run_with_cfg(df, cached, start_idx, end_idx, cfg)
                                row = {
                                    "stable_score": stable_score,
                                    "stable_slope_48_atr_min": slope48,
                                    "stable_slope_24_atr_min": slope24,
                                    "stable_persistence_fast_min": persist,
                                    "upgrade_profit_atr": upgrade,
                                    "stable_position_ratio": pos_ratio,
                                    "exit_votes": exit_votes,
                                    "metrics": metrics,
                                    "score": _score(
                                        metrics,
                                        baseline_return=args.baseline_return,
                                        baseline_missed_slow=args.baseline_missed_slow,
                                        trade_count_cap=args.trade_count_cap,
                                    ),
                                }
                                rows.append(row)
                                if done % 50 == 0 or done == total:
                                    print(f"  progress {done}/{total}")

    rows.sort(key=lambda r: r["score"], reverse=True)
    best = rows[0]
    best_cfg = replace(
        base_cfg,
        slow_uptrend=replace(
            base_cfg.slow_uptrend,
            stable_score=best["stable_score"],
            stable_slope_48_atr_min=best["stable_slope_48_atr_min"],
            stable_slope_24_atr_min=best["stable_slope_24_atr_min"],
            stable_persistence_fast_min=best["stable_persistence_fast_min"],
        ),
        slow_up_position=replace(
            base_cfg.slow_up_position,
            upgrade_profit_atr=best["upgrade_profit_atr"],
            stable_position_ratio=best["stable_position_ratio"],
            position_ratio=min(best["stable_position_ratio"], base_cfg.slow_up_position.position_ratio),
            exit_votes=best["exit_votes"],
        ),
    )

    payload = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "best": {k: v for k, v in best.items() if k != "metrics"},
        "best_metrics": best["metrics"],
        "top": [{k: v for k, v in r.items() if k != "metrics"} | {"metrics": r["metrics"]} for r in rows[: args.top_k]],
    }
    (out_dir / "tuning_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    tuned_dict = _cfg_to_json_dict(best_cfg)
    tuned_dict["_tuning_meta"] = {
        "source": "tune_backtest_trading_system_v019.py",
        "checkpoint": args.checkpoint,
        "tune_split": args.split,
        "score": best["score"],
        "valid_metrics": best["metrics"],
        "base": "v018_lifecycle_tuned2 + slow_uptrend",
    }
    Path(args.tuned_config_out).write_text(json.dumps(tuned_dict, indent=2), encoding="utf-8")
    m = best["metrics"]
    print(
        f"best score={best['score']:.4f} return={m['total_return']:.2%} "
        f"slow_ret={m.get('slow_up_trade_total_return',0):.2%} "
        f"slow_opens={int(m.get('slow_up_open_count',0))} trades={int(m['trade_count'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
