#!/usr/bin/env python3
"""Grid-search BestPoint (017c major_legs) integration on v020 base config."""

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
from trading_system.adapters.best_point_model import BestPointSignalProvider
from trading_system.adapters.market_state_model import ModelSignalProvider
from trading_system.backtest.runner import BacktestResult, run_backtest
from trading_system.config import TradingSystemConfig, load_config
from trading_system.signal import TradingSignal


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune BestPoint entry/exit integration (017c on v020)")
    add_data_args(p)
    add_feature_args(p)
    add_segment_args(p)
    add_vq_args(p)
    p.add_argument("--checkpoint", default="prod/v0.0.0/checkpoint/market_state_best.pt")
    p.add_argument(
        "--best-point-checkpoint",
        default="checkpoints/017_best_point_signal/017c_best_point_major_legs/best.pt",
    )
    p.add_argument("--best-point-context-bars", type=int, default=96)
    p.add_argument("--base-config", default="configs/trading_rule_v020_trend_segment_tuned_0062e.json")
    p.add_argument("--split", choices=["train", "valid", "test"], default="valid")
    p.add_argument("--trunk-layers", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default="backtest/020_best_point_tuning_valid")
    p.add_argument("--tuned-config-out", default="configs/trading_rule_v020_best_point_tuned_0062e.json")
    p.add_argument("--grid-long-entry", default="0.40,0.50,0.60")
    p.add_argument("--grid-short-entry", default="0.40,0.50,0.60")
    p.add_argument("--grid-exit-prob", default="0.65,0.75")
    p.add_argument("--grid-bp-exit-bars", default="2,3")
    p.add_argument("--grid-min-opportunity", default="0.0,0.08")
    p.add_argument("--baseline-return", type=float, default=0.1049)
    p.add_argument("--baseline-mdd", type=float, default=-0.0123)
    p.add_argument("--trade-count-cap", type=int, default=14)
    p.add_argument("--top-k", type=int, default=12)
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
    pf = float(metrics.get("profit_factor", 0.0))
    if not np.isfinite(pf):
        pf = 20.0
    pf = min(pf, 20.0)
    trend_ret = float(metrics.get("trend_trade_total_return", 0.0))
    short_cap = float(metrics.get("short_trend_capture_ratio", 0.0))

    score = (
        14.0 * total_return
        + 4.0 * max_drawdown
        + 0.06 * pf
        + 0.10 * min(avg_bars, 24.0)
        + 0.30 * short_cap
        + 0.9 * trend_ret
    )
    if total_return < baseline_return * 0.85:
        score -= 0.40
    if max_drawdown < baseline_mdd - 0.025:
        score -= 0.30
    if trades > trade_count_cap:
        score -= (trades - trade_count_cap) * 0.05
    if avg_bars < 4.0:
        score -= 0.12
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
    bp_provider: BestPointSignalProvider,
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
            best_point_provider=bp_provider,
        )
    return result.metrics


def _cfg_to_json_dict(cfg: TradingSystemConfig) -> dict:
    return json.loads(json.dumps(cfg, default=lambda o: o.__dict__ if hasattr(o, "__dict__") else o))


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

    print("loading market-state signals ...")
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
    cached = _CachedSignalProvider(model_provider.atr, _collect_signals(model_provider, start_idx, end_idx))

    print("loading 017c best-point signals ...")
    bp_provider = BestPointSignalProvider.from_checkpoint(
        checkpoint=args.best_point_checkpoint,
        df=df,
        context_bars=args.best_point_context_bars,
        device=args.device,
    )

    long_grid = _parse_float_grid(args.grid_long_entry)
    short_grid = _parse_float_grid(args.grid_short_entry)
    exit_grid = _parse_float_grid(args.grid_exit_prob)
    exit_bars_grid = _parse_int_grid(args.grid_bp_exit_bars)
    opp_grid = _parse_float_grid(args.grid_min_opportunity)

    rows: list[dict] = []
    done = 0
    total = len(long_grid) * len(short_grid) * len(exit_grid) * len(exit_bars_grid) * len(opp_grid)
    for long_th in long_grid:
        for short_th in short_grid:
            for exit_th in exit_grid:
                for exit_bars in exit_bars_grid:
                    for min_opp in opp_grid:
                        done += 1
                        cfg = replace(
                            base_cfg,
                            best_point=replace(
                                base_cfg.best_point,
                                enabled=True,
                                observe_only=False,
                                long_entry_confirm_threshold=long_th,
                                short_entry_confirm_threshold=short_th,
                                exit_prob_threshold=exit_th,
                                hold_min_prob=max(0.15, 1.0 - exit_th - 0.05),
                                min_opportunity_roi=min_opp,
                                require_entry_confirm_for_crash=False,
                            ),
                            trend_lifecycle=replace(
                                base_cfg.trend_lifecycle,
                                bp_exit_confirm_bars=exit_bars,
                            ),
                        )
                        metrics = _run_with_cfg(df, cached, bp_provider, start_idx, end_idx, cfg)
                        row = {
                            "long_entry_confirm_threshold": long_th,
                            "short_entry_confirm_threshold": short_th,
                            "exit_prob_threshold": exit_th,
                            "bp_exit_confirm_bars": exit_bars,
                            "min_opportunity_roi": min_opp,
                            "metrics": metrics,
                            "score": _score(
                                metrics,
                                baseline_return=args.baseline_return,
                                baseline_mdd=args.baseline_mdd,
                                trade_count_cap=args.trade_count_cap,
                            ),
                        }
                        rows.append(row)
                        if done % 10 == 0 or done == total:
                            print(f"  progress {done}/{total}")

    rows.sort(key=lambda r: r["score"], reverse=True)
    best = rows[0]
    best_cfg = replace(
        base_cfg,
        best_point=replace(
            base_cfg.best_point,
            enabled=True,
            observe_only=False,
            long_entry_confirm_threshold=best["long_entry_confirm_threshold"],
            short_entry_confirm_threshold=best["short_entry_confirm_threshold"],
            exit_prob_threshold=best["exit_prob_threshold"],
            hold_min_prob=max(0.15, 1.0 - best["exit_prob_threshold"] - 0.05),
            min_opportunity_roi=best["min_opportunity_roi"],
            require_entry_confirm_for_crash=False,
        ),
        trend_lifecycle=replace(
            base_cfg.trend_lifecycle,
            bp_exit_confirm_bars=best["bp_exit_confirm_bars"],
        ),
    )

    results_path = out_dir / "tuning_results.json"
    payload = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "best_point_checkpoint": args.best_point_checkpoint,
        "base_config": args.base_config,
        "best": {k: v for k, v in best.items() if k != "metrics"},
        "best_metrics": best["metrics"],
        "top": [{k: v for k, v in r.items() if k != "metrics"} | {"metrics": r["metrics"]} for r in rows[: args.top_k]],
    }
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    tuned_path = Path(args.tuned_config_out)
    base_dict = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    tuned_dict = {**base_dict}
    tuned_dict["best_point"] = best_cfg.best_point.__dict__
    tuned_dict["trend_lifecycle"] = {
        **base_dict.get("trend_lifecycle", {}),
        **{k: v for k, v in best_cfg.trend_lifecycle.__dict__.items() if k in ("bp_exit_confirm_bars",)},
    }
    tuned_dict["_tuning_meta"] = {
        "source": "tune_backtest_trading_system_v017_bp.py",
        "checkpoint": args.checkpoint,
        "best_point_checkpoint": args.best_point_checkpoint,
        "tune_split": args.split,
        "score": best["score"],
        "best_params": {k: best[k] for k in best if k not in ("metrics", "score")},
    }
    tuned_path.write_text(json.dumps(tuned_dict, indent=2), encoding="utf-8")

    print(f"best score={best['score']:.4f} return={best['metrics']['total_return']:.4f} mdd={best['metrics']['max_drawdown']:.4f}")
    print(f"saved tuning: {results_path}")
    print(f"saved config: {tuned_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
