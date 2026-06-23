#!/usr/bin/env python3
"""Backtest entry for modular trading system v014."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EX = Path(__file__).resolve().parent
_ROOT = _EX.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))

from _train_common import add_data_args, add_feature_args, add_segment_args, add_vq_args, apply_real_data_defaults, fetch_ohlcv_df, prepare_bar_series_from_args
from trading_system.backtest.report import write_report
from trading_system.backtest.runner import run_backtest
from trading_system.config import load_config
from trading_system.adapters.best_point_model import BestPointSignalProvider
from trading_system.adapters.market_state_model import ModelSignalProvider
from trading_system.adapters.csv_signal import CsvSignalProvider


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest trading system v014")
    add_data_args(p)
    add_feature_args(p)
    add_segment_args(p)
    add_vq_args(p)
    p.add_argument("--checkpoint", default="checkpoints/0065a_multi_seed_s45_market_state_stability/market_state_best.pt")
    p.add_argument("--config", default="configs/trading_rule_v014_conservative.json")
    p.add_argument("--signal-csv", default="")
    p.add_argument("--best-point-checkpoint", default="")
    p.add_argument("--best-point-context-bars", type=int, default=96)
    p.add_argument("--split", choices=["train", "valid", "test"], default="test")
    p.add_argument("--trunk-layers", type=int, default=2)
    p.add_argument("--output-dir", default="backtest/backtest_rule_v014_conservative")
    p.add_argument("--device", default="cpu")
    p.set_defaults(source="binance_vision", symbol="BTCUSDT", interval="1h", days=365)
    return p.parse_args()


def _split_idx(bundle, split: str):
    if split == "train":
        return bundle.train_idx
    if split == "valid":
        return bundle.valid_idx
    return bundle.test_idx


def main() -> int:
    args = parse_args()
    apply_real_data_defaults(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)

    df = fetch_ohlcv_df(args)
    bundle = prepare_bar_series_from_args(df, args)
    idx = _split_idx(bundle, args.split)
    start_idx = max(int(idx.min()), args.context_bars + 1)
    end_idx = min(int(idx.max()), len(df) - 2)

    if args.signal_csv:
        provider = CsvSignalProvider(args.signal_csv, cfg)
        if len(provider.atr) < len(df):
            raise ValueError("signal csv rows are shorter than bar series")
    else:
        provider = ModelSignalProvider.from_checkpoint(
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
            cfg=cfg,
            device=args.device,
        )

    bp_provider = None
    if args.best_point_checkpoint:
        bp_provider = BestPointSignalProvider.from_checkpoint(
            checkpoint=args.best_point_checkpoint,
            df=df,
            context_bars=args.best_point_context_bars,
            device=args.device,
        )

    result = run_backtest(
        df,
        signal_provider=provider,
        start_idx=start_idx,
        end_idx=end_idx,
        cfg=cfg,
        out_dir=out_dir,
        best_point_provider=bp_provider,
    )
    write_report(
        out_dir,
        metrics=result.metrics,
        config_path=args.config,
        checkpoint=args.checkpoint if not args.signal_csv else "csv_signal",
    )
    print(f"saved backtest outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

