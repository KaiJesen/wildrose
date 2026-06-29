#!/usr/bin/env python3
"""Run zone observation/position state machine on OHLCV data."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from market_data.schema import COL_TIME, normalize_ohlcv_df
from trend_strategy.rails import Backend, resolve_backend
from trend_strategy.zone_engine import EngineConfig, Trade, run_zone_strategy


def load_ohlcv(args: argparse.Namespace) -> pd.DataFrame:
    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        raise ValueError("--csv is required")
    df = normalize_ohlcv_df(df)
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values(COL_TIME).reset_index(drop=True)
    ts = pd.to_datetime(df[COL_TIME], utc=True)
    if args.interval == "1d" and ts.diff().dropna().median() < pd.Timedelta(hours=12):
        df = df.set_index(COL_TIME)
        df = df.resample("1D").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna(how="any").reset_index()
    if args.start:
        df = df[df[COL_TIME].astype(str).str[:10] >= args.start]
    if args.end:
        df = df[df[COL_TIME].astype(str).str[:10] <= args.end]
    return df.reset_index(drop=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zone strategy backtest (算法说明.md)")
    p.add_argument("--csv", required=True, help="OHLCV CSV path")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1d", choices=["1d", "4h", "1h"])
    p.add_argument("--start", help="YYYY-MM-DD inclusive")
    p.add_argument("--end", help="YYYY-MM-DD inclusive")
    p.add_argument("--warmup", type=int, default=40)
    p.add_argument("--flat-threshold", type=float, default=0.02)
    p.add_argument("--backend", choices=["auto", "cpu", "gpu"], default="auto")
    p.add_argument("--output-dir", default="backtest/trend_strategy_zone")
    p.add_argument("--plot", default="backtest/trend_strategy_zone/trade_points.png")
    p.add_argument("--show", action="store_true")
    return p.parse_args(argv)


def run(df: pd.DataFrame, args: argparse.Namespace) -> tuple[list[Trade], object, dict]:
    backend = Backend(args.backend)
    used = resolve_backend(backend)
    print(f"Rail backend: {used} (requested={args.backend})")

    cfg = EngineConfig(
        warmup_bars=args.warmup,
        flat_threshold=args.flat_threshold,
        backend=backend,
    )
    trades, engine = run_zone_strategy(df, config=cfg)

    total_ret = 0.0
    for tr in trades:
        if tr.side == "long":
            total_ret += (tr.exit_price - tr.entry_price) / tr.entry_price
        else:
            total_ret += (tr.entry_price - tr.exit_price) / tr.entry_price

    wins = sum(
        1 for tr in trades
        if (tr.exit_price > tr.entry_price) == (tr.side == "long")
    )
    summary = {
        "bars": len(df),
        "trade_count": len(trades),
        "win_count": wins,
        "simple_return_pct": total_ret * 100.0,
        "backend": used,
        "trades": [asdict(t) for t in trades],
    }
    return trades, engine, summary


def main() -> int:
    args = parse_args()
    df = load_ohlcv(args)
    print(f"Loaded {len(df)} bars: {df.iloc[0][COL_TIME]} -> {df.iloc[-1][COL_TIME]}")

    trades, engine, summary = run(df, args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Trades: {summary['trade_count']}  simple return: {summary['simple_return_pct']:+.2f}%")
    for tr in trades:
        pnl = (
            (tr.exit_price - tr.entry_price) / tr.entry_price * 100
            if tr.side == "long"
            else (tr.entry_price - tr.exit_price) / tr.entry_price * 100
        )
        print(f"  {tr.side:5s} {tr.entry_ts[:10]} -> {tr.exit_ts[:10]}  {pnl:+.2f}%")
    print(f"Summary -> {summary_path}")

    if args.plot:
        from trend_strategy.plot_zone_strategy import plot_zone_strategy
        import matplotlib.pyplot as plt

        plot_zone_strategy(
            df, engine, trades,
            title=f"{args.symbol} {args.interval}",
            save_path=args.plot,
        )
        if args.show:
            plt.show()
        else:
            plt.close()
        print(f"Plot -> {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
