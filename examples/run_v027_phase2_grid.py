#!/usr/bin/env python3
"""027 Phase 2: Satellite threshold grid on valid (satellite_only arm)."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(_ROOT / "examples"))

from _train_common import add_data_args, add_feature_args, add_segment_args, add_vq_args, apply_real_data_defaults, fetch_ohlcv_df, prepare_bar_series_from_args
from _v027_common import CORE_BASELINE, M2_CKPT, kline_backtest_args, verify_pw20_checkpoint

from trading_system.adapters.best_point_model import BestPointSignalProvider
from trading_system.adapters.market_state_model import ModelSignalProvider
from trading_system.backtest.runner import run_backtest
from trading_system.config import load_config

OUT = _ROOT / "backtest/v027_phase2/grid"
BP_CKPT = _ROOT / "checkpoints/017_best_point_signal/017c_best_point_major_legs/best.pt"
BASE_SAT = _ROOT / "configs/trading_rule_v027_satellite.json"


def _split_idx(bundle, split: str):
    if split == "train":
        return bundle.train_idx
    if split == "valid":
        return bundle.valid_idx
    return bundle.test_idx


def _sharpe(equity: list[float]) -> float:
    if len(equity) < 3:
        return 0.0
    arr = np.asarray(equity, dtype=np.float64)
    rets = np.diff(arr) / np.maximum(arr[:-1], 1e-12)
    if rets.std() < 1e-12:
        return 0.0
    return float(np.sqrt(252 * 24) * rets.mean() / rets.std())


def _write_sat_cfg(path: Path, overrides: dict) -> None:
    base = json.loads(BASE_SAT.read_text(encoding="utf-8"))
    sat = dict(base.get("satellite", base))
    sat.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"satellite": sat, "_meta": base.get("_meta", {})}, indent=2), encoding="utf-8")


def _run_satellite_only(args, sat_path: Path) -> dict:
    cfg = load_config(args.core_config)
    df = fetch_ohlcv_df(args)
    bundle = prepare_bar_series_from_args(df, args)
    idx = _split_idx(bundle, args.split)
    start_idx = max(int(idx.min()), args.context_bars + 1)
    end_idx = min(int(idx.max()), len(df) - 2)
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
    bp = BestPointSignalProvider.from_checkpoint(
        checkpoint=args.best_point_checkpoint,
        df=df,
        context_bars=args.best_point_context_bars,
        device=args.device,
    )
    out = OUT / sat_path.stem
    result = run_backtest(
        df,
        signal_provider=provider,
        start_idx=start_idx,
        end_idx=end_idx,
        cfg=cfg,
        out_dir=out,
        best_point_provider=bp,
        satellite_mode="only",
        satellite_config_path=sat_path,
    )
    sat_trades = [t for t in result.logger.trades if t.get("slot_id") == "satellite"]
    return {
        "return": float(result.metrics.get("total_return", 0)),
        "max_drawdown": float(result.metrics.get("max_drawdown", 0)),
        "trade_count": len(sat_trades),
        "sharpe": _sharpe(result.strategy_equity),
        "win_rate": float(result.metrics.get("win_rate", 0)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="027 Phase 2 Satellite grid")
    add_data_args(p)
    add_feature_args(p)
    add_segment_args(p)
    add_vq_args(p)
    p.add_argument("--core-config", default=str(CORE_BASELINE.relative_to(_ROOT)))
    p.add_argument("--checkpoint", default=str(M2_CKPT.relative_to(_ROOT)))
    p.add_argument("--best-point-checkpoint", default=str(BP_CKPT.relative_to(_ROOT)))
    p.add_argument("--best-point-context-bars", type=int, default=96)
    p.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    p.add_argument("--trunk-layers", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.set_defaults(source="binance_vision", symbol="BTCUSDT", interval="1h", days=365)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ka = kline_backtest_args()
    if len(ka) >= 2 and ka[0] == "--csv":
        args.csv = ka[1]
    apply_real_data_defaults(args)
    verify_pw20_checkpoint()
    OUT.mkdir(parents=True, exist_ok=True)

    entry_thresholds = [0.55, 0.60, 0.65, 0.70]
    min_opps = [0.0, 0.03]
    max_daily = [2]
    risk_caps = [0.40]

    rows: list[dict] = []
    for tau, opp, daily, risk in itertools.product(entry_thresholds, min_opps, max_daily, risk_caps):
        tag = f"tau{tau:.2f}_opp{opp:.2f}_d{daily}_r{risk:.2f}"
        sat_path = OUT / f"sat_{tag}.json"
        overrides = {
            "long_entry_threshold": tau,
            "short_entry_threshold": tau,
            "min_opportunity_roi": opp,
            "max_daily_opens": daily,
            "risk_open_max": risk,
        }
        _write_sat_cfg(sat_path, overrides)
        metrics = _run_satellite_only(args, sat_path)
        row = {"tag": tag, **overrides, **metrics}
        row["gate_pass"] = row["trade_count"] >= 8 and row["sharpe"] >= 0.0
        rows.append(row)
        print(json.dumps(row))

    rows.sort(key=lambda r: (r["gate_pass"], r["sharpe"], r["return"]), reverse=True)
    summary = {
        "split": args.split,
        "grid_size": len(rows),
        "pass_count": sum(1 for r in rows if r["gate_pass"]),
        "best": rows[0] if rows else None,
        "top5": rows[:5],
        "all": rows,
    }
    (OUT / "grid_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if rows and rows[0]["gate_pass"]:
        best = rows[0]
        best_cfg = OUT / "sat_best.json"
        _write_sat_cfg(
            best_cfg,
            {k: best[k] for k in ("long_entry_threshold", "short_entry_threshold", "min_opportunity_roi", "max_daily_opens", "risk_open_max")},
        )
        (OUT / "BEST_CONFIG.json").write_text(best_cfg.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"\nBest config written: {best_cfg}")
    else:
        print("\nNo grid point passed satellite gate (trades>=8, sharpe>=0)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
