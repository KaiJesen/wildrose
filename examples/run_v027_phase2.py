#!/usr/bin/env python3
"""027 Phase 2: Satellite slot (017 best-point) + Core+Sat combined on valid."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(_ROOT / "examples"))

from _train_common import add_data_args, add_feature_args, add_segment_args, add_vq_args, apply_real_data_defaults, fetch_ohlcv_df, prepare_bar_series_from_args
from _v027_common import CORE_BASELINE, M2_CKPT, OUT_PHASE1, kline_backtest_args, repo_rel, verify_pw20_checkpoint

from trading_system.adapters.best_point_model import BestPointSignalProvider
from trading_system.adapters.market_state_model import ModelSignalProvider
from trading_system.backtest.runner import run_backtest
from trading_system.config import load_config
from trading_system.metrics import compute_metrics

OUT = _ROOT / "backtest/v027_phase2"
SAT_CONFIG = _ROOT / "configs/trading_rule_v027_satellite.json"
BP_CKPT = _ROOT / "checkpoints/017_best_point_signal/017c_best_point_major_legs/best.pt"


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


def _run_arm(
    args,
    *,
    name: str,
    split: str,
    satellite_mode: str,
) -> dict:
    out = OUT / f"{name}_{split}"
    cfg = load_config(args.core_config)
    df = fetch_ohlcv_df(args)
    bundle = prepare_bar_series_from_args(df, args)
    idx = _split_idx(bundle, split)
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
    result = run_backtest(
        df,
        signal_provider=provider,
        start_idx=start_idx,
        end_idx=end_idx,
        cfg=cfg,
        out_dir=out,
        best_point_provider=bp,
        satellite_mode=satellite_mode,
        satellite_config_path=SAT_CONFIG,
    )
    eq = result.strategy_equity
    sharpe = _sharpe(eq)
    sat_trades = [t for t in result.logger.trades if t.get("slot_id") == "satellite"]
    row = {
        "name": name,
        "split": split,
        "satellite_mode": satellite_mode,
        "return": float(result.metrics.get("total_return", 0)),
        "max_drawdown": float(result.metrics.get("max_drawdown", 0)),
        "trade_count": len(result.logger.trades),
        "satellite_trade_count": len(sat_trades),
        "sharpe": sharpe,
        "satellite_pnl": float(sum(float(t.get("net_pnl", 0)) for t in sat_trades)),
    }
    (out / "metrics.json").write_text(json.dumps({**result.metrics, **row}, indent=2), encoding="utf-8")
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="027 Phase 2 Satellite")
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
    OUT.mkdir(parents=True, exist_ok=True)
    verify_pw20_checkpoint()
    if not BP_CKPT.is_file():
        raise FileNotFoundError(BP_CKPT)

    core_row = _run_arm(args, name="core_only", split=args.split, satellite_mode="off")
    # core baseline from phase1 if exists
    phase1_base = OUT_PHASE1 / f"baseline_{args.split}" / "metrics.json"
    if phase1_base.is_file():
        core_row["phase1_baseline_return"] = float(json.loads(phase1_base.read_text()).get("total_return", 0))

    sat_row = _run_arm(args, name="satellite_only", split=args.split, satellite_mode="only")
    comb_row = _run_arm(args, name="core_sat", split=args.split, satellite_mode="combined")

    gates = {
        "satellite_trades_min": 8,
        "satellite_sharpe_min": 0.0,
        "combined_return_ge_core": True,
    }
    sat_pass = sat_row["satellite_trade_count"] >= gates["satellite_trades_min"] and sat_row["sharpe"] >= gates["satellite_sharpe_min"]
    comb_pass = comb_row["return"] >= core_row["return"]

    summary = {
        "split": args.split,
        "core_only": core_row,
        "satellite_only": sat_row,
        "core_sat_combined": comb_row,
        "gates": gates,
        "satellite_valid_pass": sat_pass,
        "combined_valid_pass": comb_pass,
        "phase2_pass": sat_pass and comb_pass,
    }
    (OUT / "phase2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# 027 Phase 2 报告",
        "",
        f"| 臂 | return | trades | sat trades | Sharpe |",
        f"|----|--------|--------|------------|--------|",
        f"| core_only | {core_row['return']*100:.2f}% | {core_row['trade_count']} | — | — |",
        f"| satellite_only | {sat_row['return']*100:.2f}% | {sat_row['trade_count']} | {sat_row['satellite_trade_count']} | {sat_row['sharpe']:.2f} |",
        f"| core+sat | {comb_row['return']*100:.2f}% | {comb_row['trade_count']} | {comb_row['satellite_trade_count']} | {comb_row['sharpe']:.2f} |",
        "",
        f"**Satellite valid gate**: {'PASS' if sat_pass else 'FAIL'}",
        f"**Combined ≥ core**: {'PASS' if comb_pass else 'FAIL'}",
        f"**Phase 2**: {'PASS' if summary['phase2_pass'] else 'FAIL'}",
        "",
        "```bash",
        "python examples/run_v027_phase2.py --split valid",
        "```",
    ]
    (OUT / "REPORT_027_PHASE2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
