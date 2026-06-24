#!/usr/bin/env python3
"""Grid search v022 trend_signal params on valid split (module + backtest)."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_EX = Path(__file__).resolve().parent
_ROOT = _EX.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))

from _train_common import apply_real_data_defaults, fetch_ohlcv_df, prepare_bar_series_from_args
from examples.eval_trend_modules import (  # noqa: E402
    _compute_metrics,
    _run_bias_audit,
    _run_trend_signals,
    _split_idx,
)
from trading_system.config import load_config

BASE_CONFIG = _ROOT / "configs/trading_rule_v022_trend_quality_0062e.json"
CHECKPOINT = "prod/v0.0.0/checkpoint/market_state_best.pt"
V021_CONFIG = _ROOT / "configs/trading_rule_v021_full_bias_0062e.json"


@dataclass
class Candidate:
    name: str
    params: dict
    module_score: float = 0.0
    backtest_return: float = 0.0
    backtest_dd: float = 0.0
    metrics: dict | None = None


def _module_score(m: dict) -> float:
    cov = m.get("teacher_trend_coverage", 0.0)
    prec = m.get("confirmed_precision_vs_teacher", 0.0)
    broken = m.get("broken_ratio", 1.0)
    false_rng = m.get("false_confirm_on_range_teacher", 1.0)
    choppy = m.get("choppy_false_confirm_rate", 1.0)
    hard_long = m.get("hard_block_long_ratio", 1.0)
    score = (
        2.0 * cov
        + 1.5 * prec
        - 1.2 * broken
        - 1.0 * false_rng
        - 2.0 * max(0.0, choppy - 0.05)
        - 0.5 * max(0.0, hard_long - 0.30)
    )
    return score


def _eval_module(cfg_path: Path, df, idx, labels) -> dict:
    cfg = load_config(cfg_path)
    from trading_system.trend_signal import TrendSignalProvider

    provider = TrendSignalProvider(cfg.trend_signal)
    eval_df = _run_trend_signals(df, idx, provider=provider, atr_period=cfg.execution.atr_period)
    eval_df = eval_df.merge(labels[["time", "trend_leg_type", "is_leg_confirmed"]], left_on="time", right_on="time", how="left")
    metrics = _compute_metrics(eval_df)
    bias_rows = _run_bias_audit(df, idx, cfg)
    from trading_system.trend_bias_audit import aggregate_block_stats

    block_stats = aggregate_block_stats(bias_rows)
    metrics.update({k: float(v) for k, v in block_stats.items() if isinstance(v, (int, float))})
    return metrics


def _run_backtest(cfg_path: Path, split: str, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_ROOT / "examples/backtest_trading_system_v014.py"),
        "--config",
        str(cfg_path),
        "--checkpoint",
        CHECKPOINT,
        "--output-dir",
        str(out),
        "--split",
        split,
        "--symbol",
        "BTCUSDT",
        "--interval",
        "1h",
        "--days",
        "365",
    ]
    subprocess.check_call(cmd, cwd=_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.loads((out / "metrics.json").read_text(encoding="utf-8"))


def _grid() -> list[Candidate]:
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    out: list[Candidate] = []
    for invalid_reset in (4, 5, 6):
        for hold_score in (2, 3):
            for min_age in (2, 3):
                for chop_range in (2.5, 3.0, 3.5):
                    for chop_flip in (4, 5):
                        params = {
                            "invalid_reset_bars": invalid_reset,
                            "hold_confirm_score": hold_score,
                            "min_trend_age_for_hold": min_age,
                            "chop_range_atr_max": chop_range,
                            "chop_flip_max": chop_flip,
                        }
                        name = f"ir{invalid_reset}_h{hold_score}_a{min_age}_cr{chop_range}_cf{chop_flip}"
                        out.append(Candidate(name=name, params=params))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="valid")
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--skip-backtest", action="store_true")
    p.add_argument("--out-dir", default="backtest/v022_tune")
    args = p.parse_args()

    import pandas as pd
    from market_data.schema import COL_TIME

    ns = argparse.Namespace(
        synthetic=False,
        source="binance_vision",
        symbol="BTCUSDT",
        interval="1h",
        days=365,
        csv="",
        cache_dir="data/cache/kline",
        no_cache=False,
        force_download=False,
        seed=42,
        context_bars=128,
        min_seg_len=4,
        max_seg_len=32,
        max_segments=16,
    )
    apply_real_data_defaults(ns)
    df = fetch_ohlcv_df(ns)
    bundle = prepare_bar_series_from_args(df, ns)
    idx = _split_idx(bundle, args.split)
    labels = pd.read_csv(_ROOT / "data/labels/trend_leg_v020_teacher/teacher_labels.csv", parse_dates=["time"])

    base_cfg = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    candidates = _grid()
    print(f"grid size={len(candidates)} split={args.split}")

    with tempfile.TemporaryDirectory(prefix="v022_tune_") as tmp:
        tmp_path = Path(tmp)
        for c in candidates:
            cfg = copy.deepcopy(base_cfg)
            cfg["trend_signal"].update(c.params)
            path = tmp_path / f"{c.name}.json"
            path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            c.metrics = _eval_module(path, df, idx, labels)
            c.module_score = _module_score(c.metrics)
            print(
                f"{c.name} score={c.module_score:.3f} cov={c.metrics['teacher_trend_coverage']:.3f} "
                f"broken={c.metrics['broken_ratio']:.3f} prec={c.metrics['confirmed_precision_vs_teacher']:.3f}"
            )

    ranked = sorted(candidates, key=lambda x: x.module_score, reverse=True)
    top = ranked[: args.top_k]

    v021_m = {}
    if not args.skip_backtest:
        v021_m = _run_backtest(V021_CONFIG, args.split, Path(args.out_dir) / "v021_ref")
        v021_ret = float(v021_m.get("total_return", 0))
        v021_dd = float(v021_m.get("max_drawdown", 0))
        print(f"v021 valid ref return={v021_ret:.4f} dd={v021_dd:.4f}")

        with tempfile.TemporaryDirectory(prefix="v022_tune_bt_") as tmp:
            tmp_path = Path(tmp)
            base_cfg = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
            for c in top:
                cfg = copy.deepcopy(base_cfg)
                cfg["trend_signal"].update(c.params)
                path = tmp_path / f"{c.name}.json"
                path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
                m = _run_backtest(path, args.split, Path(args.out_dir) / c.name)
                c.backtest_return = float(m.get("total_return", 0))
                c.backtest_dd = float(m.get("max_drawdown", 0))
                bt_score = c.backtest_return - 0.5 * abs(c.backtest_dd - v021_dd)
                c.module_score = 0.6 * c.module_score + 0.4 * (bt_score / max(abs(v021_ret), 1e-6))
                print(f"BT {c.name} ret={c.backtest_return:.4f} dd={c.backtest_dd:.4f} combo={c.module_score:.3f}")

    final = sorted(top, key=lambda x: x.module_score, reverse=True)[0]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "best": final.name,
        "params": final.params,
        "module_metrics": final.metrics,
        "backtest_return": final.backtest_return,
        "backtest_dd": final.backtest_dd,
        "top_module": [{"name": c.name, "score": c.module_score, "params": c.params} for c in ranked[:10]],
    }
    (out_dir / "tune_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BEST", final.name, final.params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
