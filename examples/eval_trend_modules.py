#!/usr/bin/env python3
"""Trend module evaluation harness (022 WP-E Phase 1)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_EX = Path(__file__).resolve().parent
_ROOT = _EX.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))

from _train_common import add_data_args, apply_real_data_defaults, fetch_ohlcv_df, prepare_bar_series_from_args
from market_data.schema import COL_CLOSE, COL_HIGH, COL_LOW, COL_TIME
from trading_system.adapters.market_state_model import compute_atr
from trading_system.config import load_config
from trading_system.trend_signal import TrendMemory, TrendSignalProvider
from trading_system.trend_bias_audit import aggregate_block_stats, derive_block_reason

DEFAULT_CONFIG = "configs/trading_rule_v022_trend_quality_0062e.json"
DEFAULT_LABELS = "data/labels/trend_leg_v020_teacher/teacher_labels.csv"
TEACHER_LABEL_VERSION = "020_major_legs_v1"

_UP_LEGS = frozenset({"FAST_UP_LEG", "SLOW_UP_LEG", "SURGE_LEG"})
_DOWN_LEGS = frozenset({"FAST_DOWN_LEG", "SLOW_DOWN_LEG"})
_RANGE_LEGS = frozenset({"RANGE_LEG", "TRANSITION_LEG"})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate TrendSignal / Teacher alignment (022)")
    add_data_args(p)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--labels-file", default=DEFAULT_LABELS)
    p.add_argument("--teacher-label-version", default=TEACHER_LABEL_VERSION)
    p.add_argument("--split", choices=["train", "valid", "test", "all"], default="test")
    p.add_argument("--output-dir", default="reports/022_trend_eval/phase1_baseline")
    p.add_argument("--atr-period", type=int, default=14)
    p.add_argument("--perf-only", action="store_true")
    p.add_argument("--skip-segment-perf", action="store_true")
    p.add_argument("--skip-bias-audit", action="store_true")
    p.set_defaults(source="binance_vision", symbol="BTCUSDT", interval="1h", days=365)
    return p.parse_args()


def _split_idx(bundle, split: str) -> np.ndarray:
    if split == "train":
        return bundle.train_idx
    if split == "valid":
        return bundle.valid_idx
    if split == "all":
        return np.arange(len(bundle.bars))
    return bundle.test_idx


def _teacher_direction(leg_type: str) -> str:
    if leg_type in _UP_LEGS:
        return "UP"
    if leg_type in _DOWN_LEGS:
        return "DOWN"
    return "NONE"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _align_teacher(df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    merged = df[[COL_TIME]].merge(labels, left_on=COL_TIME, right_on="time", how="left")
    return merged


def _segment_runtime_8745(cfg_path: Path) -> float:
    import time

    import numpy as np

    from trading_system.config import load_config
    from trading_system.trend_segment import TrendSegmentEngine

    n = 8745
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.1, n))
    engine = TrendSegmentEngine(load_config(cfg_path).trend_segment)
    t0 = time.perf_counter()
    for i in range(n):
        engine.update(bar_idx=i, high=float(close[i] + 0.5), low=float(close[i] - 0.5), close=float(close[i]), atr=1.0)
    return time.perf_counter() - t0


def _run_trend_signals(
    df: pd.DataFrame,
    idx: np.ndarray,
    *,
    provider: TrendSignalProvider,
    atr_period: int,
) -> pd.DataFrame:
    high = df[COL_HIGH].to_numpy(dtype=np.float64)
    low = df[COL_LOW].to_numpy(dtype=np.float64)
    close = df[COL_CLOSE].to_numpy(dtype=np.float64)
    atr_arr = compute_atr(high, low, close, atr_period)

    memory = TrendMemory()
    rows: list[dict] = []
    close_hist: list[float] = []
    high_hist: list[float] = []
    low_hist: list[float] = []
    atr_hist: list[float] = []
    idx_set = set(int(i) for i in idx.tolist())
    end_i = int(idx.max())

    for i in range(end_i + 1):
        close_hist.append(float(close[i]))
        high_hist.append(float(high[i]))
        low_hist.append(float(low[i]))
        atr_hist.append(float(atr_arr[i]))
        sig = provider.compute(
            close_hist=close_hist,
            high_hist=high_hist,
            low_hist=low_hist,
            atr_hist=atr_hist,
            memory=memory,
        )
        if i not in idx_set:
            continue
        rows.append(
            {
                "idx": i,
                COL_TIME: df.iloc[i][COL_TIME],
                "direction": sig.direction.value,
                "is_confirmed": bool(sig.is_confirmed),
                "is_sustained": bool(sig.is_sustained),
                "is_broken": bool(sig.is_broken),
                "confirm_tier": sig.confirm_tier.value,
                "reason_codes": list(sig.reason_codes),
                "chop_triggered": any(
                    code in sig.reason_codes for code in ("CHOP_HARD", "CHOP_SOFT", "CHOP_RANGE", "CHOP_FLIP", "CHOP_INEFFICIENT")
                ),
            }
        )
    return pd.DataFrame(rows)


def _neutral_model_signal(ts, price: float, atr: float):
    from datetime import datetime

    from trading_system.signal import TradingSignal

    return TradingSignal(
        ts=ts if ts is not None else datetime(2026, 1, 1),
        price=price,
        atr=atr,
        p_up=0.33,
        p_down=0.33,
        p_flat=0.34,
        p_risk=0.2,
        pred_ret_1=0.0,
        pred_ret_2=0.0,
        pred_ret_3=0.0,
        pred_ret_4=0.0,
        pred_ret_5=0.0,
        pred_cum_ret_5=0.0,
    ).finalize(0.45)


def _run_bias_audit(df: pd.DataFrame, idx: np.ndarray, cfg) -> list[dict]:
    from trading_system.crash import CrashRegimeDetector
    from trading_system.slow_trend import SlowUptrendDetector
    from trading_system.trend import TrendRegimeFilter
    from trading_system.trend_bias import TrendBiasBuilder
    from trading_system.trend_segment import TrendSegmentEngine

    high = df[COL_HIGH].to_numpy(dtype=np.float64)
    low = df[COL_LOW].to_numpy(dtype=np.float64)
    close = df[COL_CLOSE].to_numpy(dtype=np.float64)
    atr_arr = compute_atr(high, low, close, cfg.execution.atr_period)
    times = df[COL_TIME].tolist()

    trend_filter = TrendRegimeFilter(cfg.trend)
    signal_provider = TrendSignalProvider(cfg.trend_signal)
    segment_engine = TrendSegmentEngine(cfg.trend_segment)
    crash_detector = CrashRegimeDetector(cfg.crash)
    slow_detector = SlowUptrendDetector(cfg.slow_uptrend)
    bias_builder = TrendBiasBuilder(cfg.trend_bias)
    memory = TrendMemory()

    close_hist: list[float] = []
    high_hist: list[float] = []
    low_hist: list[float] = []
    atr_hist: list[float] = []
    idx_set = set(int(i) for i in idx.tolist())
    rows: list[dict] = []

    for i in range(int(idx.max()) + 1):
        close_hist.append(float(close[i]))
        high_hist.append(float(high[i]))
        low_hist.append(float(low[i]))
        atr_hist.append(float(atr_arr[i]))
        if i not in idx_set:
            continue
        atr_v = float(atr_arr[i])
        trend_context = trend_filter.compute(close_hist, high_hist, low_hist, atr_v)
        trend_signal = signal_provider.compute(
            close_hist=close_hist,
            high_hist=high_hist,
            low_hist=low_hist,
            atr_hist=atr_hist,
            memory=memory,
        )
        model_sig = _neutral_model_signal(times[i], float(close[i]), atr_v)
        crash_context = crash_detector.compute(
            close_hist,
            high_hist,
            low_hist,
            atr_hist,
            model_sig,
            standard_open_short=False,
            is_flat=True,
        )
        slow_context = slow_detector.compute(
            close_hist,
            high_hist,
            low_hist,
            atr_v,
            p_risk=model_sig.p_risk,
            p_flat=model_sig.p_flat,
        )
        segment_context = segment_engine.update(
            bar_idx=i,
            high=float(high[i]),
            low=float(low[i]),
            close=float(close[i]),
            atr=atr_v,
            trend_signal=trend_signal,
            slow_ctx=slow_context,
            crash_ctx=crash_context,
            is_model_blind=bool(crash_context.is_model_blind_crash),
        )
        bias = bias_builder.build(
            trend_signal=trend_signal,
            segment_context=segment_context,
            slow_context=slow_context,
            crash_context=crash_context,
            trend_context=trend_context,
        )
        rows.append(
            {
                "allow_open_long": bias.allow_open_long,
                "allow_open_short": bias.allow_open_short,
                "block_reason_long": derive_block_reason(bias, side="long"),
                "block_reason_short": derive_block_reason(bias, side="short"),
                "chop_soft_active": "CHOP_SOFT_MICRO_WEAK" in bias.reason_codes,
                "alignment_score_long": bias.alignment_score_long,
            }
        )
    return rows


def _compute_metrics(eval_df: pd.DataFrame) -> dict[str, float]:
    teacher_conf = eval_df["is_leg_confirmed"].fillna(0).astype(int) == 1
    teacher_dir = eval_df["trend_leg_type"].map(_teacher_direction).fillna("NONE")
    teacher_trend = teacher_conf & teacher_dir.isin(["UP", "DOWN"])

    sig_dir = eval_df["direction"]
    same_dir = sig_dir == teacher_dir

    teacher_trend_bars = int(teacher_trend.sum())
    confirmed_cov = 0.0
    sustained_cov = 0.0
    if teacher_trend_bars > 0:
        m_conf = teacher_trend & eval_df["is_confirmed"] & same_dir
        m_sus = teacher_trend & (eval_df["is_confirmed"] | eval_df["is_sustained"]) & same_dir
        confirmed_cov = float(m_conf.sum()) / teacher_trend_bars
        sustained_cov = float(m_sus.sum()) / teacher_trend_bars

    sig_confirmed = eval_df["is_confirmed"]
    n_confirmed = int(sig_confirmed.sum())
    precision = 0.0
    if n_confirmed > 0:
        precision = float((sig_confirmed & same_dir & teacher_dir.isin(["UP", "DOWN"])).sum()) / n_confirmed

    teacher_range = eval_df["trend_leg_type"].isin(_RANGE_LEGS)
    false_range = 0.0
    if n_confirmed > 0:
        false_range = float((sig_confirmed & teacher_range).sum()) / n_confirmed

    chop_bars = eval_df["chop_triggered"]
    choppy_false = 0.0
    if int(chop_bars.sum()) > 0:
        choppy_false = float((chop_bars & sig_confirmed).sum()) / int(chop_bars.sum())

    broken_ratio = float(eval_df["is_broken"].mean()) if len(eval_df) else 0.0

    f1_scores: list[float] = []
    for label in ("UP", "DOWN"):
        pred = sig_confirmed & (sig_dir == label)
        truth = teacher_trend & (teacher_dir == label)
        tp = int((pred & truth).sum())
        fp = int((pred & ~truth).sum())
        fn = int((~pred & truth).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        f1_scores.append(f1)
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    return {
        "teacher_trend_coverage": sustained_cov,
        "teacher_coverage_confirmed_only": confirmed_cov,
        "teacher_coverage_confirmed_or_sustained": sustained_cov,
        "confirmed_precision_vs_teacher": precision,
        "false_confirm_on_range_teacher": false_range,
        "choppy_false_confirm_rate": choppy_false,
        "broken_ratio": broken_ratio,
        "confirmed_direction_macro_f1": macro_f1,
        "bar_count": float(len(eval_df)),
        "teacher_trend_bar_count": float(teacher_trend_bars),
        "signal_confirmed_bar_count": float(n_confirmed),
        "chop_triggered_bar_count": float(chop_bars.sum()),
    }


def _thresholds() -> dict[str, tuple[float, str]]:
    return {
        "teacher_trend_coverage": (0.65, ">="),
        "confirmed_precision_vs_teacher": (0.55, ">="),
        "false_confirm_on_range_teacher": (0.20, "<="),
        "choppy_false_confirm_rate": (0.05, "<="),
        "broken_ratio": (0.45, "<="),
        "segment_runtime_8745": (30.0, "<="),
        "hard_block_long_ratio": (0.30, "<="),
    }


def _passes(metric: str, value: float, rules: dict[str, tuple[float, str]]) -> bool | None:
    if metric not in rules:
        return None
    thr, op = rules[metric]
    if op == ">=":
        return value >= thr
    return value <= thr


def main() -> int:
    args = parse_args()
    apply_real_data_defaults(args)
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        raise SystemExit(f"config not found: {cfg_path}")

    cfg = load_config(cfg_path)
    provider = TrendSignalProvider(cfg.trend_signal)

    df = fetch_ohlcv_df(args)
    bundle = prepare_bar_series_from_args(df, args)
    idx = _split_idx(bundle, args.split)

    labels_path = Path(args.labels_file)
    labels = pd.read_csv(labels_path, parse_dates=["time"])
    teacher_params_sha = _file_sha256(labels_path)

    eval_df = _run_trend_signals(df, idx, provider=provider, atr_period=args.atr_period)
    eval_df = eval_df.merge(
        labels[["time", "trend_leg_type", "is_leg_confirmed"]],
        left_on=COL_TIME,
        right_on="time",
        how="left",
    )

    metrics = _compute_metrics(eval_df)
    if not args.skip_segment_perf:
        metrics["segment_runtime_8745"] = _segment_runtime_8745(cfg_path)
    if not args.skip_bias_audit:
        bias_rows = _run_bias_audit(df, idx, cfg)
        teacher_regime = eval_df["trend_leg_type"].isin({"RANGE_LEG", "TRANSITION_LEG"})
        for i, row in enumerate(bias_rows):
            row["teacher_regime"] = "RANGE_TRANSITION" if bool(teacher_regime.iloc[i]) else "TREND"
        block_stats = aggregate_block_stats(bias_rows)
        metrics.update({k: float(v) if isinstance(v, (int, float)) else v for k, v in block_stats.items() if isinstance(v, (int, float))})
        metrics["block_long_by_reason"] = block_stats["block_long_by_reason"]
        metrics["block_short_by_reason"] = block_stats["block_short_by_reason"]
        chop_soft_rows = [r for r in bias_rows if r.get("chop_soft_active")]
        if chop_soft_rows:
            metrics["chop_soft_micro_exposure_rate"] = float(
                sum(1 for r in chop_soft_rows if r["alignment_score_long"] != 0 or r.get("alignment_score_short", 0) != 0)
                / len(chop_soft_rows)
            )
    rules = _thresholds()
    gates = {
        k: _passes(k, metrics[k], rules)
        for k in rules
        if k in metrics
    }

    missing_dates: list[str] = []
    if hasattr(args, "missing_data_dates"):
        missing_dates = list(args.missing_data_dates or [])

    meta = {
        "symbol": args.symbol,
        "interval": args.interval,
        "start": str(df[COL_TIME].iloc[int(idx.min())]),
        "end": str(df[COL_TIME].iloc[int(idx.max())]),
        "split": args.split if args.split != "all" else "all",
        "data_source": args.source,
        "data_cache_path": str(getattr(args, "cache_path", "") or ""),
        "teacher_label_version": args.teacher_label_version,
        "teacher_params_sha256": teacher_params_sha,
        "config_path": str(cfg_path),
        "config_sha256": _file_sha256(cfg_path),
        "git_commit": _git_commit(),
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "missing_data_dates": missing_dates,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": meta, "metrics": metrics, "gates": gates}
    (out_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 022 Trend Module Eval Report",
        "",
        "## Metadata",
        "```json",
        json.dumps(meta, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Metrics",
        "| metric | value | gate |",
        "|--------|-------|------|",
    ]
    for k, v in metrics.items():
        gate = gates.get(k)
        gate_s = "—" if gate is None else ("PASS" if gate else "FAIL")
        if isinstance(v, dict):
            lines.append(f"| {k} | {json.dumps(v, ensure_ascii=False)} | {gate_s} |")
        else:
            lines.append(f"| {k} | {v:.4f} | {gate_s} |")
    lines.append("")
    report = "\n".join(lines) + "\n"
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"saved: {out_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
