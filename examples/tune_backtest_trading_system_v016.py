#!/usr/bin/env python3
"""Grid-search v016 trend-signal and rule parameters."""

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
    p = argparse.ArgumentParser(description="Tune v016 trend signal parameters")
    add_data_args(p)
    add_feature_args(p)
    add_segment_args(p)
    add_vq_args(p)
    p.add_argument("--checkpoint", default="checkpoints/0062e_market_state_return_ic_recovery/market_state_best.pt")
    p.add_argument("--base-config", default="configs/trading_rule_v016_trend_signal_0062e.json")
    p.add_argument("--split", choices=["train", "valid", "test"], default="valid")
    p.add_argument("--trunk-layers", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default="backtest/backtest_rule_v016_trend_signal_tuning")
    p.add_argument("--tuned-config-out", default="", help="optional path for tuned config json")
    p.add_argument("--grid-edge", default="0.04")
    p.add_argument("--grid-prob", default="0.34")
    p.add_argument("--grid-flat", default="0.40")
    p.add_argument("--grid-risk-open", default="0.45")
    p.add_argument("--grid-risk-exit", default="0.52")
    p.add_argument("--grid-confirmed-score", default="3,4")
    p.add_argument("--grid-strong-score", default="4,5")
    p.add_argument("--grid-extreme-score", default="5,6")
    p.add_argument("--grid-invalid-confirm-bars", default="2,3")
    p.add_argument("--grid-upgrade-profit-atr", default="0.5,0.8,1.0,1.2")
    p.add_argument("--grid-crash-upgrade-profit-atr", default="1.2,1.5,2.0")
    p.add_argument("--grid-min-trend-age-for-upgrade", default="0,1,2,3")
    p.add_argument("--grid-add-profit-atr", default="1.5,2.0")
    p.add_argument("--grid-max-trend-hold-bars", default="48,72")
    p.add_argument("--grid-strong-trend-hold-bars", default="72,96")
    p.add_argument("--grid-exhaustion-reduce-scale", default="0.3,0.5")
    p.add_argument("--grid-trail-start-atr", default="2.0")
    p.add_argument("--grid-trail-back-atr", default="1.2")
    p.add_argument(
        "--score-mode",
        choices=["balanced", "return_focus"],
        default="balanced",
        help="balanced favors trend acceptance metrics; return_focus weights total_return more",
    )
    p.add_argument("--trade-count-anchor", type=int, default=4, help="target trade count anchor from v014b baseline")
    p.add_argument("--trade-count-cap", type=int, default=9, help="soft cap from design doc")
    p.add_argument("--top-k", type=int, default=20)
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
    trade_count_anchor: int,
    trade_count_cap: int,
    score_mode: str,
) -> float:
    trades = int(metrics.get("trade_count", 0.0))
    total_return = metrics.get("total_return", 0.0)
    excess_return = metrics.get("excess_return", 0.0)
    max_drawdown = metrics.get("max_drawdown", 0.0)
    pf = metrics.get("profit_factor", 0.0)
    if not np.isfinite(pf):
        pf = 20.0
    pf = min(pf, 20.0)
    trend_upgrades = metrics.get("trend_upgrade_count", 0.0)
    trend_ret = metrics.get("trend_trade_total_return", 0.0)
    trend_hold = metrics.get("avg_trend_hold_bars", 0.0)
    missed_trend = metrics.get("missed_confirmed_trend_bars", 0.0)
    short_capture = metrics.get("short_trend_capture_ratio", 0.0)

    if score_mode == "return_focus":
        score = (
            10.0 * total_return
            + 2.5 * excess_return
            + 4.0 * max_drawdown
            + 0.06 * pf
            + 0.06 * min(trend_upgrades, 3.0)
            + 0.6 * trend_ret
            + 0.08 * short_capture
            - 0.0001 * missed_trend
        )
        if trend_upgrades < 1:
            score -= 0.20
        if trend_ret <= 0:
            score -= 0.10
        if trades > trade_count_cap:
            score -= (trades - trade_count_cap) * 0.05
        score -= abs(trades - trade_count_anchor) * 0.012
        return score

    score = (
        6.0 * total_return
        + 3.0 * excess_return
        + 0.08 * pf
        + 3.5 * max_drawdown
        + 0.10 * min(trend_upgrades, 3.0)
        + 1.5 * trend_ret
        + 0.01 * min(trend_hold, 24.0)
        + 0.20 * short_capture
        - 0.0002 * missed_trend
    )
    if trend_upgrades < 1:
        score -= 0.18
    if trend_ret <= 0:
        score -= 0.12
    if trades > trade_count_cap:
        score -= (trades - trade_count_cap) * 0.03
    score -= abs(trades - trade_count_anchor) * 0.01
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


def _finalize_signals(raw: list[TradingSignal | None], risk_open_max: float) -> list[TradingSignal | None]:
    out: list[TradingSignal | None] = [None] * len(raw)
    for i, sig in enumerate(raw):
        if sig is None:
            continue
        clone = TradingSignal(
            ts=sig.ts,
            price=sig.price,
            atr=sig.atr,
            p_up=sig.p_up,
            p_down=sig.p_down,
            p_flat=sig.p_flat,
            p_risk=sig.p_risk,
            pred_ret_1=sig.pred_ret_1,
            pred_ret_2=sig.pred_ret_2,
            pred_ret_3=sig.pred_ret_3,
            pred_ret_4=sig.pred_ret_4,
            pred_ret_5=sig.pred_ret_5,
            pred_cum_ret_5=sig.pred_cum_ret_5,
            source=sig.source,
            raw=dict(sig.raw),
        )
        out[i] = clone.finalize(risk_open_max)
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
    finalized_by_risk: dict[float, list[TradingSignal | None]] = {}

    edge_grid = _parse_float_grid(args.grid_edge)
    prob_grid = _parse_float_grid(args.grid_prob)
    flat_grid = _parse_float_grid(args.grid_flat)
    risk_open_grid = _parse_float_grid(args.grid_risk_open)
    risk_exit_grid = _parse_float_grid(args.grid_risk_exit)
    confirmed_grid = _parse_int_grid(args.grid_confirmed_score)
    strong_grid = _parse_int_grid(args.grid_strong_score)
    extreme_grid = _parse_int_grid(args.grid_extreme_score)
    invalid_grid = _parse_int_grid(args.grid_invalid_confirm_bars)
    upgrade_grid = _parse_float_grid(args.grid_upgrade_profit_atr)
    crash_upgrade_grid = _parse_float_grid(args.grid_crash_upgrade_profit_atr)
    min_trend_age_grid = _parse_int_grid(args.grid_min_trend_age_for_upgrade)
    add_grid = _parse_float_grid(args.grid_add_profit_atr)
    hold_grid = _parse_int_grid(args.grid_max_trend_hold_bars)
    strong_hold_grid = _parse_int_grid(args.grid_strong_trend_hold_bars)
    exhaustion_grid = _parse_float_grid(args.grid_exhaustion_reduce_scale)
    trail_start_grid = _parse_float_grid(args.grid_trail_start_atr)
    trail_back_grid = _parse_float_grid(args.grid_trail_back_atr)

    total = (
        len(edge_grid)
        * len(prob_grid)
        * len(flat_grid)
        * len(risk_open_grid)
        * len(risk_exit_grid)
        * len(confirmed_grid)
        * len(strong_grid)
        * len(extreme_grid)
        * len(invalid_grid)
        * len(upgrade_grid)
        * len(crash_upgrade_grid)
        * len(min_trend_age_grid)
        * len(add_grid)
        * len(hold_grid)
        * len(strong_hold_grid)
        * len(exhaustion_grid)
        * len(trail_start_grid)
        * len(trail_back_grid)
    )

    rows: list[dict] = []
    done = 0
    for edge in edge_grid:
        for prob in prob_grid:
            for flat in flat_grid:
                for risk_open in risk_open_grid:
                    if risk_open not in finalized_by_risk:
                        finalized_by_risk[risk_open] = _finalize_signals(raw_signals, risk_open)
                    cached = _CachedSignalProvider(model_provider.atr, finalized_by_risk[risk_open])
                    for risk_exit in risk_exit_grid:
                        rule = replace(
                            base_cfg.rule,
                            open_edge_threshold=edge,
                            open_prob_threshold=prob,
                            open_flat_max=flat,
                            risk_open_max=risk_open,
                            risk_exit_threshold=risk_exit,
                        )
                        for confirmed_score in confirmed_grid:
                            for strong_score in strong_grid:
                                if strong_score < confirmed_score:
                                    continue
                                for extreme_score in extreme_grid:
                                    if extreme_score < strong_score:
                                        continue
                                    trend_signal = replace(
                                        base_cfg.trend_signal,
                                        confirmed_score=confirmed_score,
                                        strong_score=strong_score,
                                        extreme_score=extreme_score,
                                    )
                                    for invalid_confirm_bars in invalid_grid:
                                        trend_signal2 = replace(
                                            trend_signal,
                                            invalid_confirm_bars=invalid_confirm_bars,
                                        )
                                        for upgrade_profit_atr in upgrade_grid:
                                            for crash_upgrade_profit_atr in crash_upgrade_grid:
                                                if crash_upgrade_profit_atr < upgrade_profit_atr:
                                                    continue
                                                for min_trend_age_for_upgrade in min_trend_age_grid:
                                                    for add_profit_atr in add_grid:
                                                        if add_profit_atr < upgrade_profit_atr:
                                                            continue
                                                        for max_trend_hold_bars in hold_grid:
                                                            for strong_trend_hold_bars in strong_hold_grid:
                                                                if strong_trend_hold_bars < max_trend_hold_bars:
                                                                    continue
                                                                for exhaustion_reduce_scale in exhaustion_grid:
                                                                    for trail_start_atr in trail_start_grid:
                                                                        for trail_back_atr in trail_back_grid:
                                                                            done += 1
                                                                            trend_position = replace(
                                                                                base_cfg.trend_position,
                                                                                upgrade_profit_atr=upgrade_profit_atr,
                                                                                crash_upgrade_profit_atr=crash_upgrade_profit_atr,
                                                                                min_trend_age_for_upgrade=min_trend_age_for_upgrade,
                                                                                add_profit_atr=add_profit_atr,
                                                                                max_trend_hold_bars=max_trend_hold_bars,
                                                                                strong_trend_hold_bars=strong_trend_hold_bars,
                                                                                exhaustion_reduce_scale=exhaustion_reduce_scale,
                                                                                trail_start_atr=trail_start_atr,
                                                                                trail_back_atr=trail_back_atr,
                                                                            )
                                                                            cfg = replace(
                                                                                base_cfg,
                                                                                rule=rule,
                                                                                trend_signal=trend_signal2,
                                                                                trend_position=trend_position,
                                                                            )
                                                                            metrics = _run_with_cfg(df, cached, start_idx, end_idx, cfg)
                                                                            row = {
                                                                                "open_edge_threshold": edge,
                                                                                "open_prob_threshold": prob,
                                                                                "open_flat_max": flat,
                                                                                "risk_open_max": risk_open,
                                                                                "risk_exit_threshold": risk_exit,
                                                                                "confirmed_score": confirmed_score,
                                                                                "strong_score": strong_score,
                                                                                "extreme_score": extreme_score,
                                                                                "invalid_confirm_bars": invalid_confirm_bars,
                                                                                "upgrade_profit_atr": upgrade_profit_atr,
                                                                                "crash_upgrade_profit_atr": crash_upgrade_profit_atr,
                                                                                "min_trend_age_for_upgrade": min_trend_age_for_upgrade,
                                                                                "add_profit_atr": add_profit_atr,
                                                                                "max_trend_hold_bars": max_trend_hold_bars,
                                                                                "strong_trend_hold_bars": strong_trend_hold_bars,
                                                                                "exhaustion_reduce_scale": exhaustion_reduce_scale,
                                                                                "trail_start_atr": trail_start_atr,
                                                                                "trail_back_atr": trail_back_atr,
                                                                                **metrics,
                                                                            }
                                                                            row["score"] = _score(
                                                                                metrics,
                                                                                trade_count_anchor=args.trade_count_anchor,
                                                                                trade_count_cap=args.trade_count_cap,
                                                                                score_mode=args.score_mode,
                                                                            )
                                                                            rows.append(row)
                                                                            if done % 100 == 0 or done == total:
                                                                                print(f"grid progress: {done}/{total}")

    rows_sorted = sorted(rows, key=lambda x: x["score"], reverse=True)
    viable = [
        r
        for r in rows_sorted
        if r.get("trend_upgrade_count", 0.0) >= 1.0
        and r.get("trend_trade_total_return", 0.0) > 0.0
        and r.get("position_limit_violations", 0.0) == 0.0
        and r.get("risk_rule_violations", 0.0) == 0.0
    ]
    best = viable[0] if viable else rows_sorted[0]
    top_rows = (viable if viable else rows_sorted)[: args.top_k]

    payload = {
        "checkpoint": args.checkpoint,
        "base_config": args.base_config,
        "split": args.split,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "grid_size": total,
        "score_mode": args.score_mode,
        "best": best,
        "top": top_rows,
    }
    (out_dir / "tuning_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    keys = (
        "open_edge_threshold",
        "open_prob_threshold",
        "open_flat_max",
        "risk_open_max",
        "risk_exit_threshold",
        "confirmed_score",
        "strong_score",
        "extreme_score",
        "invalid_confirm_bars",
        "upgrade_profit_atr",
        "crash_upgrade_profit_atr",
        "min_trend_age_for_upgrade",
        "add_profit_atr",
        "max_trend_hold_bars",
        "strong_trend_hold_bars",
        "exhaustion_reduce_scale",
        "trail_start_atr",
        "trail_back_atr",
        "total_return",
        "excess_return",
        "max_drawdown",
        "trade_count",
        "profit_factor",
        "trend_upgrade_count",
        "trend_trade_total_return",
        "avg_trend_hold_bars",
        "short_trend_capture_ratio",
        "missed_confirmed_trend_bars",
        "score",
    )
    lines = [
        f"checkpoint={args.checkpoint}",
        f"base_config={args.base_config}",
        f"split={args.split}",
        f"bars=[{start_idx},{end_idx})",
        f"grid_size={total}",
        "",
        "[best]",
    ]
    for k in keys:
        v = best.get(k)
        if isinstance(v, float):
            lines.append(f"{k}={v:.6f}")
        else:
            lines.append(f"{k}={v}")
    lines.append("")
    lines.append("[top]")
    for i, r in enumerate(top_rows, start=1):
        lines.append(
            f"{i:02d}) ret={r['total_return']:.2%} mdd={r['max_drawdown']:.2%} trades={int(r['trade_count'])} "
            f"upgrades={int(r.get('trend_upgrade_count', 0))} trend_ret={r.get('trend_trade_total_return', 0.0):.2%} "
            f"missed={int(r.get('missed_confirmed_trend_bars', 0))} "
            f"edge={r['open_edge_threshold']:.3f} prob={r['open_prob_threshold']:.3f} "
            f"confirm={int(r['confirmed_score'])} strong={int(r['strong_score'])} extreme={int(r['extreme_score'])} "
            f"invalid={int(r['invalid_confirm_bars'])} up_atr={r['upgrade_profit_atr']:.2f} "
            f"crash_up_atr={r['crash_upgrade_profit_atr']:.2f} age={int(r['min_trend_age_for_upgrade'])} "
            f"exh={r['exhaustion_reduce_scale']:.2f} "
            f"add_atr={r['add_profit_atr']:.2f} hold={int(r['max_trend_hold_bars'])}/{int(r['strong_trend_hold_bars'])} "
            f"score={r['score']:.4f}"
        )
    (out_dir / "tuning_results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    tuned_cfg_path = (
        Path(args.tuned_config_out)
        if args.tuned_config_out
        else _ROOT / "configs" / "trading_rule_v016_trend_signal_tuned_0062e.json"
    )
    tuned_payload = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    tuned_payload["rule"]["open_edge_threshold"] = float(best["open_edge_threshold"])
    tuned_payload["rule"]["open_prob_threshold"] = float(best["open_prob_threshold"])
    tuned_payload["rule"]["open_flat_max"] = float(best["open_flat_max"])
    tuned_payload["rule"]["risk_open_max"] = float(best["risk_open_max"])
    tuned_payload["rule"]["risk_exit_threshold"] = float(best["risk_exit_threshold"])
    tuned_payload["trend_signal"]["confirmed_score"] = int(best["confirmed_score"])
    tuned_payload["trend_signal"]["strong_score"] = int(best["strong_score"])
    tuned_payload["trend_signal"]["extreme_score"] = int(best["extreme_score"])
    tuned_payload["trend_signal"]["invalid_confirm_bars"] = int(best["invalid_confirm_bars"])
    tuned_payload["trend_position"]["upgrade_profit_atr"] = float(best["upgrade_profit_atr"])
    tuned_payload["trend_position"]["crash_upgrade_profit_atr"] = float(best["crash_upgrade_profit_atr"])
    tuned_payload["trend_position"]["allow_crash_trend_upgrade"] = bool(
        base_cfg.trend_position.allow_crash_trend_upgrade
        if "allow_crash_trend_upgrade" not in best
        else best.get("allow_crash_trend_upgrade", base_cfg.trend_position.allow_crash_trend_upgrade)
    )
    tuned_payload["trend_position"]["min_trend_age_for_upgrade"] = int(best["min_trend_age_for_upgrade"])
    tuned_payload["trend_position"]["add_profit_atr"] = float(best["add_profit_atr"])
    tuned_payload["trend_position"]["max_trend_hold_bars"] = int(best["max_trend_hold_bars"])
    tuned_payload["trend_position"]["strong_trend_hold_bars"] = int(best["strong_trend_hold_bars"])
    tuned_payload["trend_position"]["exhaustion_reduce_scale"] = float(best["exhaustion_reduce_scale"])
    tuned_payload["trend_position"]["trail_start_atr"] = float(best["trail_start_atr"])
    tuned_payload["trend_position"]["trail_back_atr"] = float(best["trail_back_atr"])
    tuned_payload["_tuning_meta"] = {
        "source": "tune_backtest_trading_system_v016.py",
        "checkpoint": args.checkpoint,
        "tune_split": args.split,
        "score": best.get("score"),
        "valid_metrics": {
            k: best.get(k)
            for k in (
                "total_return",
                "max_drawdown",
                "trade_count",
                "profit_factor",
                "trend_upgrade_count",
                "trend_trade_total_return",
                "avg_trend_hold_bars",
            )
        },
    }
    tuned_cfg_path.write_text(json.dumps(tuned_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"saved: {out_dir / 'tuning_results.txt'}")
    print(f"saved: {out_dir / 'tuning_results.json'}")
    print(f"saved tuned config: {tuned_cfg_path}")
    print(
        f"best ret={best.get('total_return', 0.0):.2%} "
        f"mdd={best.get('max_drawdown', 0.0):.2%} "
        f"trades={int(best.get('trade_count', 0))} "
        f"upgrades={int(best.get('trend_upgrade_count', 0))} "
        f"trend_ret={best.get('trend_trade_total_return', 0.0):.2%} "
        f"score={best.get('score', 0.0):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
