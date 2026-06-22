#!/usr/bin/env python3
"""Grid-search rule thresholds for trading system v014."""

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

from _train_common import add_data_args, add_feature_args, add_segment_args, add_vq_args, apply_real_data_defaults, fetch_ohlcv_df, prepare_bar_series_from_args
from trading_system.adapters.market_state_model import ModelSignalProvider
from trading_system.backtest.runner import BacktestResult, run_backtest
from trading_system.config import TradingSystemConfig, load_config
from trading_system.signal import TradingSignal


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune v014 trading rule thresholds")
    add_data_args(p)
    add_feature_args(p)
    add_segment_args(p)
    add_vq_args(p)
    p.add_argument("--checkpoint", default="checkpoints/0065a_multi_seed_s45_market_state_stability/market_state_best.pt")
    p.add_argument("--base-config", default="configs/trading_rule_v014_aggressive.json")
    p.add_argument("--split", choices=["train", "valid", "test"], default="valid")
    p.add_argument("--trunk-layers", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default="backtest/backtest_rule_v014_aggressive_tuning")
    p.add_argument("--tuned-config-out", default="", help="optional path for tuned config json")
    p.add_argument("--grid-edge", default="0.02,0.03,0.04,0.05")
    p.add_argument("--grid-prob", default="0.28,0.30,0.32,0.34")
    p.add_argument("--grid-flat", default="0.40,0.42,0.44,0.46")
    p.add_argument("--grid-risk-open", default="0.45,0.48,0.50")
    p.add_argument("--grid-risk-exit", default="0.52,0.55,0.58")
    p.add_argument("--min-trades", type=int, default=8)
    p.add_argument("--top-k", type=int, default=20)
    p.set_defaults(source="binance_vision", symbol="BTCUSDT", interval="1h", days=365)
    return p.parse_args()


def _parse_grid(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _split_idx(bundle, split: str):
    if split == "train":
        return bundle.train_idx
    if split == "valid":
        return bundle.valid_idx
    return bundle.test_idx


def _score(metrics: dict[str, float]) -> float:
    trades = metrics.get("trade_count", 0.0)
    trade_penalty = 0.0 if trades >= 8 else (8 - trades) * 0.015
    pf = metrics.get("profit_factor", 0.0)
    if not np.isfinite(pf):
        pf = 3.0
    return (
        2.0 * metrics.get("total_return", 0.0)
        + 1.5 * metrics.get("excess_return", 0.0)
        + 0.5 * pf
        + 2.0 * metrics.get("max_drawdown", 0.0)
        - trade_penalty
    )


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

    edge_grid = _parse_grid(args.grid_edge)
    prob_grid = _parse_grid(args.grid_prob)
    flat_grid = _parse_grid(args.grid_flat)
    risk_open_grid = _parse_grid(args.grid_risk_open)
    risk_exit_grid = _parse_grid(args.grid_risk_exit)
    total = len(edge_grid) * len(prob_grid) * len(flat_grid) * len(risk_open_grid) * len(risk_exit_grid)

    rows: list[dict] = []
    done = 0
    for edge in edge_grid:
        for prob in prob_grid:
            for flat in flat_grid:
                for risk_open in risk_open_grid:
                    for risk_exit in risk_exit_grid:
                        done += 1
                        rule = replace(
                            base_cfg.rule,
                            open_edge_threshold=edge,
                            open_prob_threshold=prob,
                            open_flat_max=flat,
                            risk_open_max=risk_open,
                            risk_exit_threshold=risk_exit,
                        )
                        cfg = replace(base_cfg, rule=rule)
                        if risk_open not in finalized_by_risk:
                            finalized_by_risk[risk_open] = _finalize_signals(raw_signals, risk_open)
                        cached = _CachedSignalProvider(model_provider.atr, finalized_by_risk[risk_open])
                        metrics = _run_with_cfg(df, cached, start_idx, end_idx, cfg)
                        row = {
                            "open_edge_threshold": edge,
                            "open_prob_threshold": prob,
                            "open_flat_max": flat,
                            "risk_open_max": risk_open,
                            "risk_exit_threshold": risk_exit,
                            **metrics,
                        }
                        row["score"] = _score(metrics)
                        rows.append(row)
                        if done % 100 == 0 or done == total:
                            print(f"grid progress: {done}/{total}")

    rows_sorted = sorted(rows, key=lambda x: x["score"], reverse=True)
    viable = [r for r in rows_sorted if int(r.get("trade_count", 0)) >= args.min_trades]
    best = viable[0] if viable else rows_sorted[0]
    top_rows = (viable if viable else rows_sorted)[: args.top_k]

    payload = {
        "checkpoint": args.checkpoint,
        "base_config": args.base_config,
        "split": args.split,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "grid_size": total,
        "min_trades_filter": args.min_trades,
        "best": best,
        "top": top_rows,
    }
    (out_dir / "tuning_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"checkpoint={args.checkpoint}",
        f"base_config={args.base_config}",
        f"split={args.split}",
        f"bars=[{start_idx},{end_idx})",
        f"grid_size={total}",
        f"min_trades_filter={args.min_trades}",
        "",
        "[best]",
    ]
    for k in (
        "open_edge_threshold",
        "open_prob_threshold",
        "open_flat_max",
        "risk_open_max",
        "risk_exit_threshold",
        "total_return",
        "benchmark_return",
        "excess_return",
        "max_drawdown",
        "trade_count",
        "win_rate",
        "profit_factor",
        "score",
    ):
        v = best.get(k)
        if isinstance(v, float):
            lines.append(f"{k}={v:.6f}")
        else:
            lines.append(f"{k}={v}")
    lines.append("")
    lines.append("[top]")
    for i, r in enumerate(top_rows, start=1):
        lines.append(
            f"{i:02d}) edge={r['open_edge_threshold']:.3f} prob={r['open_prob_threshold']:.3f} "
            f"flat={r['open_flat_max']:.3f} risk_open={r['risk_open_max']:.3f} risk_exit={r['risk_exit_threshold']:.3f} "
            f"ret={r['total_return']:.2%} mdd={r['max_drawdown']:.2%} trades={int(r['trade_count'])} "
            f"win={r['win_rate']:.1%} pf={r['profit_factor']:.3f} score={r['score']:.4f}"
        )
    (out_dir / "tuning_results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Emit tuned config JSON for downstream backtest.
    tuned_rule = replace(
        base_cfg.rule,
        open_edge_threshold=float(best["open_edge_threshold"]),
        open_prob_threshold=float(best["open_prob_threshold"]),
        open_flat_max=float(best["open_flat_max"]),
        risk_open_max=float(best["risk_open_max"]),
        risk_exit_threshold=float(best["risk_exit_threshold"]),
    )
    tuned_cfg_path = Path(args.tuned_config_out) if args.tuned_config_out else _ROOT / "configs" / "trading_rule_v014_aggressive_tuned.json"
    tuned_payload = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    tuned_payload["rule"]["open_edge_threshold"] = tuned_rule.open_edge_threshold
    tuned_payload["rule"]["open_prob_threshold"] = tuned_rule.open_prob_threshold
    tuned_payload["rule"]["open_flat_max"] = tuned_rule.open_flat_max
    tuned_payload["rule"]["risk_open_max"] = tuned_rule.risk_open_max
    tuned_payload["rule"]["risk_exit_threshold"] = tuned_rule.risk_exit_threshold
    tuned_payload["_tuning_meta"] = {
        "source": "tune_backtest_trading_system_v014.py",
        "checkpoint": args.checkpoint,
        "tune_split": args.split,
        "score": best.get("score"),
        "valid_metrics": {k: best.get(k) for k in ("total_return", "max_drawdown", "trade_count", "win_rate", "profit_factor")},
    }
    tuned_cfg_path.write_text(json.dumps(tuned_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"saved: {out_dir / 'tuning_results.txt'}")
    print(f"saved: {out_dir / 'tuning_results.json'}")
    print(f"saved tuned config: {tuned_cfg_path}")
    print(
        f"best ret={best.get('total_return', 0.0):.2%} "
        f"mdd={best.get('max_drawdown', 0.0):.2%} "
        f"trades={int(best.get('trade_count', 0))} "
        f"score={best.get('score', 0.0):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
