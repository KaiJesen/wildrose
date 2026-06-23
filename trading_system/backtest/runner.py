from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from market_data.schema import COL_CLOSE, COL_HIGH, COL_LOW, COL_OPEN, COL_TIME
from trading_system.config import TradingSystemConfig
from trading_system.engine import Bar, TradingEngine
from trading_system.logger import TradeLogger
from trading_system.metrics import compute_metrics


@dataclass
class BacktestResult:
    metrics: dict[str, float]
    logger: TradeLogger


def run_backtest(
    df,
    *,
    signal_provider,
    start_idx: int,
    end_idx: int,
    cfg: TradingSystemConfig,
    out_dir,
    best_point_provider=None,
) -> BacktestResult:
    logger = TradeLogger(out_dir=out_dir)
    engine = TradingEngine(cfg, logger)
    open_px = df[COL_OPEN].to_numpy(dtype=np.float64)
    close = df[COL_CLOSE].to_numpy(dtype=np.float64)
    for i in range(start_idx, end_idx):
        cur = Bar(
            idx=i,
            ts=df[COL_TIME].iloc[i],
            open=float(open_px[i]),
            high=float(df[COL_HIGH].iloc[i]),
            low=float(df[COL_LOW].iloc[i]),
            close=float(close[i]),
            atr=float(signal_provider.atr[i]),
        )
        nxt = Bar(
            idx=i + 1,
            ts=df[COL_TIME].iloc[i + 1],
            open=float(open_px[i + 1]),
            high=float(df[COL_HIGH].iloc[i + 1]),
            low=float(df[COL_LOW].iloc[i + 1]),
            close=float(close[i + 1]),
            atr=float(signal_provider.atr[i + 1]),
        )
        sig = signal_provider.signal_at(i)
        bp_sig = best_point_provider.signal_at(i) if best_point_provider is not None else None
        engine.on_bar_close(sig, cur, nxt, best_point_signal=bp_sig)

    logger.flush()
    eq = [row["equity"] for row in logger.equity_curve]
    # Benchmark open-to-open on same bars.
    bh = [1.0]
    for i in range(start_idx, end_idx):
        ret = (open_px[i + 1] - open_px[i]) / max(1e-12, open_px[i])
        bh.append(bh[-1] * (1.0 + float(ret)))
    wins = sum(1 for t in logger.trades if float(t["net_pnl"]) > 0)
    gross_profit = sum(max(0.0, float(t["net_pnl"])) for t in logger.trades)
    gross_loss = sum(max(0.0, -float(t["net_pnl"])) for t in logger.trades)
    avg_hold = float(np.mean([float(t["bars_held"]) for t in logger.trades])) if logger.trades else 0.0
    avg_fee = float(np.mean([float(t["fee"]) for t in logger.trades])) if logger.trades else 0.0
    metrics = compute_metrics(
        eq,
        benchmark_return=float(bh[-1] - 1.0),
        trade_count=len(logger.trades),
        wins=wins,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        avg_bars_held=avg_hold,
        avg_fee_per_trade=avg_fee,
        max_margin_loss_ratio_observed=engine.max_margin_loss_ratio_observed,
        position_limit_violations=engine.position_limit_violations,
        risk_rule_violations=engine.risk_rule_violations,
    )
    probe_trades = [t for t in logger.trades if int(t.get("entry_was_probe", 0)) == 1]
    sentinel_trades = [t for t in logger.trades if int(t.get("entry_was_sentinel", 0)) == 1]
    probe_short_count = len(probe_trades)
    probe_wins = sum(1 for t in probe_trades if float(t.get("net_pnl", 0.0)) > 0.0)
    probe_total_return = float(sum(float(t.get("net_pnl", 0.0)) for t in probe_trades))
    sentinel_short_count = len(sentinel_trades)
    sentinel_upgrade_count = sum(1 for d in logger.decisions if d.get("reason_code") == "UPGRADE_SENTINEL_TO_MODEL_SHORT")
    sentinel_not_confirmed_close_count = sum(1 for d in logger.decisions if d.get("reason_code") == "CLOSE_SENTINEL_NOT_CONFIRMED")
    sentinel_total_return = float(sum(float(t.get("net_pnl", 0.0)) for t in sentinel_trades))
    blocked_long_downtrend_count = sum(
        1 for d in logger.decisions if d.get("reason_code") == "BLOCK_LONG_DOWNTREND"
    )
    downtrend_rows = [d for d in logger.decisions if int(d.get("trend_is_downtrend", 0)) == 1]
    missed_downtrend_bars = sum(
        1
        for d in downtrend_rows
        if d.get("state") == "FLAT" and d.get("action") not in ("OPEN_SHORT", "REVERSE")
    )
    short_cover_bars = sum(1 for d in downtrend_rows if d.get("state") == "SHORT")
    model_short_trades = [t for t in logger.trades if t.get("entry_reason") in ("OPEN_SHORT_SIGNAL", "UPGRADE_SENTINEL_TO_MODEL_SHORT")]
    model_short_trend_hold_count = sum(1 for t in model_short_trades if str(t.get("hold_mode", "NORMAL")) == "TREND")
    avg_model_short_hold_bars = (
        float(np.mean([float(t.get("bars_held", 0.0)) for t in model_short_trades])) if model_short_trades else 0.0
    )
    close_max_hold_bars_in_downtrend_count = sum(
        1
        for d in logger.decisions
        if d.get("reason_code") in ("CLOSE_MAX_HOLD_BARS", "CLOSE_TREND_MAX_HOLD_BARS") and int(d.get("trend_is_downtrend", 0)) == 1
    )
    close_short_trend_broken_count = sum(1 for d in logger.decisions if d.get("reason_code") == "CLOSE_SHORT_TREND_BROKEN")
    crash_short_count = sum(1 for d in logger.decisions if d.get("reason_code") == "OPEN_SHORT_CRASH")
    crash_upgrade_count = sum(1 for d in logger.decisions if d.get("reason_code") == "UPGRADE_CRASH_TO_MODEL_SHORT")
    same_regime_reentry_count = sum(1 for d in logger.decisions if d.get("reason_code") == "BLOCK_CRASH_ONCE_PER_REGIME")
    model_blind_crash_count = sum(1 for d in logger.decisions if int(d.get("is_model_blind_crash", 0)) == 1)
    trend_upgrade_count = sum(
        1 for d in logger.decisions if d.get("reason_code") in ("UPGRADE_TO_TREND_LONG", "UPGRADE_TO_TREND_SHORT")
    )
    trend_trades = [t for t in logger.trades if int(t.get("trend_upgrade_done", 0)) == 1 or str(t.get("hold_mode", "")) == "TREND"]
    trend_trade_total_return = float(sum(float(t.get("net_pnl", 0.0)) for t in trend_trades))
    avg_trend_hold_bars = float(np.mean([float(t.get("bars_held", 0.0)) for t in trend_trades])) if trend_trades else 0.0
    add_trend_continuation_count = sum(1 for d in logger.decisions if d.get("reason_code") == "ADD_TREND_CONTINUATION")
    reduce_trend_exhaustion_count = sum(1 for d in logger.decisions if d.get("reason_code") == "REDUCE_TREND_EXHAUSTION")
    close_trend_broken_count = sum(1 for d in logger.decisions if d.get("reason_code") == "CLOSE_TREND_BROKEN")
    confirmed_up_rows = [d for d in logger.decisions if d.get("trend_direction") == "UP" and int(d.get("trend_is_confirmed", 0)) == 1]
    confirmed_down_rows = [d for d in logger.decisions if d.get("trend_direction") == "DOWN" and int(d.get("trend_is_confirmed", 0)) == 1]
    long_capture = sum(1 for d in confirmed_up_rows if d.get("state") == "LONG")
    short_capture = sum(1 for d in confirmed_down_rows if d.get("state") == "SHORT")
    missed_confirmed_trend_bars = sum(
        1
        for d in logger.decisions
        if int(d.get("trend_is_confirmed", 0)) == 1 and d.get("state") == "FLAT" and d.get("action") == "HOLD"
    )
    metrics.update(
        {
            "probe_short_count": float(probe_short_count),
            "probe_short_win_rate": float(probe_wins / max(1, probe_short_count)),
            "probe_short_total_return": probe_total_return,
            "sentinel_short_count": float(sentinel_short_count),
            "sentinel_upgrade_count": float(sentinel_upgrade_count),
            "sentinel_not_confirmed_close_count": float(sentinel_not_confirmed_close_count),
            "sentinel_short_total_return": sentinel_total_return,
            "blocked_long_downtrend_count": float(blocked_long_downtrend_count),
            "missed_downtrend_bars": float(missed_downtrend_bars),
            "short_coverage_downtrend_ratio": float(short_cover_bars / max(1, len(downtrend_rows))),
            "model_short_trend_hold_count": float(model_short_trend_hold_count),
            "avg_model_short_hold_bars": float(avg_model_short_hold_bars),
            "close_max_hold_bars_in_downtrend_count": float(close_max_hold_bars_in_downtrend_count),
            "close_short_trend_broken_count": float(close_short_trend_broken_count),
            "crash_short_count": float(crash_short_count),
            "crash_upgrade_count": float(crash_upgrade_count),
            "same_regime_reentry_count": float(same_regime_reentry_count),
            "model_blind_crash_count": float(model_blind_crash_count),
            "trend_upgrade_count": float(trend_upgrade_count),
            "trend_trade_count": float(len(trend_trades)),
            "trend_trade_total_return": float(trend_trade_total_return),
            "avg_trend_hold_bars": float(avg_trend_hold_bars),
            "close_trend_broken_count": float(close_trend_broken_count),
            "reduce_trend_exhaustion_count": float(reduce_trend_exhaustion_count),
            "add_trend_continuation_count": float(add_trend_continuation_count),
            "short_trend_capture_ratio": float(short_capture / max(1, len(confirmed_down_rows))),
            "long_trend_capture_ratio": float(long_capture / max(1, len(confirmed_up_rows))),
            "missed_confirmed_trend_bars": float(missed_confirmed_trend_bars),
        }
    )
    return BacktestResult(metrics=metrics, logger=logger)

