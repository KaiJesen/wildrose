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
    strategy_equity: list[float]
    benchmark_equity: list[float]


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
        bar_count=end_idx - start_idx,
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
    upgrade_crash_to_trend_short_count = sum(1 for d in logger.decisions if d.get("reason_code") == "UPGRADE_CRASH_TO_TREND_SHORT")
    hold_crash_trend_confirming_count = sum(1 for d in logger.decisions if d.get("reason_code") == "HOLD_CRASH_TREND_CONFIRMING")
    close_trend_exit_confirmed_count = sum(1 for d in logger.decisions if d.get("reason_code") == "CLOSE_TREND_EXIT_CONFIRMED")
    reduce_trend_profit_lock_count = sum(1 for d in logger.decisions if d.get("reason_code") == "REDUCE_TREND_PROFIT_LOCK")
    hold_trend_runner_count = sum(1 for d in logger.decisions if d.get("reason_code") == "HOLD_TREND_RUNNER")
    trend_upgrade_count = sum(
        1
        for d in logger.decisions
        if d.get("reason_code")
        in (
            "UPGRADE_TO_TREND_LONG",
            "UPGRADE_TO_TREND_SHORT",
            "UPGRADE_CRASH_TO_TREND_SHORT",
            "UPGRADE_SLOW_LONG_TO_TREND",
        )
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
    slow_up_rows = [d for d in logger.decisions if int(d.get("is_slow_uptrend", 0)) == 1]
    missed_slow_uptrend_bars = sum(
        1 for d in slow_up_rows if d.get("state") == "FLAT" and d.get("reason_code") in ("HOLD_NO_ENTRY", "WATCH_SLOW_UPTREND")
    )
    slow_up_open_count = sum(1 for d in logger.decisions if d.get("reason_code") == "OPEN_LONG_SLOW_TREND")
    trend_qualified_open_count = sum(
        1
        for d in logger.decisions
        if d.get("reason_code") in ("OPEN_LONG_TREND_QUALIFIED", "OPEN_SHORT_TREND_QUALIFIED")
    )
    watch_slow_uptrend_count = sum(1 for d in logger.decisions if d.get("reason_code") == "WATCH_SLOW_UPTREND")
    upgrade_slow_long_to_trend_count = sum(1 for d in logger.decisions if d.get("reason_code") == "UPGRADE_SLOW_LONG_TO_TREND")
    close_slow_uptrend_broken_count = sum(1 for d in logger.decisions if d.get("reason_code") == "CLOSE_SLOW_UPTREND_BROKEN")
    reduce_slow_up_profit_lock_count = sum(1 for d in logger.decisions if d.get("reason_code") == "REDUCE_SLOW_UP_PROFIT_LOCK")
    hold_slow_up_runner_count = sum(1 for d in logger.decisions if d.get("reason_code") == "HOLD_SLOW_UP_RUNNER")
    slow_up_trades = [t for t in logger.trades if int(t.get("entry_was_slow_up", 0)) == 1]
    slow_up_trade_total_return = float(sum(float(t.get("net_pnl", 0.0)) for t in slow_up_trades))
    trend_qualified_trades = [t for t in logger.trades if int(t.get("entry_was_trend_qualified", 0)) == 1]
    trend_qualified_pnl = float(sum(float(t.get("net_pnl", 0.0)) for t in trend_qualified_trades))
    avg_slow_up_hold_bars = float(np.mean([float(t.get("bars_held", 0.0)) for t in slow_up_trades])) if slow_up_trades else 0.0
    confirmed_leg_rows = [
        d
        for d in logger.decisions
        if int(d.get("is_leg_confirmed", 0)) == 1
        and d.get("leg_type") in ("SLOW_UP_LEG", "FAST_UP_LEG", "SLOW_DOWN_LEG", "FAST_DOWN_LEG", "CRASH_LEG")
    ]
    covered_leg_bars = sum(
        1
        for d in confirmed_leg_rows
        if d.get("state") in ("LONG", "SHORT")
    )
    leg_coverage_ratio = float(covered_leg_bars / max(1, len(confirmed_leg_rows)))
    slow_up_leg_ids = {int(d.get("leg_id", -1)) for d in logger.decisions if d.get("leg_type") == "SLOW_UP_LEG" and int(d.get("is_leg_confirmed", 0)) == 1}
    traded_slow_up_legs = {
        int(t.get("entry_leg_id", -1))
        for t in logger.trades
        if str(t.get("entry_leg_type", "")) == "SLOW_UP_LEG" or int(t.get("entry_was_slow_up", 0)) == 1
    }
    missed_slow_up_legs = float(max(0, len(slow_up_leg_ids) - len({i for i in traded_slow_up_legs if i >= 0})))
    down_leg_rows = [d for d in logger.decisions if d.get("leg_type") in ("FAST_DOWN_LEG", "CRASH_LEG", "SLOW_DOWN_LEG") and int(d.get("is_leg_confirmed", 0)) == 1]
    short_cover_down_legs = sum(1 for d in down_leg_rows if d.get("state") == "SHORT")
    missed_fast_down_legs = float(max(0, len({int(d.get("leg_id", -1)) for d in down_leg_rows}) - (1 if short_cover_down_legs > 0 else 0)))
    avg_hold_vs_leg_duration = float(
        np.mean(
            [
                float(t.get("bars_held", 0.0))
                / max(
                    1.0,
                    float(
                        next(
                            (
                                d.get("bars_since_leg_start", 1.0)
                                for d in logger.decisions
                                if int(d.get("leg_id", -2)) == int(t.get("entry_leg_id", -1))
                            ),
                            1.0,
                        )
                    ),
                )
                for t in logger.trades
                if int(t.get("entry_leg_id", -1)) >= 0
            ]
        )
    ) if logger.trades else 0.0
    false_leg_entry_count = sum(
        1
        for d in logger.decisions
        if d.get("reason_code") in ("OPEN_LONG_SLOW_TREND", "OPEN_SHORT_CRASH")
        and d.get("leg_state") == "LEG_FORMING"
        and float(d.get("position_ratio", 0.0)) > 0.06
    )
    close_trend_leg_end_count = sum(1 for d in logger.decisions if d.get("reason_code") == "CLOSE_TREND_LEG_END")
    block_counter_trend_count = sum(
        1 for d in logger.decisions if d.get("reason_code") in ("BLOCK_COUNTER_TREND_LONG", "BLOCK_COUNTER_TREND_SHORT")
    )
    bias_rows = logger.decisions
    bias_field_nonempty = sum(1 for d in bias_rows if d.get("decision_scope"))
    bias_reason_nonempty = sum(1 for d in bias_rows if d.get("bias_reason_codes"))
    hard_counter_open_count = float(engine.hard_counter_open_count)
    legacy_trend_direct_block_count = float(engine.legacy_trend_direct_block_count)
    open_actions = [
        d
        for d in logger.decisions
        if d.get("action") in ("OPEN_LONG", "OPEN_SHORT") and d.get("reason_code", "").startswith("OPEN_")
    ]
    bias_reason_coverage = (
        sum(1 for d in open_actions if d.get("bias_reason_codes")) / max(1, len(open_actions))
    )
    max_position_ratio_observed = float(max((float(d.get("position_ratio", 0.0)) for d in logger.decisions), default=0.0))
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
            "upgrade_crash_to_trend_short_count": float(upgrade_crash_to_trend_short_count),
            "hold_crash_trend_confirming_count": float(hold_crash_trend_confirming_count),
            "close_trend_exit_confirmed_count": float(close_trend_exit_confirmed_count),
            "reduce_trend_profit_lock_count": float(reduce_trend_profit_lock_count),
            "hold_trend_runner_count": float(hold_trend_runner_count),
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
            "missed_slow_uptrend_bars": float(missed_slow_uptrend_bars),
            "slow_up_open_count": float(slow_up_open_count),
            "trend_qualified_open_count": float(trend_qualified_open_count),
            "watch_slow_uptrend_count": float(watch_slow_uptrend_count),
            "upgrade_slow_long_to_trend_count": float(upgrade_slow_long_to_trend_count),
            "close_slow_uptrend_broken_count": float(close_slow_uptrend_broken_count),
            "reduce_slow_up_profit_lock_count": float(reduce_slow_up_profit_lock_count),
            "hold_slow_up_runner_count": float(hold_slow_up_runner_count),
            "slow_up_trade_count": float(len(slow_up_trades)),
            "slow_up_trade_total_return": float(slow_up_trade_total_return),
            "trend_qualified_trade_count": float(len(trend_qualified_trades)),
            "trend_qualified_pnl": float(trend_qualified_pnl),
            "avg_slow_up_hold_bars": float(avg_slow_up_hold_bars),
            "leg_coverage_ratio": float(leg_coverage_ratio),
            "missed_slow_up_legs": float(missed_slow_up_legs),
            "missed_fast_down_legs": float(missed_fast_down_legs),
            "avg_hold_vs_leg_duration": float(avg_hold_vs_leg_duration),
            "false_leg_entry_count": float(false_leg_entry_count),
            "close_trend_leg_end_count": float(close_trend_leg_end_count),
            "block_counter_trend_count": float(block_counter_trend_count),
            "bias_field_nonempty_ratio": float(bias_field_nonempty / max(1, len(bias_rows))),
            "bias_reason_nonempty_ratio": float(bias_reason_nonempty / max(1, len(bias_rows))),
            "hard_counter_open_count": hard_counter_open_count,
            "legacy_trend_direct_block_count": legacy_trend_direct_block_count,
            "legacy_trend_direct_read_count": float(engine.legacy_trend_direct_read_count),
            "bias_reason_codes_coverage": float(bias_reason_coverage),
            "max_position_ratio_observed": max_position_ratio_observed,
            "trend_add_candidate_count": float(engine.trend_add_candidate_count),
            "trend_add_risk_evaluated_count": float(engine.trend_add_risk_evaluated_count),
            "trend_add_rejected_by_risk_count": float(engine.trend_add_rejected_by_risk_count),
            "trend_add_allowed_count": float(engine.trend_add_allowed_count),
        }
    )
    return BacktestResult(metrics=metrics, logger=logger, strategy_equity=eq, benchmark_equity=bh)

