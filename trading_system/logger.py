from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path

from trading_system.adapters.best_point_model import BestPointSignal
from trading_system.crash import CrashContext
from trading_system.execution import FillEvent
from trading_system.portfolio import PortfolioState
from trading_system.rules import TradingAction
from trading_system.signal import TradingSignal
from trading_system.trend import TrendContext
from trading_system.trend_signal import TrendSignal
from trading_system.slow_trend import SlowTrendContext
from trading_system.trend_bias import RiskBudget, TrendBiasContext
from trading_system.trend_entry_qualifier import TrendEntryQualification
from trading_system.trend_segment import SegmentContext


@dataclass
class TradeLogger:
    out_dir: Path
    decisions: list[dict] = field(default_factory=list)
    orders: list[dict] = field(default_factory=list)
    fills: list[dict] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)

    def record_decision(
        self,
        signal: TradingSignal,
        action: TradingAction,
        portfolio: PortfolioState,
        trend_context: TrendContext | None = None,
        trend_signal: TrendSignal | None = None,
        crash_context: CrashContext | None = None,
        best_point_signal: BestPointSignal | None = None,
        slow_context: SlowTrendContext | None = None,
        segment_context: SegmentContext | None = None,
        trend_bias: TrendBiasContext | None = None,
        risk_budget: RiskBudget | None = None,
        decision_scope: str = "",
        blocked_reason: str = "",
        trend_entry_qualification: TrendEntryQualification | None = None,
    ) -> None:
        tc = trend_context
        ts = trend_signal
        cc = crash_context
        bs = best_point_signal
        sc = slow_context
        seg = segment_context
        tb = trend_bias
        rb = risk_budget
        teq = trend_entry_qualification
        self.decisions.append(
            {
                "ts": signal.ts,
                "price": signal.price,
                "atr": signal.atr,
                "p_up": signal.p_up,
                "p_down": signal.p_down,
                "p_flat": signal.p_flat,
                "p_risk": signal.p_risk,
                "edge": signal.edge,
                "conf": signal.conf,
                "pred_cum_ret_5": signal.pred_cum_ret_5,
                "action": action.action.value,
                "reason_code": action.reason_code,
                "blocked": int(action.action.value == "BLOCK"),
                "blocked_reason": blocked_reason,
                "portfolio_equity": portfolio.equity,
                "position_ratio": portfolio.position.position_ratio,
                "state": portfolio.position.side.value,
                "trend_is_downtrend": int(tc.is_downtrend) if tc else 0,
                "trend_is_strong_downtrend": int(tc.is_strong_downtrend) if tc else 0,
                "trend_score": tc.trend_score if tc else 0.0,
                "ret_3_atr": tc.ret_3_atr if tc else 0.0,
                "ret_6_atr": tc.ret_6_atr if tc else 0.0,
                "ema_fast": tc.ema_fast if tc else 0.0,
                "ema_slow": tc.ema_slow if tc else 0.0,
                "breakdown_low_n": int(tc.breakdown_low_n) if tc else 0,
                "trend_reason_codes": "|".join(tc.reason_codes) if tc else "",
                "fallback_action": action.reason_code
                if ("SENTINEL" in action.reason_code or "DOWNTREND" in action.reason_code or "TREND" in action.reason_code)
                else "",
                "reverse_confirm_count": portfolio.position.short_reverse_confirm_count,
                "hold_mode": portfolio.position.hold_mode,
                "trend_hold_bars": portfolio.position.trend_hold_bars,
                "trend_break_count": portfolio.position.trend_break_count,
                "entry_was_sentinel": int(portfolio.position.entry_was_sentinel),
                "sentinel_bars": portfolio.position.sentinel_bars,
                "peak_profit_atr": portfolio.position.peak_profit_atr,
                "is_crash": int(cc.is_crash) if cc else 0,
                "is_model_blind_crash": int(cc.is_model_blind_crash) if cc else 0,
                "crash_score": cc.crash_score if cc else 0.0,
                "crash_reason_codes": "|".join(cc.reason_codes) if cc else "",
                "drawdown_24h": cc.drawdown_24h if cc else 0.0,
                "ret_12_atr": cc.ret_12_atr if cc else 0.0,
                "range_expansion": cc.range_expansion if cc else 0.0,
                "entry_was_crash": int(portfolio.position.entry_was_crash),
                "crash_regime_id": portfolio.position.crash_regime_id,
                "trend_direction": ts.direction.value if ts else "NONE",
                "trend_strength": ts.strength.value if ts else "NONE",
                "trend_phase": ts.phase.value if ts else "NONE",
                "trend_score_up": ts.score_up if ts else 0.0,
                "trend_score_down": ts.score_down if ts else 0.0,
                "trend_confidence": ts.confidence if ts else 0.0,
                "trend_age": ts.trend_age if ts else 0,
                "trend_invalid_count": ts.invalid_count if ts else 0,
                "trend_is_confirmed": int(ts.is_confirmed) if ts else 0,
                "trend_is_broken": int(ts.is_broken) if ts else 0,
                "trend_is_accelerating": int(ts.is_accelerating) if ts else 0,
                "trend_is_exhausted": int(ts.is_exhausted) if ts else 0,
                "trend_signal_reason_codes": "|".join(ts.reason_codes) if ts else "",
                "bp_p_long_entry_zone": bs.p_long_entry_zone if bs else 0.0,
                "bp_p_short_entry_zone": bs.p_short_entry_zone if bs else 0.0,
                "bp_p_hold_long": bs.p_hold_long if bs else 0.0,
                "bp_p_hold_short": bs.p_hold_short if bs else 0.0,
                "bp_p_exit_long": bs.p_exit_long if bs else 0.0,
                "bp_p_exit_short": bs.p_exit_short if bs else 0.0,
                "bp_expected_opportunity_roi": bs.expected_opportunity_roi if bs else 0.0,
                "is_slow_uptrend": int(sc.is_slow_uptrend) if sc else 0,
                "is_stable_slow_uptrend": int(sc.is_stable_slow_uptrend) if sc else 0,
                "slow_up_score": sc.slow_up_score if sc else 0.0,
                "slope_24_atr": sc.slope_24_atr if sc else 0.0,
                "slope_48_atr": sc.slope_48_atr if sc else 0.0,
                "persistence_above_ema_fast": sc.persistence_above_ema_fast if sc else 0.0,
                "persistence_above_ema_mid": sc.persistence_above_ema_mid if sc else 0.0,
                "slow_up_exit_votes": portfolio.position.slow_up_exit_votes,
                "entry_was_slow_up": int(portfolio.position.entry_was_slow_up),
                "slow_up_reason_codes": "|".join(sc.reason_codes) if sc else "",
                "leg_id": seg.active_leg.leg_id if seg and seg.active_leg else -1,
                "leg_type": seg.leg_type.value if seg else "NONE",
                "leg_state": seg.leg_state.value if seg else "NO_LEG",
                "is_leg_confirmed": int(seg.active_leg.is_confirmed) if seg and seg.active_leg else 0,
                "sub_phase": seg.sub_phase.value if seg else "NONE",
                "bars_since_leg_start": seg.bars_since_leg_start if seg else 0,
                "leg_progress_ratio": seg.leg_progress_ratio if seg else 0.0,
                "market_regime": seg.regime.value if seg else "NEUTRAL",
                "should_hold_trend": int(seg.should_hold_trend) if seg else 0,
                "should_avoid_counter": int(seg.should_avoid_counter) if seg else 0,
                "segment_reason_codes": "|".join(seg.reason_codes) if seg else "",
                "decision_scope": decision_scope,
                "open_bias_long": tb.open_bias_long if tb else 1.0,
                "alignment_score_long": tb.alignment_score_long if tb else 0,
                "alignment_score_short": tb.alignment_score_short if tb else 0,
                "open_bias_short": tb.open_bias_short if tb else 1.0,
                "size_bias_long": tb.size_bias_long if tb else 1.0,
                "size_bias_short": tb.size_bias_short if tb else 1.0,
                "hold_bias_long": tb.hold_bias_long if tb else 1.0,
                "hold_bias_short": tb.hold_bias_short if tb else 1.0,
                "time_exit_permission_long": int(tb.time_exit_permission_long) if tb else 1,
                "time_exit_permission_short": int(tb.time_exit_permission_short) if tb else 1,
                "exit_bias_long": tb.exit_bias_long if tb else 1.0,
                "exit_bias_short": tb.exit_bias_short if tb else 1.0,
                "counter_level_long": tb.counter_level_long.value if tb else "NONE",
                "counter_level_short": tb.counter_level_short.value if tb else "NONE",
                "allow_open_long": int(tb.allow_open_long) if tb else 1,
                "allow_open_short": int(tb.allow_open_short) if tb else 1,
                "allow_add_long": int(tb.allow_add_long) if tb else 0,
                "allow_add_short": int(tb.allow_add_short) if tb else 0,
                "bias_reason_codes": "|".join(tb.reason_codes) if tb else "",
                "risk_budget_allow_open_long": int(rb.allow_open_long) if rb else 1,
                "risk_budget_allow_open_short": int(rb.allow_open_short) if rb else 1,
                "risk_budget_allow_add_long": int(rb.allow_add_long) if rb else 0,
                "risk_budget_allow_add_short": int(rb.allow_add_short) if rb else 0,
                "risk_budget_remaining_position_ratio": rb.remaining_position_ratio if rb else 0.0,
                "risk_budget_remaining_loss_budget_ratio": rb.remaining_loss_budget_ratio if rb else 1.0,
                "lifecycle": portfolio.position.lifecycle,
                "lifecycle_bars": portfolio.position.lifecycle_bars,
                "min_hold_until": portfolio.position.min_hold_until,
                "trend_exit_votes": portfolio.position.trend_exit_votes,
                "bp_exit_count": portfolio.position.best_point_exit_count,
                "runner_active": int(portfolio.position.runner_active),
                "profit_atr": (
                    ((portfolio.position.entry_price - signal.price) / max(signal.atr, 1e-12))
                    if portfolio.position.side.value == "SHORT"
                    else ((signal.price - portfolio.position.entry_price) / max(signal.atr, 1e-12))
                    if portfolio.position.side.value == "LONG"
                    else 0.0
                ),
                "teq_allow_long": int(teq.allow_trend_entry_long) if teq else 0,
                "teq_allow_short": int(teq.allow_trend_entry_short) if teq else 0,
                "teq_entry_tier": teq.entry_tier if teq else "NONE",
                "teq_relax_edge_mult": teq.relax_edge_mult if teq else 0.0,
                "teq_relax_prob_delta": teq.relax_prob_delta if teq else 0.0,
                "teq_reason_codes": "|".join(teq.reason_codes) if teq else "",
            }
        )

    def record_order(self, row: dict) -> None:
        self.orders.append(row)

    def record_fill(self, fill: FillEvent) -> None:
        self.fills.append(asdict(fill))

    def record_trade(self, row: dict) -> None:
        self.trades.append(row)

    def record_equity(self, ts, equity: float) -> None:
        self.equity_curve.append({"ts": ts, "equity": equity})

    def flush(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._write_csv(self.out_dir / "decisions.csv", self.decisions)
        self._write_csv(self.out_dir / "orders.csv", self.orders)
        self._write_csv(self.out_dir / "fills.csv", self.fills)
        self._write_csv(self.out_dir / "trades.csv", self.trades)
        self._write_csv(self.out_dir / "equity_curve.csv", self.equity_curve)

    @staticmethod
    def _write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        keys = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)

