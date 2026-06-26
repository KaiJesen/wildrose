from __future__ import annotations

import math
from dataclasses import dataclass, field

from trading_system.config import TradingSystemConfig
from trading_system.adapters.best_point_model import BestPointSignal
from trading_system.crash import CrashContext
from trading_system.enums import ActionType, Side
from trading_system.portfolio import PortfolioState
from trading_system.signal import TradingSignal
from trading_system.slow_trend import SlowTrendContext
from trading_system.trend import TrendContext
from trading_system.trend_signal import TrendDirection, TrendPhase, TrendSignal
from trading_system.trend_bias import CounterTrendLevel, RiskBudget, TrendBiasContext
from trading_system.trend_entry_qualifier import TrendEntryQualification
from trading_system.trend_segment import SegmentContext, SubLegPhase, TrendLegType


@dataclass
class TradingAction:
    action: ActionType
    target_side: Side
    reason_code: str
    blocked_by: str = ""
    diagnostics: dict = field(default_factory=dict)


class RuleEngine:
    _SCOPE_ORDER = {"observe": 0, "open_only": 1, "open_size": 2, "full": 3}
    _STD_TREND_LEG_TYPES = frozenset({
        TrendLegType.FAST_UP_LEG,
        TrendLegType.SLOW_UP_LEG,
        TrendLegType.FAST_DOWN_LEG,
        TrendLegType.SLOW_DOWN_LEG,
    })

    def __init__(self, cfg: TradingSystemConfig) -> None:
        self.cfg = cfg

    def _bias_scope_at_least(self, scope: str) -> bool:
        tb = self.cfg.trend_bias
        if not tb.enabled:
            return False
        return self._SCOPE_ORDER.get(tb.decision_scope, 0) >= self._SCOPE_ORDER.get(scope, 0)

    def _legacy_segment_counter_blocks(self) -> bool:
        tb = self.cfg.trend_bias
        if tb.enabled and tb.disable_legacy_trend_rules and self._bias_scope_at_least("open_only"):
            return False
        return True

    def _regime_threshold_relax(
        self,
        *,
        side: str,
        trend_signal: TrendSignal | None = None,
        slow_context: SlowTrendContext | None = None,
        crash_context: CrashContext | None = None,
        for_trend_qualified: bool = False,
    ) -> tuple[float, float]:
        rt = self.cfg.regime_threshold
        if not rt.enabled:
            return 1.0, 0.0
        if for_trend_qualified and not rt.apply_to_trend_qualified:
            return 1.0, 0.0
        if not for_trend_qualified and not rt.apply_to_standard_opens:
            return 1.0, 0.0
        if side == "short" and crash_context and crash_context.is_model_blind_crash:
            return rt.crash_edge_mult, rt.crash_prob_delta
        if side == "long" and slow_context and slow_context.is_stable_slow_uptrend:
            return rt.slow_up_edge_mult, rt.slow_up_prob_delta
        if trend_signal and trend_signal.is_confirmed:
            if side == "long" and trend_signal.direction == TrendDirection.UP:
                return rt.trend_confirmed_edge_mult, rt.trend_confirmed_prob_delta
            if side == "short" and trend_signal.direction == TrendDirection.DOWN:
                return rt.trend_confirmed_edge_mult, rt.trend_confirmed_prob_delta
        return 1.0, 0.0

    def _effective_open_thresholds(
        self,
        trend_bias: TrendBiasContext | None,
        *,
        side: str,
        trend_signal: TrendSignal | None = None,
        slow_context: SlowTrendContext | None = None,
        crash_context: CrashContext | None = None,
    ) -> tuple[float, float]:
        r = self.cfg.rule
        if not self._bias_scope_at_least("open_only") or trend_bias is None:
            if side == "long":
                return r.open_edge_threshold, r.open_prob_threshold
            return r.open_edge_threshold, r.open_prob_threshold
        if side == "long":
            bias = max(trend_bias.open_bias_long, 1e-6)
            counter = trend_bias.counter_level_long
            align = trend_bias.alignment_score_long
        else:
            bias = max(trend_bias.open_bias_short, 1e-6)
            counter = trend_bias.counter_level_short
            align = trend_bias.alignment_score_short
        if bias > 1.0 and align < 1:
            bias = 1.0
        eff_edge = r.open_edge_threshold / bias
        eff_prob = r.open_prob_threshold / bias
        if counter == CounterTrendLevel.LIGHT:
            eff_prob = r.open_prob_threshold
            if align < 1:
                eff_edge = r.open_edge_threshold
        edge_mult, prob_delta = self._regime_threshold_relax(
            side=side,
            trend_signal=trend_signal,
            slow_context=slow_context,
            crash_context=crash_context,
        )
        eff_edge *= edge_mult
        eff_prob = max(0.0, eff_prob + prob_delta)
        return eff_edge, eff_prob

    def _trend_qualified_thresholds(
        self,
        trend_bias: TrendBiasContext | None,
        qualification: TrendEntryQualification,
        *,
        side: str,
        trend_signal: TrendSignal | None = None,
        slow_context: SlowTrendContext | None = None,
        crash_context: CrashContext | None = None,
    ) -> tuple[float, float]:
        qcfg = self.cfg.trend_entry_qualifier
        if side == "long":
            base_edge = qcfg.min_edge_long
            base_prob = qcfg.min_prob_long
            if self._bias_scope_at_least("open_only") and trend_bias is not None:
                bias = max(trend_bias.open_bias_long, 1e-6) * qcfg.open_bias_penalty
                if bias > 1.0 and trend_bias.alignment_score_long < 1:
                    bias = 1.0
            else:
                bias = 1.0
        else:
            base_edge = abs(qcfg.min_edge_short)
            base_prob = qcfg.min_prob_short
            if self._bias_scope_at_least("open_only") and trend_bias is not None:
                bias = max(trend_bias.open_bias_short, 1e-6) * qcfg.open_bias_penalty
                if bias > 1.0 and trend_bias.alignment_score_short < 1:
                    bias = 1.0
            else:
                bias = 1.0
        edge_mult, prob_delta = self._regime_threshold_relax(
            side=side,
            trend_signal=trend_signal,
            slow_context=slow_context,
            crash_context=crash_context,
            for_trend_qualified=True,
        )
        eff_edge = base_edge * qualification.relax_edge_mult * edge_mult / bias
        eff_prob = max(0.0, base_prob + qualification.relax_prob_delta + prob_delta) / bias
        return eff_edge, eff_prob

    def _try_trend_qualified_open(
        self,
        signal: TradingSignal,
        trend_bias: TrendBiasContext | None,
        risk_budget: RiskBudget | None,
        qualification: TrendEntryQualification | None,
        best_point_signal: BestPointSignal | None,
        *,
        side: str,
        trend_signal: TrendSignal | None = None,
        slow_context: SlowTrendContext | None = None,
        crash_context: CrashContext | None = None,
    ) -> TradingAction | None:
        qcfg = self.cfg.trend_entry_qualifier
        if not qcfg.enabled or qualification is None:
            return None
        if side == "long":
            if not qualification.allow_trend_entry_long:
                return None
            eff_edge, eff_prob = self._trend_qualified_thresholds(
                trend_bias, qualification, side="long",
                trend_signal=trend_signal, slow_context=slow_context, crash_context=crash_context,
            )
            teq_edge = signal.teq_edge_long if self.cfg.teq_edge.enabled else signal.edge
            if not (
                teq_edge >= eff_edge
                and signal.p_up >= eff_prob
                and signal.p_flat <= self.cfg.rule.open_flat_max
                and signal.risk_ok
            ):
                return None
            bias_block = self._bias_open_gate(trend_bias, risk_budget, side="long")
            if bias_block is not None:
                return bias_block
            if qcfg.require_best_point and not self._bp_entry_ok(best_point_signal, side="long"):
                return TradingAction(ActionType.BLOCK, Side.FLAT, "BLOCK_BP_TREND_QUAL_LONG", blocked_by="best_point")
            return TradingAction(
                ActionType.OPEN_LONG,
                Side.LONG,
                "OPEN_LONG_TREND_QUALIFIED",
                diagnostics={
                    "entry_tier": qualification.entry_tier,
                    "teq_reasons": qualification.reason_codes,
                    "edge_source": "teq",
                    "channel_threshold_snapshot": self.cfg.teq_edge.weight_participation,
                },
            )
        if not qualification.allow_trend_entry_short:
            return None
        eff_edge, eff_prob = self._trend_qualified_thresholds(
            trend_bias, qualification, side="short",
            trend_signal=trend_signal, slow_context=slow_context, crash_context=crash_context,
        )
        teq_edge = signal.teq_edge_short if self.cfg.teq_edge.enabled else signal.edge
        if not (
            teq_edge <= -eff_edge
            and signal.p_down >= eff_prob
            and signal.p_flat <= self.cfg.rule.open_flat_max
            and signal.risk_ok
        ):
            return None
        bias_block = self._bias_open_gate(trend_bias, risk_budget, side="short")
        if bias_block is not None:
            return bias_block
        if qcfg.require_best_point and not self._bp_entry_ok(best_point_signal, side="short"):
            return TradingAction(ActionType.BLOCK, Side.FLAT, "BLOCK_BP_TREND_QUAL_SHORT", blocked_by="best_point")
        return TradingAction(
            ActionType.OPEN_SHORT,
            Side.SHORT,
            "OPEN_SHORT_TREND_QUALIFIED",
            diagnostics={
                "entry_tier": qualification.entry_tier,
                "teq_reasons": qualification.reason_codes,
                "edge_source": "teq",
                "channel_threshold_snapshot": self.cfg.teq_edge.weight_participation,
            },
        )

    def _bias_open_gate(
        self,
        trend_bias: TrendBiasContext | None,
        risk_budget: RiskBudget | None,
        *,
        side: str,
    ) -> TradingAction | None:
        if not self._bias_scope_at_least("open_only") or trend_bias is None:
            return None
        if side == "long":
            if not trend_bias.allow_open_long:
                return TradingAction(ActionType.BLOCK, Side.FLAT, "BLOCK_BIAS_LONG_OPEN", blocked_by="trend_bias")
            if (
                self._bias_scope_at_least("open_size")
                and risk_budget is not None
                and not risk_budget.allow_open_long
            ):
                reason = risk_budget.open_reject_reason_long or "BLOCK_RISK_BUDGET_LONG"
                return TradingAction(ActionType.BLOCK, Side.FLAT, reason, blocked_by="risk_budget")
            if (
                self.cfg.trend_bias.disable_legacy_trend_rules
                and trend_bias.counter_level_long == CounterTrendLevel.HARD_BLOCK
                and not self.cfg.trend_bias.allow_hard_counter_probe
            ):
                return TradingAction(
                    ActionType.BLOCK, Side.FLAT, "BLOCK_BIAS_HARD_COUNTER_LONG", blocked_by="trend_bias"
                )
        elif side == "short":
            if not trend_bias.allow_open_short:
                return TradingAction(ActionType.BLOCK, Side.FLAT, "BLOCK_BIAS_SHORT_OPEN", blocked_by="trend_bias")
            if (
                self._bias_scope_at_least("open_size")
                and risk_budget is not None
                and not risk_budget.allow_open_short
            ):
                reason = risk_budget.open_reject_reason_short or "BLOCK_RISK_BUDGET_SHORT"
                return TradingAction(ActionType.BLOCK, Side.FLAT, reason, blocked_by="risk_budget")
            if (
                self.cfg.trend_bias.disable_legacy_trend_rules
                and trend_bias.counter_level_short == CounterTrendLevel.HARD_BLOCK
                and not self.cfg.trend_bias.allow_hard_counter_probe
            ):
                return TradingAction(
                    ActionType.BLOCK, Side.FLAT, "BLOCK_BIAS_HARD_COUNTER_SHORT", blocked_by="trend_bias"
                )
        return None

    def _trend_exit_vote_threshold(
        self,
        trend_bias: TrendBiasContext | None,
        *,
        side: Side,
    ) -> int:
        base = self.cfg.trend_lifecycle.exit_confirm_votes
        if not self._bias_scope_at_least("full") or trend_bias is None:
            return base
        if side == Side.LONG:
            return max(1, math.ceil(base * trend_bias.exit_bias_long))
        return max(1, math.ceil(base * trend_bias.exit_bias_short))

    def _is_chop_hard(self, trend_signal: TrendSignal | None) -> bool:
        if trend_signal is None:
            return False
        return "CHOP_HARD" in trend_signal.reason_codes

    def _use_std_trend_edge(
        self,
        *,
        segment_context: SegmentContext | None,
        trend_signal: TrendSignal | None,
        side: str,
    ) -> bool:
        ch = self.cfg.participation_channel
        if not ch.enabled or not ch.std_trend_gate.enabled:
            return False
        if not self.cfg.trend_segment.enabled or segment_context is None:
            return False
        leg = segment_context.active_leg
        if leg is None or not leg.is_confirmed:
            return False
        if segment_context.leg_type not in self._STD_TREND_LEG_TYPES:
            return False
        if self._is_chop_hard(trend_signal):
            return False
        if side == "long" and leg.direction != TrendDirection.UP:
            return False
        if side == "short" and leg.direction != TrendDirection.DOWN:
            return False
        return True

    def _standard_long_edge(self, signal: TradingSignal, *, use_std_trend: bool) -> float:
        if use_std_trend:
            return signal.std_trend_edge_long
        return signal.edge

    def _standard_short_edge(self, signal: TradingSignal, *, use_std_trend: bool) -> float:
        if use_std_trend:
            return signal.std_trend_edge_short
        return signal.edge

    def _std_trend_open_diag(self) -> dict:
        gate = self.cfg.participation_channel.std_trend_gate
        return {
            "edge_source": "std_trend",
            "channel_threshold_snapshot": gate.weight_part,
        }

    def _is_standard_short_entry(
        self,
        signal: TradingSignal,
        trend_bias: TrendBiasContext | None = None,
        *,
        trend_signal: TrendSignal | None = None,
        slow_context: SlowTrendContext | None = None,
        crash_context: CrashContext | None = None,
        segment_context: SegmentContext | None = None,
        for_flat_open: bool = False,
    ) -> bool:
        r = self.cfg.rule
        eff_edge_th, eff_prob_th = self._effective_open_thresholds(
            trend_bias,
            side="short",
            trend_signal=trend_signal,
            slow_context=slow_context,
            crash_context=crash_context,
        )
        use_std = for_flat_open and self._use_std_trend_edge(
            segment_context=segment_context,
            trend_signal=trend_signal,
            side="short",
        )
        edge = self._standard_short_edge(signal, use_std_trend=use_std)
        return (
            edge <= -eff_edge_th
            and signal.p_down >= eff_prob_th
            and signal.p_flat <= r.open_flat_max
            and signal.pred_cum_ret_5 < 0
            and signal.risk_ok
        )

    def _is_sentinel_short_entry(self, signal: TradingSignal, tc: TrendContext | None) -> bool:
        if tc is None or not tc.is_strong_downtrend:
            return False
        s = self.cfg.sentinel_short
        if tc.ret_6_atr > s.sentinel_ret6_atr_threshold:
            return False
        if not (signal.price < tc.ema_fast < tc.ema_slow):
            return False
        return (
            signal.p_risk <= s.sentinel_risk_max
            and signal.p_flat <= s.sentinel_flat_max
            and signal.edge <= 0.0
            and signal.pred_cum_ret_5 <= 0.0
        )

    def _slow_up_open_diagnostics(self, sc: SlowTrendContext | None, *, watch_probe: bool = False, extra: dict | None = None) -> dict:
        gate = self.cfg.participation_channel.slow_up_gate
        diag: dict = {
            "stable_slow_up": True,
            "slow_up_score": sc.slow_up_score if sc else 0.0,
            "edge_source": "slow_up",
            "channel_threshold_snapshot": gate.tau_slow if gate.enabled else 0.0,
        }
        if extra:
            diag.update(extra)
        return diag

    def _model_opposes_slow_up(self, signal: TradingSignal) -> bool:
        sp = self.cfg.slow_up_position
        return (
            signal.pred_cum_ret_5 < sp.model_opp_cum_ret_min
            or signal.edge <= sp.model_opp_edge_min
            or signal.p_risk >= self.cfg.rule.risk_exit_threshold
        )

    def _can_open_slow_up(
        self,
        signal: TradingSignal,
        portfolio: PortfolioState,
        crash_context: CrashContext | None,
        trend_signal: TrendSignal | None,
        trend_context: TrendContext | None,
        slow_context: SlowTrendContext | None,
        segment_context: SegmentContext | None = None,
        *,
        watch_probe: bool = False,
    ) -> bool:
        sp = self.cfg.slow_up_position
        if slow_context is None or not sp.enabled or not self.cfg.slow_uptrend.enabled:
            return False
        if not slow_context.is_stable_slow_uptrend:
            return False
        if watch_probe and portfolio.slow_up_watch_streak < sp.watch_min_bars:
            return False
        if signal.p_risk > sp.risk_max:
            return False
        if portfolio.daily_open_block or portfolio.account_circuit_breaker:
            return False
        if crash_context and crash_context.is_crash:
            return False
        if trend_context and trend_context.is_downtrend:
            return False
        if trend_signal and trend_signal.direction == TrendDirection.DOWN:
            return False
        if trend_signal and trend_signal.phase == TrendPhase.REVERSAL_RISK:
            return False
        if not watch_probe and trend_signal and trend_signal.direction != TrendDirection.UP:
            return False
        if self._model_opposes_slow_up(signal):
            return False
        gate = self.cfg.participation_channel.slow_up_gate
        if self.cfg.participation_channel.enabled and gate.enabled:
            edge_ok = (
                gate.edge_threshold_slow != 0.0
                and signal.slow_up_edge_long >= gate.edge_threshold_slow
            )
            part_ok = signal.participate_score_long >= gate.tau_slow
            if not (edge_ok or part_ok):
                return False
        if not watch_probe and signal.pred_cum_ret_5 < 0.0:
            return False
        if watch_probe:
            if trend_signal and trend_signal.direction != TrendDirection.UP:
                return False
            if signal.pred_cum_ret_5 < sp.watch_probe_min_cum_ret:
                return False
            if signal.p_up < sp.watch_probe_min_p_up:
                return False
            if sp.watch_probe_require_up_leg and self.cfg.trend_segment.enabled and segment_context is not None:
                if segment_context.leg_type not in (
                    TrendLegType.SLOW_UP_LEG,
                    TrendLegType.FAST_UP_LEG,
                ):
                    return False
                if segment_context.bars_since_leg_start < sp.segment_min_bars:
                    return False
            return True
        if self.cfg.trend_segment.enabled and segment_context is not None:
            leg = segment_context.active_leg
            if leg is None or not leg.is_confirmed:
                return False
            if segment_context.leg_type != TrendLegType.SLOW_UP_LEG:
                return False
            min_bars = sp.segment_min_bars
            if segment_context.bars_since_leg_start < min_bars:
                return False
        return True

    def _segment_allows_upgrade(self, segment_context: SegmentContext | None, side: Side) -> bool:
        if not self.cfg.trend_segment.enabled or segment_context is None:
            return True
        leg = segment_context.active_leg
        if leg is None or not leg.is_confirmed:
            return False
        if segment_context.bars_since_leg_start < self.cfg.trend_segment.upgrade_min_bars:
            return False
        tradable = {
            TrendLegType.SLOW_UP_LEG,
            TrendLegType.FAST_UP_LEG,
            TrendLegType.SLOW_DOWN_LEG,
            TrendLegType.FAST_DOWN_LEG,
            TrendLegType.CRASH_LEG,
        }
        if segment_context.leg_type not in tradable:
            return False
        if side == Side.LONG and leg.direction != TrendDirection.UP:
            return False
        if side == Side.SHORT and leg.direction != TrendDirection.DOWN:
            return False
        return True

    def _crash_segment_allows_upgrade(self, segment_context: SegmentContext | None, side: Side) -> bool:
        if not self.cfg.trend_segment.enabled or segment_context is None:
            return True
        leg = segment_context.active_leg
        if leg is None:
            return False
        if segment_context.leg_type not in (
            TrendLegType.FAST_DOWN_LEG,
            TrendLegType.SLOW_DOWN_LEG,
            TrendLegType.CRASH_LEG,
        ):
            return False
        if side == Side.SHORT and leg.direction != TrendDirection.DOWN:
            return False
        min_bars = min(self.cfg.trend_position.min_trend_age_for_upgrade, self.cfg.trend_segment.upgrade_min_bars)
        return segment_context.bars_since_leg_start >= min_bars

    def _bp_exit_vote_weight(self, segment_context: SegmentContext | None) -> int:
        if not self.cfg.trend_segment.enabled or segment_context is None:
            return 1
        if not self.cfg.trend_segment.exit_vote_requires_exhaustion:
            return 1
        if segment_context.sub_phase in (SubLegPhase.EXHAUSTION, SubLegPhase.LEG_END):
            return 1
        if segment_context.leg_progress_ratio > 0.6:
            return 1
        return 0

    def _bp_entry_ok(self, best_point_signal: BestPointSignal | None, *, side: str, crash: bool = False) -> bool:
        bp = self.cfg.best_point
        if not bp.enabled or bp.observe_only:
            return True
        if best_point_signal is None:
            return False
        if best_point_signal.expected_opportunity_roi < bp.min_opportunity_roi:
            return False
        if side == "long":
            return best_point_signal.p_long_entry_zone >= bp.long_entry_confirm_threshold
        if crash and bp.require_entry_confirm_for_crash:
            return best_point_signal.p_short_entry_zone >= bp.crash_short_entry_confirm_threshold
        return best_point_signal.p_short_entry_zone >= bp.short_entry_confirm_threshold

    def _bp_exit_triggered(self, best_point_signal: BestPointSignal | None, *, side: Side) -> bool:
        bp = self.cfg.best_point
        if best_point_signal is None:
            return False
        exit_th = bp.exit_prob_threshold if bp.enabled else 0.70
        hold_min = bp.hold_min_prob if bp.enabled else 0.30
        if side == Side.SHORT:
            return best_point_signal.p_exit_short >= exit_th or best_point_signal.p_hold_short <= hold_min
        return best_point_signal.p_exit_long >= exit_th or best_point_signal.p_hold_long <= hold_min

    def _slow_up_profit_atr(self, signal: TradingSignal, pos) -> float:
        return (signal.price - pos.entry_price) / max(signal.atr, 1e-12)

    def _slow_up_exit_votes(
        self,
        signal: TradingSignal,
        pos,
        slow_context: SlowTrendContext,
        best_point_signal: BestPointSignal | None,
    ) -> int:
        votes = 0
        if signal.price < slow_context.ema_mid:
            pos.slow_up_below_ema_mid_count += 1
        else:
            pos.slow_up_below_ema_mid_count = 0
        if pos.slow_up_below_ema_mid_count >= 2:
            votes += 2
        if slow_context.slope_24_atr < 0:
            votes += 1
        if slow_context.ret_6_atr <= -1.2:
            votes += 1
        if slow_context.max_drawdown_24_atr > self.cfg.slow_uptrend.exit_drawdown_24_atr:
            votes += 1
        if signal.p_risk >= self.cfg.rule.risk_exit_threshold:
            votes += 2
        if signal.pred_cum_ret_5 < self.cfg.slow_up_position.model_opp_cum_ret_min:
            pos.slow_up_weak_model_count += 1
        else:
            pos.slow_up_weak_model_count = 0
        if pos.slow_up_weak_model_count >= 2:
            votes += 1
        if best_point_signal is not None and best_point_signal.p_exit_long >= self.cfg.best_point.exit_prob_threshold:
            votes += 1
        return votes

    def _manage_slow_up_long(
        self,
        signal: TradingSignal,
        pos,
        bar_index: int,
        slow_context: SlowTrendContext | None,
        trend_signal: TrendSignal | None,
        best_point_signal: BestPointSignal | None,
    ) -> TradingAction | None:
        if not pos.entry_was_slow_up or slow_context is None:
            return None
        sp = self.cfg.slow_up_position
        su = self.cfg.slow_uptrend
        profit_atr = self._slow_up_profit_atr(signal, pos)

        if (
            sp.allow_trend_upgrade
            and pos.hold_mode != "TREND"
            and profit_atr >= sp.upgrade_profit_atr
            and slow_context.is_slow_uptrend
            and signal.p_risk < self.cfg.rule.risk_exit_threshold
            and (
                (trend_signal is not None and trend_signal.direction == TrendDirection.UP)
                or slow_context.is_stable_slow_uptrend
            )
        ):
            return TradingAction(ActionType.HOLD, Side.LONG, "UPGRADE_SLOW_LONG_TO_TREND")

        if profit_atr >= sp.runner_profit_atr and not pos.runner_active:
            return TradingAction(ActionType.REDUCE, Side.LONG, "REDUCE_SLOW_UP_PROFIT_LOCK")
        if pos.runner_active:
            votes = self._slow_up_exit_votes(signal, pos, slow_context, best_point_signal)
            pos.slow_up_exit_votes = votes
            if votes >= sp.exit_votes:
                return TradingAction(ActionType.CLOSE, Side.FLAT, "CLOSE_SLOW_UPTREND_BROKEN")
            return TradingAction(ActionType.HOLD, Side.LONG, "HOLD_SLOW_UP_RUNNER")

        if (
            slow_context.is_slow_uptrend
            and slow_context.max_drawdown_24_atr <= su.exit_drawdown_24_atr
            and signal.p_risk < self.cfg.rule.risk_exit_threshold
        ):
            if profit_atr >= 3.0:
                votes = self._slow_up_exit_votes(signal, pos, slow_context, best_point_signal)
                pos.slow_up_exit_votes = votes
                if bar_index >= pos.min_hold_until and votes >= sp.exit_votes:
                    return TradingAction(ActionType.CLOSE, Side.FLAT, "CLOSE_SLOW_UPTREND_BROKEN")
                return TradingAction(ActionType.HOLD, Side.LONG, "HOLD_SLOW_UPTREND")
            if bar_index < pos.min_hold_until:
                return TradingAction(ActionType.HOLD, Side.LONG, "HOLD_SLOW_UPTREND")
            votes = self._slow_up_exit_votes(signal, pos, slow_context, best_point_signal)
            pos.slow_up_exit_votes = votes
            if votes >= sp.exit_votes:
                return TradingAction(ActionType.CLOSE, Side.FLAT, "CLOSE_SLOW_UPTREND_BROKEN")
            return TradingAction(ActionType.HOLD, Side.LONG, "HOLD_SLOW_UPTREND")

        votes = self._slow_up_exit_votes(signal, pos, slow_context, best_point_signal)
        pos.slow_up_exit_votes = votes
        if bar_index >= pos.min_hold_until and votes >= sp.exit_votes:
            return TradingAction(ActionType.CLOSE, Side.FLAT, "CLOSE_SLOW_UPTREND_BROKEN")
        return None

    def decide(
        self,
        signal: TradingSignal,
        portfolio: PortfolioState,
        bar_index: int = 0,
        trend_context: TrendContext | None = None,
        crash_context: CrashContext | None = None,
        trend_signal: TrendSignal | None = None,
        best_point_signal: BestPointSignal | None = None,
        slow_context: SlowTrendContext | None = None,
        segment_context: SegmentContext | None = None,
        trend_bias: TrendBiasContext | None = None,
        risk_budget: RiskBudget | None = None,
        trend_entry_qualification: TrendEntryQualification | None = None,
    ) -> TradingAction:
        pos = portfolio.position
        r = self.cfg.rule
        protection = self.cfg.protection
        tc = trend_context
        seg = segment_context
        if pos.is_flat:
            eff_long_edge, eff_long_prob = self._effective_open_thresholds(
                trend_bias,
                side="long",
                trend_signal=trend_signal,
                slow_context=slow_context,
                crash_context=crash_context,
            )
            use_std_long = self._use_std_trend_edge(
                segment_context=seg,
                trend_signal=trend_signal,
                side="long",
            )
            long_edge = self._standard_long_edge(signal, use_std_trend=use_std_long)
            if (
                self._legacy_segment_counter_blocks()
                and seg
                and self.cfg.trend_segment.enabled
                and seg.should_avoid_counter
                and long_edge >= eff_long_edge
                and signal.p_up >= eff_long_prob
                and seg.active_leg
                and seg.active_leg.direction == TrendDirection.DOWN
            ):
                return TradingAction(ActionType.BLOCK, Side.FLAT, "BLOCK_COUNTER_TREND_LONG", blocked_by="segment")
            if (
                long_edge >= eff_long_edge
                and signal.p_up >= eff_long_prob
                and signal.p_flat <= r.open_flat_max
                and signal.pred_cum_ret_5 > 0
                and signal.risk_ok
            ):
                bias_block = self._bias_open_gate(trend_bias, risk_budget, side="long")
                if bias_block is not None:
                    return bias_block
                if not self._bp_entry_ok(best_point_signal, side="long"):
                    return TradingAction(ActionType.BLOCK, Side.FLAT, "BLOCK_BP_LONG_ENTRY", blocked_by="best_point")
                return TradingAction(
                    ActionType.OPEN_LONG,
                    Side.LONG,
                    "OPEN_LONG_SIGNAL",
                    diagnostics=self._std_trend_open_diag() if use_std_long else {},
                )
            if self._is_standard_short_entry(
                signal,
                trend_bias,
                trend_signal=trend_signal,
                slow_context=slow_context,
                crash_context=crash_context,
                segment_context=seg,
                for_flat_open=True,
            ):
                if (
                    self._legacy_segment_counter_blocks()
                    and seg
                    and self.cfg.trend_segment.enabled
                    and seg.should_avoid_counter
                    and seg.active_leg
                    and seg.active_leg.direction == TrendDirection.UP
                    and seg.active_leg.is_confirmed
                ):
                    return TradingAction(ActionType.BLOCK, Side.FLAT, "BLOCK_COUNTER_TREND_SHORT", blocked_by="segment")
                bias_block = self._bias_open_gate(trend_bias, risk_budget, side="short")
                if bias_block is not None:
                    return bias_block
                if not self._bp_entry_ok(best_point_signal, side="short"):
                    return TradingAction(ActionType.BLOCK, Side.FLAT, "BLOCK_BP_SHORT_ENTRY", blocked_by="best_point")
                use_std_short = self._use_std_trend_edge(
                    segment_context=seg,
                    trend_signal=trend_signal,
                    side="short",
                )
                return TradingAction(
                    ActionType.OPEN_SHORT,
                    Side.SHORT,
                    "OPEN_SHORT_SIGNAL",
                    diagnostics=self._std_trend_open_diag() if use_std_short else {},
                )
            cc = crash_context
            if (
                cc
                and self.cfg.crash_short.enabled
                and cc.is_model_blind_crash
                and signal.p_risk <= self.cfg.crash_short.risk_max
                and signal.p_flat <= self.cfg.crash_short.flat_max
            ):
                bias_block = self._bias_open_gate(trend_bias, risk_budget, side="short")
                if bias_block is not None:
                    return bias_block
                if not self._bp_entry_ok(best_point_signal, side="short", crash=True):
                    return TradingAction(ActionType.BLOCK, Side.FLAT, "BLOCK_BP_CRASH_ENTRY", blocked_by="best_point")
                return TradingAction(
                    ActionType.OPEN_SHORT,
                    Side.SHORT,
                    "OPEN_SHORT_CRASH",
                    diagnostics={"is_crash_short": True, "strong_crash": cc.strong_crash},
                )
            teq_long = self._try_trend_qualified_open(
                signal,
                trend_bias,
                risk_budget,
                trend_entry_qualification,
                best_point_signal,
                side="long",
                trend_signal=trend_signal,
                slow_context=slow_context,
                crash_context=crash_context,
            )
            if teq_long is not None:
                return teq_long
            teq_short = self._try_trend_qualified_open(
                signal,
                trend_bias,
                risk_budget,
                trend_entry_qualification,
                best_point_signal,
                side="short",
                trend_signal=trend_signal,
                slow_context=slow_context,
                crash_context=crash_context,
            )
            if teq_short is not None:
                return teq_short
            if protection.allow_sentinel_short and self.cfg.sentinel_short.enabled and self._is_sentinel_short_entry(signal, tc):
                return TradingAction(
                    ActionType.OPEN_SHORT,
                    Side.SHORT,
                    "OPEN_SHORT_SENTINEL",
                    diagnostics={"is_sentinel_short": True},
                )
            sc = slow_context
            standard_long = (
                signal.edge >= eff_long_edge
                and signal.p_up >= eff_long_prob
                and signal.p_flat <= r.open_flat_max
                and signal.pred_cum_ret_5 > 0
                and signal.risk_ok
            )
            if self._can_open_slow_up(signal, portfolio, crash_context, trend_signal, tc, sc, seg):
                portfolio.slow_up_watch_streak = 0
                bias_block = self._bias_open_gate(trend_bias, risk_budget, side="long")
                if bias_block is not None:
                    return bias_block
                return TradingAction(
                    ActionType.OPEN_LONG,
                    Side.LONG,
                    "OPEN_LONG_SLOW_TREND",
                    diagnostics=self._slow_up_open_diagnostics(sc),
                )
            if sc and sc.is_slow_uptrend and not standard_long:
                if sc.is_stable_slow_uptrend:
                    portfolio.slow_up_watch_streak += 1
                else:
                    portfolio.slow_up_watch_streak = 0
                if (
                    portfolio.slow_up_watch_streak >= self.cfg.slow_up_position.watch_min_bars
                    and self._can_open_slow_up(
                        signal, portfolio, crash_context, trend_signal, tc, sc, seg, watch_probe=True
                    )
                ):
                    watch_streak = portfolio.slow_up_watch_streak
                    portfolio.slow_up_watch_streak = 0
                    bias_block = self._bias_open_gate(trend_bias, risk_budget, side="long")
                    if bias_block is not None:
                        return bias_block
                    return TradingAction(
                        ActionType.OPEN_LONG,
                        Side.LONG,
                        "OPEN_LONG_SLOW_TREND",
                        diagnostics=self._slow_up_open_diagnostics(
                            sc,
                            watch_probe=True,
                            extra={"watch_probe": True, "watch_streak": watch_streak},
                        ),
                    )
                return TradingAction(
                    ActionType.HOLD,
                    Side.FLAT,
                    "WATCH_SLOW_UPTREND",
                    diagnostics={
                        "slow_up_score": sc.slow_up_score,
                        "leg_type": seg.leg_type.value if seg else "NONE",
                        "watch_streak": portfolio.slow_up_watch_streak,
                    },
                )
            portfolio.slow_up_watch_streak = 0
            return TradingAction(ActionType.HOLD, Side.FLAT, "HOLD_NO_ENTRY")

        side = pos.side
        ts = trend_signal
        if side == Side.LONG and pos.entry_was_slow_up:
            slow_action = self._manage_slow_up_long(signal, pos, bar_index, slow_context, ts, best_point_signal)
            if slow_action is not None:
                return slow_action
        if not pos.is_flat and ts is not None:
            # Upgrade profitable position into trend-hold mode.
            profit_atr = (
                (signal.price - pos.entry_price) / max(signal.atr, 1e-12)
                if side == Side.LONG
                else (pos.entry_price - signal.price) / max(signal.atr, 1e-12)
            )
            upgrade_profit_atr = (
                self.cfg.trend_position.crash_upgrade_profit_atr
                if pos.hold_mode == "CRASH" and self.cfg.trend_position.allow_crash_trend_upgrade
                else self.cfg.trend_position.upgrade_profit_atr
            )
            can_upgrade = pos.hold_mode != "TREND"
            if pos.hold_mode == "CRASH" and not self.cfg.trend_position.allow_crash_trend_upgrade:
                can_upgrade = False
            if (
                can_upgrade
                and profit_atr >= upgrade_profit_atr
                and self._segment_allows_upgrade(seg, side)
                and ts.is_confirmed
                and ts.trend_age >= self.cfg.trend_position.min_trend_age_for_upgrade
                and ts.phase in (TrendPhase.CONTINUATION, TrendPhase.ACCELERATION)
                and signal.p_risk < self.cfg.rule.risk_exit_threshold
            ):
                if side == Side.LONG and ts.direction == TrendDirection.UP:
                    return TradingAction(ActionType.HOLD, Side.LONG, "UPGRADE_TO_TREND_LONG")
                if side == Side.SHORT and ts.direction == TrendDirection.DOWN:
                    return TradingAction(ActionType.HOLD, Side.SHORT, "UPGRADE_TO_TREND_SHORT")
            if pos.hold_mode == "TREND":
                if seg and seg.sub_phase == SubLegPhase.LEG_END:
                    return TradingAction(ActionType.CLOSE, Side.FLAT, "CLOSE_TREND_LEG_END")
                if seg and seg.sub_phase == SubLegPhase.EXHAUSTION and not pos.exhaustion_reduce_done:
                    return TradingAction(ActionType.REDUCE, side, "REDUCE_TREND_EXHAUSTION")
                if seg and seg.should_hold_trend and seg.sub_phase == SubLegPhase.PULLBACK:
                    return TradingAction(ActionType.HOLD, side, "HOLD_TREND_PULLBACK")
                # Lifecycle: TREND -> PROTECT_PROFIT -> RUNNER
                if profit_atr >= self.cfg.trend_lifecycle.runner_profit_atr and not pos.runner_active:
                    return TradingAction(ActionType.REDUCE, side, "REDUCE_TREND_PROFIT_LOCK")
                score_drop = max(0.0, pos.trend_peak_score - ts.score_abs)
                if not pos.exhaustion_reduce_done and (
                    ts.phase == TrendPhase.EXHAUSTION or score_drop >= 2.0
                ):
                    return TradingAction(ActionType.REDUCE, side, "REDUCE_TREND_EXHAUSTION")
                if pos.runner_active:
                    return TradingAction(ActionType.HOLD, side, "HOLD_TREND_RUNNER")
                add_ok = (
                    ts.phase == TrendPhase.ACCELERATION
                    and pos.add_count < self.cfg.base.max_add_count
                    and pos.position_ratio < self.cfg.base.max_position_ratio
                    and signal.p_risk < self.cfg.rule.risk_open_max
                    and profit_atr >= self.cfg.trend_position.add_profit_atr
                )
                ext = self.cfg.trend_hold_extension
                if add_ok and ext.enabled and seg and seg.leg_progress_ratio > ext.no_add_after_leg_progress_gt:
                    add_ok = False
                if add_ok and self._bias_scope_at_least("open_size") and trend_bias is not None:
                    if side == Side.LONG:
                        add_ok = trend_bias.allow_add_long and (risk_budget is None or risk_budget.allow_add_long)
                    else:
                        add_ok = trend_bias.allow_add_short and (risk_budget is None or risk_budget.allow_add_short)
                if add_ok:
                    return TradingAction(ActionType.ADD, side, "ADD_TREND_CONTINUATION")
                # Exit vote: best-point exit only contributes one vote.
                votes = 0
                if ts.is_broken:
                    votes += 2
                if side == Side.SHORT:
                    if signal.price > ts.ema_fast:
                        pos.trend_pullback_count += 1
                    else:
                        pos.trend_pullback_count = 0
                    if pos.trend_pullback_count >= 2:
                        votes += 1
                    if ts.ret_6_atr >= 1.2:
                        votes += 1
                    if signal.pred_cum_ret_5 > 0:
                        pos.exit_pending_count += 1
                    else:
                        pos.exit_pending_count = 0
                    if pos.exit_pending_count >= 2:
                        votes += 1
                    if best_point_signal is not None and self._bp_exit_triggered(best_point_signal, side=Side.SHORT):
                        pos.best_point_exit_count += 1
                    else:
                        pos.best_point_exit_count = 0
                    if pos.best_point_exit_count >= self.cfg.trend_lifecycle.bp_exit_confirm_bars:
                        votes += self._bp_exit_vote_weight(seg)
                elif side == Side.LONG and best_point_signal is not None:
                    if self._bp_exit_triggered(best_point_signal, side=Side.LONG):
                        pos.best_point_exit_count += 1
                    else:
                        pos.best_point_exit_count = 0
                    if pos.best_point_exit_count >= self.cfg.trend_lifecycle.bp_exit_confirm_bars:
                        votes += self._bp_exit_vote_weight(seg)
                pos.trend_exit_votes = votes
                exit_threshold = self._trend_exit_vote_threshold(trend_bias, side=side)
                if bar_index >= pos.min_hold_until and votes >= exit_threshold:
                    return TradingAction(ActionType.CLOSE, Side.FLAT, "CLOSE_TREND_EXIT_CONFIRMED")
                return TradingAction(ActionType.HOLD, side, "HOLD_TREND_CONTINUATION")
        if side == Side.LONG and tc and tc.is_downtrend and not pos.entry_was_slow_up:
            if signal.edge <= 0.0 or signal.pred_cum_ret_5 <= 0.0:
                return TradingAction(ActionType.CLOSE, Side.FLAT, "CLOSE_LONG_DOWNTREND_CONFIRMED")
            return TradingAction(ActionType.REDUCE, Side.LONG, "REDUCE_LONG_DOWNTREND_RISK")

        # Hard reverse signal.
        if side == Side.LONG and not pos.entry_was_slow_up and (signal.edge <= -r.reverse_edge_threshold or signal.pred_cum_ret_5 < 0):
            if r.allow_reverse and signal.edge <= -r.open_edge_threshold and signal.p_down >= r.open_prob_threshold:
                return TradingAction(ActionType.REVERSE, Side.SHORT, "REVERSE_SIGNAL")
            return TradingAction(ActionType.CLOSE, Side.FLAT, "CLOSE_REVERSE_SIGNAL")
        if side == Side.SHORT:
            if (
                pos.entry_was_crash
                and ts is not None
                and self.cfg.trend_position.allow_crash_trend_upgrade
                and self._crash_segment_allows_upgrade(seg, Side.SHORT)
                and ts.direction == TrendDirection.DOWN
                and ts.is_confirmed
                and ts.phase in (TrendPhase.CONTINUATION, TrendPhase.ACCELERATION)
                and ts.trend_age >= self.cfg.trend_position.min_trend_age_for_upgrade
                and ((pos.entry_price - signal.price) / max(signal.atr, 1e-12)) >= self.cfg.trend_position.crash_upgrade_profit_atr
                and signal.p_risk < self.cfg.rule.risk_exit_threshold
            ):
                return TradingAction(ActionType.HOLD, Side.SHORT, "UPGRADE_CRASH_TO_TREND_SHORT")
            if pos.entry_was_crash and self._is_standard_short_entry(
                signal, trend_bias, trend_signal=ts, crash_context=crash_context
            ):
                return TradingAction(
                    ActionType.ADD,
                    Side.SHORT,
                    "UPGRADE_CRASH_TO_MODEL_SHORT",
                    diagnostics={"upgrade_crash": True},
                )
            if pos.entry_was_crash and crash_context is not None:
                down_confirmed = (
                    ts is not None
                    and ts.direction == TrendDirection.DOWN
                    and ts.is_confirmed
                    and ts.phase in (TrendPhase.CONTINUATION, TrendPhase.ACCELERATION)
                )
                fail_by_price = (
                    signal.price > pos.entry_price + self.cfg.crash_short.fail_stop_atr * max(signal.atr, 1e-12)
                    and (ts is None or ts.is_broken)
                )
                fail_by_votes = (
                    crash_context.crash_votes < self.cfg.crash.min_crash_votes
                    and pos.crash_bars >= 2
                    and (ts is None or ts.direction != TrendDirection.DOWN)
                )
                if down_confirmed:
                    return TradingAction(ActionType.HOLD, Side.SHORT, "HOLD_CRASH_TREND_CONFIRMING")
                if fail_by_price or fail_by_votes:
                    return TradingAction(ActionType.CLOSE, Side.FLAT, "CLOSE_CRASH_FAILED")
                if pos.peak_profit_atr >= self.cfg.crash_short.trail_start_atr:
                    cur_profit_atr = (pos.entry_price - signal.price) / max(1e-12, signal.atr)
                    if (pos.peak_profit_atr - cur_profit_atr) >= self.cfg.crash_short.trail_back_atr:
                        return TradingAction(ActionType.CLOSE, Side.FLAT, "CLOSE_CRASH_TRAIL")

            # Sentinel must be upgraded quickly, otherwise exit to avoid probe-like drag.
            if pos.entry_was_sentinel and self._is_standard_short_entry(
                signal, trend_bias, trend_signal=ts, slow_context=slow_context, crash_context=crash_context
            ):
                return TradingAction(
                    ActionType.ADD,
                    Side.SHORT,
                    "UPGRADE_SENTINEL_TO_MODEL_SHORT",
                    diagnostics={"upgrade_sentinel": True},
                )
            if pos.entry_was_sentinel and pos.sentinel_bars >= self.cfg.sentinel_short.sentinel_max_hold_bars:
                return TradingAction(ActionType.CLOSE, Side.FLAT, "CLOSE_SENTINEL_NOT_CONFIRMED")

            in_trend_hold = pos.hold_mode == "TREND"
            if in_trend_hold and tc:
                trend_break = (
                    signal.price > tc.ema_fast
                    or signal.edge >= r.reverse_edge_threshold
                    or signal.pred_cum_ret_5 > 0
                    or tc.ret_3_atr > 0.8
                )
                trend_restore = signal.price < tc.ema_fast and signal.pred_cum_ret_5 <= 0 and signal.edge < r.reverse_edge_threshold
                if trend_restore:
                    pos.trend_break_count = 0
                elif trend_break:
                    pos.trend_break_count += 1
                if pos.trend_break_count >= self.cfg.trend_hold.trend_break_confirm_bars:
                    return TradingAction(ActionType.CLOSE, Side.FLAT, "CLOSE_SHORT_TREND_BROKEN")
                if pos.peak_profit_atr >= 2.0:
                    cur_profit_atr = (pos.entry_price - signal.price) / max(1e-12, signal.atr)
                    if (pos.peak_profit_atr - cur_profit_atr) >= 1.0:
                        return TradingAction(ActionType.CLOSE, Side.FLAT, "CLOSE_SHORT_TREND_TRAIL")

            if signal.edge >= r.reverse_edge_threshold or signal.pred_cum_ret_5 > 0:
                if tc and tc.is_downtrend:
                    if protection.short_exit_require_edge_and_cum:
                        reverse_confirm = signal.edge >= r.reverse_edge_threshold and signal.pred_cum_ret_5 > 0
                    else:
                        reverse_confirm = signal.edge >= r.reverse_edge_threshold or signal.pred_cum_ret_5 > 0
                    if reverse_confirm:
                        pos.short_reverse_confirm_count += 1
                    else:
                        pos.short_reverse_confirm_count = 0
                    if pos.short_reverse_confirm_count < protection.short_exit_confirm_bars:
                        return TradingAction(
                            ActionType.HOLD,
                            Side.SHORT,
                            "HOLD_SHORT_REVERSE_WAIT_CONFIRM",
                            diagnostics={"reverse_confirm_count": pos.short_reverse_confirm_count},
                        )
                    pos.short_reverse_confirm_count = 0
                    return TradingAction(ActionType.CLOSE, Side.FLAT, "CLOSE_SHORT_REVERSE_CONFIRMED")
                if r.allow_reverse and signal.edge >= r.open_edge_threshold and signal.p_up >= r.open_prob_threshold:
                    return TradingAction(ActionType.REVERSE, Side.LONG, "REVERSE_SIGNAL")
                return TradingAction(ActionType.CLOSE, Side.FLAT, "CLOSE_REVERSE_SIGNAL")

        # Continue condition.
        if side == Side.LONG:
            keep = signal.edge > r.long_continue_edge_min and signal.pred_cum_ret_5 >= 0
        else:
            keep = signal.edge < r.short_continue_edge_max and signal.pred_cum_ret_5 <= 0
        if keep:
            pos.continue_fail_count = 0
            pos.short_reverse_confirm_count = 0
            if (
                side == Side.SHORT
                and tc
                and self.cfg.trend_hold.enabled
                and tc.is_downtrend
                and signal.pred_cum_ret_5 < 0
                and signal.p_risk <= self.cfg.rule.risk_exit_threshold
                and (
                    not self.cfg.trend_hold.allow_extend_only_for_model_short
                    or (
                        pos.entry_signal_snapshot.get("reason_code")
                        in ("OPEN_SHORT_SIGNAL", "UPGRADE_SENTINEL_TO_MODEL_SHORT", "UPGRADE_CRASH_TO_MODEL_SHORT")
                    )
                )
            ):
                pos.hold_mode = "TREND"
            if (
                pos.add_count < self.cfg.base.max_add_count
                and signal.risk_ok
                and ((side == Side.LONG and signal.pred_cum_ret_5 > 0) or (side == Side.SHORT and signal.pred_cum_ret_5 < 0))
            ):
                if pos.entry_was_sentinel and side == Side.SHORT and not self._is_standard_short_entry(
                    signal, trend_bias, trend_signal=ts, slow_context=slow_context, crash_context=crash_context
                ):
                    return TradingAction(ActionType.HOLD, side, "HOLD_SENTINEL_WAIT_STANDARD_CONFIRM")
                return TradingAction(ActionType.ADD, side, "ADD_SIGNAL_CONFIRMED")
            return TradingAction(ActionType.HOLD, side, "HOLD_SIGNAL_VALID")

        if side == Side.LONG and pos.entry_was_slow_up and slow_context and slow_context.is_slow_uptrend:
            return TradingAction(ActionType.HOLD, Side.LONG, "HOLD_SLOW_UPTREND")

        pos.continue_fail_count += 1
        if side == Side.SHORT and (signal.edge < r.reverse_edge_threshold or signal.pred_cum_ret_5 <= 0.0):
            pos.short_reverse_confirm_count = 0
        if pos.continue_fail_count >= r.continue_fail_limit:
            return TradingAction(ActionType.CLOSE, Side.FLAT, "CLOSE_CONTINUE_SIGNAL_FAILED")
        return TradingAction(ActionType.REDUCE, side, "REDUCE_WEAK_CONTINUE_SIGNAL")

