from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BaseConfig:
    max_position_ratio: float = 0.20
    fixed_leverage: float = 20.0
    max_add_count: int = 2
    decision_interval: str = "1h"
    execute_timing: str = "next_bar_open"
    max_margin_loss_ratio: float = 1.00
    catastrophe_margin_loss_buffer: float = 0.90


@dataclass(frozen=True)
class RuleConfig:
    open_edge_threshold: float = 0.08
    open_prob_threshold: float = 0.42
    open_flat_max: float = 0.34
    risk_open_max: float = 0.38
    risk_exit_threshold: float = 0.48
    long_continue_edge_min: float = -0.03
    short_continue_edge_max: float = 0.03
    reverse_edge_threshold: float = 0.05
    max_hold_bars: int = 6
    continue_fail_limit: int = 2
    reduce_scale: float = 0.5
    allow_reverse: bool = True
    risk_exit_mode: str = "full_close"
    time_exit_mode: str = "full_close"


@dataclass(frozen=True)
class RiskConfig:
    stop_atr_mult: float = 1.2
    tp1_atr_mult: float = 1.0
    tp2_atr_mult: float = 2.0
    trail_atr_mult: float = 0.8
    day_drawdown_stop: float = 0.02
    week_drawdown_defensive: float = 0.05
    defensive_size_scale: float = 0.30
    loss_streak_limit: int = 3
    cooldown_bars: int = 12


@dataclass(frozen=True)
class SizingConfig:
    weak_conf_min: float = 0.08
    medium_conf_min: float = 0.14
    strong_conf_min: float = 0.20
    weak_range: tuple[float, float] = (0.05, 0.08)
    medium_range: tuple[float, float] = (0.08, 0.12)
    strong_range: tuple[float, float] = (0.12, 0.15)
    very_strong_range: tuple[float, float] = (0.15, 0.20)
    reference_atr_ratio: float = 0.015


@dataclass(frozen=True)
class ExecutionConfig:
    fee_bps: float = 4.0
    slippage_bps: float = 2.0
    atr_period: int = 14


@dataclass(frozen=True)
class TrendConfig:
    enabled: bool = True
    ema_fast: int = 12
    ema_slow: int = 36
    ret_lookback_fast: int = 3
    ret_lookback_slow: int = 6
    down_ret_atr_threshold: float = -1.2
    strong_down_ret_atr_threshold: float = -1.8
    breakdown_lookback: int = 12
    min_downtrend_votes: int = 2
    min_strong_downtrend_votes: int = 3


@dataclass(frozen=True)
class ProtectionConfig:
    block_long_in_downtrend: bool = True
    allow_probe_short: bool = False
    probe_short_position_ratio: float = 0.05
    probe_short_max_position_ratio: float = 0.08
    probe_short_risk_max: float = 0.45
    probe_short_flat_max: float = 0.45
    probe_short_require_cum_ret_negative: bool = False
    probe_short_min_edge: float = -0.015
    probe_short_min_p_down: float = 0.30
    short_exit_confirm_bars: int = 2
    short_exit_require_edge_and_cum: bool = True
    allow_sentinel_short: bool = False


@dataclass(frozen=True)
class TrendHoldConfig:
    enabled: bool = True
    normal_max_hold_bars: int = 8
    short_trend_max_hold_bars: int = 24
    strong_short_trend_max_hold_bars: int = 36
    trend_break_confirm_bars: int = 2
    min_profit_to_extend_atr: float = 0.5
    allow_extend_only_for_model_short: bool = True


@dataclass(frozen=True)
class SentinelShortConfig:
    enabled: bool = True
    sentinel_position_ratio: float = 0.03
    sentinel_max_position_ratio: float = 0.05
    sentinel_max_hold_bars: int = 4
    sentinel_cooldown_bars: int = 12
    sentinel_risk_max: float = 0.42
    sentinel_flat_max: float = 0.38
    sentinel_ret6_atr_threshold: float = -2.0


@dataclass(frozen=True)
class CrashConfig:
    enabled: bool = True
    ret6_atr_threshold: float = -3.0
    ret12_atr_threshold: float = -5.0
    drawdown_24h_threshold: float = -0.06
    range_expansion_threshold: float = 1.5
    lower_low_lookback: int = 24
    min_crash_votes: int = 2
    strong_crash_votes: int = 3
    regime_release_bars: int = 6


@dataclass(frozen=True)
class CrashShortConfig:
    enabled: bool = True
    position_ratio: float = 0.04
    strong_position_ratio: float = 0.06
    max_position_ratio: float = 0.08
    risk_max: float = 0.48
    flat_max: float = 0.42
    max_hold_bars: int = 18
    strong_max_hold_bars: int = 30
    fail_stop_atr: float = 1.2
    trail_start_atr: float = 2.0
    trail_back_atr: float = 1.0
    same_regime_once: bool = True


@dataclass(frozen=True)
class TrendSignalConfig:
    enabled: bool = True
    ema_fast: int = 12
    ema_mid: int = 24
    ema_slow: int = 72
    ret_fast: int = 6
    ret_mid: int = 12
    ret_slow: int = 24
    structure_lookback: int = 24
    persistence_lookback: int = 12
    confirmed_score: int = 4
    strong_score: int = 5
    extreme_score: int = 6
    invalid_confirm_bars: int = 2
    invalid_reset_bars: int = 4
    hold_confirm_score: int = 3
    min_trend_age_for_hold: int = 3
    direction_margin: int = 2
    acceleration_range_expansion: float = 1.4
    exhaustion_distance_atr: float = 5.0
    chop_guard_enabled: bool = True
    chop_range_atr_max: float = 3.0
    chop_ret24_atr_max: float = 1.0
    chop_flip_lookback: int = 12
    chop_flip_max: int = 4
    chop_efficiency_lookback: int = 12
    chop_efficiency_min: float = 0.25
    chop_slow_trend_exception_enabled: bool = True
    chop_efficiency_48_min: float = 0.35
    chop_ema_slope_bars: int = 48
    chop_persistence_ema_mid_min: float = 0.6
    chop_ema_slope_atr_min: float = 0.02


@dataclass(frozen=True)
class TrendPositionConfig:
    upgrade_profit_atr: float = 1.5
    crash_upgrade_profit_atr: float = 1.5
    allow_crash_trend_upgrade: bool = False
    min_trend_age_for_upgrade: int = 0
    add_profit_atr: float = 2.0
    max_trend_hold_bars: int = 48
    strong_trend_hold_bars: int = 72
    exhaustion_reduce_scale: float = 0.5
    trail_start_atr: float = 2.0
    trail_back_atr: float = 1.2


@dataclass(frozen=True)
class TrendLifecycleConfig:
    min_trend_hold_bars: int = 6
    strong_min_trend_hold_bars: int = 10
    exit_confirm_votes: int = 3
    bp_exit_confirm_bars: int = 3
    protect_profit_atr: float = 3.0
    runner_profit_atr: float = 5.0
    runner_reduce_scale: float = 0.5


@dataclass(frozen=True)
class BestPointConfig:
    enabled: bool = False
    observe_only: bool = True
    long_entry_confirm_threshold: float = 0.55
    short_entry_confirm_threshold: float = 0.55
    crash_short_entry_confirm_threshold: float = 0.45
    require_entry_confirm_for_crash: bool = False
    min_opportunity_roi: float = 0.0
    watch_entry_threshold: float = 0.70
    exit_prob_threshold: float = 0.70
    hold_min_prob: float = 0.30


@dataclass(frozen=True)
class LongHorizonLabelConfig:
    enabled: bool = True
    min_net_roi: float = 0.2
    min_holding_bars: int = 8
    max_holding_bars: int = 72
    cooldown_after_trade: int = 3


@dataclass(frozen=True)
class SlowUptrendConfig:
    enabled: bool = True
    ema_fast: int = 12
    ema_mid: int = 24
    ema_slow: int = 72
    slope_24_atr_min: float = 1.0
    slope_48_atr_min: float = 1.8
    stable_slope_24_atr_min: float = 1.8
    stable_slope_48_atr_min: float = 2.5
    persistence_fast_min: float = 0.62
    persistence_mid_min: float = 0.55
    stable_persistence_fast_min: float = 0.75
    stable_persistence_mid_min: float = 0.67
    max_drawdown_24_atr: float = 3.0
    exit_drawdown_24_atr: float = 3.5
    pullback_depth_atr_max: float = 0.8
    min_score: int = 5
    stable_score: int = 7
    risk_score_max: float = 0.45
    flat_range_max: float = 0.50


@dataclass(frozen=True)
class SlowUpPositionConfig:
    enabled: bool = True
    position_ratio: float = 0.04
    stable_position_ratio: float = 0.06
    max_position_ratio: float = 0.08
    risk_max: float = 0.45
    upgrade_profit_atr: float = 0.8
    min_hold_bars: int = 8
    max_hold_bars: int = 72
    exit_votes: int = 3
    runner_profit_atr: float = 5.0
    runner_reduce_scale: float = 0.7
    model_opp_cum_ret_min: float = -0.10
    model_opp_edge_min: float = -0.04
    watch_min_bars: int = 6
    segment_min_bars: int = 4
    watch_probe_require_up_leg: bool = True
    watch_probe_min_cum_ret: float = -0.05
    watch_probe_min_p_up: float = 0.28
    allow_trend_upgrade: bool = True
    stop_atr_mult: float = 2.2


@dataclass(frozen=True)
class TrendSegmentChangepointConfig:
    enabled: bool = False
    bocpd_hazard: float = 0.02
    cp_threshold: float = 0.45
    cooldown_bars: int = 6
    eff_min: float = 0.20


@dataclass(frozen=True)
class TrendSegmentConfig:
    enabled: bool = True
    swing_small_left_bars: int = 3
    swing_small_right_bars: int = 3
    swing_large_left_bars: int = 8
    swing_large_right_bars: int = 8
    merge_pullback_atr: float = 2.0
    min_leg_bars: int = 12
    min_move_atr: float = 1.2
    min_efficiency: float = 0.30
    upgrade_min_bars: int = 6
    leg_end_pullback_atr: float = 2.0
    use_regime_filter: bool = True
    counter_trend_block: bool = True
    exit_vote_requires_exhaustion: bool = True
    history_buffer_bars: int = 512
    changepoint: TrendSegmentChangepointConfig = TrendSegmentChangepointConfig()


@dataclass(frozen=True)
class SlowUpLongHorizonLabelConfig:
    enabled: bool = True
    min_net_roi: float = 0.10
    min_holding_bars: int = 8
    max_holding_bars: int = 72
    max_adverse_excursion_ratio: float = 0.5
    cooldown_after_trade: int = 3


@dataclass(frozen=True)
class TrendEntryQualifierConfig:
    enabled: bool = False
    min_edge_long: float = 0.02
    min_edge_short: float = -0.02
    min_prob_long: float = 0.30
    min_prob_short: float = 0.30
    position_ratio: float = 0.04
    max_position_ratio: float = 0.05
    open_bias_penalty: float = 0.85
    relax_edge_mult: float = 0.65
    relax_prob_delta: float = -0.04
    require_best_point: bool = True
    require_segment: bool = True
    require_confirmed_leg: bool = True
    block_long_in_crash: bool = True
    stop_atr_mult: float = 2.0


@dataclass(frozen=True)
class RegimeThresholdConfig:
    enabled: bool = False
    apply_to_standard_opens: bool = True
    apply_to_trend_qualified: bool = True
    trend_confirmed_edge_mult: float = 0.65
    trend_confirmed_prob_delta: float = -0.04
    slow_up_edge_mult: float = 0.55
    slow_up_prob_delta: float = -0.05
    crash_edge_mult: float = 0.50
    crash_prob_delta: float = -0.06


@dataclass(frozen=True)
class TrendHoldExtensionConfig:
    enabled: bool = False
    leg_progress_hold_boost_below: float = 0.5
    hold_bias_boost_mult: float = 1.2
    leg_progress_time_exit_above: float = 0.75
    no_add_after_leg_progress_gt: float = 0.6


@dataclass(frozen=True)
class ParticipationFloorConfig:
    enabled: bool = False


@dataclass(frozen=True)
class SlowUpParticipationGateConfig:
    enabled: bool = False
    tau_slow: float = 0.55
    edge_threshold_slow: float = 0.0
    probe_ratio: float = 0.5
    weight_legacy: float = 0.6
    weight_teq: float = 0.25
    weight_part: float = 0.15
    calibration_path: str = ""
    use_calibrated: bool = True


@dataclass(frozen=True)
class ParticipationChannelConfig:
    enabled: bool = False
    slow_up_gate: SlowUpParticipationGateConfig = SlowUpParticipationGateConfig()


@dataclass(frozen=True)
class TeqEdgeConfig:
    enabled: bool = False
    weight_edge_5: float = 0.35
    weight_edge_24: float = 0.45
    weight_participation: float = 0.20
    calibration_path: str = ""
    use_calibrated: bool = True
    model_checkpoint: str = ""


@dataclass(frozen=True)
class TrendBiasConfig:
    enabled: bool = True
    decision_scope: str = "observe"
    disable_legacy_trend_rules: bool = False
    long_open_relax: float = 1.2
    short_open_relax: float = 1.2
    trend_size_boost: float = 1.2
    light_counter_size_penalty: float = 0.7
    medium_counter_size_penalty: float = 0.4
    light_counter_tighten: float = 0.85
    medium_counter_tighten: float = 0.6
    crash_short_open_boost: float = 1.35
    slow_up_size_boost: float = 1.15
    slow_up_hold_boost: float = 1.2
    trend_exit_vote_bonus: float = 1.2
    counter_exit_vote_penalty: float = 0.85
    allow_hard_counter_probe: bool = False
    hard_counter_probe_ratio: float = 0.01
    legacy_down_hard_block: bool = False
    crash_hard_block_long: bool = True
    chop_soft_micro_weight: float = 0.5
    max_hard_block_ratio: float = 0.35


@dataclass(frozen=True)
class TradingSystemConfig:
    base: BaseConfig = BaseConfig()
    rule: RuleConfig = RuleConfig()
    risk: RiskConfig = RiskConfig()
    sizing: SizingConfig = SizingConfig()
    execution: ExecutionConfig = ExecutionConfig()
    trend: TrendConfig = TrendConfig()
    protection: ProtectionConfig = ProtectionConfig()
    trend_hold: TrendHoldConfig = TrendHoldConfig()
    sentinel_short: SentinelShortConfig = SentinelShortConfig()
    crash: CrashConfig = CrashConfig()
    crash_short: CrashShortConfig = CrashShortConfig()
    trend_signal: TrendSignalConfig = TrendSignalConfig()
    trend_position: TrendPositionConfig = TrendPositionConfig()
    trend_lifecycle: TrendLifecycleConfig = TrendLifecycleConfig()
    long_horizon_label: LongHorizonLabelConfig = LongHorizonLabelConfig()
    slow_uptrend: SlowUptrendConfig = SlowUptrendConfig()
    slow_up_position: SlowUpPositionConfig = SlowUpPositionConfig()
    slow_up_long_horizon_label: SlowUpLongHorizonLabelConfig = SlowUpLongHorizonLabelConfig()
    trend_segment: TrendSegmentConfig = TrendSegmentConfig()
    best_point: BestPointConfig = BestPointConfig()
    trend_bias: TrendBiasConfig = TrendBiasConfig()
    trend_entry_qualifier: TrendEntryQualifierConfig = TrendEntryQualifierConfig()
    trend_hold_extension: TrendHoldExtensionConfig = TrendHoldExtensionConfig()
    regime_threshold: RegimeThresholdConfig = RegimeThresholdConfig()
    participation_floor: ParticipationFloorConfig = ParticipationFloorConfig()
    participation_channel: ParticipationChannelConfig = ParticipationChannelConfig()
    teq_edge: TeqEdgeConfig = TeqEdgeConfig()


def _tuple2(v: list[float] | tuple[float, float], default: tuple[float, float]) -> tuple[float, float]:
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return float(v[0]), float(v[1])
    return default


def load_config(path: str | Path) -> TradingSystemConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    base_p = payload.get("base", {})
    rule_p = payload.get("rule", {})
    risk_p = payload.get("risk", {})
    sizing_p = payload.get("sizing", {})
    exec_p = payload.get("execution", {})
    trend_p = payload.get("trend", {})
    protection_p = payload.get("protection", {})
    trend_hold_p = payload.get("trend_hold", {})
    sentinel_p = payload.get("sentinel_short", {})
    crash_p = payload.get("crash", {})
    crash_short_p = payload.get("crash_short", {})
    trend_signal_p = payload.get("trend_signal", {})
    trend_position_p = payload.get("trend_position", {})
    trend_lifecycle_p = payload.get("trend_lifecycle", {})
    long_horizon_label_p = payload.get("long_horizon_label", {})
    slow_uptrend_p = payload.get("slow_uptrend", {})
    slow_up_position_p = payload.get("slow_up_position", {})
    slow_up_long_horizon_label_p = payload.get("slow_up_long_horizon_label", {})
    best_point_p = payload.get("best_point", {})
    trend_bias_p = payload.get("trend_bias", {})
    trend_entry_qualifier_p = payload.get("trend_entry_qualifier", {})
    trend_hold_extension_p = payload.get("trend_hold_extension", {})
    regime_threshold_p = payload.get("regime_threshold", {})
    participation_floor_p = payload.get("participation_floor", {})
    participation_channel_p = payload.get("participation_channel", {})
    slow_up_gate_p = participation_channel_p.get("slow_up_gate", {})
    teq_edge_p = payload.get("teq_edge", {})
    trend_segment_p = dict(payload.get("trend_segment", {}))
    changepoint_p = trend_segment_p.pop("changepoint", {})
    ts_defaults = {**TrendSegmentConfig().__dict__, **trend_segment_p}
    ts_defaults.pop("changepoint", None)
    trend_segment = TrendSegmentConfig(
        **ts_defaults,
        changepoint=TrendSegmentChangepointConfig(**{**TrendSegmentChangepointConfig().__dict__, **changepoint_p}),
    )
    return TradingSystemConfig(
        base=BaseConfig(**{**BaseConfig().__dict__, **base_p}),
        rule=RuleConfig(**{**RuleConfig().__dict__, **rule_p}),
        risk=RiskConfig(**{**RiskConfig().__dict__, **risk_p}),
        sizing=SizingConfig(
            **{
                **SizingConfig().__dict__,
                **{
                    **sizing_p,
                    "weak_range": _tuple2(sizing_p.get("weak_range", SizingConfig().weak_range), SizingConfig().weak_range),
                    "medium_range": _tuple2(sizing_p.get("medium_range", SizingConfig().medium_range), SizingConfig().medium_range),
                    "strong_range": _tuple2(sizing_p.get("strong_range", SizingConfig().strong_range), SizingConfig().strong_range),
                    "very_strong_range": _tuple2(
                        sizing_p.get("very_strong_range", SizingConfig().very_strong_range),
                        SizingConfig().very_strong_range,
                    ),
                },
            }
        ),
        execution=ExecutionConfig(**{**ExecutionConfig().__dict__, **exec_p}),
        trend=TrendConfig(**{**TrendConfig().__dict__, **trend_p}),
        protection=ProtectionConfig(**{**ProtectionConfig().__dict__, **protection_p}),
        trend_hold=TrendHoldConfig(**{**TrendHoldConfig().__dict__, **trend_hold_p}),
        sentinel_short=SentinelShortConfig(**{**SentinelShortConfig().__dict__, **sentinel_p}),
        crash=CrashConfig(**{**CrashConfig().__dict__, **crash_p}),
        crash_short=CrashShortConfig(**{**CrashShortConfig().__dict__, **crash_short_p}),
        trend_signal=TrendSignalConfig(**{**TrendSignalConfig().__dict__, **trend_signal_p}),
        trend_position=TrendPositionConfig(**{**TrendPositionConfig().__dict__, **trend_position_p}),
        trend_lifecycle=TrendLifecycleConfig(**{**TrendLifecycleConfig().__dict__, **trend_lifecycle_p}),
        long_horizon_label=LongHorizonLabelConfig(**{**LongHorizonLabelConfig().__dict__, **long_horizon_label_p}),
        slow_uptrend=SlowUptrendConfig(**{**SlowUptrendConfig().__dict__, **slow_uptrend_p}),
        slow_up_position=SlowUpPositionConfig(**{**SlowUpPositionConfig().__dict__, **slow_up_position_p}),
        slow_up_long_horizon_label=SlowUpLongHorizonLabelConfig(
            **{**SlowUpLongHorizonLabelConfig().__dict__, **slow_up_long_horizon_label_p}
        ),
        trend_segment=trend_segment,
        best_point=BestPointConfig(**{**BestPointConfig().__dict__, **best_point_p}),
        trend_bias=TrendBiasConfig(**{**TrendBiasConfig().__dict__, **trend_bias_p}),
        trend_entry_qualifier=TrendEntryQualifierConfig(
            **{**TrendEntryQualifierConfig().__dict__, **trend_entry_qualifier_p}
        ),
        trend_hold_extension=TrendHoldExtensionConfig(
            **{**TrendHoldExtensionConfig().__dict__, **trend_hold_extension_p}
        ),
        regime_threshold=RegimeThresholdConfig(
            **{**RegimeThresholdConfig().__dict__, **regime_threshold_p}
        ),
        participation_floor=ParticipationFloorConfig(
            **{**ParticipationFloorConfig().__dict__, **participation_floor_p}
        ),
        participation_channel=ParticipationChannelConfig(
            enabled=bool(participation_channel_p.get("enabled", False)),
            slow_up_gate=SlowUpParticipationGateConfig(
                **{**SlowUpParticipationGateConfig().__dict__, **slow_up_gate_p}
            ),
        ),
        teq_edge=TeqEdgeConfig(**{**TeqEdgeConfig().__dict__, **teq_edge_p}),
    )

