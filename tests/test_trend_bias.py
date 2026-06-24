from __future__ import annotations

from trading_system.config import TrendBiasConfig
from trading_system.crash import CrashContext
from trading_system.trend import TrendContext
from trading_system.trend_bias import BiasDirection, CounterTrendLevel, TrendBiasBuilder, neutral_trend_bias
from trading_system.trend_signal import TrendDirection, TrendPhase, TrendSignal, TrendStrength


def _minimal_trend_signal(*, direction: TrendDirection = TrendDirection.UP) -> TrendSignal:
    return TrendSignal(
        direction=direction,
        strength=TrendStrength.NORMAL,
        phase=TrendPhase.CONTINUATION,
        score_up=3.0,
        score_down=0.0,
        score_abs=3.0,
        confidence=0.8,
        trend_age=10,
        invalid_count=0,
        is_confirmed=True,
        is_broken=False,
        is_accelerating=False,
        is_exhausted=False,
        ret_6_atr=0.5,
        ret_12_atr=0.4,
        ret_24_atr=0.3,
        ema_fast=100.0,
        ema_mid=99.0,
        ema_slow=98.0,
        close_to_ema_fast_atr=0.1,
        distance_from_ema_slow_atr=0.2,
        rolling_high_break=False,
        rolling_low_break=False,
        higher_high_low=True,
        lower_high_low=False,
        persistence_ratio=0.7,
        range_expansion=1.0,
        reason_codes=["TEST"],
    )


def test_builder_disabled_returns_neutral() -> None:
    cfg = TrendBiasConfig(enabled=False)
    bias = TrendBiasBuilder(cfg).build(
        trend_signal=_minimal_trend_signal(),
        segment_context=None,
        slow_context=None,
        crash_context=None,
        trend_context=None,
    )
    assert bias == neutral_trend_bias()


def test_crash_blocks_long_opens() -> None:
    cfg = TrendBiasConfig(enabled=True)
    crash = CrashContext(
        is_crash=True,
        is_model_blind_crash=True,
        crash_score=5.0,
        drawdown_24h=-0.1,
        ret_6_atr=-2.0,
        ret_12_atr=-2.5,
        range_expansion=1.5,
        consecutive_down_bars=3,
        lower_low_break=True,
        model_disagrees=True,
        crash_votes=3,
        strong_crash=True,
        reason_codes=["CRASH"],
    )
    bias = TrendBiasBuilder(cfg).build(
        trend_signal=_minimal_trend_signal(direction=TrendDirection.DOWN),
        segment_context=None,
        slow_context=None,
        crash_context=crash,
        trend_context=TrendContext(
            is_downtrend=True,
            is_strong_downtrend=False,
            is_uptrend=False,
            trend_score=-2.0,
            ret_3_atr=-0.5,
            ret_6_atr=-1.0,
            ema_fast=95.0,
            ema_slow=100.0,
            breakdown_low_n=True,
            lower_high_low=True,
            reason_codes=[],
        ),
    )
    assert bias.allow_open_long is False
    assert bias.open_bias_long == 0.0
    assert bias.crash_short_active is True
    assert bias.open_bias_short >= cfg.crash_short_open_boost


def test_legacy_downtrend_soft_when_hard_block_disabled() -> None:
    cfg = TrendBiasConfig(enabled=True, legacy_down_hard_block=False)
    trend_ctx = TrendContext(
        is_downtrend=True,
        is_strong_downtrend=False,
        is_uptrend=False,
        trend_score=-2.0,
        ret_3_atr=-0.5,
        ret_6_atr=-1.0,
        ema_fast=95.0,
        ema_slow=100.0,
        breakdown_low_n=True,
        lower_high_low=True,
        reason_codes=[],
    )
    bias = TrendBiasBuilder(cfg).build(
        trend_signal=_minimal_trend_signal(direction=TrendDirection.UP),
        segment_context=None,
        slow_context=None,
        crash_context=None,
        trend_context=trend_ctx,
    )
    assert bias.allow_open_long is True
    assert "LEGACY_DOWNTREND_SOFT_LONG" in bias.reason_codes


def test_chop_soft_micro_weak_reduces_alignment() -> None:
    from dataclasses import replace

    cfg = TrendBiasConfig(enabled=True, chop_soft_micro_weight=0.0)
    base_ts = _minimal_trend_signal(direction=TrendDirection.UP)
    chop_ts = replace(base_ts, reason_codes=["CHOP_SOFT"])
    base_bias = TrendBiasBuilder(cfg).build(
        trend_signal=base_ts,
        segment_context=None,
        slow_context=None,
        crash_context=None,
        trend_context=None,
    )
    chop_bias = TrendBiasBuilder(cfg).build(
        trend_signal=chop_ts,
        segment_context=None,
        slow_context=None,
        crash_context=None,
        trend_context=None,
    )
    assert "CHOP_SOFT_MICRO_WEAK" in chop_bias.reason_codes
    assert chop_bias.alignment_score_long <= base_bias.alignment_score_long


def test_hard_block_invariant() -> None:
    cfg = TrendBiasConfig(enabled=True)
    bias = TrendBiasBuilder(cfg).build(
        trend_signal=_minimal_trend_signal(direction=TrendDirection.DOWN),
        segment_context=None,
        slow_context=None,
        crash_context=None,
        trend_context=None,
    )
    if bias.counter_level_long == CounterTrendLevel.HARD_BLOCK:
        assert bias.allow_open_long is False
        assert bias.open_bias_long == 0.0
        assert bias.allow_add_long is False
