from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from trading_system.config import CrashConfig
from trading_system.signal import TradingSignal


@dataclass
class CrashContext:
    is_crash: bool
    is_model_blind_crash: bool
    crash_score: float
    drawdown_24h: float
    ret_6_atr: float
    ret_12_atr: float
    range_expansion: float
    consecutive_down_bars: int
    lower_low_break: bool
    model_disagrees: bool
    crash_votes: int
    strong_crash: bool
    reason_codes: list[str] = field(default_factory=list)


class CrashRegimeDetector:
    def __init__(self, cfg: CrashConfig) -> None:
        self.cfg = cfg

    def compute(
        self,
        close_hist: list[float],
        high_hist: list[float],
        low_hist: list[float],
        atr_hist: list[float],
        signal: TradingSignal,
        *,
        standard_open_short: bool,
        is_flat: bool,
    ) -> CrashContext:
        if not self.cfg.enabled or len(close_hist) < 30:
            return CrashContext(
                is_crash=False,
                is_model_blind_crash=False,
                crash_score=0.0,
                drawdown_24h=0.0,
                ret_6_atr=0.0,
                ret_12_atr=0.0,
                range_expansion=1.0,
                consecutive_down_bars=0,
                lower_low_break=False,
                model_disagrees=False,
                crash_votes=0,
                strong_crash=False,
                reason_codes=[],
            )
        c = float(close_hist[-1])
        atr = max(float(atr_hist[-1]), 1e-12)
        ret_6_atr = (c - float(close_hist[-7])) / atr
        ret_12_atr = (c - float(close_hist[-13])) / atr
        rolling_high_24 = float(np.max(np.asarray(high_hist[-24:], dtype=np.float64)))
        drawdown_24h = c / max(1e-12, rolling_high_24) - 1.0
        lower_low_break = c < float(np.min(np.asarray(low_hist[-self.cfg.lower_low_lookback :], dtype=np.float64)))

        tr = np.maximum(
            np.asarray(high_hist[-24:], dtype=np.float64) - np.asarray(low_hist[-24:], dtype=np.float64),
            1e-12,
        )
        recent_tr = float(np.mean(tr[-3:]))
        base_tr = float(np.mean(tr))
        range_expansion = recent_tr / max(base_tr, 1e-12)

        consecutive_down = 0
        for i in range(len(close_hist) - 1, max(0, len(close_hist) - 8), -1):
            if close_hist[i] < close_hist[i - 1]:
                consecutive_down += 1
            else:
                break

        votes = 0
        reasons: list[str] = []
        if ret_6_atr <= self.cfg.ret6_atr_threshold:
            votes += 1
            reasons.append("RET6_ATR_CRASH")
        if ret_12_atr <= self.cfg.ret12_atr_threshold:
            votes += 1
            reasons.append("RET12_ATR_CRASH")
        if drawdown_24h <= self.cfg.drawdown_24h_threshold:
            votes += 1
            reasons.append("DD24H_CRASH")
        if lower_low_break:
            votes += 1
            reasons.append("LOWER_LOW_BREAK")
        if consecutive_down >= 4 and range_expansion >= self.cfg.range_expansion_threshold:
            votes += 1
            reasons.append("CONSEC_DOWN_RANGE_EXPAND")

        is_crash = votes >= self.cfg.min_crash_votes
        strong_crash = votes >= self.cfg.strong_crash_votes
        strong_model_long = (
            signal.p_up >= 0.38
            and signal.edge >= 0.06
            and float(signal.pred_cum_ret_5) > 0.25
            and signal.p_risk < 0.40
        )
        model_disagrees = not standard_open_short
        is_model_blind_crash = is_crash and is_flat and (not standard_open_short) and (not strong_model_long)
        score = float(votes + max(0.0, -ret_6_atr - 2.0))
        return CrashContext(
            is_crash=is_crash,
            is_model_blind_crash=is_model_blind_crash,
            crash_score=score,
            drawdown_24h=float(drawdown_24h),
            ret_6_atr=float(ret_6_atr),
            ret_12_atr=float(ret_12_atr),
            range_expansion=float(range_expansion),
            consecutive_down_bars=int(consecutive_down),
            lower_low_break=bool(lower_low_break),
            model_disagrees=bool(model_disagrees),
            crash_votes=votes,
            strong_crash=strong_crash,
            reason_codes=reasons,
        )

