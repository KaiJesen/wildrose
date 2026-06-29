"""State machine for long/short observation and position zones (算法说明.md)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from market_data.schema import COL_CLOSE, COL_HIGH, COL_LOW, COL_TIME
from trend_strategy.rails import (
    Backend,
    TrendLine,
    find_long_close_line,
    find_lower_rail,
    find_short_close_line,
    find_upper_rail,
)


class Zone(str, Enum):
    IDLE = "idle"
    LONG_OBS = "long_obs"
    SHORT_OBS = "short_obs"
    LONG_POS = "long_pos"
    SHORT_POS = "short_pos"


@dataclass
class Trade:
    side: str
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    entry_ts: str
    exit_ts: str


@dataclass
class ZoneSnapshot:
    idx: int
    zone: Zone
    upper_rail: TrendLine | None = None
    lower_rail: TrendLine | None = None
    long_close: TrendLine | None = None
    short_close: TrendLine | None = None


@dataclass
class EngineConfig:
    warmup_bars: int = 40
    flat_threshold: float = 0.02
    backend: Backend = Backend.AUTO


@dataclass
class ZoneEngine:
    config: EngineConfig = field(default_factory=EngineConfig)
    zone: Zone = Zone.IDLE
    anchor_a: int = -1
    zone_enter_idx: int = -1
    zone_extreme_idx: int = -1
    zone_extreme_val: float = float("inf")
    upper_rail: TrendLine | None = None
    lower_rail: TrendLine | None = None
    long_close: TrendLine | None = None
    short_close: TrendLine | None = None
    pending_cross_idx: int | None = None
    pending_side: str | None = None
    trades: list[Trade] = field(default_factory=list)
    snapshots: list[ZoneSnapshot] = field(default_factory=list)
    _open_side: str | None = None
    _open_idx: int = -1
    _open_price: float = 0.0

    def _backend(self) -> Backend:
        return self.config.backend

    def _cross_down(self, line: TrendLine, t: int, closes: np.ndarray) -> bool:
        if t < 1:
            return False
        prev = closes[t - 1]
        cur = closes[t]
        lp, lc = line.value_at(t - 1), line.value_at(t)
        return prev >= lp - 1e-8 and cur < lc - 1e-8

    def _cross_up(self, line: TrendLine, t: int, closes: np.ndarray) -> bool:
        if t < 1:
            return False
        prev = closes[t - 1]
        cur = closes[t]
        lp, lc = line.value_at(t - 1), line.value_at(t)
        return prev <= lp + 1e-8 and cur > lc + 1e-8

    def _try_init(self, t: int, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> bool:
        if t < self.config.warmup_bars:
            return False
        start = 0
        seg_h = highs[start : t + 1]
        seg_l = lows[start : t + 1]
        i_h = start + int(np.argmax(seg_h))
        i_l = start + int(np.argmin(seg_l))
        if abs(closes[t] - closes[start]) / max(closes[start], 1e-12) < self.config.flat_threshold:
            return False

        self.zone_enter_idx = t
        if i_h < i_l:
            self.zone = Zone.LONG_OBS
            self.anchor_a = i_h
            self.zone_extreme_idx = i_l
            self.zone_extreme_val = float(lows[i_l])
            self.upper_rail = find_upper_rail(highs, lows, self.anchor_a, t, backend=self._backend())
        else:
            self.zone = Zone.SHORT_OBS
            self.anchor_a = i_l
            self.zone_extreme_idx = i_h
            self.zone_extreme_val = float(highs[i_h])
            self.lower_rail = find_lower_rail(highs, lows, self.anchor_a, t, backend=self._backend())
        return True

    def _enter_long_pos(self, t: int, closes: np.ndarray) -> None:
        self.zone = Zone.LONG_POS
        self.zone_enter_idx = t
        self.zone_extreme_idx = t
        self.zone_extreme_val = float(closes[t])
        self.long_close = None
        self.pending_cross_idx = None
        self.pending_side = None
        self._open_side = "long"
        self._open_idx = t
        self._open_price = float(closes[t])

    def _enter_short_pos(self, t: int, closes: np.ndarray) -> None:
        self.zone = Zone.SHORT_POS
        self.zone_enter_idx = t
        self.zone_extreme_idx = t
        self.zone_extreme_val = float(closes[t])
        self.short_close = None
        self.pending_cross_idx = None
        self.pending_side = None
        self._open_side = "short"
        self._open_idx = t
        self._open_price = float(closes[t])

    def _close_position(self, t: int, closes: np.ndarray, times: np.ndarray) -> None:
        if self._open_side is None:
            return
        self.trades.append(
            Trade(
                side=self._open_side,
                entry_idx=self._open_idx,
                exit_idx=t,
                entry_price=self._open_price,
                exit_price=float(closes[t]),
                entry_ts=str(times[self._open_idx]),
                exit_ts=str(times[t]),
            )
        )
        self._open_side = None
        self._open_idx = -1
        self._open_price = 0.0

    def _on_long_obs(self, t: int, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, times: np.ndarray) -> None:
        if lows[t] < self.zone_extreme_val - 1e-12:
            self.zone_extreme_val = float(lows[t])
            self.zone_extreme_idx = t
            self.upper_rail = find_upper_rail(highs, lows, self.anchor_a, t, backend=self._backend())

        if self.upper_rail is None:
            return

        if self.pending_cross_idx is not None and self.pending_side == "long":
            if t == self.pending_cross_idx + 1:
                if closes[t] > self.upper_rail.value_at(t) + 1e-8:
                    obs_low = self.zone_extreme_idx
                    self._enter_long_pos(t, closes)
                    self.long_close = find_long_close_line(
                        highs, lows, obs_low, t, backend=self._backend()
                    )
                self.pending_cross_idx = None
                self.pending_side = None
            return

        if self._cross_up(self.upper_rail, t, closes):
            self.pending_cross_idx = t
            self.pending_side = "long"

    def _on_short_obs(self, t: int, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, times: np.ndarray) -> None:
        if highs[t] > self.zone_extreme_val + 1e-12:
            self.zone_extreme_val = float(highs[t])
            self.zone_extreme_idx = t
            self.lower_rail = find_lower_rail(highs, lows, self.anchor_a, t, backend=self._backend())

        if self.lower_rail is None:
            return

        if self.pending_cross_idx is not None and self.pending_side == "short":
            if t == self.pending_cross_idx + 1:
                if closes[t] < self.lower_rail.value_at(t) - 1e-8:
                    obs_high = self.zone_extreme_idx
                    self._enter_short_pos(t, closes)
                    self.short_close = find_short_close_line(
                        highs, lows, obs_high, t, backend=self._backend()
                    )
                self.pending_cross_idx = None
                self.pending_side = None
            return

        if self._cross_down(self.lower_rail, t, closes):
            self.pending_cross_idx = t
            self.pending_side = "short"

    def _on_long_pos(self, t: int, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, times: np.ndarray) -> None:
        if self.long_close is not None and self._cross_down(self.long_close, t, closes):
            self._close_position(t, closes, times)
            pos_start = self.zone_enter_idx
            self.zone = Zone.SHORT_OBS
            self.zone_enter_idx = t
            self.anchor_a = pos_start + int(np.argmax(highs[pos_start : t + 1]))
            self.zone_extreme_idx = t
            self.zone_extreme_val = float(highs[t])
            self.upper_rail = find_upper_rail(highs, lows, self.anchor_a, t, backend=self._backend())
            self.long_close = None
            return

        if highs[t] > self.zone_extreme_val + 1e-12 or self.long_close is None:
            self.zone_extreme_val = float(highs[t])
            self.zone_extreme_idx = t
            a = self.zone_enter_idx + int(np.argmin(lows[self.zone_enter_idx : t + 1]))
            self.long_close = find_long_close_line(highs, lows, a, t, backend=self._backend())

    def _on_short_pos(self, t: int, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, times: np.ndarray) -> None:
        if self.short_close is not None and self._cross_up(self.short_close, t, closes):
            self._close_position(t, closes, times)
            pos_start = self.zone_enter_idx
            self.zone = Zone.LONG_OBS
            self.zone_enter_idx = t
            self.anchor_a = pos_start + int(np.argmin(lows[pos_start : t + 1]))
            self.zone_extreme_idx = t
            self.zone_extreme_val = float(lows[t])
            self.lower_rail = find_lower_rail(highs, lows, self.anchor_a, t, backend=self._backend())
            self.short_close = None
            return

        if lows[t] < self.zone_extreme_val - 1e-12 or self.short_close is None:
            self.zone_extreme_val = float(lows[t])
            self.zone_extreme_idx = t
            a = self.zone_enter_idx + int(np.argmax(highs[self.zone_enter_idx : t + 1]))
            self.short_close = find_short_close_line(highs, lows, a, t, backend=self._backend())

    def step(self, t: int, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, times: np.ndarray) -> None:
        if self.zone == Zone.IDLE:
            self._try_init(t, highs, lows, closes)
        elif self.zone == Zone.LONG_OBS:
            self._on_long_obs(t, highs, lows, closes, times)
        elif self.zone == Zone.SHORT_OBS:
            self._on_short_obs(t, highs, lows, closes, times)
        elif self.zone == Zone.LONG_POS:
            self._on_long_pos(t, highs, lows, closes, times)
        elif self.zone == Zone.SHORT_POS:
            self._on_short_pos(t, highs, lows, closes, times)

        self.snapshots.append(
            ZoneSnapshot(
                idx=t,
                zone=self.zone,
                upper_rail=self.upper_rail,
                lower_rail=self.lower_rail,
                long_close=self.long_close,
                short_close=self.short_close,
            )
        )

    def run(self, df: pd.DataFrame) -> list[Trade]:
        highs = df[COL_HIGH].values.astype(np.float64)
        lows = df[COL_LOW].values.astype(np.float64)
        closes = df[COL_CLOSE].values.astype(np.float64)
        times = df[COL_TIME].values
        n = len(df)
        for t in range(n):
            self.step(t, highs, lows, closes, times)
        if self._open_side is not None:
            self._close_position(n - 1, closes, times)
        return self.trades


def run_zone_strategy(df: pd.DataFrame, *, config: EngineConfig | None = None) -> tuple[list[Trade], ZoneEngine]:
    engine = ZoneEngine(config=config or EngineConfig())
    trades = engine.run(df)
    return trades, engine
