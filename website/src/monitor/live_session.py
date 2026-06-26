"""Live paper-trading session: model inference + TradingEngine per new bar."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_data.schema import COL_CLOSE, COL_HIGH, COL_LOW, COL_OPEN, COL_TIME, COL_VOLUME
from transformer_kit.segment_dataset import prepare_bar_series
from trading_system.adapters.market_state_model import ModelSignalProvider
from trading_system.config import load_config
from trading_system.engine import Bar, TradingEngine
from trading_system.logger import TradeLogger

from monitor.adapters.registry import _resolve
from monitor.exporter import MonitorExporter, _serialize
from monitor.monitor_logger import _utc_ts
from monitor.schemas import BackendProfile

logger = logging.getLogger(__name__)

_DECISION_KEYS = (
    "price", "p_up", "p_down", "p_flat", "p_risk", "edge", "conf",
    "pred_cum_ret_5", "action", "reason_code", "blocked", "blocked_reason",
    "portfolio_equity", "position_ratio", "state",
)


def ohlcv_rows_to_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df[COL_TIME] = pd.to_datetime(df["ts"], utc=True)
    df[COL_OPEN] = df["open"].astype(float)
    df[COL_HIGH] = df["high"].astype(float)
    df[COL_LOW] = df["low"].astype(float)
    df[COL_CLOSE] = df["close"].astype(float)
    df[COL_VOLUME] = df["volume"].astype(float)
    return df.sort_values(COL_TIME).reset_index(drop=True)


class LivePaperSession:
    """Incrementally advance TradingEngine; export decisions on new closed bars."""

    def __init__(
        self,
        profile: BackendProfile,
        api_url: str,
        *,
        device: str = "cpu",
        trend_features: bool = True,
        trend_windows: tuple[int, ...] = (20, 60, 120),
    ) -> None:
        self.profile = profile
        self.device = device
        self.trend_features = trend_features
        self.trend_windows = trend_windows
        self.exporter = MonitorExporter(api_url, profile.id)
        cfg_path = _resolve(profile.config_path)
        if not cfg_path or not cfg_path.is_file():
            raise FileNotFoundError(f"config not found: {profile.config_path}")
        self.cfg = load_config(str(cfg_path))
        ckpt = _resolve(profile.checkpoint)
        if not ckpt or not ckpt.is_file():
            raise FileNotFoundError(f"checkpoint not found: {profile.checkpoint}")
        self.checkpoint = str(ckpt)
        self._engine: TradingEngine | None = None
        self._last_processed_idx: int | None = None
        self._last_df_len = 0
        self._decisions_seen = 0
        self._trades_seen = 0
        self._equity_seen = 0

    def _build_provider(self, df: pd.DataFrame) -> ModelSignalProvider:
        bundle = prepare_bar_series(
            df,
            use_trend_features=self.trend_features,
            trend_windows=self.trend_windows,
        )
        return ModelSignalProvider.from_checkpoint(
            checkpoint=self.checkpoint,
            bars=bundle.bars,
            df=df,
            context_bars=96,
            d_model=128,
            n_heads=4,
            trunk_layers=2,
            trend_features=self.trend_features,
            trend_windows=self.trend_windows,
            max_seg_len=32,
            max_segments=16,
            min_seg_len=4,
            num_codes=512,
            vq_beta=0.25,
            vq_inverse_freq_ema=True,
            cfg=self.cfg,
            device=self.device,
        )

    def _step(self, provider: ModelSignalProvider, df: pd.DataFrame, idx: int) -> None:
        open_px = df[COL_OPEN].to_numpy(dtype=np.float64)
        close = df[COL_CLOSE].to_numpy(dtype=np.float64)
        cur = Bar(
            idx=idx,
            ts=df[COL_TIME].iloc[idx],
            open=float(open_px[idx]),
            high=float(df[COL_HIGH].iloc[idx]),
            low=float(df[COL_LOW].iloc[idx]),
            close=float(close[idx]),
            atr=float(provider.atr[idx]),
        )
        nxt = Bar(
            idx=idx + 1,
            ts=df[COL_TIME].iloc[idx + 1],
            open=float(open_px[idx + 1]),
            high=float(df[COL_HIGH].iloc[idx + 1]),
            low=float(df[COL_LOW].iloc[idx + 1]),
            close=float(close[idx + 1]),
            atr=float(provider.atr[idx + 1]),
        )
        sig = provider.signal_at(idx)
        assert self._engine is not None
        self._engine.on_bar_close(sig, cur, nxt)

    def _export_new_records(self) -> None:
        assert self._engine is not None
        log = self._engine.logger
        sym, iv = self.profile.symbol, self.profile.interval
        for row in log.decisions[self._decisions_seen :]:
            payload = {k: row.get(k) for k in _DECISION_KEYS}
            self.exporter.emit("decision", _utc_ts(row.get("ts")), payload, symbol=sym, interval=iv)
        self._decisions_seen = len(log.decisions)
        for row in log.trades[self._trades_seen :]:
            self.exporter.emit(
                "trade",
                _utc_ts(row.get("exit_ts") or row.get("entry_ts")),
                _serialize(row),
                symbol=sym,
                interval=iv,
            )
        self._trades_seen = len(log.trades)
        for row in log.equity_curve[self._equity_seen :]:
            self.exporter.emit(
                "equity",
                _utc_ts(row.get("ts")),
                {"equity": row.get("equity")},
                symbol=sym,
                interval=iv,
            )
        self._equity_seen = len(log.equity_curve)

    def on_bars(self, rows: list[dict]) -> dict[str, Any] | None:
        df = ohlcv_rows_to_df(rows)
        n = len(df)
        if n < 3:
            return None

        provider = self._build_provider(df)
        start = provider.context_bars + 1
        end_exclusive = n - 1
        if end_exclusive <= start:
            return None

        if self._engine is None or n < self._last_df_len:
            logger.info("live session warmup bars=%d", n)
            self._engine = TradingEngine(self.cfg, TradeLogger(out_dir=Path("/tmp/wildrose_live")))
            self._decisions_seen = self._trades_seen = self._equity_seen = 0
            for i in range(start, end_exclusive):
                self._step(provider, df, i)
            self._last_processed_idx = end_exclusive - 1
            self._last_df_len = n
            return {"warmup": True, "bars": n, "processed_idx": self._last_processed_idx}

        if n == self._last_df_len:
            return None

        for i in range((self._last_processed_idx or start) + 1, end_exclusive):
            self._step(provider, df, i)
        self._export_new_records()
        self._last_processed_idx = end_exclusive - 1
        self._last_df_len = n
        ts = pd.Timestamp(df[COL_TIME].iloc[self._last_processed_idx]).strftime("%Y-%m-%dT%H:%M:%SZ")
        return {
            "warmup": False,
            "ts": ts,
            "processed_idx": self._last_processed_idx,
            "close": float(df[COL_CLOSE].iloc[self._last_processed_idx]),
        }
