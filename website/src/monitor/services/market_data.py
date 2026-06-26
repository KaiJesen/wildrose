from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from monitor.db import connect, insert_event, set_runner_meta
from monitor.settings import REPO_ROOT

logger = logging.getLogger(__name__)

# market_data lives in repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from market_data.schema import COL_CLOSE, COL_CLOSE_TIME, COL_HIGH, COL_LOW, COL_OPEN, COL_TIME, COL_VOLUME  # noqa: E402
from market_data.sources.binance_data_api import BinanceDataApiKlineProvider  # noqa: E402
from market_data.sources.binance_futures import BinanceFuturesKlineProvider  # noqa: E402
from market_data.sources.binance_vision import BinanceVisionKlineProvider  # noqa: E402

# 尾部 K 线：优先 fapi 永续，不可达时用 data-api 现货 REST（国内可直连）
_TAIL_LIMIT = 72
_HISTORY_DAYS = 30


def _filter_closed_bars(df: pd.DataFrame) -> pd.DataFrame:
    """去掉尚未收盘的最后一根 K 线。"""
    if df.empty:
        return df
    now = pd.Timestamp.now(tz="UTC")
    if COL_CLOSE_TIME in df.columns:
        return df[df[COL_CLOSE_TIME] <= now].reset_index(drop=True)
    return df.iloc[:-1].reset_index(drop=True)


def _merge_ohlcv(primary: pd.DataFrame, supplemental: pd.DataFrame) -> pd.DataFrame:
    """按时间合并；primary（vision 永续归档）优先于 supplemental（尾部 REST）。"""
    if primary.empty:
        return supplemental.reset_index(drop=True)
    if supplemental.empty:
        return primary.reset_index(drop=True)
    out = pd.concat([primary, supplemental], ignore_index=True)
    out = out.sort_values(COL_TIME).drop_duplicates(subset=[COL_TIME], keep="first")
    return out.reset_index(drop=True)


class MarketDataService:
    def __init__(self, backend_id: str, symbol: str = "BTCUSDT", interval: str = "1h") -> None:
        self.backend_id = backend_id
        self.symbol = symbol
        self.interval = interval
        self._archive = BinanceVisionKlineProvider(
            verbose=False, retries=2, request_timeout=20.0,
        )
        self._data_api = BinanceDataApiKlineProvider(
            retries=2, request_timeout=12.0,
        )
        self._last_tail_source: str | None = None

    @property
    def last_tail_source(self) -> str | None:
        return self._last_tail_source

    def _fetch_tail(self) -> pd.DataFrame:
        """最近几根 K 线：fapi 永续 → data-api 现货。"""
        self._last_tail_source = None
        try:
            df = BinanceFuturesKlineProvider.fetch_recent_with_fallback(
                self.symbol, self.interval, limit=_TAIL_LIMIT,
                request_timeout=4.0, retries=1,
            )
            if not df.empty:
                self._last_tail_source = "binance_futures"
                return _filter_closed_bars(df)
        except Exception as e:
            logger.info("fapi tail unavailable: %s", e)

        try:
            df = self._data_api.fetch_recent_klines(
                self.symbol, self.interval, limit=_TAIL_LIMIT,
            )
            if not df.empty:
                self._last_tail_source = "binance_data_api"
                logger.info(
                    "using data-api spot tail for %s (fapi unreachable); "
                    "prices track USDT-M closely",
                    self.symbol,
                )
                return _filter_closed_bars(df)
        except Exception as e:
            logger.warning("data-api tail failed: %s", e)
        return pd.DataFrame()

    def _fetch_full_history(self, limit: int = 200) -> pd.DataFrame:
        """vision 永续归档 + REST 尾部（冷启动 / 回填）。"""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=_HISTORY_DAYS)

        df_hist = pd.DataFrame()
        try:
            df_hist = self._archive.fetch_kline(self.symbol, self.interval, start, end)
        except Exception as e:
            logger.warning("vision history fetch failed: %s", e)

        df_tail = self._fetch_tail()
        df = _merge_ohlcv(df_hist, df_tail)
        if df.empty:
            try:
                live = BinanceFuturesKlineProvider(
                    verbose_retry=False, retries=2, request_timeout=8.0,
                )
                df = live.fetch_kline(self.symbol, self.interval, start, end)
                self._last_tail_source = "binance_futures"
            except Exception as e:
                logger.warning("full fapi fetch failed: %s", e)
        if df.empty:
            return df
        return _filter_closed_bars(df).tail(limit).reset_index(drop=True)

    def _fetch_recent(self, limit: int = 200) -> pd.DataFrame:
        """常规轮询只拉尾部 REST；库内不足时再全量回填。"""
        with connect() as conn:
            n = conn.execute(
                "SELECT COUNT(*) AS c FROM ohlcv_bars WHERE backend_id=?",
                (self.backend_id,),
            ).fetchone()["c"]
        if n < 48:
            return self._fetch_full_history(limit)
        df_tail = self._fetch_tail()
        if df_tail.empty:
            return self._fetch_full_history(limit)
        return df_tail.tail(limit).reset_index(drop=True)

    def poll_and_store(self) -> str | None:
        """Fetch latest bars from Binance futures; return new bar ts if any."""
        try:
            df = self._fetch_recent()
        except Exception as e:
            logger.warning("market data fetch failed: %s", e)
            return None
        if df.empty:
            return None

        latest_known = self._latest_stored_ts()
        new_bar_ts: str | None = None

        with connect() as conn:
            for _, row in df.iterrows():
                ts = pd.Timestamp(row[COL_TIME]).strftime("%Y-%m-%dT%H:%M:%SZ")
                conn.execute(
                    """
                    INSERT OR REPLACE INTO ohlcv_bars
                    (backend_id, ts, open, high, low, close, volume)
                    VALUES (?,?,?,?,?,?,?)
                    """,
                    (
                        self.backend_id, ts,
                        float(row[COL_OPEN]), float(row[COL_HIGH]),
                        float(row[COL_LOW]), float(row[COL_CLOSE]),
                        float(row[COL_VOLUME]),
                    ),
                )
                if latest_known is None or ts > latest_known:
                    new_bar_ts = ts

            if new_bar_ts:
                set_runner_meta(conn, self.backend_id, latest_bar_ts=new_bar_ts)
                insert_event(
                    conn,
                    {
                        "backend_id": self.backend_id,
                        "event_type": "bar_close",
                        "ts": new_bar_ts,
                        "symbol": self.symbol,
                        "interval": self.interval,
                        "payload": {
                            "open": float(df.iloc[-1][COL_OPEN]),
                            "high": float(df.iloc[-1][COL_HIGH]),
                            "low": float(df.iloc[-1][COL_LOW]),
                            "close": float(df.iloc[-1][COL_CLOSE]),
                            "volume": float(df.iloc[-1][COL_VOLUME]),
                            "tail_source": self._last_tail_source,
                        },
                    },
                )
        return new_bar_ts

    def _latest_stored_ts(self) -> str | None:
        with connect() as conn:
            row = conn.execute(
                "SELECT MAX(ts) AS ts FROM ohlcv_bars WHERE backend_id=?",
                (self.backend_id,),
            ).fetchone()
            return row["ts"] if row and row["ts"] else None

    def get_ohlcv(self, limit: int = 500) -> list[dict]:
        with connect() as conn:
            rows = conn.execute(
                """
                SELECT ts, open, high, low, close, volume
                FROM ohlcv_bars WHERE backend_id=?
                ORDER BY ts DESC LIMIT ?
                """,
                (self.backend_id, limit),
            ).fetchall()
        out = [dict(r) for r in reversed(rows)]
        return out
