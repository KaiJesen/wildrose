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

from market_data.schema import COL_CLOSE, COL_HIGH, COL_LOW, COL_OPEN, COL_TIME, COL_VOLUME  # noqa: E402
from market_data.sources.binance_futures import BinanceFuturesKlineProvider  # noqa: E402
from market_data.sources.binance_vision import BinanceVisionKlineProvider  # noqa: E402


class MarketDataService:
    def __init__(self, backend_id: str, symbol: str = "BTCUSDT", interval: str = "1h") -> None:
        self.backend_id = backend_id
        self.symbol = symbol
        self.interval = interval
        self._live = BinanceFuturesKlineProvider(
            verbose_retry=False, retries=2, request_timeout=8.0,
        )
        self._archive = BinanceVisionKlineProvider(
            verbose=False, retries=2, request_timeout=20.0,
        )

    def _fetch_recent(self, limit: int = 200) -> pd.DataFrame:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=30)
        df = pd.DataFrame()
        try:
            df = self._live.fetch_kline(self.symbol, self.interval, start, end)
        except Exception as e:
            logger.info("fapi unavailable, fallback to binance_vision: %s", e)
        if df.empty:
            df = self._archive.fetch_kline(self.symbol, self.interval, start, end)
        if df.empty:
            return df
        return df.tail(limit).reset_index(drop=True)

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
