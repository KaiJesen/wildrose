"""Binance 公开 Data API（data-api.binance.vision）现货 REST K 线。

国内多数网络可直连，用于 fapi 不可达时补齐最近几根 K 线尾部。
注意：底层为现货 /api/v3/klines，与 USDT-M 永续 OHLC 极为接近但非同一合约。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from market_data.base import KlineProvider
from market_data.http_utils import retry_transient
from market_data.schema import (
    COL_AMOUNT,
    COL_CLOSE,
    COL_CLOSE_TIME,
    COL_HIGH,
    COL_LOW,
    COL_OPEN,
    COL_QUOTE_VOLUME,
    COL_TAKER_BUY_BASE,
    COL_TAKER_BUY_QUOTE,
    COL_TIME,
    COL_TRADES,
    COL_VOLUME,
    normalize_ohlcv_df,
)


class BinanceDataApiKlineProvider(KlineProvider):
    """GET https://data-api.binance.vision/api/v3/klines（现货，免代理）。"""

    id = "binance_data_api"
    description = "Binance Data API (data-api.binance.vision, spot REST tail)"

    _INTERVAL_ALIAS = {"60m": "1h", "1day": "1d"}
    _INTERVALS = frozenset(
        ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M")
    )
    MAX_LIMIT = 1000
    DEFAULT_BASE_URL = "https://data-api.binance.vision"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        retries: int = 3,
        retry_base_sleep_s: float = 1.0,
        request_timeout: float = 12.0,
        trust_env: bool = True,
    ) -> None:
        self._base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self._retries = max(1, retries)
        self._retry_base_sleep_s = retry_base_sleep_s
        self._request_timeout = request_timeout
        self._session = requests.Session()
        self._session.trust_env = trust_env
        self._session.headers.update({"User-Agent": "market-data-kit/0.1 (+binance_data_api)"})

    @property
    def supported_intervals(self) -> frozenset[str]:
        return self._INTERVALS | frozenset(self._INTERVAL_ALIAS.keys())

    def fetch_kline(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        *,
        adjust: str = "",
    ) -> pd.DataFrame:
        if adjust:
            raise ValueError(f"{self.id} 不支持复权参数 adjust={adjust!r}")
        norm = self._normalize_interval(interval)
        sym = symbol.strip().upper()
        start_ms = self._to_ms(start)
        end_ms = self._to_ms(end)
        if end_ms <= start_ms:
            raise ValueError(f"end 必须晚于 start：start={start}, end={end}")

        rows: list[list[Any]] = []
        cursor = start_ms
        while cursor < end_ms:
            batch = self._get(
                {
                    "symbol": sym,
                    "interval": norm,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": self.MAX_LIMIT,
                }
            )
            if not batch:
                break
            rows.extend(batch)
            last_open = int(batch[-1][0])
            if len(batch) < self.MAX_LIMIT:
                break
            cursor = last_open + 1

        return normalize_ohlcv_df(self._rows_to_df(rows))

    def fetch_recent_klines(self, symbol: str, interval: str, *, limit: int = 100) -> pd.DataFrame:
        """拉最近 limit 根 K 线（含当前未收盘的一根，需调用方过滤）。"""
        norm = self._normalize_interval(interval)
        sym = symbol.strip().upper()
        lim = max(1, min(int(limit), self.MAX_LIMIT))
        rows = self._get({"symbol": sym, "interval": norm, "limit": lim})
        return normalize_ohlcv_df(self._rows_to_df(rows))

    def _get(self, params: dict[str, Any]) -> list[list[Any]]:
        def call() -> list[list[Any]]:
            url = f"{self._base_url}/api/v3/klines"
            resp = self._session.get(url, params=params, timeout=self._request_timeout)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                raise RuntimeError(f"非预期返回: {data!r}")
            return data

        return retry_transient(
            call,
            retries=self._retries,
            retry_base_sleep_s=self._retry_base_sleep_s,
            exceptions=(
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.HTTPError,
            ),
            on_retry=lambda *_: None,
        )

    def _normalize_interval(self, interval: str) -> str:
        key = interval.strip()
        if key == "1M":
            return "1M"
        key = self._INTERVAL_ALIAS.get(key.lower(), key.lower())
        if key not in self._INTERVALS:
            raise ValueError(f"不支持的周期 {interval!r}")
        return key

    @staticmethod
    def _to_ms(dt: datetime) -> int:
        if dt.tzinfo is None:
            dt = dt.astimezone()
        return int(dt.astimezone(timezone.utc).timestamp() * 1000)

    @staticmethod
    def _rows_to_df(rows: list[list[Any]]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(
            rows,
            columns=[
                "open_time_ms",
                COL_OPEN,
                COL_HIGH,
                COL_LOW,
                COL_CLOSE,
                COL_VOLUME,
                "close_time_ms",
                COL_QUOTE_VOLUME,
                COL_TRADES,
                COL_TAKER_BUY_BASE,
                COL_TAKER_BUY_QUOTE,
                "_ignore",
            ],
        )
        df[COL_TIME] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
        df[COL_CLOSE_TIME] = pd.to_datetime(df["close_time_ms"], unit="ms", utc=True)
        for c in (COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME, COL_QUOTE_VOLUME):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[COL_AMOUNT] = df[COL_QUOTE_VOLUME]
        return df.drop(columns=["open_time_ms", "close_time_ms", "_ignore"])
