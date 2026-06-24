"""Binance USDT-M 永续合约 K 线（fapi /fapi/v1/klines）。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import time
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


class BinanceFuturesKlineProvider(KlineProvider):
    """
    Binance USDT-M Futures K 线（永续 + 交割），底层接口 GET /fapi/v1/klines。

    - symbol：合约符号，如 DOGEUSDT、BTCUSDT、ETHUSDT_240927（交割合约）
    - interval：1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,1M
      为了和 A 股套件保持一致，也接受 60m（自动映射成 1h）
    - 长区间会自动按 1500 根/批分页
    """

    id = "binance_futures"
    description = "Binance USDT-M Futures (fapi /fapi/v1/klines)"

    # 标准 interval（含 A 股 60m 别名）→ Binance 接口的 interval
    _INTERVAL_ALIAS = {
        "60m": "1h",
        "120m": "2h",
        "240m": "4h",
        "1day": "1d",
    }
    _BINANCE_INTERVALS = frozenset(
        ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M")
    )

    # 单次最多返回 1500 根
    MAX_LIMIT = 1500

    DEFAULT_BASE_URL = "https://fapi.binance.com"
    # 部分网络环境下可改用其它端点，如 https://fapi.binance.com、https://www.binance.com 等
    FALLBACK_BASE_URL = "https://www.binance.com"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        retries: int = 5,
        retry_base_sleep_s: float = 2.0,
        request_timeout: float = 15.0,
        verbose_retry: bool = True,
        proxies: dict[str, str] | None = None,
        trust_env: bool = True,
        page_sleep_s: float = 0.0,
    ) -> None:
        self._base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self._retries = max(1, retries)
        self._retry_base_sleep_s = retry_base_sleep_s
        self._request_timeout = request_timeout
        self._verbose_retry = verbose_retry
        self._page_sleep_s = max(0.0, page_sleep_s)

        # 一次性构造 Session，复用连接、统一代理/UA/header
        self._session = requests.Session()
        # trust_env=True 时会自动读 HTTP_PROXY / HTTPS_PROXY / ALL_PROXY 环境变量
        self._session.trust_env = trust_env
        self._session.headers.update(
            {"User-Agent": "market-data-kit/0.1 (+binance_futures)"}
        )
        if proxies:
            # 显式 proxies 覆盖 trust_env 的环境变量
            self._session.proxies.update(proxies)

    @property
    def supported_intervals(self) -> frozenset[str]:
        return self._BINANCE_INTERVALS | frozenset(self._INTERVAL_ALIAS.keys())

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
            # 加密货币没有复权概念，显式提示而不是静默忽略
            raise ValueError(f"{self.id} 不支持复权参数 adjust={adjust!r}")

        norm_interval = self._normalize_interval(interval)
        sym = symbol.strip().upper()
        if not sym:
            raise ValueError("symbol 不能为空")

        start_ms = self._to_ms(start)
        end_ms = self._to_ms(end)
        if end_ms <= start_ms:
            raise ValueError(f"end 必须晚于 start：start={start}, end={end}")

        try:
            rows = self._fetch_paginated(sym, norm_interval, start_ms, end_ms)
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"无法连接 {self._base_url}（{e}）\n"
                "排查建议：\n"
                "  1) 检查网络/代理是否能访问 fapi.binance.com（GFW 区域通常需要代理）；\n"
                "  2) 设置环境变量 HTTPS_PROXY=http://<host>:<port>，本类默认 trust_env=True 会自动读取；\n"
                "  3) 或显式传 proxies={'https': '...'}，SOCKS5 需要先 pip install 'requests[socks]'；\n"
                "  4) 完全无法走代理时，可改用历史归档 provider：binance_vision（直连免代理）。"
            ) from e
        raw = self._rows_to_dataframe(rows)
        return normalize_ohlcv_df(raw)

    def _normalize_interval(self, interval: str) -> str:
        key = interval.strip().lower()
        # 1M（月线）大小写敏感，单独处理
        if key == "1mo" or interval.strip() == "1M":
            return "1M"
        key = self._INTERVAL_ALIAS.get(key, key)
        if key not in self._BINANCE_INTERVALS:
            raise ValueError(
                f"不支持的周期 {interval!r}；{self.id} 支持: {sorted(self.supported_intervals)}"
            )
        return key

    @staticmethod
    def _to_ms(dt: datetime) -> int:
        if dt.tzinfo is None:
            # 视作本地时间转 UTC
            dt = dt.astimezone()
        return int(dt.astimezone(timezone.utc).timestamp() * 1000)

    def _fetch_paginated(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list[list[Any]]:
        all_rows: list[list[Any]] = []
        cursor = start_ms
        last_open: int | None = None

        while cursor < end_ms:
            batch = self._fetch_one_batch(symbol, interval, cursor, end_ms, self.MAX_LIMIT)
            if not batch:
                break

            # 防御：避免极端情况下死循环（接口返回同一根 K 线）
            first_open = int(batch[0][0])
            if last_open is not None and first_open <= last_open:
                # 已经退化，强行向前推
                cursor = last_open + 1
                continue

            all_rows.extend(batch)
            last_open = int(batch[-1][0])

            if len(batch) < self.MAX_LIMIT:
                # 这一批没拉满，说明已经到边界
                break
            cursor = last_open + 1
            if self._page_sleep_s:
                time.sleep(self._page_sleep_s)

        return all_rows

    def _fetch_one_batch(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        limit: int,
    ) -> list[list[Any]]:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": limit,
        }

        def call() -> list[list[Any]]:
            url = f"{self._base_url}/fapi/v1/klines"
            resp = self._session.get(url, params=params, timeout=self._request_timeout)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                raise RuntimeError(f"非预期返回（{type(data).__name__}）: {data!r}")
            return data

        def on_retry(e: BaseException, attempt: int) -> None:
            if self._verbose_retry:
                wait = self._retry_base_sleep_s * (2**attempt)
                print(
                    f"[{self.id}] 网络/接口失败（{type(e).__name__}: {e}），"
                    f"{wait:.1f}s 后重试 ({attempt + 1}/{self._retries})…"
                )

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
            on_retry=on_retry if self._verbose_retry else lambda *_: None,
        )

    @staticmethod
    def _rows_to_dataframe(rows: list[list[Any]]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(
                columns=[
                    COL_TIME,
                    COL_OPEN,
                    COL_HIGH,
                    COL_LOW,
                    COL_CLOSE,
                    COL_VOLUME,
                    COL_CLOSE_TIME,
                    COL_QUOTE_VOLUME,
                    COL_TRADES,
                    COL_TAKER_BUY_BASE,
                    COL_TAKER_BUY_QUOTE,
                    COL_AMOUNT,
                ]
            )

        # Binance kline schema:
        # [openTime, open, high, low, close, volume,
        #  closeTime, quoteAssetVolume, numberOfTrades,
        #  takerBuyBaseAssetVolume, takerBuyQuoteAssetVolume, ignore]
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
        df = df.drop(columns=["_ignore"])

        df[COL_TIME] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
        df[COL_CLOSE_TIME] = pd.to_datetime(df["close_time_ms"], unit="ms", utc=True)
        df = df.drop(columns=["open_time_ms", "close_time_ms"])

        numeric_cols = [
            COL_OPEN,
            COL_HIGH,
            COL_LOW,
            COL_CLOSE,
            COL_VOLUME,
            COL_QUOTE_VOLUME,
            COL_TAKER_BUY_BASE,
            COL_TAKER_BUY_QUOTE,
        ]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[COL_TRADES] = pd.to_numeric(df[COL_TRADES], errors="coerce").astype("Int64")

        # 与 A 股 schema 对齐：amount 在加密语境下等价于以计价币计的成交额
        df[COL_AMOUNT] = df[COL_QUOTE_VOLUME]
        return df
