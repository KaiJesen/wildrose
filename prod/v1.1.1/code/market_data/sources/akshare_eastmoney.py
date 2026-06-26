"""AkShare → 东方财富：A 股分钟 K 线与日线。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import akshare as ak
import pandas as pd

from market_data.base import KlineProvider
from market_data.http_utils import eastmoney_friendly_requests_get, retry_transient
from market_data.schema import normalize_ohlcv_df


class AkShareEastmoneyKlineProvider(KlineProvider):
    id = "akshare_em"
    description = "AkShare 封装东方财富（stock_zh_a_hist_min_em / stock_zh_a_hist）"

    INTERVAL_TO_EM_PERIOD = {
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "60m": "60",
    }

    def __init__(
        self,
        *,
        retries: int = 5,
        retry_base_sleep_s: float = 2.0,
        verbose_retry: bool = True,
    ) -> None:
        self._retries = max(1, retries)
        self._retry_base_sleep_s = retry_base_sleep_s
        self._verbose_retry = verbose_retry

    @property
    def supported_intervals(self) -> frozenset[str]:
        return frozenset((*self.INTERVAL_TO_EM_PERIOD.keys(), "1d"))

    def fetch_kline(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        *,
        adjust: str = "",
    ) -> pd.DataFrame:
        interval = interval.strip().lower()
        if interval not in self.supported_intervals:
            raise ValueError(
                f"不支持的周期 {interval!r}；"
                f"{self.id} 支持: {sorted(self.supported_intervals)}"
            )
        if interval == "1d":
            raw = self._fetch_daily_raw(symbol, start, end, adjust=adjust)
        else:
            raw = self._fetch_minute_raw(symbol, interval, start, end, adjust=adjust)
        return normalize_ohlcv_df(raw)

    def _with_retry(self, call: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        def on_retry(e: BaseException, attempt: int) -> None:
            if self._verbose_retry:
                wait = self._retry_base_sleep_s * (2**attempt)
                print(f"网络请求失败（{type(e).__name__}），{wait:.1f}s 后重试 ({attempt + 1}/{self._retries})…")

        return retry_transient(
            call,
            retries=self._retries,
            retry_base_sleep_s=self._retry_base_sleep_s,
            on_retry=on_retry if self._verbose_retry else lambda *_: None,
        )

    def _fetch_minute_raw(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        *,
        adjust: str,
    ) -> pd.DataFrame:
        period = self.INTERVAL_TO_EM_PERIOD[interval]
        fmt = "%Y-%m-%d %H:%M:%S"
        params = dict(
            symbol=symbol,
            start_date=start.strftime(fmt),
            end_date=end.strftime(fmt),
            period=period,
            adjust=adjust,
        )

        def call():
            with eastmoney_friendly_requests_get():
                return ak.stock_zh_a_hist_min_em(**params)

        return self._with_retry(call)

    def _fetch_daily_raw(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        adjust: str,
    ) -> pd.DataFrame:
        fmt = "%Y%m%d"
        params = dict(
            symbol=symbol,
            period="daily",
            start_date=start.strftime(fmt),
            end_date=end.strftime(fmt),
            adjust=adjust,
        )

        def call():
            with eastmoney_friendly_requests_get():
                return ak.stock_zh_a_hist(**params)

        return self._with_retry(call)
