"""Binance Vision 历史归档 K 线（公开 ZIP，免代理可达）。

URL 模式：
    monthly: https://data.binance.vision/data/futures/um/monthly/klines/{SYMBOL}/{interval}/
             {SYMBOL}-{interval}-{YYYY-MM}.zip
    daily  : https://data.binance.vision/data/futures/um/daily/klines/{SYMBOL}/{interval}/
             {SYMBOL}-{interval}-{YYYY-MM-DD}.zip

ZIP 内是 CSV：
    open_time, open, high, low, close, volume, close_time,
    quote_volume, count, taker_buy_volume, taker_buy_quote_volume, ignore

特点：
- **不需要代理**：很多 GFW 网络都能直连 data.binance.vision
- **不存在"非会员只能下 4h"的限制**：所有 interval（1m..1mo）都对所有人开放下载
- **覆盖范围**：DOGEUSDT 永续合约从 2020-07 至今
- **时延**：daily 一般延迟 1~2 天；最新一两天的实时数据需要 fapi REST
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import io
from typing import Literal
import zipfile

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

ContractType = Literal["um", "cm", "spot"]

# CSV schema（Binance Vision 自 2020 年起列结构稳定）
_CSV_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]


class BinanceVisionKlineProvider(KlineProvider):
    """从 data.binance.vision 拉历史 K 线归档 ZIP。"""

    id = "binance_vision"
    description = "Binance Vision archive (data.binance.vision, monthly+daily ZIP)"

    # 标准/A 股 interval → vision 接口 interval
    _INTERVAL_ALIAS = {
        "60m": "1h",
        "120m": "2h",
        "240m": "4h",
        "1day": "1d",
        # vision 月线写作 1mo（注意：fapi 是 1M）
        "1M": "1mo",
    }
    _VISION_INTERVALS = frozenset(
        ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1mo")
    )

    DEFAULT_BASE_URL = "https://data.binance.vision"

    def __init__(
        self,
        *,
        contract_type: ContractType = "um",
        base_url: str | None = None,
        retries: int = 5,
        retry_base_sleep_s: float = 1.5,
        request_timeout: float = 30.0,
        verbose: bool = True,
        proxies: dict[str, str] | None = None,
        trust_env: bool = True,
        prefer_monthly: bool = True,
    ) -> None:
        """
        :param contract_type: 'um' = USDT-M 永续/交割（默认）；'cm' = 币本位；'spot' = 现货
        :param prefer_monthly: 完整月份用 monthly zip（更省请求），尾部不完整月份退化到 daily
        """
        if contract_type not in ("um", "cm", "spot"):
            raise ValueError(f"contract_type 仅支持 um/cm/spot，收到 {contract_type!r}")
        self._contract_type = contract_type
        self._base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self._retries = max(1, retries)
        self._retry_base_sleep_s = retry_base_sleep_s
        self._request_timeout = request_timeout
        self._verbose = verbose
        self._prefer_monthly = prefer_monthly

        self._session = requests.Session()
        self._session.trust_env = trust_env
        self._session.headers.update(
            {"User-Agent": "market-data-kit/0.1 (+binance_vision)"}
        )
        if proxies:
            self._session.proxies.update(proxies)

    @property
    def supported_intervals(self) -> frozenset[str]:
        return self._VISION_INTERVALS | frozenset(self._INTERVAL_ALIAS.keys())

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
        sym = symbol.strip().upper()
        if not sym:
            raise ValueError("symbol 不能为空")
        norm_interval = self._normalize_interval(interval)

        start_utc = self._to_utc(start)
        end_utc = self._to_utc(end)
        if end_utc <= start_utc:
            raise ValueError(f"end 必须晚于 start：start={start}, end={end}")

        chunks = self._fetch_archives(sym, norm_interval, start_utc, end_utc)
        if not chunks:
            return normalize_ohlcv_df(pd.DataFrame())

        df = pd.concat(chunks, ignore_index=True)
        df = self._post_process(df, start_utc, end_utc)
        return normalize_ohlcv_df(df)

    def _normalize_interval(self, interval: str) -> str:
        raw = interval.strip()
        # 月线大小写敏感
        if raw == "1M":
            return "1mo"
        key = raw.lower()
        key = self._INTERVAL_ALIAS.get(key, key)
        if key not in self._VISION_INTERVALS:
            raise ValueError(
                f"不支持的周期 {interval!r}；{self.id} 支持: {sorted(self.supported_intervals)}"
            )
        return key

    @staticmethod
    def _to_utc(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.astimezone().astimezone(timezone.utc)
        return dt.astimezone(timezone.utc)

    # ----- 归档下载与拼接 ---------------------------------------------------

    def _fetch_archives(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> list[pd.DataFrame]:
        chunks: list[pd.DataFrame] = []
        # 拆成 (monthly_full_months, daily_days) 两段
        monthly_months, daily_days = self._plan_files(start, end)

        if self._verbose:
            print(
                f"[{self.id}] plan: monthly={len(monthly_months)} 个月份, "
                f"daily={len(daily_days)} 天 (symbol={symbol}, interval={interval})"
            )

        for ym in monthly_months:
            url = self._build_url(symbol, interval, "monthly", ym)
            df = self._download_csv_zip(url, required=False)
            if df is not None and not df.empty:
                chunks.append(df)

        for ymd in daily_days:
            url = self._build_url(symbol, interval, "daily", ymd)
            df = self._download_csv_zip(url, required=False)
            if df is not None and not df.empty:
                chunks.append(df)

        return chunks

    def _plan_files(
        self,
        start: datetime,
        end: datetime,
    ) -> tuple[list[str], list[str]]:
        """
        返回 (monthly_keys, daily_keys)：
        - monthly_keys: ['2024-01', '2024-02', ...] 仅包含完全落入 [start, end] 内的整月
        - daily_keys: ['2024-03-01', ...] 用于 monthly 没覆盖到的日期（区间两端的不完整月）
        """
        # 把 start/end 都规整到 UTC 日期边界
        s_date = start.date()
        e_date = end.date()

        if not self._prefer_monthly:
            # 直接全部走 daily
            days: list[str] = []
            d = s_date
            while d <= e_date:
                days.append(d.strftime("%Y-%m-%d"))
                d += timedelta(days=1)
            return [], days

        monthly: list[str] = []
        daily: list[str] = []

        # 一个月内（start.month==end.month）：直接 daily
        if (s_date.year, s_date.month) == (e_date.year, e_date.month):
            d = s_date
            while d <= e_date:
                daily.append(d.strftime("%Y-%m-%d"))
                d += timedelta(days=1)
            return monthly, daily

        # 1) 起始月：start 不在月初 → 起始月走 daily 到月底
        first_month_end = self._month_end(s_date)
        if s_date.day == 1:
            monthly.append(f"{s_date.year:04d}-{s_date.month:02d}")
        else:
            d = s_date
            while d <= first_month_end:
                daily.append(d.strftime("%Y-%m-%d"))
                d += timedelta(days=1)

        # 2) 中间月份：完整月走 monthly
        cursor = self._month_start_after(s_date)
        last_full_month_start = self._month_start(e_date)
        while cursor < last_full_month_start:
            monthly.append(f"{cursor.year:04d}-{cursor.month:02d}")
            cursor = self._month_start_after(cursor)

        # 3) 结束月：end 不在月末 → 结束月走 daily 从 1 号到 end
        last_month_start = self._month_start(e_date)
        if e_date == self._month_end(e_date):
            monthly.append(f"{e_date.year:04d}-{e_date.month:02d}")
        else:
            d = last_month_start
            while d <= e_date:
                daily.append(d.strftime("%Y-%m-%d"))
                d += timedelta(days=1)

        # 去重并排序
        monthly = sorted(set(monthly))
        daily = sorted(set(daily))
        return monthly, daily

    @staticmethod
    def _month_start(d):
        return d.replace(day=1)

    @staticmethod
    def _month_end(d):
        # next month day1 - 1 day
        if d.month == 12:
            nxt = d.replace(year=d.year + 1, month=1, day=1)
        else:
            nxt = d.replace(month=d.month + 1, day=1)
        return nxt - timedelta(days=1)

    @staticmethod
    def _month_start_after(d):
        ms = d.replace(day=1)
        if ms.month == 12:
            return ms.replace(year=ms.year + 1, month=1, day=1)
        return ms.replace(month=ms.month + 1, day=1)

    def _build_url(
        self,
        symbol: str,
        interval: str,
        granularity: str,  # 'monthly' | 'daily'
        period_key: str,   # '2024-01' or '2024-03-15'
    ) -> str:
        # spot vs um/cm 路径不同
        if self._contract_type == "spot":
            base_path = f"data/spot/{granularity}/klines/{symbol}/{interval}"
        else:
            base_path = f"data/futures/{self._contract_type}/{granularity}/klines/{symbol}/{interval}"
        fname = f"{symbol}-{interval}-{period_key}.zip"
        return f"{self._base_url}/{base_path}/{fname}"

    # ----- 实际 HTTP + ZIP/CSV 解析 ----------------------------------------

    def _download_csv_zip(self, url: str, *, required: bool) -> pd.DataFrame | None:
        def call() -> pd.DataFrame | None:
            resp = self._session.get(url, timeout=self._request_timeout, stream=False)
            if resp.status_code == 404:
                if self._verbose:
                    print(f"[{self.id}] 404 跳过：{url}")
                return None
            resp.raise_for_status()
            return self._parse_zip_bytes(resp.content)

        def on_retry(e: BaseException, attempt: int) -> None:
            if self._verbose:
                wait = self._retry_base_sleep_s * (2**attempt)
                print(
                    f"[{self.id}] 下载失败（{type(e).__name__}: {e}），"
                    f"{wait:.1f}s 后重试 ({attempt + 1}/{self._retries}) — {url}"
                )

        try:
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
                on_retry=on_retry if self._verbose else lambda *_: None,
            )
        except requests.exceptions.HTTPError as e:
            # 404 已经吞掉了，这里只剩 5xx 之类
            if not required:
                if self._verbose:
                    print(f"[{self.id}] 跳过（{e}）：{url}")
                return None
            raise

    @staticmethod
    def _parse_zip_bytes(data: bytes) -> pd.DataFrame:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            # 期望只有一个 csv 文件
            names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not names:
                raise RuntimeError(f"ZIP 内未发现 CSV：{zf.namelist()}")
            with zf.open(names[0]) as f:
                raw = f.read().decode("utf-8", errors="replace")

        # 自动判断有无表头：首行能转成 int 就是无表头
        first_line = raw.split("\n", 1)[0].strip()
        first_cell = first_line.split(",", 1)[0]
        has_header = not first_cell.lstrip("-").isdigit()

        if has_header:
            df = pd.read_csv(io.StringIO(raw))
            # 旧 dump 有时少列，按现有列名匹配标准列
            df = df.rename(columns={c: c.strip() for c in df.columns})
        else:
            df = pd.read_csv(io.StringIO(raw), header=None, names=_CSV_COLUMNS[: len(_CSV_COLUMNS)])
        return df

    # ----- 后处理：标准列、时间过滤、数值类型 ----------------------------

    @staticmethod
    def _post_process(df: pd.DataFrame, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
        # 兼容老归档可能列数不齐
        for c in _CSV_COLUMNS:
            if c not in df.columns:
                df[c] = pd.NA

        df = df.copy()
        df[COL_TIME] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df[COL_CLOSE_TIME] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        rename = {
            "open": COL_OPEN,
            "high": COL_HIGH,
            "low": COL_LOW,
            "close": COL_CLOSE,
            "volume": COL_VOLUME,
            "quote_volume": COL_QUOTE_VOLUME,
            "count": COL_TRADES,
            "taker_buy_volume": COL_TAKER_BUY_BASE,
            "taker_buy_quote_volume": COL_TAKER_BUY_QUOTE,
        }
        df = df.rename(columns=rename)

        keep = [
            COL_TIME,
            COL_OPEN,
            COL_HIGH,
            COL_LOW,
            COL_CLOSE,
            COL_VOLUME,
            COL_QUOTE_VOLUME,
            COL_TRADES,
            COL_TAKER_BUY_BASE,
            COL_TAKER_BUY_QUOTE,
            COL_CLOSE_TIME,
        ]
        df = df[[c for c in keep if c in df.columns]]

        for c in (COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME, COL_QUOTE_VOLUME,
                  COL_TAKER_BUY_BASE, COL_TAKER_BUY_QUOTE):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if COL_TRADES in df.columns:
            df[COL_TRADES] = pd.to_numeric(df[COL_TRADES], errors="coerce").astype("Int64")

        # 区间过滤（左闭右闭，按 open_time 比较；右端点最多比较到 end_utc 起点的那根）
        df = df[(df[COL_TIME] >= start_utc) & (df[COL_TIME] <= end_utc)]
        df = df.drop_duplicates(subset=[COL_TIME], keep="last").sort_values(COL_TIME).reset_index(drop=True)

        # 与 fapi 一致：amount 是 quote_volume 的别名
        if COL_QUOTE_VOLUME in df.columns:
            df[COL_AMOUNT] = df[COL_QUOTE_VOLUME]
        return df
