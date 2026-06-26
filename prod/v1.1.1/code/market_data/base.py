"""K 线数据源抽象。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class KlineProvider(ABC):
    """多数据源切换的统一接口。返回列见 `market_data.schema`（time, open, high, low, close, volume 等）。"""

    id: str
    description: str = ""

    @abstractmethod
    def fetch_kline(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        *,
        adjust: str = "",
    ) -> pd.DataFrame:
        """
        :param symbol: 标的代码（含义由具体数据源定义，如 A 股 600519）
        :param interval: 周期，如 1m, 5m, 15m, 30m, 60m, 1d
        :param adjust: 复权，'' | qfq | hfq（日线/分钟若数据源支持）
        """

    @property
    def supported_intervals(self) -> frozenset[str]:
        """当前实现支持的 interval 集合。"""
        return frozenset()
