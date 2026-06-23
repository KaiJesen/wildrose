"""多数据源 K 线抓取套件：统一 schema + 可插拔 Provider。"""

from typing import TYPE_CHECKING, Any

from market_data.base import KlineProvider
from market_data.registry import get_kline_provider, list_kline_providers, register_provider
from market_data.schema import (
    COL_AMOUNT,
    COL_CLOSE,
    COL_HIGH,
    COL_LOW,
    COL_OPEN,
    COL_TIME,
    COL_VOLUME,
    CORE_COLUMNS,
    normalize_ohlcv_df,
)

if TYPE_CHECKING:  # 仅为类型提示，运行期不强依赖 matplotlib
    from market_data.plotting import (
        plot_candlestick,
        plot_candlestick_volume,
        plot_volume,
    )

# 绘图函数走懒加载：未装 matplotlib 时不会因 import market_data 而出错
_PLOTTING_EXPORTS = {"plot_candlestick", "plot_candlestick_volume", "plot_volume"}


def __getattr__(name: str) -> Any:
    if name in _PLOTTING_EXPORTS:
        from market_data import plotting

        return getattr(plotting, name)
    raise AttributeError(f"module 'market_data' has no attribute {name!r}")


__all__ = [
    "KlineProvider",
    "COL_AMOUNT",
    "COL_CLOSE",
    "COL_HIGH",
    "COL_LOW",
    "COL_OPEN",
    "COL_TIME",
    "COL_VOLUME",
    "CORE_COLUMNS",
    "normalize_ohlcv_df",
    "get_kline_provider",
    "list_kline_providers",
    "register_provider",
    "plot_candlestick",
    "plot_candlestick_volume",
    "plot_volume",
]
