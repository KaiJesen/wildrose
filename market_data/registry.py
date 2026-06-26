"""数据源注册与工厂。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from market_data.base import KlineProvider

_REGISTRY: dict[str, type["KlineProvider"]] = {}

_ALIASES: dict[str, str] = {
    "eastmoney": "akshare_em",
    "akshare": "akshare_em",
    "em": "akshare_em",
    # Binance USDT-M 永续/交割合约（实时 REST）
    "binance": "binance_futures",
    "binance_um": "binance_futures",
    "binance_usdm": "binance_futures",
    "binance_perp": "binance_futures",
    "fapi": "binance_futures",
    # Binance Vision 历史归档（公开 ZIP，国内多数网络可直连）
    "vision": "binance_vision",
    "binance_archive": "binance_vision",
    "binance_history": "binance_vision",
    # Binance Data API（data-api.binance.vision，现货 REST 尾部补齐）
    "data_api": "binance_data_api",
    "binance_data_api": "binance_data_api",
}


def register_provider(cls: type["KlineProvider"]) -> None:
    """注册自定义 Provider（往切换表里加一个 id）。"""
    _REGISTRY[cls.id] = cls


def _ensure_default_providers() -> None:
    if _REGISTRY:
        return
    from market_data.sources.akshare_eastmoney import AkShareEastmoneyKlineProvider
    from market_data.sources.binance_futures import BinanceFuturesKlineProvider
    from market_data.sources.binance_vision import BinanceVisionKlineProvider
    from market_data.sources.binance_data_api import BinanceDataApiKlineProvider

    register_provider(AkShareEastmoneyKlineProvider)
    register_provider(BinanceFuturesKlineProvider)
    register_provider(BinanceVisionKlineProvider)
    register_provider(BinanceDataApiKlineProvider)


def list_kline_providers() -> list[tuple[str, str, frozenset[str]]]:
    """返回 [(id, description, supported_intervals), ...]。"""
    _ensure_default_providers()
    out: list[tuple[str, str, frozenset[str]]] = []
    for cls in _REGISTRY.values():
        p = cls()
        try:
            intervals = p.supported_intervals
        except Exception:
            intervals = frozenset()
        out.append((cls.id, getattr(cls, "description", ""), intervals))
    return sorted(out, key=lambda x: x[0])


def get_kline_provider(name: str, **kwargs) -> "KlineProvider":
    """
    按名称实例化数据源。

    :param name: 如 akshare_em、eastmoney（别名）、或你通过 register_provider 注册的 id
    :param kwargs: 传给具体 Provider 构造参数（如 retries=8）
    """
    _ensure_default_providers()
    key = name.strip().lower().replace("-", "_")
    key = _ALIASES.get(key, key)
    cls = _REGISTRY.get(key)
    if cls is None:
        known = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"未知数据源 {name!r}。已注册: {known}")
    return cls(**kwargs)
