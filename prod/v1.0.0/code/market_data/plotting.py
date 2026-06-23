"""K 线绘图工具：基于标准 OHLCV schema 的蜡烛图 / 量价图。

只依赖 matplotlib + pandas。任何符合 `market_data.schema` 标准列的 DataFrame
（含 time/open/high/low/close，可选 volume）都能直接喂进来。

最小用法::

    from market_data.plotting import plot_candlestick, plot_candlestick_volume

    # 单图：只画蜡烛
    plot_candlestick(df, title="DOGEUSDT 1h").show()

    # 双图：上方蜡烛 + 下方成交量
    plot_candlestick_volume(df, title="DOGEUSDT 1h", save_path="doge.png")

    # 嵌入到自己已有的 Axes 中
    fig, ax = plt.subplots()
    plot_candlestick(df, ax=ax, color_style="ashare")

color_style:
    - "crypto" : 绿涨红跌（默认，国际 / 加密圈惯例）
    - "ashare" : 红涨绿跌（A 股惯例）
    - (up_hex, down_hex) : 自定义二元组
"""

from __future__ import annotations

from collections.abc import Sequence
import os
from typing import TYPE_CHECKING, Literal, Union

import pandas as pd

from market_data.schema import (
    COL_CLOSE,
    COL_HIGH,
    COL_LOW,
    COL_OPEN,
    COL_TIME,
    COL_VOLUME,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

ColorStyle = Union[Literal["crypto", "ashare"], Sequence[str]]
PathLike = Union[str, "os.PathLike[str]"]

_COLOR_STYLES: dict[str, tuple[str, str]] = {
    # (上涨颜色, 下跌颜色)
    "crypto": ("#26a69a", "#ef5350"),
    "ashare": ("#e4393c", "#00a854"),
}


def _resolve_colors(style: ColorStyle) -> tuple[str, str]:
    if isinstance(style, str):
        if style not in _COLOR_STYLES:
            raise ValueError(
                f"未知 color_style={style!r}；支持: {list(_COLOR_STYLES)}，或 (up, down) 二元组"
            )
        return _COLOR_STYLES[style]
    seq = list(style)
    if len(seq) != 2:
        raise ValueError(f"color_style 自定义需为长度 2 的序列 (up, down)，收到 {seq!r}")
    return seq[0], seq[1]


def _validate_ohlc(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        raise ValueError("DataFrame 为空，无法绘图")
    required = (COL_TIME, COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"缺少必要列 {missing}（需要 {list(required)}）；"
            f"请先用 market_data.schema.normalize_ohlcv_df 规范列名"
        )


def _spacing_and_width(df: pd.DataFrame, body_width_ratio: float) -> tuple[Sequence[float], float, float]:
    """计算 X 轴时间数值、相邻 K 线中位间距、蜡烛体宽度。"""
    import matplotlib.dates as mdates

    times_num = mdates.date2num(df[COL_TIME].to_numpy())
    if len(times_num) > 1:
        spacing = float(pd.Series(times_num).diff().median())
    else:
        spacing = 1.0 / 24.0
    width = max(spacing * body_width_ratio, 1e-6)
    return times_num, spacing, width


def plot_candlestick(
    df: pd.DataFrame,
    *,
    ax: "Axes | None" = None,
    color_style: ColorStyle = "crypto",
    title: str | None = None,
    body_width_ratio: float = 0.55,
    grid: bool = True,
    ylabel: str | None = "Price",
) -> tuple["Figure", "Axes"]:
    """
    在一个 Axes 上画蜡烛图（不含成交量子图）。

    :param df: 标准 OHLCV DataFrame（必须含 time/open/high/low/close）
    :param ax: 复用已有 Axes；为 None 时新建 figure
    :param color_style: 见模块级文档
    :param title: 图表标题，None 则不设置
    :param body_width_ratio: 实体宽度占相邻 K 线间距的比例
    :param grid: 是否显示网格
    :param ylabel: Y 轴标签，传 None 则不设置
    :return: (fig, ax)
    """
    _validate_ohlc(df)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    up_color, down_color = _resolve_colors(color_style)
    times_num, spacing, width = _spacing_and_width(df, body_width_ratio)

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5))
    else:
        fig = ax.figure

    o_arr = df[COL_OPEN].to_numpy(dtype=float)
    h_arr = df[COL_HIGH].to_numpy(dtype=float)
    l_arr = df[COL_LOW].to_numpy(dtype=float)
    c_arr = df[COL_CLOSE].to_numpy(dtype=float)

    for i in range(len(df)):
        t = times_num[i]
        o, h, l, c = o_arr[i], h_arr[i], l_arr[i], c_arr[i]
        color = up_color if c >= o else down_color

        ax.plot([t, t], [l, h], color=color, linewidth=1.0, zorder=1)
        body_low, body_high = (o, c) if o <= c else (c, o)
        body_h = body_high - body_low
        if body_h < 1e-12:
            ax.plot([t - width / 2, t + width / 2], [o, o], color=color, linewidth=1.5, zorder=2)
        else:
            ax.add_patch(
                Rectangle(
                    (t - width / 2, body_low),
                    width,
                    body_h,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=1,
                    zorder=2,
                )
            )

    ax.set_xlim(times_num[0] - spacing * 0.6, times_num[-1] + spacing * 0.6)
    y_pad = (df[COL_HIGH].max() - df[COL_LOW].min()) * 0.03 + 1e-9
    ax.set_ylim(df[COL_LOW].min() - y_pad, df[COL_HIGH].max() + y_pad)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.3)
    ax.xaxis_date()
    return fig, ax


def plot_volume(
    df: pd.DataFrame,
    *,
    ax: "Axes | None" = None,
    color_style: ColorStyle = "crypto",
    body_width_ratio: float = 0.55,
    ylabel: str | None = "Volume",
    grid: bool = True,
) -> tuple["Figure", "Axes"]:
    """单独画成交量柱（颜色按当根 K 线涨跌着色）。"""
    _validate_ohlc(df)
    if COL_VOLUME not in df.columns:
        raise ValueError(f"缺少列 {COL_VOLUME!r}")
    import matplotlib.pyplot as plt

    up_color, down_color = _resolve_colors(color_style)
    times_num, _spacing, width = _spacing_and_width(df, body_width_ratio)

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 2.5))
    else:
        fig = ax.figure

    up_mask = (df[COL_CLOSE].to_numpy() >= df[COL_OPEN].to_numpy())
    colors = [up_color if up else down_color for up in up_mask]

    ax.bar(times_num, df[COL_VOLUME].to_numpy(), width=width, color=colors, align="center", alpha=0.9)
    if ylabel:
        ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, axis="y", alpha=0.3)
    ax.xaxis_date()
    return fig, ax


def plot_candlestick_volume(
    df: pd.DataFrame,
    *,
    title: str | None = None,
    save_path: PathLike | None = None,
    color_style: ColorStyle = "crypto",
    figsize: tuple[float, float] = (11, 6),
    height_ratios: tuple[float, float] = (3.0, 1.0),
    price_label: str = "Price",
    volume_label: str = "Volume",
    date_format: str = "%m-%d\n%H:%M",
    body_width_ratio: float = 0.55,
    show: bool = True,
    tight_layout: bool = True,
) -> "Figure":
    """
    K 线 + 成交量双子图（最常用的总览视图）。

    :param df: 标准 OHLCV DataFrame
    :param title: 主图标题
    :param save_path: 给定则保存到该路径（PNG/PDF/SVG 由后缀决定）
    :param color_style: 涨跌配色
    :param figsize: (宽, 高) inch
    :param height_ratios: (蜡烛图, 成交量) 高度比
    :param price_label: 价格 Y 轴标签
    :param volume_label: 成交量 Y 轴标签
    :param date_format: X 轴时间格式（matplotlib strftime）
    :param body_width_ratio: 蜡烛体相对相邻 K 线间距的宽度
    :param show: 没有 save_path 时是否调用 plt.show()；True 阻塞展示，False 仅返回 fig
    :param tight_layout: 是否调用 fig.tight_layout()
    :return: matplotlib Figure
    """
    _validate_ohlc(df)
    has_volume = COL_VOLUME in df.columns
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    if has_volume:
        fig, (ax_price, ax_vol) = plt.subplots(
            2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": list(height_ratios)}
        )
    else:
        fig, ax_price = plt.subplots(figsize=figsize)
        ax_vol = None

    plot_candlestick(
        df,
        ax=ax_price,
        color_style=color_style,
        title=title,
        body_width_ratio=body_width_ratio,
        ylabel=price_label,
    )

    if ax_vol is not None:
        plot_volume(
            df,
            ax=ax_vol,
            color_style=color_style,
            body_width_ratio=body_width_ratio,
            ylabel=volume_label,
        )
        ax_vol.set_xlim(ax_price.get_xlim())
        ax_vol.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        fig.autofmt_xdate()
    else:
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        fig.autofmt_xdate()

    if tight_layout:
        plt.tight_layout()

    if save_path:
        fig.savefig(os.fspath(save_path), dpi=120, bbox_inches="tight")
        print(f"Figure saved: {save_path}")
        if not show:
            plt.close(fig)
    elif show:
        plt.show()

    return fig


__all__ = [
    "ColorStyle",
    "plot_candlestick",
    "plot_candlestick_volume",
    "plot_volume",
]
