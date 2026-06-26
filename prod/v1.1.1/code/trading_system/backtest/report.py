from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading_system.backtest.plotting import plot_equity_curve, plot_trade_points


def write_report(
    out_dir: Path,
    *,
    metrics: dict[str, float],
    config_path: str,
    checkpoint: str,
    symbol: str = "",
    split: str = "test",
    df: pd.DataFrame | None = None,
    strategy_eq: np.ndarray | None = None,
    benchmark_eq: np.ndarray | None = None,
    plot: bool = True,
    dpi: int = 160,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Backtest Report",
        "",
        f"- symbol: `{symbol or 'N/A'}`",
        f"- split: `{split}`",
        f"- config: `{config_path}`",
        f"- checkpoint: `{checkpoint}`",
        "",
        "## 核心指标",
        "",
    ]
    core = (
        ("annualized_return", "年化收益率"),
        ("total_return", "总收益率"),
        ("benchmark_return", "基准收益率"),
        ("excess_return", "超额收益率"),
        ("max_drawdown", "最大回撤"),
        ("win_rate", "胜率"),
        ("profit_factor", "盈亏比"),
        ("trade_count", "交易次数"),
        ("avg_bars_held", "平均持仓周期"),
    )
    shown: set[str] = set()
    for key, label in core:
        if key not in metrics:
            continue
        shown.add(key)
        lines.append(f"- {label}: `{_fmt(metrics[key], key)}`")
    lines.extend(["", "## 全部指标", ""])
    for k, v in metrics.items():
        if k in shown:
            continue
        if isinstance(v, float):
            lines.append(f"- {k}: `{_fmt(v, k)}`")
        else:
            lines.append(f"- {k}: `{v}`")
    if plot:
        lines.extend(
            [
                "",
                "## 图表",
                "",
                f"- 资金曲线: `{out_dir / 'equity_curve.png'}`",
                f"- 买卖点: `{out_dir / 'trade_points.png'}`",
            ]
        )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "metrics.txt").write_text("\n".join(f"{k}={v}" for k, v in metrics.items()) + "\n", encoding="utf-8")

    if not plot:
        return
    eq_path = out_dir / "equity_curve.csv"
    if strategy_eq is None and eq_path.exists() and eq_path.stat().st_size > 0:
        strategy_eq = pd.read_csv(eq_path)["equity"].to_numpy(dtype=np.float64)
    if strategy_eq is not None:
        plot_equity_curve(
            out_dir / "equity_curve.png",
            strategy_eq=np.asarray(strategy_eq, dtype=np.float64),
            benchmark_eq=benchmark_eq,
            title=f"{symbol} equity ({split})" if symbol else f"Equity ({split})",
            dpi=dpi,
        )
    trades_path = out_dir / "trades.csv"
    if df is not None and trades_path.exists() and trades_path.stat().st_size > 0:
        plot_trade_points(
            out_dir / "trade_points.png",
            df=df,
            trades=pd.read_csv(trades_path),
            title=f"{symbol} buy/sell points" if symbol else "Buy/sell points",
            dpi=dpi,
        )


def _pct(v: float) -> str:
    return f"{100.0 * v:.2f}%"


def _fmt(v: float, key: str) -> str:
    if key in {"win_rate"}:
        return _pct(v)
    if key.endswith("_return") or key in {"max_drawdown", "excess_return"}:
        return _pct(v)
    if key in {"trade_count"}:
        return str(int(v))
    return f"{v:.6f}"

