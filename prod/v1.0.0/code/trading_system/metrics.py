from __future__ import annotations

import numpy as np

BARS_PER_YEAR_1H = 24 * 365


def annualize_return(total_return: float, bar_count: int, *, bars_per_year: float = BARS_PER_YEAR_1H) -> float:
    if bar_count <= 0:
        return 0.0
    growth = 1.0 + float(total_return)
    if growth <= 0.0:
        return -1.0
    return float(growth ** (bars_per_year / bar_count) - 1.0)


def compute_metrics(
    equity_curve: list[float],
    *,
    benchmark_return: float,
    trade_count: int,
    wins: int,
    gross_profit: float,
    gross_loss: float,
    avg_bars_held: float,
    avg_fee_per_trade: float,
    max_margin_loss_ratio_observed: float,
    position_limit_violations: int,
    risk_rule_violations: int,
    bar_count: int | None = None,
    bars_per_year: float = BARS_PER_YEAR_1H,
) -> dict[str, float]:
    if not equity_curve:
        return {}
    eq = np.asarray(equity_curve, dtype=np.float64)
    total_return = float(eq[-1] - 1.0)
    n_bars = bar_count if bar_count is not None else max(1, len(eq) - 1)
    annualized_return = annualize_return(total_return, n_bars, bars_per_year=bars_per_year)
    benchmark_annualized_return = annualize_return(benchmark_return, n_bars, bars_per_year=bars_per_year)
    peak = np.maximum.accumulate(eq)
    max_dd = float(((eq - peak) / np.clip(peak, 1e-12, None)).min())
    win_rate = float(wins / max(1, trade_count))
    pf = float(gross_profit / max(1e-12, gross_loss)) if gross_loss > 0 else float("inf")
    return {
        "annualized_return": annualized_return,
        "benchmark_annualized_return": benchmark_annualized_return,
        "excess_annualized_return": annualized_return - benchmark_annualized_return,
        "total_return": total_return,
        "benchmark_return": benchmark_return,
        "excess_return": total_return - benchmark_return,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": pf,
        "trade_count": float(trade_count),
        "avg_bars_held": float(avg_bars_held),
        "avg_fee_per_trade": float(avg_fee_per_trade),
        "bar_count": float(n_bars),
        "max_margin_loss_ratio_observed": float(max_margin_loss_ratio_observed),
        "position_limit_violations": float(position_limit_violations),
        "risk_rule_violations": float(risk_rule_violations),
    }

