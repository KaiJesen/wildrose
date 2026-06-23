from __future__ import annotations

import math

import numpy as np


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
) -> dict[str, float]:
    if not equity_curve:
        return {}
    eq = np.asarray(equity_curve, dtype=np.float64)
    total_return = float(eq[-1] - 1.0)
    peak = np.maximum.accumulate(eq)
    max_dd = float(((eq - peak) / np.clip(peak, 1e-12, None)).min())
    win_rate = float(wins / max(1, trade_count))
    pf = float(gross_profit / max(1e-12, gross_loss)) if gross_loss > 0 else float("inf")
    return {
        "total_return": total_return,
        "benchmark_return": benchmark_return,
        "excess_return": total_return - benchmark_return,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": pf,
        "trade_count": float(trade_count),
        "avg_bars_held": float(avg_bars_held),
        "avg_fee_per_trade": float(avg_fee_per_trade),
        "max_margin_loss_ratio_observed": float(max_margin_loss_ratio_observed),
        "position_limit_violations": float(position_limit_violations),
        "risk_rule_violations": float(risk_rule_violations),
    }

