"""Prod monitoring: FLAT / no-trade decision attribution from backtest logs."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _episode_lengths(mask: pd.Series) -> list[int]:
    lengths: list[int] = []
    run = 0
    for active in mask.astype(bool):
        if active:
            run += 1
        elif run:
            lengths.append(run)
            run = 0
    if run:
        lengths.append(run)
    return lengths


def analyze_decisions(
    decisions: pd.DataFrame,
    *,
    trades: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Summarize why the strategy stayed flat or did not open."""
    df = decisions.copy()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)

    total_bars = len(df)
    flat = df[df["state"].astype(str) == "FLAT"]
    flat_bars = len(flat)
    flat_ratio = float(flat_bars / max(1, total_bars))

    flat_hold = flat[~flat["reason_code"].astype(str).str.startswith("OPEN_")]
    reason_counts = flat_hold["reason_code"].value_counts().head(20).to_dict()
    reason_counts = {str(k): int(v) for k, v in reason_counts.items()}

    watch_mask = df["reason_code"].astype(str) == "WATCH_SLOW_UPTREND"
    watch_bars = int(watch_mask.sum())
    watch_ep_lens = _episode_lengths(watch_mask)
    slow_open = int((df["reason_code"].astype(str) == "OPEN_LONG_SLOW_TREND").sum())

    teq_cols = [c for c in ("teq_reason_codes", "teq_allow_long", "teq_allow_short") if c in df.columns]
    teq_flat = flat
    teq_reject_rows = teq_flat[
        teq_flat.get("teq_reason_codes", pd.Series("", index=teq_flat.index)).astype(str).str.len() > 0
    ] if "teq_reason_codes" in teq_flat.columns else teq_flat.iloc[0:0]
    teq_reason_tokens: dict[str, int] = {}
    if not teq_reject_rows.empty and "teq_reason_codes" in teq_reject_rows.columns:
        for raw in teq_reject_rows["teq_reason_codes"].astype(str):
            for tok in raw.split("|"):
                tok = tok.strip()
                if tok:
                    teq_reason_tokens[tok] = teq_reason_tokens.get(tok, 0) + 1

    hold_no_entry = int((flat["reason_code"].astype(str) == "HOLD_NO_ENTRY").sum())
    blocked = int(flat["action"].astype(str).isin(["BLOCK", "HOLD"]).sum()) if "action" in flat.columns else 0

    open_rows = df[df["reason_code"].astype(str).str.startswith("OPEN_")]
    open_by_reason = open_rows["reason_code"].value_counts().to_dict()
    open_by_reason = {str(k): int(v) for k, v in open_by_reason.items()}

    trade_count = 0
    if trades is not None and not trades.empty:
        trade_count = len(trades)

    return {
        "total_bars": total_bars,
        "flat_bars": flat_bars,
        "flat_ratio": flat_ratio,
        "trade_count": trade_count,
        "open_by_reason": open_by_reason,
        "flat_top_reasons": reason_counts,
        "watch_slow_uptrend_bars": watch_bars,
        "watch_slow_episode_count": len(watch_ep_lens),
        "watch_slow_episode_max_bars": max(watch_ep_lens) if watch_ep_lens else 0,
        "watch_slow_episode_avg_bars": float(sum(watch_ep_lens) / len(watch_ep_lens)) if watch_ep_lens else 0.0,
        "slow_up_open_count": slow_open,
        "slow_up_watch_to_open_ratio": float(slow_open / max(1, watch_bars)),
        "hold_no_entry_bars": hold_no_entry,
        "teq_reject_bars": int(len(teq_reject_rows)),
        "teq_reject_reason_tokens": dict(sorted(teq_reason_tokens.items(), key=lambda x: -x[1])[:15]),
        "blocked_or_hold_flat_bars": blocked,
    }


def compare_to_baseline(
    current: dict[str, Any],
    baseline: dict[str, Any],
    *,
    return_tol_pp: float = 0.5,
    mdd_tol_pp: float = 0.3,
    trade_count_tol: int = 1,
) -> dict[str, Any]:
    """Health check vs frozen prod metrics (architect P0)."""
    checks: dict[str, Any] = {}

    cur_ret = float(current.get("total_return", 0))
    base_ret = float(baseline.get("total_return", 0))
    ret_delta_pp = (cur_ret - base_ret) * 100
    checks["total_return"] = {
        "current": cur_ret,
        "baseline": base_ret,
        "delta_pp": ret_delta_pp,
        "pass": abs(ret_delta_pp) <= return_tol_pp,
    }

    cur_mdd = float(current.get("max_drawdown", 0))
    base_mdd = float(baseline.get("max_drawdown", 0))
    mdd_delta_pp = (cur_mdd - base_mdd) * 100
    checks["max_drawdown"] = {
        "current": cur_mdd,
        "baseline": base_mdd,
        "delta_pp": mdd_delta_pp,
        "pass": mdd_delta_pp >= -mdd_tol_pp,
    }

    cur_tc = int(current.get("trade_count", 0))
    base_tc = int(baseline.get("trade_count", 0))
    checks["trade_count"] = {
        "current": cur_tc,
        "baseline": base_tc,
        "delta": cur_tc - base_tc,
        "pass": abs(cur_tc - base_tc) <= trade_count_tol,
    }

    cur_viol = int(current.get("risk_rule_violations", 0))
    checks["risk_rule_violations"] = {"current": cur_viol, "pass": cur_viol == 0}

    all_pass = all(c.get("pass", True) for c in checks.values())
    return {"pass": all_pass, "checks": checks}
