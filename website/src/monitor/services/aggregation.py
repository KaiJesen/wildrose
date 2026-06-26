from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from monitor.adapters.registry import load_reference_metrics
from monitor.db import connect


def _parse_dt(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _period_start(period: str) -> datetime | None:
    now = datetime.now(timezone.utc)
    if period == "today":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "7d":
        return now - timedelta(days=7)
    if period == "30d":
        return now - timedelta(days=30)
    return None  # all


def get_health(backend_id: str) -> dict[str, Any]:
    with connect() as conn:
        b = conn.execute("SELECT * FROM backends WHERE id=?", (backend_id,)).fetchone()
        m = conn.execute("SELECT * FROM runner_meta WHERE backend_id=?", (backend_id,)).fetchone()
    if not b:
        return {"error": "backend not found"}
    latest_bar = m["latest_bar_ts"] if m else None
    lag = None
    if latest_bar:
        lag = (datetime.now(timezone.utc) - _parse_dt(latest_bar)).total_seconds()
    return {
        "backend_id": backend_id,
        "mode": b["mode"],
        "runner_alive": bool(m["runner_alive"]) if m else False,
        "latest_bar_ts": latest_bar,
        "last_decision_ts": b["last_decision_ts"],
        "last_import_ts": b["last_import_ts"],
        "lag_seconds": lag,
        "symbol": b["symbol"],
        "interval": b["interval"],
    }


def get_position(backend_id: str) -> dict[str, Any]:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT * FROM decisions WHERE backend_id=?
            ORDER BY ts DESC LIMIT 1
            """,
            (backend_id,),
        ).fetchone()
        b = conn.execute("SELECT config_path FROM backends WHERE id=?", (backend_id,)).fetchone()
    if not row:
        return {"state": "FLAT", "position_ratio": 0.0}
    leverage = 20.0
    if b and b["config_path"]:
        try:
            from monitor.adapters.registry import _resolve
            cfg_path = _resolve(b["config_path"])
            if cfg_path and cfg_path.exists():
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                leverage = float(cfg.get("base", {}).get("fixed_leverage", 20.0))
        except Exception:
            pass
    return {
        "ts": row["ts"],
        "state": row["state"],
        "position_ratio": row["position_ratio"],
        "portfolio_equity": row["portfolio_equity"],
        "price": row["price"],
        "leverage": leverage,
    }


def get_dashboard(backend_id: str, period: str = "30d") -> dict[str, Any]:
    start = _period_start(period)
    with connect() as conn:
        b = conn.execute("SELECT * FROM backends WHERE id=?", (backend_id,)).fetchone()
        if not b:
            return {}
        dec_where = "backend_id=?"
        dec_args: list[Any] = [backend_id]
        if start:
            dec_where += " AND ts >= ?"
            dec_args.append(start.strftime("%Y-%m-%dT%H:%M:%SZ"))

        decisions = conn.execute(
            f"SELECT * FROM decisions WHERE {dec_where}", dec_args
        ).fetchall()
        trades = conn.execute(
            f"SELECT * FROM trades WHERE backend_id=?" + (" AND entry_ts >= ?" if start else ""),
            [backend_id] + ([start.strftime("%Y-%m-%dT%H:%M:%SZ")] if start else []),
        ).fetchall()
        equity = conn.execute(
            "SELECT ts, equity FROM equity_curve WHERE backend_id=? ORDER BY ts",
            (backend_id,),
        ).fetchall()

    n_block = sum(1 for d in decisions if int(d["blocked"] or 0) == 1)
    edges = [float(d["edge"]) for d in decisions if d["edge"] is not None]
    p_up = [float(d["p_up"]) for d in decisions if d["p_up"] is not None]

    wins = sum(1 for t in trades if t["net_pnl"] and float(t["net_pnl"]) > 0)
    trade_count = len(trades)
    total_pnl = sum(float(t["net_pnl"] or 0) for t in trades)

    eq = [float(r["equity"]) for r in equity]
    max_dd = 0.0
    if eq:
        arr = np.array(eq)
        peak = np.maximum.accumulate(arr)
        max_dd = float(((arr - peak) / np.clip(peak, 1e-12, None)).min())

    ref = load_reference_metrics(b["manifest_path"]) if b["manifest_path"] else {}

    block_reasons: dict[str, int] = {}
    for d in decisions:
        if int(d["blocked"] or 0) == 1:
            rc = d["reason_code"] or "UNKNOWN"
            block_reasons[rc] = block_reasons.get(rc, 0) + 1
    top_blocks = sorted(block_reasons.items(), key=lambda x: -x[1])[:8]

    return {
        "backend_id": backend_id,
        "display_name": b["display_name"],
        "mode": b["mode"],
        "period": period,
        "signals": {
            "count": len(decisions),
            "avg_edge": float(np.mean(edges)) if edges else 0.0,
            "avg_p_up": float(np.mean(p_up)) if p_up else 0.0,
            "block_rate": n_block / max(1, len(decisions)),
            "block_topn": [{"reason": k, "count": v} for k, v in top_blocks],
        },
        "pnl": {
            "trade_count": trade_count,
            "win_rate": wins / max(1, trade_count),
            "total_net_pnl": total_pnl,
            "total_return": eq[-1] - 1.0 if eq else 0.0,
            "max_drawdown": max_dd,
        },
        "position": get_position(backend_id),
        "reference_metrics": ref,
        "health": get_health(backend_id),
    }


def get_trades(backend_id: str, limit: int = 200) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT entry_ts, exit_ts, side, entry_price, exit_price, net_pnl, bars_held
            FROM trades WHERE backend_id=? ORDER BY entry_ts DESC LIMIT ?
            """,
            (backend_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_equity(backend_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT ts, equity FROM equity_curve WHERE backend_id=? ORDER BY ts",
            (backend_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_decisions(
    backend_id: str,
    *,
    blocked_only: bool = False,
    limit: int = 100,
) -> list[dict]:
    with connect() as conn:
        q = "SELECT ts, price, action, reason_code, blocked, edge, conf, state FROM decisions WHERE backend_id=?"
        args: list[Any] = [backend_id]
        if blocked_only:
            q += " AND blocked=1"
        q += " ORDER BY ts DESC LIMIT ?"
        args.append(limit)
        rows = conn.execute(q, args).fetchall()
    return [dict(r) for r in rows]


def ingest_events(events: list[dict]) -> int:
    from monitor.db import insert_event, set_runner_meta

    n = 0
    with connect() as conn:
        for ev in events:
            insert_event(conn, ev)
            if ev["event_type"] == "decision":
                payload = ev.get("payload") or {}
                conn.execute(
                    """
                    INSERT OR REPLACE INTO decisions
                    (backend_id, ts, price, p_up, p_down, p_flat, p_risk, edge, conf,
                     pred_cum_ret_5, action, reason_code, blocked, blocked_reason,
                     portfolio_equity, position_ratio, state, raw_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        ev["backend_id"], ev["ts"],
                        payload.get("price"), payload.get("p_up"), payload.get("p_down"),
                        payload.get("p_flat"), payload.get("p_risk"), payload.get("edge"),
                        payload.get("conf"), payload.get("pred_cum_ret_5"),
                        payload.get("action"), payload.get("reason_code"),
                        int(payload.get("blocked") or 0), payload.get("blocked_reason"),
                        payload.get("portfolio_equity"), payload.get("position_ratio"),
                        payload.get("state"), json.dumps(payload),
                    ),
                )
                conn.execute(
                    "UPDATE backends SET last_decision_ts=? WHERE id=?",
                    (ev["ts"], ev["backend_id"]),
                )
            elif ev["event_type"] == "equity":
                payload = ev.get("payload") or {}
                conn.execute(
                    "INSERT OR REPLACE INTO equity_curve (backend_id, ts, equity) VALUES (?,?,?)",
                    (ev["backend_id"], ev["ts"], payload.get("equity")),
                )
            elif ev["event_type"] == "trade":
                payload = ev.get("payload") or {}
                conn.execute(
                    """
                    INSERT OR REPLACE INTO trades
                    (backend_id, entry_ts, exit_ts, side, entry_price, exit_price,
                     net_pnl, bars_held, entry_reason, exit_reason, raw_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        ev["backend_id"],
                        payload.get("entry_ts"), payload.get("exit_ts"),
                        payload.get("side"), payload.get("entry_price"), payload.get("exit_price"),
                        payload.get("net_pnl"), payload.get("bars_held"),
                        payload.get("entry_reason"), payload.get("exit_reason"),
                        json.dumps(payload),
                    ),
                )
            elif ev["event_type"] == "runner_heartbeat":
                set_runner_meta(conn, ev["backend_id"], runner_alive=True, heartbeat=True)
            elif ev["event_type"] == "bar_close":
                set_runner_meta(
                    conn, ev["backend_id"], latest_bar_ts=ev["ts"], runner_alive=True
                )
            n += 1
    return n
