from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from monitor.db import connect, insert_event, upsert_backend


@dataclass
class ImportResult:
    backend_id: str
    decisions: int
    trades: int
    equity_points: int
    imported_at: str


_CORE_DECISION_COLS = [
    "ts", "price", "p_up", "p_down", "p_flat", "p_risk", "edge", "conf",
    "pred_cum_ret_5", "action", "reason_code", "blocked", "blocked_reason",
    "portfolio_equity", "position_ratio", "state",
]


def _parse_ts(val: Any) -> str:
    if val is None or val == "":
        return ""
    if isinstance(val, str):
        return val
    return str(val)


def import_archive(
    backend_id: str,
    data_root: Path,
    *,
    display_name: str = "",
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    mode: str = "archive",
    manifest_path: str | None = None,
    config_path: str | None = None,
) -> ImportResult:
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    decisions_path = data_root / "decisions.csv"
    trades_path = data_root / "trades.csv"
    equity_path = data_root / "equity_curve.csv"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    n_dec = n_tr = n_eq = 0

    with connect() as conn:
        upsert_backend(
            conn,
            {
                "id": backend_id,
                "display_name": display_name or backend_id,
                "symbol": symbol,
                "interval": interval,
                "mode": mode,
                "manifest_path": manifest_path,
                "config_path": config_path,
                "data_root": str(data_root),
                "checkpoint": None,
            },
        )
        conn.execute("DELETE FROM decisions WHERE backend_id=?", (backend_id,))
        conn.execute("DELETE FROM trades WHERE backend_id=?", (backend_id,))
        conn.execute("DELETE FROM equity_curve WHERE backend_id=?", (backend_id,))

        if decisions_path.exists():
            with decisions_path.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    ts = _parse_ts(row.get("ts"))
                    if not ts:
                        continue
                    core = {k: row.get(k) for k in _CORE_DECISION_COLS}
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO decisions
                        (backend_id, ts, price, p_up, p_down, p_flat, p_risk, edge, conf,
                         pred_cum_ret_5, action, reason_code, blocked, blocked_reason,
                         portfolio_equity, position_ratio, state, raw_json)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            backend_id, ts,
                            _f(core.get("price")), _f(core.get("p_up")), _f(core.get("p_down")),
                            _f(core.get("p_flat")), _f(core.get("p_risk")), _f(core.get("edge")),
                            _f(core.get("conf")), _f(core.get("pred_cum_ret_5")),
                            core.get("action"), core.get("reason_code"),
                            int(float(core.get("blocked") or 0)),
                            core.get("blocked_reason"),
                            _f(core.get("portfolio_equity")), _f(core.get("position_ratio")),
                            core.get("state"), json.dumps(row, ensure_ascii=False),
                        ),
                    )
                    n_dec += 1

        if trades_path.exists():
            with trades_path.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    entry_ts = _parse_ts(row.get("entry_ts"))
                    exit_ts = _parse_ts(row.get("exit_ts"))
                    if not entry_ts:
                        continue
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO trades
                        (backend_id, entry_ts, exit_ts, side, entry_price, exit_price,
                         net_pnl, bars_held, entry_reason, exit_reason, raw_json)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            backend_id, entry_ts, exit_ts, row.get("side"),
                            _f(row.get("entry_price")), _f(row.get("exit_price")),
                            _f(row.get("net_pnl")), int(float(row.get("bars_held") or 0)),
                            row.get("entry_reason"), row.get("exit_reason"),
                            json.dumps(row, ensure_ascii=False),
                        ),
                    )
                    n_tr += 1

        if equity_path.exists():
            with equity_path.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    ts = _parse_ts(row.get("ts"))
                    if not ts:
                        continue
                    conn.execute(
                        "INSERT OR REPLACE INTO equity_curve (backend_id, ts, equity) VALUES (?,?,?)",
                        (backend_id, ts, _f(row.get("equity"))),
                    )
                    n_eq += 1

        conn.execute(
            "UPDATE backends SET last_import_ts=? WHERE id=?",
            (now, backend_id),
        )
        insert_event(
            conn,
            {
                "backend_id": backend_id,
                "event_type": "archive_import",
                "ts": now,
                "symbol": symbol,
                "interval": interval,
                "payload": {"decisions": n_dec, "trades": n_tr, "equity": n_eq},
            },
        )

    return ImportResult(backend_id, n_dec, n_tr, n_eq, now)


def _f(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
