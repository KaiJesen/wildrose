from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from monitor.settings import DB_PATH

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS backends (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    mode TEXT NOT NULL,
    manifest_path TEXT,
    config_path TEXT,
    data_root TEXT,
    checkpoint TEXT,
    last_import_ts TEXT,
  runner_alive INTEGER DEFAULT 0,
    last_decision_ts TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS ohlcv_bars (
    backend_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (backend_id, ts)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backend_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    ts TEXT NOT NULL,
    symbol TEXT,
    interval TEXT,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_backend_ts ON events(backend_id, ts);

CREATE TABLE IF NOT EXISTS decisions (
    backend_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    price REAL,
    p_up REAL, p_down REAL, p_flat REAL, p_risk REAL,
    edge REAL, conf REAL,
    pred_cum_ret_5 REAL,
    action TEXT, reason_code TEXT,
    blocked INTEGER,
    blocked_reason TEXT,
    portfolio_equity REAL,
    position_ratio REAL,
    state TEXT,
    raw_json TEXT,
    PRIMARY KEY (backend_id, ts)
);

CREATE TABLE IF NOT EXISTS trades (
    backend_id TEXT NOT NULL,
    entry_ts TEXT NOT NULL,
    exit_ts TEXT,
    side TEXT,
    entry_price REAL,
    exit_price REAL,
    net_pnl REAL,
    bars_held INTEGER,
    entry_reason TEXT,
    exit_reason TEXT,
    raw_json TEXT,
    PRIMARY KEY (backend_id, entry_ts, exit_ts)
);

CREATE TABLE IF NOT EXISTS equity_curve (
    backend_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    equity REAL NOT NULL,
    PRIMARY KEY (backend_id, ts)
);

CREATE TABLE IF NOT EXISTS runner_meta (
    backend_id TEXT PRIMARY KEY,
    runner_alive INTEGER DEFAULT 0,
    latest_bar_ts TEXT,
    last_heartbeat_ts TEXT
);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def init_db(db_path: Path | None = None) -> Path:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(_SCHEMA)
    return path


@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def upsert_backend(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO backends (id, display_name, symbol, interval, mode,
            manifest_path, config_path, data_root, checkpoint, updated_at)
        VALUES (:id, :display_name, :symbol, :interval, :mode,
            :manifest_path, :config_path, :data_root, :checkpoint, :updated_at)
        ON CONFLICT(id) DO UPDATE SET
            display_name=excluded.display_name,
            symbol=excluded.symbol,
            interval=excluded.interval,
            mode=excluded.mode,
            manifest_path=excluded.manifest_path,
            config_path=excluded.config_path,
            data_root=excluded.data_root,
            checkpoint=excluded.checkpoint,
            updated_at=excluded.updated_at
        """,
        {**row, "updated_at": _utc_now()},
    )


def insert_event(conn: sqlite3.Connection, ev: dict[str, Any]) -> int:
    cur = conn.execute(
        """
        INSERT INTO events (backend_id, event_type, ts, symbol, interval, payload, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ev["backend_id"],
            ev["event_type"],
            ev["ts"],
            ev.get("symbol"),
            ev.get("interval"),
            json.dumps(ev.get("payload") or {}, ensure_ascii=False),
            _utc_now(),
        ),
    )
    return int(cur.lastrowid)


def set_runner_meta(
    conn: sqlite3.Connection,
    backend_id: str,
    *,
    runner_alive: bool | None = None,
    latest_bar_ts: str | None = None,
    heartbeat: bool = False,
) -> None:
    row = conn.execute("SELECT * FROM runner_meta WHERE backend_id=?", (backend_id,)).fetchone()
    alive = int(runner_alive) if runner_alive is not None else (row["runner_alive"] if row else 0)
    bar_ts = latest_bar_ts if latest_bar_ts is not None else (row["latest_bar_ts"] if row else None)
    hb = _utc_now() if heartbeat else (row["last_heartbeat_ts"] if row else None)
    conn.execute(
        """
        INSERT INTO runner_meta (backend_id, runner_alive, latest_bar_ts, last_heartbeat_ts)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(backend_id) DO UPDATE SET
            runner_alive=excluded.runner_alive,
            latest_bar_ts=COALESCE(excluded.latest_bar_ts, runner_meta.latest_bar_ts),
            last_heartbeat_ts=COALESCE(excluded.last_heartbeat_ts, runner_meta.last_heartbeat_ts)
        """,
        (backend_id, alive, bar_ts, hb),
    )
