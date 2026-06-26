from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from monitor.adapters.archive_import import import_archive
from monitor.adapters.registry import bootstrap_archives, load_backends_yaml, register_backends
from monitor.db import connect, init_db
from monitor.schemas import IngestBatch
from monitor.services.aggregation import (
    get_dashboard,
    get_decisions,
    get_equity,
    get_health,
    get_position,
    get_trades,
    ingest_events,
)
from monitor.services.market_data import MarketDataService
from monitor.settings import POLL_INTERVAL_S, REPO_ROOT
from monitor.ws.manager import ws_manager

logger = logging.getLogger(__name__)

_poll_task: asyncio.Task | None = None
_live_backend_id: str | None = None


async def _poll_market_loop() -> None:
    while True:
        try:
            if _live_backend_id:
                svc = MarketDataService(_live_backend_id)
                new_ts = await asyncio.to_thread(svc.poll_and_store)
                if new_ts:
                    await ws_manager.broadcast(
                        _live_backend_id,
                        {"type": "bar_close", "ts": new_ts},
                    )
        except Exception as e:
            logger.exception("poll error: %s", e)
        await asyncio.sleep(POLL_INTERVAL_S)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _poll_task, _live_backend_id
    init_db()
    profiles = register_backends()
    bootstrap_archives(profiles)
    for p in profiles:
        if p.mode == "live":
            _live_backend_id = p.id
            break
    _poll_task = asyncio.create_task(_poll_market_loop())
    yield
    if _poll_task:
        _poll_task.cancel()
        try:
            await _poll_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="wildrose monitor", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/backends")
def list_backends():
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, display_name, symbol, interval, mode FROM backends ORDER BY id"
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/backends/{backend_id}/health")
def backend_health(backend_id: str):
    h = get_health(backend_id)
    if h.get("error"):
        raise HTTPException(404, h["error"])
    return h


@app.get("/api/backends/{backend_id}/dashboard")
def backend_dashboard(backend_id: str, period: str = Query("30d")):
    d = get_dashboard(backend_id, period)
    if not d:
        raise HTTPException(404, "backend not found")
    return d


@app.get("/api/backends/{backend_id}/position")
def backend_position(backend_id: str):
    return get_position(backend_id)


@app.get("/api/backends/{backend_id}/trades")
def backend_trades(backend_id: str, limit: int = Query(200, le=500)):
    return get_trades(backend_id, limit)


@app.get("/api/backends/{backend_id}/equity")
def backend_equity(backend_id: str):
    return get_equity(backend_id)


@app.get("/api/backends/{backend_id}/decisions")
def backend_decisions(
    backend_id: str,
    blocked_only: bool = Query(False),
    limit: int = Query(100, le=500),
):
    return get_decisions(backend_id, blocked_only=blocked_only, limit=limit)


@app.get("/api/backends/{backend_id}/ohlcv")
def backend_ohlcv(backend_id: str, limit: int = Query(500, le=2000)):
    with connect() as conn:
        b = conn.execute("SELECT symbol, interval FROM backends WHERE id=?", (backend_id,)).fetchone()
    if not b:
        raise HTTPException(404, "backend not found")
    svc = MarketDataService(backend_id, b["symbol"], b["interval"])
    return svc.get_ohlcv(limit)


@app.post("/api/backends/{backend_id}/import")
def backend_import(backend_id: str):
    profiles = {p.id: p for p in load_backends_yaml()}
    p = profiles.get(backend_id)
    if not p or not p.data_root:
        raise HTTPException(400, "no data_root configured")
    root = Path(p.data_root)
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    r = import_archive(
        backend_id, root,
        display_name=p.display_name, symbol=p.symbol, interval=p.interval,
        mode=p.mode, manifest_path=p.manifest_path, config_path=p.config_path,
    )
    return r.__dict__


@app.post("/api/ingest")
async def ingest(batch: IngestBatch):
    events = [e.model_dump() for e in batch.events]
    n = ingest_events(events)
    for ev in events:
        await ws_manager.broadcast(ev["backend_id"], {"type": ev["event_type"], **ev})
    return {"ingested": n}


@app.websocket("/ws/backends/{backend_id}")
async def ws_backend(backend_id: str, websocket: WebSocket):
    await ws_manager.connect(backend_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(backend_id, websocket)


# Static frontend
_FRONTEND_DIST = REPO_ROOT / "website" / "src" / "frontend" / "dist"


@app.get("/")
def index_page():
    if (_FRONTEND_DIST / "index.html").is_file():
        return FileResponse(_FRONTEND_DIST / "index.html")
    return {"message": "wildrose monitor API", "docs": "/docs"}


@app.get("/styles.css")
def styles_css():
    return FileResponse(_FRONTEND_DIST / "styles.css")


@app.get("/app.js")
def app_js():
    return FileResponse(_FRONTEND_DIST / "app.js")
