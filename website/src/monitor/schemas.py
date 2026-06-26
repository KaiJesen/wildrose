from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class BackendProfile(BaseModel):
    id: str
    display_name: str
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    mode: Literal["live", "archive", "replay"] = "live"
    manifest_path: str | None = None
    config_path: str | None = None
    data_root: str | None = None
    checkpoint: str | None = None


class IngestEvent(BaseModel):
    backend_id: str
    event_type: str
    ts: str
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    payload: dict[str, Any] = Field(default_factory=dict)


class IngestBatch(BaseModel):
    events: list[IngestEvent]


class HealthStatus(BaseModel):
    backend_id: str
    mode: str
    runner_alive: bool = False
    latest_bar_ts: str | None = None
    last_decision_ts: str | None = None
    last_import_ts: str | None = None
    lag_seconds: float | None = None
    symbol: str
    interval: str
