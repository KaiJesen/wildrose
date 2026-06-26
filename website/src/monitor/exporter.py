"""Push trading events to the monitor website ingest API."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any

import requests


def _utc_ts(ts: Any) -> str:
    if isinstance(ts, str):
        return ts if ts.endswith("Z") else ts
    if hasattr(ts, "isoformat"):
        dt = ts
        if getattr(dt, "tzinfo", None) is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class MonitorExporter:
    def __init__(self, base_url: str, backend_id: str, *, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.backend_id = backend_id
        self.timeout = timeout

    def emit(
        self,
        event_type: str,
        ts: Any,
        payload: dict[str, Any],
        *,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
    ) -> bool:
        body = {
            "events": [
                {
                    "backend_id": self.backend_id,
                    "event_type": event_type,
                    "ts": _utc_ts(ts),
                    "symbol": symbol,
                    "interval": interval,
                    "payload": _serialize(payload),
                }
            ]
        }
        try:
            r = requests.post(
                f"{self.base_url}/api/ingest",
                json=body,
                timeout=self.timeout,
            )
            r.raise_for_status()
            return True
        except Exception:
            return False

    def heartbeat(self) -> bool:
        return self.emit(
            "runner_heartbeat",
            datetime.now(timezone.utc),
            {"status": "ok"},
        )


def _serialize(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    if hasattr(obj, "value"):  # Enum
        return obj.value
    return obj
