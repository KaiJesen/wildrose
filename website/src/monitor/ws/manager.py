from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self) -> None:
        self._rooms: dict[str, set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, backend_id: str, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._rooms.setdefault(backend_id, set()).add(ws)

    async def disconnect(self, backend_id: str, ws: WebSocket) -> None:
        async with self._lock:
            room = self._rooms.get(backend_id)
            if room and ws in room:
                room.remove(ws)

    async def broadcast(self, backend_id: str, message: dict[str, Any]) -> None:
        async with self._lock:
            sockets = list(self._rooms.get(backend_id, set()))
        dead: list[WebSocket] = []
        text = json.dumps(message, ensure_ascii=False)
        for ws in sockets:
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(backend_id, ws)


ws_manager = ConnectionManager()
