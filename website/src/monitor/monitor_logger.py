"""TradeLogger that streams decisions/trades/equity to the monitor ingest API."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_system.logger import TradeLogger

from monitor.exporter import MonitorExporter


def _utc_ts(ts: Any) -> str:
    if isinstance(ts, str):
        return ts if ts.endswith("Z") else ts
    if hasattr(ts, "isoformat"):
        dt = ts
        if getattr(dt, "tzinfo", None) is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class MonitorTradeLogger(TradeLogger):
    def __init__(
        self,
        exporter: MonitorExporter,
        *,
        out_dir: Path | None = None,
        export: bool = True,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
    ) -> None:
        super().__init__(out_dir=out_dir or Path("/tmp/wildrose_monitor_logs"))
        self._exporter = exporter
        self._export = export
        self._symbol = symbol
        self._interval = interval

    def record_decision(self, *args, **kwargs) -> None:
        super().record_decision(*args, **kwargs)
        if not self._export or not self.decisions:
            return
        row = self.decisions[-1]
        payload = {
            k: row.get(k)
            for k in (
                "price", "p_up", "p_down", "p_flat", "p_risk", "edge", "conf",
                "pred_cum_ret_5", "action", "reason_code", "blocked", "blocked_reason",
                "portfolio_equity", "position_ratio", "state",
            )
        }
        self._exporter.emit(
            "decision",
            _utc_ts(row.get("ts")),
            payload,
            symbol=self._symbol,
            interval=self._interval,
        )

    def record_trade(self, row: dict) -> None:
        super().record_trade(row)
        if not self._export:
            return
        self._exporter.emit(
            "trade",
            _utc_ts(row.get("exit_ts") or row.get("entry_ts")),
            row,
            symbol=self._symbol,
            interval=self._interval,
        )

    def record_equity(self, ts, equity: float) -> None:
        super().record_equity(ts, equity)
        if not self._export:
            return
        self._exporter.emit(
            "equity",
            _utc_ts(ts),
            {"equity": equity},
            symbol=self._symbol,
            interval=self._interval,
        )
