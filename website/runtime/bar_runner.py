#!/usr/bin/env python3
"""Bar runner: poll market data, optional live engine, push events to monitor API."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "website" / "src"))

from monitor.adapters.registry import load_backends_yaml  # noqa: E402
from monitor.exporter import MonitorExporter  # noqa: E402
from monitor.services.market_data import MarketDataService  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bar_runner")


def _load_live_session(backend_id: str, api_url: str, device: str):
    from monitor.live_session import LivePaperSession

    profiles = {p.id: p for p in load_backends_yaml()}
    profile = profiles.get(backend_id)
    if not profile:
        raise SystemExit(f"backend not found in backends.yaml: {backend_id}")
    if not profile.checkpoint or not profile.config_path:
        raise SystemExit(f"backend {backend_id} missing checkpoint/config_path")
    return LivePaperSession(profile, api_url, device=device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend-id", default="prod-v1.0.0-live")
    ap.add_argument("--api-url", default="http://127.0.0.1:8765")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--poll-seconds", type=int, default=45)
    ap.add_argument("--with-engine", action="store_true", help="run TradingEngine on each new bar")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    exporter = MonitorExporter(args.api_url, args.backend_id)
    mds = MarketDataService(args.backend_id, args.symbol, args.interval)
    session = None
    if args.with_engine:
        logger.info("loading live paper session (torch + checkpoint)...")
        session = _load_live_session(args.backend_id, args.api_url, args.device)

    last_bar: str | None = None
    logger.info(
        "bar_runner started backend=%s api=%s engine=%s",
        args.backend_id, args.api_url, bool(session),
    )
    import time

    while True:
        try:
            exporter.heartbeat()
            new_ts = mds.poll_and_store()
            bars = mds.get_ohlcv(800)
            if session is not None:
                result = session.on_bars(bars)
                if result and not result.get("warmup"):
                    logger.info(
                        "engine step ts=%s close=%.2f",
                        result.get("ts"),
                        result.get("close", 0),
                    )
            if new_ts and new_ts != last_bar:
                last_bar = new_ts
                row = mds.get_ohlcv(1)[-1]
                exporter.emit("bar_close", new_ts, row, symbol=args.symbol, interval=args.interval)
                logger.info("new bar %s close=%.2f", new_ts, row["close"])
        except Exception as e:
            logger.exception("loop error: %s", e)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
