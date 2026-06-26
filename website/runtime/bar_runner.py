#!/usr/bin/env python3
"""Bar runner: poll Binance for new 1h bar, post heartbeat & bar_close to monitor API."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "website" / "src"))

from monitor.exporter import MonitorExporter  # noqa: E402
from monitor.services.market_data import MarketDataService  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bar_runner")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend-id", default="prod-v1.0.0-live")
    ap.add_argument("--api-url", default="http://127.0.0.1:8765")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--poll-seconds", type=int, default=45)
    args = ap.parse_args()

    exporter = MonitorExporter(args.api_url, args.backend_id)
    mds = MarketDataService(args.backend_id, args.symbol, args.interval)
    last_bar: str | None = None

    logger.info("bar_runner started backend=%s api=%s", args.backend_id, args.api_url)
    while True:
        try:
            exporter.heartbeat()
            new_ts = mds.poll_and_store()
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
