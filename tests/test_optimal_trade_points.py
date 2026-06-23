from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from trade.tools.optimal_trade_points import find_major_leg_trades, find_optimal_trades


def _synthetic_uptrend(n: int = 40) -> pd.DataFrame:
    close = np.linspace(100.0, 130.0, n)
    high = close + 0.5
    low = close - 0.5
    return pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=n, freq="h"),
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1.0,
        }
    )


class TestOptimalTradePoints(unittest.TestCase):
    def test_major_legs_covers_uptrend_as_single_long(self) -> None:
        df = _synthetic_uptrend()
        trades = find_major_leg_trades(df, min_net_roi=0.002, zigzag_min_move_atr=0.5)
        self.assertGreaterEqual(len(trades), 1)
        longs = [t for t in trades if t.direction == "long"]
        self.assertTrue(longs)
        best = max(longs, key=lambda t: t.holding_bars)
        self.assertGreaterEqual(best.holding_bars, 20)
        self.assertGreater(best.net_roi, 0)

    def test_dp_mode_backward_compatible(self) -> None:
        df = _synthetic_uptrend(n=10)
        trades = find_optimal_trades(df, mode="dp", min_net_roi=0.001)
        self.assertIsInstance(trades, list)

    def test_major_legs_fewer_trades_than_dp_on_doge(self) -> None:
        csv = Path(__file__).resolve().parents[1] / "trade/report/0001_DOGE永续合约全局最优多空点标注/ohlcv.csv"
        if not csv.exists():
            self.skipTest("sample ohlcv missing")
        df = pd.read_csv(csv)
        dp = find_optimal_trades(df, mode="dp", min_net_roi=0.002)
        legs = find_optimal_trades(df, mode="major_legs", min_net_roi=0.002)
        self.assertLess(len(legs), len(dp))
        self.assertGreaterEqual(
            sum(t.holding_bars for t in legs),
            sum(t.holding_bars for t in dp) * 0.35,
        )


if __name__ == "__main__":
    unittest.main()
