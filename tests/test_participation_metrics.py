"""Tests for 023 participation metrics."""

from __future__ import annotations

import unittest

import pandas as pd

from trading_system.participation import compute_participation_metrics, leg_direction_from_type


class ParticipationMetricsTest(unittest.TestCase):
    def test_leg_direction_mapping(self):
        self.assertEqual(leg_direction_from_type("FAST_UP_LEG"), "UP")
        self.assertEqual(leg_direction_from_type("FAST_DOWN_LEG"), "DOWN")

    def test_aligned_coverage_counts_long_on_up_leg(self):
        decisions = pd.DataFrame(
            {
                "ts": pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC"),
                "price": [100 + i for i in range(10)],
                "state": ["FLAT"] * 4 + ["LONG"] * 4 + ["FLAT"] * 2,
                "position_ratio": [0.0] * 4 + [0.05] * 4 + [0.0] * 2,
                "leg_id": [1] * 10,
                "leg_type": ["FAST_UP_LEG"] * 10,
                "is_leg_confirmed": [1] * 10,
                "reason_code": ["HOLD"] * 10,
            }
        )
        m = compute_participation_metrics(decisions, pd.DataFrame())
        self.assertEqual(m.leg_count, 1)
        self.assertEqual(m.legs[0].aligned_overlap_bars, 4)
        self.assertEqual(m.legs[0].counter_overlap_bars, 0)
        self.assertTrue(m.legs[0].effective_covered)

    def test_counter_overlap_not_counted_as_coverage(self):
        decisions = pd.DataFrame(
            {
                "ts": pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC"),
                "price": [100 + i for i in range(8)],
                "state": ["SHORT"] * 6 + ["FLAT"] * 2,
                "position_ratio": [0.05] * 6 + [0.0] * 2,
                "leg_id": [2] * 8,
                "leg_type": ["FAST_UP_LEG"] * 8,
                "is_leg_confirmed": [1] * 8,
                "reason_code": ["HOLD"] * 8,
            }
        )
        m = compute_participation_metrics(decisions, pd.DataFrame())
        self.assertEqual(m.legs[0].aligned_overlap_bars, 0)
        self.assertEqual(m.legs[0].counter_overlap_bars, 6)
        self.assertFalse(m.legs[0].effective_covered)
        self.assertEqual(m.counter_leg_participation_count, 1)


if __name__ == "__main__":
    unittest.main()
