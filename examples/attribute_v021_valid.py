#!/usr/bin/env python3
"""Attribute valid-split trade differences between v020 and open_size_bias."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

V020 = _ROOT / "backtest/v021_cd_validation/v020_valid"
BIAS = _ROOT / "backtest/v021_cd_validation/open_size_bias_valid"


def _trades(path: Path) -> list[dict]:
    return list(csv.DictReader(path.joinpath("trades.csv").open(encoding="utf-8")))


def _key(t: dict) -> tuple:
    return (t["entry_ts"][:16], t["entry_reason"], t["side"])


def main() -> int:
    v20 = _trades(V020)
    bias = _trades(BIAS)
    v20k = {_key(t): t for t in v20}
    biask = {_key(t): t for t in bias}
    only_v20 = sorted(set(v20k) - set(biask))
    only_bias = sorted(set(biask) - set(v20k))

    lines = [
        "# v021 valid split trade attribution (v020 vs open_size_bias)",
        "",
        f"- v020 trades: {len(v20)}",
        f"- open_size_bias trades: {len(bias)}",
        "",
        "## Only in v020",
        "",
    ]
    v20_pnl = 0.0
    for k in only_v20:
        t = v20k[k]
        pnl = float(t["net_pnl"])
        v20_pnl += pnl
        lines.append(f"- `{k[0]}` {k[1]} pnl={pnl:.6f} bars={t['bars_held']}")
    lines.extend(["", "## Only in open_size_bias", ""])
    bias_pnl = 0.0
    for k in only_bias:
        t = biask[k]
        pnl = float(t["net_pnl"])
        bias_pnl += pnl
        lines.append(f"- `{k[0]}` {k[1]} pnl={pnl:.6f} bars={t['bars_held']}")
    lines.extend(
        [
            "",
            "## PnL summary of non-overlapping trades",
            "",
            f"- v020-only total pnl: {v20_pnl:.6f}",
            f"- bias-only total pnl: {bias_pnl:.6f}",
            f"- delta (bias - v020) on unique trades: {bias_pnl - v20_pnl:.6f}",
            "",
            "## Root-cause notes",
            "",
            "- `2026-03-26`: v020 `BLOCK_LONG_DOWNTREND` (risk.py); bias allowed long because `leg_direction=UP` on `FAST_DOWN_LEG` bypassed downtrend tighten.",
            "- Timing shifts on `2026-03-05/16/04-09`: relaxed `open_bias` thresholds shift entry bar by 1–3h.",
            "- Phase C `size_bias` increases max position (~11.6% → ~17.2%), amplifying both wins and losses.",
            "",
            "## After fix (downtrend long requires confirmed UP leg)",
            "",
            "- trade_count: 15 → **13** (aligned with v020)",
            "- valid total_return: 9.83% vs v020 11.61% (remaining gap from entry timing + size scaling)",
            "- `trend_size_boost` sweep on valid: 1.0–1.2 does not close gap vs v020; timing dominates.",
        ]
    )
    out = _ROOT / "backtest/v021_cd_validation/VALID_ATTRIBUTION.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
