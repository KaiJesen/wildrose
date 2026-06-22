from __future__ import annotations

import csv
from datetime import datetime

from trading_system.config import TradingSystemConfig
from trading_system.signal import TradingSignal


class CsvSignalProvider:
    def __init__(self, csv_path: str, cfg: TradingSystemConfig) -> None:
        self.cfg = cfg
        self.rows: list[TradingSignal] = []
        atr_vals: list[float] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ts = datetime.fromisoformat(row["ts"].replace("Z", "+00:00"))
                sig = TradingSignal(
                    ts=ts,
                    price=float(row["price"]),
                    atr=float(row["atr"]),
                    p_up=float(row["p_up"]),
                    p_down=float(row["p_down"]),
                    p_flat=float(row["p_flat"]),
                    p_risk=float(row["p_risk"]),
                    pred_ret_1=float(row.get("pred_ret_1", 0.0)),
                    pred_ret_2=float(row.get("pred_ret_2", 0.0)),
                    pred_ret_3=float(row.get("pred_ret_3", 0.0)),
                    pred_ret_4=float(row.get("pred_ret_4", 0.0)),
                    pred_ret_5=float(row.get("pred_ret_5", 0.0)),
                    pred_cum_ret_5=float(row.get("pred_cum_ret_5")) if row.get("pred_cum_ret_5") else None,
                    source="csv_signal",
                    raw=row,
                ).finalize(cfg.rule.risk_open_max)
                self.rows.append(sig)
                atr_vals.append(sig.atr)
        import numpy as np

        self.atr = np.asarray(atr_vals, dtype=np.float64)

    def signal_at(self, i: int) -> TradingSignal:
        return self.rows[i]

