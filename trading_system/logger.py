from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path

from trading_system.adapters.best_point_model import BestPointSignal
from trading_system.crash import CrashContext
from trading_system.execution import FillEvent
from trading_system.portfolio import PortfolioState
from trading_system.rules import TradingAction
from trading_system.signal import TradingSignal
from trading_system.trend import TrendContext
from trading_system.trend_signal import TrendSignal


@dataclass
class TradeLogger:
    out_dir: Path
    decisions: list[dict] = field(default_factory=list)
    orders: list[dict] = field(default_factory=list)
    fills: list[dict] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)

    def record_decision(
        self,
        signal: TradingSignal,
        action: TradingAction,
        portfolio: PortfolioState,
        trend_context: TrendContext | None = None,
        trend_signal: TrendSignal | None = None,
        crash_context: CrashContext | None = None,
        best_point_signal: BestPointSignal | None = None,
        blocked_reason: str = "",
    ) -> None:
        tc = trend_context
        ts = trend_signal
        cc = crash_context
        bs = best_point_signal
        self.decisions.append(
            {
                "ts": signal.ts,
                "price": signal.price,
                "atr": signal.atr,
                "p_up": signal.p_up,
                "p_down": signal.p_down,
                "p_flat": signal.p_flat,
                "p_risk": signal.p_risk,
                "edge": signal.edge,
                "conf": signal.conf,
                "pred_cum_ret_5": signal.pred_cum_ret_5,
                "action": action.action.value,
                "reason_code": action.reason_code,
                "blocked": int(action.action.value == "BLOCK"),
                "blocked_reason": blocked_reason,
                "portfolio_equity": portfolio.equity,
                "position_ratio": portfolio.position.position_ratio,
                "state": portfolio.position.side.value,
                "trend_is_downtrend": int(tc.is_downtrend) if tc else 0,
                "trend_is_strong_downtrend": int(tc.is_strong_downtrend) if tc else 0,
                "trend_score": tc.trend_score if tc else 0.0,
                "ret_3_atr": tc.ret_3_atr if tc else 0.0,
                "ret_6_atr": tc.ret_6_atr if tc else 0.0,
                "ema_fast": tc.ema_fast if tc else 0.0,
                "ema_slow": tc.ema_slow if tc else 0.0,
                "breakdown_low_n": int(tc.breakdown_low_n) if tc else 0,
                "trend_reason_codes": "|".join(tc.reason_codes) if tc else "",
                "fallback_action": action.reason_code
                if ("SENTINEL" in action.reason_code or "DOWNTREND" in action.reason_code or "TREND" in action.reason_code)
                else "",
                "reverse_confirm_count": portfolio.position.short_reverse_confirm_count,
                "hold_mode": portfolio.position.hold_mode,
                "trend_hold_bars": portfolio.position.trend_hold_bars,
                "trend_break_count": portfolio.position.trend_break_count,
                "entry_was_sentinel": int(portfolio.position.entry_was_sentinel),
                "sentinel_bars": portfolio.position.sentinel_bars,
                "peak_profit_atr": portfolio.position.peak_profit_atr,
                "is_crash": int(cc.is_crash) if cc else 0,
                "is_model_blind_crash": int(cc.is_model_blind_crash) if cc else 0,
                "crash_score": cc.crash_score if cc else 0.0,
                "crash_reason_codes": "|".join(cc.reason_codes) if cc else "",
                "drawdown_24h": cc.drawdown_24h if cc else 0.0,
                "ret_12_atr": cc.ret_12_atr if cc else 0.0,
                "range_expansion": cc.range_expansion if cc else 0.0,
                "entry_was_crash": int(portfolio.position.entry_was_crash),
                "crash_regime_id": portfolio.position.crash_regime_id,
                "trend_direction": ts.direction.value if ts else "NONE",
                "trend_strength": ts.strength.value if ts else "NONE",
                "trend_phase": ts.phase.value if ts else "NONE",
                "trend_score_up": ts.score_up if ts else 0.0,
                "trend_score_down": ts.score_down if ts else 0.0,
                "trend_confidence": ts.confidence if ts else 0.0,
                "trend_age": ts.trend_age if ts else 0,
                "trend_invalid_count": ts.invalid_count if ts else 0,
                "trend_is_confirmed": int(ts.is_confirmed) if ts else 0,
                "trend_is_broken": int(ts.is_broken) if ts else 0,
                "trend_is_accelerating": int(ts.is_accelerating) if ts else 0,
                "trend_is_exhausted": int(ts.is_exhausted) if ts else 0,
                "trend_signal_reason_codes": "|".join(ts.reason_codes) if ts else "",
                "bp_p_long_entry_zone": bs.p_long_entry_zone if bs else 0.0,
                "bp_p_short_entry_zone": bs.p_short_entry_zone if bs else 0.0,
                "bp_p_hold_long": bs.p_hold_long if bs else 0.0,
                "bp_p_hold_short": bs.p_hold_short if bs else 0.0,
                "bp_p_exit_long": bs.p_exit_long if bs else 0.0,
                "bp_p_exit_short": bs.p_exit_short if bs else 0.0,
                "bp_expected_opportunity_roi": bs.expected_opportunity_roi if bs else 0.0,
            }
        )

    def record_order(self, row: dict) -> None:
        self.orders.append(row)

    def record_fill(self, fill: FillEvent) -> None:
        self.fills.append(asdict(fill))

    def record_trade(self, row: dict) -> None:
        self.trades.append(row)

    def record_equity(self, ts, equity: float) -> None:
        self.equity_curve.append({"ts": ts, "equity": equity})

    def flush(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._write_csv(self.out_dir / "decisions.csv", self.decisions)
        self._write_csv(self.out_dir / "orders.csv", self.orders)
        self._write_csv(self.out_dir / "fills.csv", self.fills)
        self._write_csv(self.out_dir / "trades.csv", self.trades)
        self._write_csv(self.out_dir / "equity_curve.csv", self.equity_curve)

    @staticmethod
    def _write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        keys = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)

