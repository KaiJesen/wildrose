"""027 dual-slot engine: Core (legacy) + optional Satellite best-point layer."""

from __future__ import annotations

import json
from pathlib import Path

from trading_system.adapters.best_point_model import BestPointSignal
from trading_system.config import TradingSystemConfig
from trading_system.engine import Bar, TradingEngine
from trading_system.logger import TradeLogger
from trading_system.portfolio import PortfolioState
from trading_system.portfolio_slots import AccountEquity
from trading_system.satellite_engine import SatelliteEngine
from trading_system.satellite_rules import SatelliteSlotConfig
from trading_system.signal import TradingSignal


def load_satellite_config(path: Path | str) -> SatelliteSlotConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    sat = payload.get("satellite", payload)
    return SatelliteSlotConfig(
        enabled=bool(sat.get("enabled", True)),
        max_position_ratio=float(sat.get("max_position_ratio", 0.08)),
        max_hold_bars=int(sat.get("max_hold_bars", 48)),
        max_daily_opens=int(sat.get("max_daily_opens", 5)),
        long_entry_threshold=float(sat.get("long_entry_threshold", 0.50)),
        short_entry_threshold=float(sat.get("short_entry_threshold", 0.50)),
        exit_prob_threshold=float(sat.get("exit_prob_threshold", 0.70)),
        hold_min_prob=float(sat.get("hold_min_prob", 0.30)),
        min_opportunity_roi=float(sat.get("min_opportunity_roi", 0.0)),
        min_pred_cum_ret_5_long=float(sat.get("min_pred_cum_ret_5_long", 0.0)),
        risk_open_max=float(sat.get("risk_open_max", 0.50)),
        stop_atr_mult=float(sat.get("stop_atr_mult", 1.4)),
        require_core_flat=bool(sat.get("require_core_flat", False)),
        require_core_watch_slow_uptrend=bool(sat.get("require_core_watch_slow_uptrend", False)),
    )


class DualSlotEngine:
    """Core delegates to ``TradingEngine``; Satellite is additive PnL wallet."""

    def __init__(
        self,
        cfg: TradingSystemConfig,
        logger: TradeLogger,
        *,
        satellite_enabled: bool = False,
        satellite_cfg: SatelliteSlotConfig | None = None,
        core_only: bool = False,
    ) -> None:
        self.cfg = cfg
        self.logger = logger
        self.core_only = core_only
        use_satellite = (satellite_enabled or core_only) and satellite_cfg is not None
        self.satellite_enabled = use_satellite
        self.core_engine = TradingEngine(cfg, logger)
        self.account = AccountEquity()
        self.satellite_engine: SatelliteEngine | None = None
        self._sat_trades_logged = 0
        self._shadow_core: TradingEngine | None = None
        if use_satellite and satellite_cfg is not None:
            self.satellite_engine = SatelliteEngine(cfg, satellite_cfg)
            if core_only and satellite_cfg.require_core_watch_slow_uptrend:
                self._shadow_core = TradingEngine(cfg, TradeLogger(out_dir=logger.out_dir / "shadow_core"))

    @property
    def portfolio(self) -> PortfolioState:
        return self.core_engine.portfolio

    @property
    def max_margin_loss_ratio_observed(self) -> float:
        return self.core_engine.max_margin_loss_ratio_observed

    @property
    def position_limit_violations(self) -> int:
        return self.core_engine.position_limit_violations

    @property
    def risk_rule_violations(self) -> int:
        return self.core_engine.risk_rule_violations

    @property
    def hard_counter_open_count(self) -> int:
        return self.core_engine.hard_counter_open_count

    @property
    def legacy_trend_direct_block_count(self) -> int:
        return self.core_engine.legacy_trend_direct_block_count

    @property
    def legacy_trend_direct_read_count(self) -> int:
        return self.core_engine.legacy_trend_direct_read_count

    @property
    def trend_add_candidate_count(self) -> int:
        return self.core_engine.trend_add_candidate_count

    @property
    def trend_add_risk_evaluated_count(self) -> int:
        return self.core_engine.trend_add_risk_evaluated_count

    @property
    def trend_add_rejected_by_risk_count(self) -> int:
        return self.core_engine.trend_add_rejected_by_risk_count

    @property
    def trend_add_allowed_count(self) -> int:
        return self.core_engine.trend_add_allowed_count

    def combined_equity(self) -> float:
        if self.core_only and self.satellite_engine is not None:
            return self.satellite_engine.portfolio.equity
        core_eq = self.core_engine.portfolio.equity
        if self.satellite_engine is None:
            return core_eq
        return core_eq + self.satellite_engine.incremental_pnl()

    def _flush_satellite_trades(self) -> None:
        if self.satellite_engine is None:
            return
        new_trades = self.satellite_engine.trades[self._sat_trades_logged :]
        for tr in new_trades:
            self.logger.record_trade(tr)
        self._sat_trades_logged = len(self.satellite_engine.trades)

    def _sync_core_account(self) -> None:
        if self._shadow_core is not None:
            self.account.sync_from_core_portfolio(self._shadow_core.portfolio)
        else:
            self.account.sync_from_core_portfolio(self.core_engine.portfolio)

    def _core_reason_code(self) -> str:
        if self._shadow_core is not None:
            return self._shadow_core.last_core_reason_code
        return self.core_engine.last_core_reason_code

    def on_bar_close(
        self,
        signal: TradingSignal,
        current_bar: Bar,
        next_bar: Bar,
        *,
        best_point_signal: BestPointSignal | None = None,
    ) -> None:
        if self.core_only:
            if self._shadow_core is not None:
                self._shadow_core.on_bar_close(
                    signal, current_bar, next_bar, best_point_signal=best_point_signal
                )
            self._sync_core_account()
            if self.satellite_engine is not None and best_point_signal is not None:
                self.satellite_engine.on_bar_close(
                    signal,
                    best_point_signal,
                    current_bar,
                    next_bar,
                    self.account,
                    core_reason_code=self._core_reason_code(),
                )
                self._flush_satellite_trades()
            self.logger.record_equity(next_bar.ts, self.combined_equity())
            self.account.update_peak_equity()
            return

        self.core_engine.on_bar_close(
            signal,
            current_bar,
            next_bar,
            best_point_signal=best_point_signal,
        )
        core_reason = self.core_engine.last_core_reason_code
        self.account.sync_from_core_portfolio(self.core_engine.portfolio)

        if self.satellite_engine is not None and best_point_signal is not None:
            self.satellite_engine.on_bar_close(
                signal,
                best_point_signal,
                current_bar,
                next_bar,
                self.account,
                core_reason_code=core_reason,
            )
            self._flush_satellite_trades()
            if self.logger.equity_curve:
                self.logger.equity_curve[-1] = {
                    "ts": next_bar.ts,
                    "equity": self.combined_equity(),
                }
            else:
                self.logger.record_equity(next_bar.ts, self.combined_equity())

        self.account.update_peak_equity()
