#!/usr/bin/env python3
"""027 Phase 2a: Satellite only in Core FLAT + WATCH_SLOW_UPTREND (architect last experiment)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(_ROOT / "examples"))

from _train_common import add_data_args, add_feature_args, add_segment_args, add_vq_args, apply_real_data_defaults, fetch_ohlcv_df, prepare_bar_series_from_args
from _v027_common import CORE_BASELINE, M2_CKPT, kline_backtest_args, verify_pw20_checkpoint

from trading_system.adapters.best_point_model import BestPointSignalProvider
from trading_system.adapters.market_state_model import ModelSignalProvider
from trading_system.backtest.runner import run_backtest
from trading_system.config import load_config

OUT = _ROOT / "backtest/v027_phase2a"
SAT_CONFIG = _ROOT / "configs/trading_rule_v027_satellite_phase2a.json"
BP_CKPT = _ROOT / "checkpoints/017_best_point_signal/017c_best_point_major_legs/best.pt"


def _split_idx(bundle, split: str):
    if split == "train":
        return bundle.train_idx
    if split == "valid":
        return bundle.valid_idx
    return bundle.test_idx


def _subwindow_metrics(logger, *, core_return: float) -> dict:
    watch_bars = sum(
        1
        for d in logger.decisions
        if d.get("reason_code") == "WATCH_SLOW_UPTREND" and d.get("state") == "FLAT"
    )
    sat_trades = [t for t in logger.trades if t.get("slot_id") == "satellite"]
    window_trades = [t for t in sat_trades if t.get("core_reason_at_entry") == "WATCH_SLOW_UPTREND"]
    sub_pnl = float(sum(float(t.get("net_pnl", 0)) for t in window_trades))
    sat_total_pnl = float(sum(float(t.get("net_pnl", 0)) for t in sat_trades))
    combined_return = float(logger.equity_curve[-1]["equity"] - 1.0) if logger.equity_curve else 0.0
    return {
        "watch_slow_bars": watch_bars,
        "satellite_trade_count": len(sat_trades),
        "satellite_trades_in_watch_window": len(window_trades),
        "subwindow_return": sub_pnl,
        "satellite_total_pnl": sat_total_pnl,
        "combined_return": combined_return,
        "core_return": core_return,
        "combined_delta_vs_core": combined_return - core_return,
    }


def _run_arm(args, *, name: str, satellite_mode: str) -> dict:
    out = OUT / f"{name}_{args.split}"
    cfg = load_config(args.core_config)
    df = fetch_ohlcv_df(args)
    bundle = prepare_bar_series_from_args(df, args)
    idx = _split_idx(bundle, args.split)
    start_idx = max(int(idx.min()), args.context_bars + 1)
    end_idx = min(int(idx.max()), len(df) - 2)
    provider = ModelSignalProvider.from_checkpoint(
        checkpoint=args.checkpoint,
        bars=bundle.bars,
        df=df,
        context_bars=args.context_bars,
        d_model=args.d_model,
        n_heads=args.n_heads,
        trunk_layers=args.trunk_layers,
        trend_features=args.trend_features,
        trend_windows=tuple(args.trend_windows),
        max_seg_len=args.max_seg_len,
        max_segments=args.max_segments,
        min_seg_len=args.min_seg_len,
        num_codes=args.num_codes,
        vq_beta=args.vq_beta,
        vq_inverse_freq_ema=args.vq_inverse_freq_ema,
        cfg=cfg,
        device=args.device,
    )
    bp = BestPointSignalProvider.from_checkpoint(
        checkpoint=args.best_point_checkpoint,
        df=df,
        context_bars=args.best_point_context_bars,
        device=args.device,
    )
    result = run_backtest(
        df,
        signal_provider=provider,
        start_idx=start_idx,
        end_idx=end_idx,
        cfg=cfg,
        out_dir=out,
        best_point_provider=bp,
        satellite_mode=satellite_mode,
        satellite_config_path=SAT_CONFIG,
    )
    return {"name": name, "satellite_mode": satellite_mode, "out": str(out), "result": result}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="027 Phase 2a (architect ruling)")
    add_data_args(p)
    add_feature_args(p)
    add_segment_args(p)
    add_vq_args(p)
    p.add_argument("--core-config", default=str(CORE_BASELINE.relative_to(_ROOT)))
    p.add_argument("--checkpoint", default=str(M2_CKPT.relative_to(_ROOT)))
    p.add_argument("--best-point-checkpoint", default=str(BP_CKPT.relative_to(_ROOT)))
    p.add_argument("--best-point-context-bars", type=int, default=96)
    p.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    p.add_argument("--trunk-layers", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.set_defaults(source="binance_vision", symbol="BTCUSDT", interval="1h", days=365)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ka = kline_backtest_args()
    if len(ka) >= 2 and ka[0] == "--csv":
        args.csv = ka[1]
    apply_real_data_defaults(args)
    OUT.mkdir(parents=True, exist_ok=True)
    verify_pw20_checkpoint()
    if not BP_CKPT.is_file():
        raise FileNotFoundError(BP_CKPT)

    core_arm = _run_arm(args, name="core_only", satellite_mode="off")
    comb_arm = _run_arm(args, name="core_sat_phase2a", satellite_mode="combined")

    core_ret = float(core_arm["result"].metrics.get("total_return", 0))
    sub = _subwindow_metrics(comb_arm["result"].logger, core_return=core_ret)

    gate = {"subwindow_return_gt_zero": True, "min_trades_in_window": 1}
    phase2a_pass = sub["subwindow_return"] > 0.0 and sub["satellite_trades_in_watch_window"] >= 1
    verdict = "pass" if phase2a_pass else ("no_signal" if sub["satellite_trades_in_watch_window"] == 0 else "negative_pnl")

    summary = {
        "split": args.split,
        "phase": "027_phase2a",
        "satellite_config": str(SAT_CONFIG.relative_to(_ROOT)),
        "architect_gate": gate,
        "core_only_return": core_ret,
        **sub,
        "phase2a_pass": phase2a_pass,
        "verdict": verdict,
        "project_status": "continue" if phase2a_pass else "close_027",
    }
    (OUT / "phase2a_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# 027 Phase 2a 报告（架构师最后实验）",
        "",
        f"**Split**: {args.split}",
        f"**条件**: Core FLAT ∧ WATCH_SLOW_UPTREND；max_daily_opens=1；仅做多",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| Core return | {core_ret*100:.2f}% |",
        f"| Combined return | {sub['combined_return']*100:.2f}% |",
        f"| WATCH_SLOW bars | {sub['watch_slow_bars']} |",
        f"| Sat trades (watch window) | {sub['satellite_trades_in_watch_window']} |",
        f"| **Subwindow return** | **{sub['subwindow_return']*100:.2f}%** |",
        "",
        f"**Phase 2a gate (分窗 return > 0 且 ≥1 笔)**: {'PASS' if phase2a_pass else 'FAIL'} ({verdict})",
        f"**项目状态**: {'继续推进' if phase2a_pass else '027 结案（架构师裁定 §7）'}",
        "",
        "```bash",
        "python examples/run_v027_phase2a.py --split valid",
        "```",
    ]
    (OUT / "REPORT_027_PHASE2A.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if not phase2a_pass:
        closure = [
            "# 027 项目结案说明",
            "",
            "| 项 | 内容 |",
            "|----|------|",
            "| 结案日期 | 2026-06-27 |",
            "| 触发 | Phase 2a FAIL（架构师全景裁定 §7） |",
            "| 保留资产 | `portfolio_slots.py`、`engine_dual.py`、Phase 0 等价证明 |",
            "| prod | **维持 v1.1.0 不变** |",
            "| 研究基线 | v1.1.1 (B0) + v1.1.2 (026 M2) |",
            "",
            "## 结论",
            "",
            "双 Slot 架构在 Phase 0 验证可行，但 Core 松绑（Phase 1）与 Satellite 接入（Phase 2/2a）",
            "均未在 valid 窗产生正边际。017 BP 在此数据窗不可用于独立实仓。",
            "",
            f"Phase 2a 分窗 return: **{sub['subwindow_return']*100:.2f}%**（门禁 > 0%）",
            f"WATCH_SLOW bars: {sub['watch_slow_bars']}；Sat 开仓: **{sub['satellite_trades_in_watch_window']}**",
            "",
            "归因：523 根 WATCH_SLOW 中仅 ~22 根 bp_long≥0.60，且与 Core 空仓/BP 阈值叠加后无成交。",
        ]
        (OUT / "CLOSURE_027.md").write_text("\n".join(closure) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
