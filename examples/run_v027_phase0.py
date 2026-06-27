#!/usr/bin/env python3
"""027 Phase 0: B0 reproduction + dual-slot (Sat FLAT) equivalence gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(_ROOT / "examples"))

from _train_common import add_data_args, add_feature_args, add_segment_args, add_vq_args, apply_real_data_defaults, fetch_ohlcv_df, prepare_bar_series_from_args
from _v025_common import PW20_CKPT, kline_backtest_args, sha256_prefix, verify_pw20_checkpoint

from trading_system.adapters.market_state_model import ModelSignalProvider
from trading_system.backtest.runner import run_backtest
from trading_system.config import load_config

OUT = _ROOT / "backtest/v027_phase0"
B0_CONFIG = _ROOT / "configs/trading_rule_v024_phase1c_teq_0065a_c1_pw20.json"
GATES = {
    "b0_return": 0.0901,
    "b0_return_tol": 0.002,
    "b0_coverage": 0.267,
    "b0_coverage_tol": 0.01,
    "b0_teq": 3,
    "equity_max_diff": 1e-8,
}


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=_ROOT)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _split_idx(bundle, split: str):
    if split == "train":
        return bundle.train_idx
    if split == "valid":
        return bundle.valid_idx
    return bundle.test_idx


def _build_provider(args, cfg, df, bundle):
    return ModelSignalProvider.from_checkpoint(
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


def _run_inline_backtest(args, *, split: str, out_dir: Path, dual_slot: bool) -> dict:
    apply_real_data_defaults(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    df = fetch_ohlcv_df(args)
    bundle = prepare_bar_series_from_args(df, args)
    idx = _split_idx(bundle, split)
    start_idx = max(int(idx.min()), args.context_bars + 1)
    end_idx = min(int(idx.max()), len(df) - 2)
    provider = _build_provider(args, cfg, df, bundle)
    result = run_backtest(
        df,
        signal_provider=provider,
        start_idx=start_idx,
        end_idx=end_idx,
        cfg=cfg,
        out_dir=out_dir,
        dual_slot=dual_slot,
    )
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
    eq = pd.DataFrame(result.logger.equity_curve)
    eq.to_csv(out_dir / "equity_curve.csv", index=False)
    return {
        "out_dir": str(out_dir),
        "metrics": result.metrics,
        "equity": [float(x) for x in result.strategy_equity],
        "trade_count": len(result.logger.trades),
        "decisions": result.logger.decisions,
    }


def _compare_equivalence(legacy: dict, dual: dict) -> dict:
    leg_eq = np.asarray(legacy["equity"], dtype=np.float64)
    dual_eq = np.asarray(dual["equity"], dtype=np.float64)
    n = min(len(leg_eq), len(dual_eq))
    leg_eq = leg_eq[:n]
    dual_eq = dual_eq[:n]
    max_diff = float(np.max(np.abs(leg_eq - dual_eq))) if n else 0.0
    reason_mismatch = 0
    for i, (a, b) in enumerate(zip(legacy["decisions"], dual["decisions"])):
        if a.get("reason_code") != b.get("reason_code") or a.get("action") != b.get("action"):
            reason_mismatch += 1
            if reason_mismatch <= 3:
                print(f"decision mismatch @ {i}: legacy={a.get('reason_code')} dual={b.get('reason_code')}")
    return {
        "equity_bars": n,
        "equity_max_abs_diff": max_diff,
        "trade_count_legacy": legacy["trade_count"],
        "trade_count_dual": dual["trade_count"],
        "decision_reason_mismatches": reason_mismatch,
        "pass": (
            max_diff <= GATES["equity_max_diff"]
            and legacy["trade_count"] == dual["trade_count"]
            and reason_mismatch == 0
        ),
    }


def _b0_gate(metrics: dict, participation: dict) -> dict:
    cov = float(participation["test"]["participation_metrics"]["leg_count_coverage_ratio"])
    ret = float(metrics["total_return"])
    teq = int(metrics.get("trend_qualified_open_count", 0))
    ret_ok = abs(ret - GATES["b0_return"]) <= GATES["b0_return_tol"]
    cov_ok = abs(cov - GATES["b0_coverage"]) <= GATES["b0_coverage_tol"]
    teq_ok = teq == GATES["b0_teq"]
    return {
        "return": ret,
        "coverage": cov,
        "teq_open": teq,
        "ret_ok": ret_ok,
        "cov_ok": cov_ok,
        "teq_ok": teq_ok,
        "pass": ret_ok and cov_ok and teq_ok,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="027 Phase 0 gate")
    add_data_args(p)
    add_feature_args(p)
    add_segment_args(p)
    add_vq_args(p)
    p.add_argument("--config", default=str(B0_CONFIG.relative_to(_ROOT)))
    p.add_argument("--checkpoint", default=str(PW20_CKPT.relative_to(_ROOT)))
    p.add_argument("--split", default="test", choices=["train", "valid", "test"])
    p.add_argument("--output-dir", default=str(OUT.relative_to(_ROOT)))
    p.add_argument("--device", default="cpu")
    p.add_argument("--trunk-layers", type=int, default=2)
    p.add_argument("--skip-b0-script", action="store_true", help="skip prod verify_phase0.sh smoke")
    p.set_defaults(source="binance_vision", symbol="BTCUSDT", interval="1h", days=365)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ka = kline_backtest_args()
    if len(ka) >= 2 and ka[0] == "--csv" and not getattr(args, "csv", None):
        args.csv = ka[1]
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = _ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    verify_pw20_checkpoint()
    ck_hash = sha256_prefix(PW20_CKPT)

    b0_smoke: dict = {}
    if not args.skip_b0_script:
        smoke_dir = out / "b0_smoke"
        _run(["bash", "prod/v1.1.1/scripts/verify_phase0.sh", str(smoke_dir)])
        b0_smoke = _b0_gate(
            _read_json(smoke_dir / "metrics.json"),
            _read_json(smoke_dir / "participation.json"),
        )
        print(f"B0 smoke gate: {'PASS' if b0_smoke['pass'] else 'FAIL'}", b0_smoke)

    # Inline legacy vs dual-slot on same split
    legacy_dir = out / f"legacy_{args.split}"
    dual_dir = out / f"dual_flat_{args.split}"
    legacy = _run_inline_backtest(args, split=args.split, out_dir=legacy_dir, dual_slot=False)
    dual = _run_inline_backtest(args, split=args.split, out_dir=dual_dir, dual_slot=True)
    equiv = _compare_equivalence(legacy, dual)
    print(f"equivalence: max_diff={equiv['equity_max_abs_diff']:.2e} trades={equiv['trade_count_legacy']} pass={equiv['pass']}")

    part_out = legacy_dir / "participation.json"
    _run([
        sys.executable,
        "examples/eval_participation.py",
        "--backtest-dir",
        str(legacy_dir.resolve().relative_to(_ROOT.resolve())),
        "--output",
        str(part_out.resolve().relative_to(_ROOT.resolve())),
    ])
    b0_inline = _b0_gate(legacy["metrics"], _read_json(legacy_dir / "participation.json"))

    summary = {
        "split": args.split,
        "checkpoint": str(PW20_CKPT),
        "checkpoint_hash": ck_hash,
        "config": args.config,
        "b0_smoke": b0_smoke,
        "b0_inline": b0_inline,
        "equivalence": equiv,
        "phase0_pass": equiv["pass"] and b0_inline["pass"] and (b0_smoke.get("pass", True) if b0_smoke else True),
    }
    summary_path = out / "phase0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "# 027 Phase 0 报告",
        "",
        f"| 项 | 值 |",
        f"|----|-----|",
        f"| split | {args.split} |",
        f"| B0 inline return | {b0_inline['return']*100:.2f}% |",
        f"| B0 inline coverage | {b0_inline['coverage']*100:.2f}% |",
        f"| teq_open | {b0_inline['teq_open']} |",
        f"| equity max diff | {equiv['equity_max_abs_diff']:.2e} |",
        f"| equivalence | {'PASS' if equiv['pass'] else 'FAIL'} |",
        f"| **Phase 0** | **{'PASS' if summary['phase0_pass'] else 'FAIL'}** |",
        "",
        "## 复现",
        "```bash",
        "python examples/run_v027_phase0.py",
        "bash scripts/verify_v027_phase0.sh",
        "```",
    ]
    (out / "REPORT_027_PHASE0.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"saved {summary_path}")
    return 0 if summary["phase0_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
