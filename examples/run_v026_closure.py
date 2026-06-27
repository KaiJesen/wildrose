#!/usr/bin/env python3
"""026 结案：test 回测 + 策略/事后最优买卖点图 + 日均交易统计。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "examples") not in sys.path:
    sys.path.insert(0, str(_ROOT / "examples"))

from _v025_common import PW20_CKPT, kline_backtest_args, sha256_prefix, verify_pw20_checkpoint

OUT = _ROOT / "backtest/v026_closure"
PHASE3 = _ROOT / "backtest/v026_phase3"
B0_CONFIG = _ROOT / "configs/trading_rule_v024_phase1c_teq_0065a_c1_pw20.json"
M2_CKPT = _ROOT / "checkpoints/026_phase1_c1d1/market_state_best.pt"
M3_CKPT = _ROOT / "checkpoints/026_phase2_a1/market_state_best.pt"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=_ROOT)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_ROOT.resolve()))
    except ValueError:
        return str(path)


def _arm_specs() -> list[dict]:
    m2_sweep = _read_json(PHASE3 / "teq_wp_sweep.json")
    m3_sweep = _read_json(PHASE3 / "teq_wp_sweep_m3.json")
    m2_cfg = _ROOT / str(m2_sweep.get("best", {}).get("config", "backtest/v026_phase3/configs/m2_wp0.30.json"))
    m3_cfg = _ROOT / str(m3_sweep.get("best", {}).get("config", "backtest/v026_phase3/configs/m3_wp0.30.json"))
    return [
        {"name": "b0", "label": "B0 研究基线 (c1_pw20)", "config": B0_CONFIG, "checkpoint": PW20_CKPT},
        {"name": "m2", "label": "M2 C3+C1+D1", "config": m2_cfg, "checkpoint": M2_CKPT},
        {"name": "m3", "label": "M3 +A1 CORAL", "config": m3_cfg, "checkpoint": M3_CKPT},
    ]


def _backtest_arm(name: str, *, config: Path, checkpoint: Path, split: str) -> Path:
    out = OUT / f"{name}_{split}"
    _run([
        sys.executable,
        "examples/backtest_trading_system_v014.py",
        "--config",
        _repo_rel(config),
        "--checkpoint",
        _repo_rel(checkpoint),
        "--split",
        split,
        "--output-dir",
        str(out.relative_to(_ROOT)),
        *kline_backtest_args(),
    ])
    return out


def _trade_daily_stats(bt_dir: Path) -> dict:
    equity = pd.read_csv(bt_dir / "equity_curve.csv")
    equity["ts"] = pd.to_datetime(equity["ts"], utc=True)
    t0, t1 = equity["ts"].iloc[0], equity["ts"].iloc[-1]
    calendar_days = max(1, int((t1 - t0).total_seconds() // 86400) + 1)

    trades = pd.read_csv(bt_dir / "trades.csv") if (bt_dir / "trades.csv").is_file() else pd.DataFrame()
    trade_count = int(len(trades))
    avg_trades_per_day = trade_count / calendar_days

    active_days = 0
    if not trades.empty and "entry_ts" in trades.columns:
        trades["entry_ts"] = pd.to_datetime(trades["entry_ts"], utc=True)
        active_days = int(trades["entry_ts"].dt.floor("D").nunique())
    avg_on_active_days = trade_count / max(1, active_days)

    optimal_n = 0
    opt_path = bt_dir / "optimal_trades_hindsight.csv"
    if opt_path.is_file():
        optimal_n = int(len(pd.read_csv(opt_path)))

    metrics = _read_json(bt_dir / "metrics.json")
    return {
        "calendar_days": calendar_days,
        "period_start": str(t0),
        "period_end": str(t1),
        "trade_count": trade_count,
        "avg_trades_per_calendar_day": round(avg_trades_per_day, 4),
        "active_trade_days": active_days,
        "avg_trades_on_active_days": round(avg_on_active_days, 4),
        "total_return": metrics.get("total_return"),
        "max_drawdown": metrics.get("max_drawdown"),
        "optimal_hindsight_trades": optimal_n,
    }


def _plot_arm(bt_dir: Path, *, title: str) -> Path:
    plot_out = bt_dir / "trades_with_optimal.png"
    _run([
        sys.executable,
        "examples/plot_backtest_trading_v014.py",
        "--backtest-dir",
        str(bt_dir.relative_to(_ROOT)),
        "--output",
        str(plot_out.relative_to(_ROOT)),
        "--title",
        title,
        "--csv",
        str(kline_backtest_args()[1]),
    ])
    return plot_out


def main() -> int:
    ap = argparse.ArgumentParser(description="026 closure backtest + trade/optimal plots")
    ap.add_argument("--split", default="test", choices=["test", "valid"])
    ap.add_argument("--skip-backtest", action="store_true", help="reuse existing OUT dirs if present")
    args = ap.parse_args()

    verify_pw20_checkpoint()
    OUT.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for spec in _arm_specs():
        name = spec["name"]
        bt_dir = OUT / f"{name}_{args.split}"
        if not args.skip_backtest or not (bt_dir / "metrics.json").is_file():
            if not spec["config"].is_file():
                raise FileNotFoundError(spec["config"])
            if not spec["checkpoint"].is_file():
                raise FileNotFoundError(spec["checkpoint"])
            _backtest_arm(name, config=spec["config"], checkpoint=spec["checkpoint"], split=args.split)
        else:
            print(f"reuse backtest: {bt_dir}")

        title = f"026 结案 · {spec['label']} ({args.split})"
        plot_path = _plot_arm(bt_dir, title=title)
        stats = _trade_daily_stats(bt_dir)
        stats.update(
            {
                "arm": name,
                "label": spec["label"],
                "config": _repo_rel(spec["config"]),
                "checkpoint": _repo_rel(spec["checkpoint"]),
                "backtest_dir": _repo_rel(bt_dir),
                "plot": _repo_rel(plot_path),
                "trade_points_png": _repo_rel(bt_dir / "trade_points.png"),
            }
        )
        rows.append(stats)
        print(
            f"[{name}] trades={stats['trade_count']} "
            f"avg/day={stats['avg_trades_per_calendar_day']:.3f} "
            f"return={float(stats.get('total_return', 0))*100:.2f}%"
        )

    summary = {"split": args.split, "arms": rows, "verdict": "026 结案 — 探索门未 PASS，不替换 prod v1.1.0"}
    (OUT / "closure_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# 026 项目结案报告",
        "",
        "| 项 | 内容 |",
        "|----|------|",
        "| 状态 | **结案** — 模型轨部分达标，探索 coverage 膝点未突破 |",
        "| 数据 | BTCUSDT 1h test split（冻结 K 线） |",
        f"| 产物目录 | `{OUT.relative_to(_ROOT)}` |",
        "",
        "## 回测与日均交易次数",
        "",
        "| 臂 | 总交易 | 日历天数 | **日均交易** | 有交易日均笔数 | 总收益 | 最大回撤 | 事后最优(DP) |",
        "|----|--------|----------|--------------|----------------|--------|----------|--------------|",
    ]
    for r in rows:
        ret = float(r.get("total_return") or 0.0)
        dd = float(r.get("max_drawdown") or 0.0)
        lines.append(
            f"| {r['arm'].upper()} | {r['trade_count']} | {r['calendar_days']} | "
            f"**{r['avg_trades_per_calendar_day']:.3f}** | {r['avg_trades_on_active_days']:.3f} | "
            f"{ret*100:.2f}% | {dd*100:.2f}% | {r['optimal_hindsight_trades']} |"
        )
    lines.extend([
        "",
        "说明：",
        "- **日均交易** = 成交笔数 ÷ 回测区间日历天数（含无交易日）。",
        "- **有交易日均笔数** = 成交笔数 ÷ 至少有一笔开仓的自然日数。",
        "- **事后最优买卖点** 来自 `trade/tools/optimal_trade_points` 动态规划（hindsight，非实盘信号）。",
        "",
        "## 图表",
        "",
    ])
    for r in rows:
        lines.extend([
            f"### {r['label']} ({r['arm'].upper()})",
            "",
            f"- 策略 + 最优 overlay：`{r['plot']}`",
            f"- 引擎默认买卖点：`{r['trade_points_png']}`",
            f"- 回测目录：`{r['backtest_dir']}`",
            "",
        ])
    lines.extend([
        "## 结案结论",
        "",
        "1. Phase 0~2 模型轨：C3/C1+D1/A1 依次 PASS（part_auc ≥ 0.62）。",
        "2. Phase 3 探索门：M2/M3 test coverage 均 **26.67%**（门禁 28%），M3 return 8.07% 低于 M2。",
        "3. **prod v1.1.0 保持不变**；026 作为研究支线结案。",
        "",
        "## 复现",
        "```bash",
        "python examples/run_v026_closure.py",
        "```",
    ])
    report = OUT / "REPORT_026_CLOSURE.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
