#!/usr/bin/env python3
"""Run backtests for BTCUSDT and DOGEUSDT with reports and trade-point plots."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest BTC and DOGE contracts")
    p.add_argument("--config", default="prod/v0.1.0/config/trading_rule.json")
    p.add_argument("--checkpoint", default="prod/v0.1.0/checkpoint/market_state_best.pt")
    p.add_argument("--split", default="test")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--python", default=str(_ROOT / ".venv" / "bin" / "python"))
    p.add_argument("--output-root", default="backtest/symbol_compare")
    return p.parse_args()


def _fmt_pct(v: float) -> str:
    return f"{100.0 * v:.2f}%"


def main() -> int:
    args = parse_args()
    py = Path(args.python)
    if not py.exists():
        py = Path(sys.executable)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    symbols = ("BTCUSDT", "DOGEUSDT")
    summaries: list[dict] = []
    for symbol in symbols:
        out_dir = out_root / symbol.lower()
        cmd = [
            str(py),
            str(_ROOT / "examples" / "backtest_trading_system_v014.py"),
            "--config",
            args.config,
            "--checkpoint",
            args.checkpoint,
            "--symbol",
            symbol,
            "--split",
            args.split,
            "--days",
            str(args.days),
            "--output-dir",
            str(out_dir),
        ]
        print(f"[run] {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=_ROOT)
        metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
        summaries.append({"symbol": symbol, "out_dir": str(out_dir), **metrics})

    lines = [
        "# BTC / DOGE 合约回测对比报告",
        "",
        f"- 策略配置: `{args.config}`",
        f"- 模型 checkpoint: `{args.checkpoint}`",
        f"- 数据窗口: 最近 `{args.days}` 天，周期 `1h`，split=`{args.split}`",
        "",
        "## 对比摘要",
        "",
        "| 标的 | 年化收益 | 总收益 | 最大回撤 | 胜率 | 交易次数 | 盈亏比 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['symbol']} | {_fmt_pct(row.get('annualized_return', 0.0))} | "
            f"{_fmt_pct(row.get('total_return', 0.0))} | {_fmt_pct(row.get('max_drawdown', 0.0))} | "
            f"{_fmt_pct(row.get('win_rate', 0.0))} | {int(row.get('trade_count', 0))} | "
            f"{row.get('profit_factor', 0.0):.2f} |"
        )
    lines.extend(["", "## 分标的产物", ""])
    for row in summaries:
        out = Path(row["out_dir"])
        lines.extend(
            [
                f"### {row['symbol']}",
                "",
                f"- 报告: `{out / 'REPORT.md'}`",
                f"- 指标: `{out / 'metrics.json'}`",
                f"- 资金曲线: `{out / 'equity_curve.png'}`",
                f"- 买卖点: `{out / 'trade_points.png'}`",
                "",
            ]
        )
    lines.append(
        "说明：模型在 BTC 上训练，DOGE 回测属于跨标的迁移验证，结果仅作参考，不代表 DOGE 专用策略表现。"
    )
    (out_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_root / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved comparison report: {out_root / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
