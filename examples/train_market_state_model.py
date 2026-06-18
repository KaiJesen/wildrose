#!/usr/bin/env python3
"""Train multi-task market-state model on real BTC data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

_EX = Path(__file__).resolve().parent
_ROOT = _EX.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))

from _train_common import (
    add_break_vol_args,
    add_data_args,
    add_feature_args,
    add_segment_args,
    add_train_args,
    add_vq_args,
    apply_real_data_defaults,
    fetch_ohlcv_df,
    prepare_bar_series_from_args,
)
from transformer_kit.causal_transformer import CausalTransformerConfig
from transformer_kit.labels import MarketStateThresholds, estimate_market_state_thresholds
from transformer_kit.pattern_encoder import pattern_config_from_args
from transformer_kit.pattern_model import KlinePatternPredictor, PatternPredictorConfig
from transformer_kit.schedulers import build_adamw_with_warmup_cosine_restarts
from transformer_kit.segment_dataset import PatternSequenceDataset, SequenceSampleIndex, build_sequence_sample_indices
from transformer_kit.train_utils import load_auto_encoder, save_checkpoint
from transformer_kit.training import evaluate_market_state, train_market_state_epoch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train multi-task market-state model")
    add_data_args(p)
    add_feature_args(p)
    add_train_args(p)
    add_segment_args(p)
    add_vq_args(p)
    add_break_vol_args(p)
    p.add_argument("--pred-horizon", type=int, default=5)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--trunk-layers", type=int, default=2)
    p.add_argument("--init-checkpoint", default="checkpoints/0041_rebalanced_real_btc/stage2_vqvae.pt")
    p.add_argument("--encoder-lr-scale", type=float, default=0.05)
    p.add_argument("--report-dir", default="reports/0050_market_state_formal")
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument("--return-weight", type=float, default=0.4)
    p.add_argument("--direction-weight", type=float, default=0.4)
    p.add_argument("--volatility-weight", type=float, default=0.15)
    p.add_argument("--risk-weight", type=float, default=0.05)
    p.add_argument("--direction-threshold-quantile", type=float, default=0.35)
    p.add_argument("--risk-threshold-quantile", type=float, default=0.8)
    p.set_defaults(epochs=36, batch_size=64, d_model=128, n_heads=4, encoder_layers=2)
    return p.parse_args()


def build_split_samples(bundle, args) -> tuple[list[SequenceSampleIndex], list[SequenceSampleIndex], list[SequenceSampleIndex]]:
    def split(idx: np.ndarray) -> list[SequenceSampleIndex]:
        return build_sequence_sample_indices(
            bundle.bars.shape[0],
            context_bars=args.context_bars,
            pred_horizon=args.pred_horizon,
            stride=args.stride,
            index_min=int(idx.min()),
            index_max=int(idx.max()),
        )

    return split(bundle.train_idx), split(bundle.valid_idx), split(bundle.test_idx)


def collect_future_train_windows(raw_log_ret: np.ndarray, samples: list[SequenceSampleIndex]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for s in samples:
        rows.append(raw_log_ret[s.context_end : s.future_end].astype(np.float32))
    return np.stack(rows, axis=0)


def make_loader(bundle, samples, args, thresholds: MarketStateThresholds, *, shuffle: bool, drop_last: bool) -> DataLoader:
    ds = PatternSequenceDataset(
        bundle.bars,
        samples,
        bundle.raw_log_ret,
        zscore_window=bundle.zscore_window,
        return_market_state_targets=True,
        direction_threshold=thresholds.direction_threshold,
        risk_vol_threshold=thresholds.risk_vol_threshold,
    )
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, drop_last=drop_last)


def composite_score(metrics: dict[str, float]) -> float:
    """与软件设计师文档一致的 valid 选模分数。"""
    return (
        metrics["cum_direction_acc"]
        + 0.5 * metrics["return_ic"]
        + 0.25 * metrics["direction_macro_f1"]
        - 0.1 * metrics["volatility_mae"]
    )


def plot_training_curves(history: list[dict[str, float]], out_path: Path, dpi: int) -> None:
    epochs = [int(h["epoch"]) for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0, 0].plot(epochs, [h["loss"] for h in history], label="valid")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.25)

    axes[0, 1].plot(epochs, [h["direction_acc"] for h in history], label="direction_acc")
    axes[0, 1].plot(epochs, [h["cum_direction_acc"] for h in history], label="cum_direction_acc")
    axes[0, 1].set_title("Direction Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.25)

    axes[1, 0].plot(epochs, [h["return_ic"] for h in history], label="return_ic")
    axes[1, 0].axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    axes[1, 0].set_title("Return IC (valid)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.25)

    axes[1, 1].plot(epochs, [h["volatility_mae"] for h in history], label="volatility_mae")
    axes[1, 1].plot(epochs, [h["risk_f1"] for h in history], label="risk_f1")
    axes[1, 1].set_title("Vol MAE / Risk F1")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.25)

    fig.suptitle("Market State Model Training Curves", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_test_metrics(test_metrics: dict[str, float], smoke_metrics: dict[str, float] | None, out_path: Path, dpi: int) -> None:
    keys = [
        ("cum_direction_acc", "Cum Dir Acc"),
        ("direction_acc", "Step Dir Acc"),
        ("return_ic", "Return IC"),
        ("direction_macro_f1", "Dir Macro F1"),
        ("volatility_mae", "Vol MAE"),
        ("risk_f1", "Risk F1"),
    ]
    x = np.arange(len(keys))
    width = 0.35 if smoke_metrics else 0.55
    vals = [test_metrics[k] for k, _ in keys]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - (width / 2 if smoke_metrics else 0), vals, width=width, label="formal test")
    if smoke_metrics:
        smoke_vals = [smoke_metrics.get(k, 0.0) for k, _ in keys]
        ax.bar(x + width / 2, smoke_vals, width=width, label="smoke test (0049)")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in keys], rotation=20, ha="right")
    ax.set_title("Test Metrics: Formal vs Smoke Baseline")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_report_md(
    path: Path,
    *,
    args: argparse.Namespace,
    thresholds: MarketStateThresholds,
    test_metrics: dict[str, float],
    best_valid: dict[str, float],
    smoke_metrics: dict[str, float] | None,
) -> None:
    lines = [
        "# 0050 多任务市场状态模型正式实验报告",
        "",
        "## 实验依据",
        "",
        "依据 `document/软件设计师_001_多任务市场状态改造实施与训练建议.md` 第一轮正式配置执行。",
        "",
        "## 数据与模型",
        "",
        f"- 数据源: `{args.source}` / `{args.symbol}` / `{args.interval}` / `{args.days}` 天",
        f"- 上下文: `{args.context_bars}` bar，预测 horizon: `{args.pred_horizon}`",
        f"- `d_model={args.d_model}`, `n_heads={args.n_heads}`, `trunk_layers={args.trunk_layers}`",
        f"- 多任务头: return / direction(3-class) / volatility / risk(2-class)",
        f"- 初始化 encoder: `{args.init_checkpoint}`",
        "",
        "## 标签阈值（仅 train 拟合）",
        "",
        f"- `direction_threshold={thresholds.direction_threshold:.8f}`",
        f"- `risk_vol_threshold={thresholds.risk_vol_threshold:.8f}`",
        "",
        "## 损失权重",
        "",
        f"- return={args.return_weight}, direction={args.direction_weight}, "
        f"volatility={args.volatility_weight}, risk={args.risk_weight}",
        "",
        "## 测试集指标",
        "",
        f"| 指标 | 正式实验 |" + (" Smoke(0049) |" if smoke_metrics else ""),
        f"|------|----------|" + ("-----------|" if smoke_metrics else ""),
    ]
    metric_rows = [
        ("cum_direction_acc", "{:.1%}"),
        ("direction_acc", "{:.1%}"),
        ("direction_macro_f1", "{:.3f}"),
        ("return_ic", "{:.3f}"),
        ("return_mae", "{:.6f}"),
        ("volatility_mae", "{:.6f}"),
        ("risk_f1", "{:.3f}"),
        ("loss", "{:.4f}"),
    ]
    for key, fmt in metric_rows:
        row = f"| {key} | {fmt.format(test_metrics[key])} |"
        if smoke_metrics and key in smoke_metrics:
            row += f" {fmt.format(smoke_metrics[key])} |"
        lines.append(row)
    lines.extend(
        [
            "",
            "## 最佳验证集（选模分数）",
            "",
            f"- composite_score={composite_score(best_valid):.4f}",
            f"- cum_direction_acc={best_valid['cum_direction_acc']:.1%}",
            f"- return_ic={best_valid['return_ic']:.3f}",
            f"- direction_macro_f1={best_valid['direction_macro_f1']:.3f}",
            f"- volatility_mae={best_valid['volatility_mae']:.6f}",
            "",
            "## 图表",
            "",
            "- `01_training_curves.png`：训练/验证损失与多任务指标曲线",
            "- `02_test_metrics.png`：测试集指标（与 smoke 对比）",
            "",
            "## 结论与下一步",
            "",
        ]
    )
    if test_metrics["cum_direction_acc"] >= (smoke_metrics or {}).get("cum_direction_acc", 0.0):
        lines.append("- 累计方向准确率不低于 smoke 基线。")
    else:
        lines.append("- 累计方向准确率低于 smoke，需加强 encoder 预训练或调整损失权重。")
    if test_metrics["return_ic"] > 0:
        lines.append("- 收益 IC 在测试集为正，回归头具备一定预测力。")
    else:
        lines.append("- 收益 IC 仍为负，建议加大 return 权重或启用 horizon-aware 市场状态头。")
    lines.append("- 后续可做 CPC 主干 vs VQ 主干同头对比（文档 5.3）。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_smoke_metrics() -> dict[str, float] | None:
    smoke_path = Path("reports/0049_market_state_smoke/metrics.json")
    if not smoke_path.is_file():
        return None
    data = json.loads(smoke_path.read_text(encoding="utf-8"))
    return data.get("test_metrics")


def main() -> int:
    args = parse_args()
    apply_real_data_defaults(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    bundle = prepare_bar_series_from_args(fetch_ohlcv_df(args), args)
    train_samples, valid_samples, test_samples = build_split_samples(bundle, args)
    thr = estimate_market_state_thresholds(
        collect_future_train_windows(bundle.raw_log_ret, train_samples),
        direction_quantile=args.direction_threshold_quantile,
        risk_quantile=args.risk_threshold_quantile,
    )
    train_loader = make_loader(bundle, train_samples, args, thr, shuffle=True, drop_last=True)
    valid_loader = make_loader(bundle, valid_samples, args, thr, shuffle=False, drop_last=False)
    test_loader = make_loader(bundle, test_samples, args, thr, shuffle=False, drop_last=False)

    auto_cfg = pattern_config_from_args(args)
    model = KlinePatternPredictor(
        PatternPredictorConfig(
            auto_segment=auto_cfg,
            trunk=CausalTransformerConfig(d_model=args.d_model, n_heads=args.n_heads, n_layers=args.trunk_layers),
            pred_horizon=args.pred_horizon,
            pred_feat_dim=1,
            pool_mode="attn",
            learnable_scale=True,
            use_horizon_head=False,
            use_market_state_head=True,
            direction_classes=3,
            risk_classes=2,
        )
    ).to(device)

    init_path = Path(args.init_checkpoint)
    if init_path.is_file():
        load_auto_encoder(model.auto_encoder, init_path)
        print(f"  loaded auto encoder from {init_path}")

    enc_params = list(model.auto_encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt, sched = build_adamw_with_warmup_cosine_restarts(
        [{"params": enc_params, "lr": args.lr * args.encoder_lr_scale}, {"params": head_params, "lr": args.lr}],
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        t0=args.cosine_t0,
        t_mult=args.cosine_t_mult,
        eta_min=args.eta_min,
    )

    best = float("-inf")
    best_valid: dict[str, float] = {}
    history: list[dict[str, float]] = []
    for ep in range(1, args.epochs + 1):
        tr = train_market_state_epoch(
            model,
            train_loader,
            opt,
            sched,
            device,
            grad_clip=args.grad_clip,
            return_weight=args.return_weight,
            direction_weight=args.direction_weight,
            volatility_weight=args.volatility_weight,
            risk_weight=args.risk_weight,
        )
        va = evaluate_market_state(
            model,
            valid_loader,
            device,
            return_weight=args.return_weight,
            direction_weight=args.direction_weight,
            volatility_weight=args.volatility_weight,
            risk_weight=args.risk_weight,
        )
        row = {"epoch": ep, "train_loss": tr.loss, **va}
        history.append(row)
        score = composite_score(va)
        mark = ""
        if score > best:
            best = score
            best_valid = dict(va)
            save_checkpoint(ckpt_dir / "market_state_best.pt", {"model": model.state_dict(), "args": vars(args)})
            mark = " *saved"
        if ep == 1 or ep % max(1, args.epochs // 6) == 0:
            print(
                f"  ep {ep:03d} tr={tr.loss:.4f} va={va['loss']:.4f} "
                f"dir={va['direction_acc']:.1%} cum={va['cum_direction_acc']:.1%} "
                f"ic={va['return_ic']:.3f} vol_mae={va['volatility_mae']:.4f}{mark}"
            )

    ck = torch.load(ckpt_dir / "market_state_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    te = evaluate_market_state(
        model,
        test_loader,
        device,
        return_weight=args.return_weight,
        direction_weight=args.direction_weight,
        volatility_weight=args.volatility_weight,
        risk_weight=args.risk_weight,
    )
    print(
        f"[TEST] loss={te['loss']:.4f} dir={te['direction_acc']:.1%} "
        f"macro_f1={te['direction_macro_f1']:.3f} cum={te['cum_direction_acc']:.1%} "
        f"ic={te['return_ic']:.3f} ret_mae={te['return_mae']:.5f} "
        f"vol_mae={te['volatility_mae']:.5f} risk_f1={te['risk_f1']:.3f}"
    )

    smoke_metrics = load_smoke_metrics()
    plot_training_curves(history, report_dir / "01_training_curves.png", args.dpi)
    plot_test_metrics(te, smoke_metrics, report_dir / "02_test_metrics.png", args.dpi)
    write_report_md(
        report_dir / "REPORT.md",
        args=args,
        thresholds=thr,
        test_metrics=te,
        best_valid=best_valid,
        smoke_metrics=smoke_metrics,
    )

    payload = {
        "args": vars(args),
        "thresholds": {
            "direction_threshold": thr.direction_threshold,
            "risk_vol_threshold": thr.risk_vol_threshold,
        },
        "history": history,
        "best_valid_metrics": best_valid,
        "best_composite_score": composite_score(best_valid) if best_valid else None,
        "test_metrics": te,
        "smoke_baseline": smoke_metrics,
    }
    (report_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (report_dir / "metrics.txt").write_text(
        "\n".join(
            [
                "=== Market State Model ===",
                f"source={args.source} symbol={args.symbol} interval={args.interval} days={args.days}",
                f"direction_threshold={thr.direction_threshold:.8f}",
                f"risk_vol_threshold={thr.risk_vol_threshold:.8f}",
                f"direction_acc={te['direction_acc']:.1%}",
                f"direction_macro_f1={te['direction_macro_f1']:.3f}",
                f"cum_direction_acc={te['cum_direction_acc']:.1%}",
                f"return_ic={te['return_ic']:.3f}",
                f"return_mae={te['return_mae']:.6f}",
                f"volatility_mae={te['volatility_mae']:.6f}",
                f"risk_f1={te['risk_f1']:.3f}",
                f"loss={te['loss']:.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"report saved: {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

