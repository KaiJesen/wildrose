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
from transformer_kit.labels import MarketStateThresholds, build_market_state_targets, estimate_market_state_thresholds
from transformer_kit.pattern_encoder import pattern_config_from_args
from transformer_kit.pattern_model import KlinePatternPredictor, PatternPredictorConfig
from transformer_kit.schedulers import build_adamw_with_warmup_cosine_restarts
from transformer_kit.segment_dataset import PatternSequenceDataset, SequenceSampleIndex, build_sequence_sample_indices
from transformer_kit.train_utils import load_auto_encoder, load_checkpoint, save_checkpoint
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
    p.add_argument("--init-checkpoint", default="checkpoints/0050_market_state_embed/stage2_vqvae.pt")
    p.add_argument(
        "--init-market-checkpoint",
        default="",
        help="加载完整 market-state 模型权重（用于从上一轮 best 微调）",
    )
    p.add_argument("--encoder-lr-scale", type=float, default=0.05)
    p.add_argument("--report-dir", default="reports/0058_market_state_usable")
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument("--return-weight", type=float, default=0.30)
    p.add_argument("--direction-weight", type=float, default=0.50)
    p.add_argument("--volatility-weight", type=float, default=0.10)
    p.add_argument("--risk-weight", type=float, default=0.10)
    p.add_argument("--direction-threshold-quantile", type=float, default=0.25)
    p.add_argument("--risk-threshold-quantile", type=float, default=0.70)
    p.add_argument("--use-class-weights", action="store_true", help="direction/risk CE 使用 train 类别权重（阶段 B）")
    p.add_argument("--risk-focal-loss", action="store_true", help="risk 头使用 focal loss（0053 阶段 C）")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--cum-direction-weight", type=float, default=0.06, help="累计方向辅助损失权重")
    p.add_argument("--min-valid-risk-f1", type=float, default=0.45, help="选模时 valid risk_f1 下限，防止 risk 头坍缩")
    p.add_argument("--early-stop-patience", type=int, default=15)
    p.set_defaults(
        epochs=60,
        batch_size=64,
        d_model=128,
        n_heads=4,
        encoder_layers=2,
        checkpoint_dir="checkpoints/0058_market_state_usable",
        report_dir="reports/0058_market_state_usable",
        use_class_weights=True,
        cum_direction_weight=0.06,
        min_valid_risk_f1=0.45,
    )
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
    """架构师-003 可用阶段选模分数 score_v1。"""
    return (
        0.45 * metrics["cum_direction_acc"]
        + 0.25 * metrics["direction_macro_f1"]
        + 0.15 * max(metrics["return_ic"], -0.05)
        + 0.10 * metrics["risk_f1"]
        - 0.05 * metrics["volatility_mae"]
    )


USABLE_GATES = {
    "cum_direction_acc>=56%": ("cum_direction_acc", lambda v: v >= 0.56),
    "direction_macro_f1>=0.30": ("direction_macro_f1", lambda v: v >= 0.30),
    "risk_f1>=0.48": ("risk_f1", lambda v: v >= 0.48),
    "return_ic>0": ("return_ic", lambda v: v > 0),
    "volatility_mae<=0.10": ("volatility_mae", lambda v: v <= 0.10),
}


def float_metrics(metrics: dict) -> dict[str, float]:
    return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and not k.startswith("_")}


def compute_train_class_weights(loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    dir_counts = torch.zeros(3, dtype=torch.float32)
    risk_counts = torch.zeros(2, dtype=torch.float32)
    for batch in loader:
        dir_counts += torch.bincount(batch["target_direction"].reshape(-1), minlength=3).float()
        risk_counts += torch.bincount(batch["target_risk"].reshape(-1).long(), minlength=2).float()
    dir_w = dir_counts.sum() / (3.0 * dir_counts.clamp(min=1.0))
    risk_w = risk_counts.sum() / (2.0 * risk_counts.clamp(min=1.0))
    return dir_w.to(device), risk_w.to(device)


def label_distribution(raw_log_ret: np.ndarray, samples: list[SequenceSampleIndex], thr: MarketStateThresholds) -> dict:
    dir_counts = np.zeros(3, dtype=np.float64)
    risk_pos = 0
    n = 0
    for s in samples:
        future = raw_log_ret[s.context_end : s.future_end]
        tgt = build_market_state_targets(
            future,
            direction_threshold=thr.direction_threshold,
            risk_vol_threshold=thr.risk_vol_threshold,
        )
        for c in tgt.direction_label.numpy():
            dir_counts[int(c)] += 1
        risk_pos += int(tgt.risk_label.numpy().max() > 0.5)
        n += 1
    return {
        "direction": {f"c{i}": float(dir_counts[i] / max(1.0, dir_counts.sum())) for i in range(3)},
        "risk_positive_rate": float(risk_pos / max(1, n)),
        "num_samples": n,
    }


def acceptance_decision(
    test: dict[str, float],
    diagnostics: dict | None = None,
    *,
    vol_cap: float = 0.10,
) -> tuple[str, list[str], str]:
    checks = {
        k: fn(test[key]) for k, (key, fn) in USABLE_GATES.items() if k != "volatility_mae<=0.10"
    }
    checks["volatility_mae<=cap"] = test["volatility_mae"] <= vol_cap
    passed = sum(checks.values())
    reasons = [k for k, ok in checks.items() if not ok]
    blocking = reasons[0] if reasons else ""
    if diagnostics:
        risk_pred = diagnostics.get("risk_positive_rate_pred")
        if risk_pred is not None and (risk_pred < 0.05 or risk_pred > 0.95):
            reasons.append("risk_prediction_collapsed")
            blocking = blocking or "risk_prediction_collapsed"
    if test["return_ic"] <= 0 or test["direction_macro_f1"] < 0.27:
        decision = "reject"
    elif "risk_prediction_collapsed" in reasons:
        decision = "reject"
    elif passed >= 4:
        decision = "accept"
    else:
        decision = "conditional"
    return decision, reasons, blocking


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


def plot_test_metrics(
    test_metrics: dict[str, float],
    baseline_metrics: dict[str, float] | None,
    out_path: Path,
    dpi: int,
    *,
    baseline_label: str = "0050 formal",
) -> None:
    keys = [
        ("cum_direction_acc", "Cum Dir Acc"),
        ("direction_acc", "Step Dir Acc"),
        ("return_ic", "Return IC"),
        ("direction_macro_f1", "Dir Macro F1"),
        ("volatility_mae", "Vol MAE"),
        ("risk_f1", "Risk F1"),
    ]
    x = np.arange(len(keys))
    width = 0.35 if baseline_metrics else 0.55
    vals = [test_metrics[k] for k, _ in keys]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - (width / 2 if baseline_metrics else 0), vals, width=width, label="current test")
    if baseline_metrics:
        base_vals = [baseline_metrics.get(k, 0.0) for k, _ in keys]
        ax.bar(x + width / 2, base_vals, width=width, label=baseline_label)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in keys], rotation=20, ha="right")
    ax.set_title("Test Metrics vs 0050 Baseline")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_report_md(
    path: Path,
    *,
    run_id: str,
    args: argparse.Namespace,
    thresholds: MarketStateThresholds,
    class_dist: dict,
    test_metrics: dict[str, float],
    test_diagnostics: dict,
    best_valid: dict[str, float],
    baseline_metrics: dict[str, float] | None,
    decision: str,
    reject_reasons: list[str],
    blocking_metric: str = "",
    target_stage: str = "usable",
) -> None:
    lines = [
        f"# {run_id} 多任务市场状态模型训练报告",
        "",
        "## 实验依据",
        "",
        "- `document/架构师-003-理想模型指标目标指导.md`",
        "- `document/架构师-002-模型指标训练指导.md`",
        "- `document/软件设计师_003_市场状态模型最终训练建议[训练建议].md`",
        "",
        f"## 目标阶段: **{target_stage}**",
        "",
        "## 本轮训练配置（架构师-003 阶段 1→可用）",
        "",
        f"- `direction_threshold_quantile={args.direction_threshold_quantile}`",
        f"- `risk_threshold_quantile={args.risk_threshold_quantile}`",
        f"- return/direction/volatility/risk = {args.return_weight}/{args.direction_weight}/"
        f"{args.volatility_weight}/{args.risk_weight}",
        f"- cum_direction_weight={args.cum_direction_weight}",
        f"- min_valid_risk_f1={args.min_valid_risk_f1}",
        f"- class_weights={args.use_class_weights}, risk_focal_loss={args.risk_focal_loss}",
        f"- score_v1 选模, epochs={args.epochs}, early_stop_patience={args.early_stop_patience}",
        "",
        "## 数据与模型",
        "",
        f"- 数据源: `{args.source}` / `{args.symbol}` / `{args.interval}` / `{args.days}` 天",
        f"- 初始化 encoder: `{args.init_checkpoint}`",
        "",
        "## 标签阈值（仅 train 拟合）",
        "",
        f"- `direction_threshold={thresholds.direction_threshold:.8f}`",
        f"- `risk_vol_threshold={thresholds.risk_vol_threshold:.8f}`",
        "",
        "## Train 类别分布",
        "",
        f"- direction: `{class_dist['train']['direction']}`",
        f"- risk_positive_rate: `{class_dist['train']['risk_positive_rate']:.3f}`",
        "",
        "## 测试集指标",
        "",
        f"| 指标 | {run_id} |" + (f" 0050 |" if baseline_metrics else ""),
        f"|------|------|" + ("------|" if baseline_metrics else ""),
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
        if baseline_metrics and key in baseline_metrics:
            row += f" {fmt.format(baseline_metrics[key])} |"
        lines.append(row)
    lines.extend(
        [
            "",
            "## 最佳验证集",
            "",
            f"- composite_score={composite_score(best_valid):.4f}",
            f"- cum_direction_acc={best_valid['cum_direction_acc']:.1%}",
            f"- direction_macro_f1={best_valid['direction_macro_f1']:.3f}",
            f"- return_ic={best_valid['return_ic']:.3f}",
            f"- risk_f1={best_valid['risk_f1']:.3f}",
            f"- volatility_mae={best_valid['volatility_mae']:.6f}",
            "",
            "## 测试诊断",
            "",
            f"- direction_pred: `{ {k: round(v,3) for k,v in test_diagnostics.items() if k.startswith('direction_pred_')} }`",
            f"- risk_positive_rate_true/pred: "
            f"{test_diagnostics.get('risk_positive_rate_true', 0):.3f} / "
            f"{test_diagnostics.get('risk_positive_rate_pred', 0):.3f}",
            "",
            "## 验收结论（可用模型 5 项至少 4 项）",
            "",
            f"- target_stage: **{target_stage}**",
            f"- decision: **{decision}**",
            f"- gates_passed: {5 - len(reject_reasons)}/5",
        ]
    )
    if blocking_metric:
        lines.append(f"- blocking_metric: `{blocking_metric}`")
    if reject_reasons:
        lines.append(f"- 未达标项: {', '.join(reject_reasons)}")
    lines.extend(["", "## 图表", "", "- `01_training_curves.png`", "- `02_test_metrics.png`", ""])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_baseline_metrics(path: str) -> dict[str, float] | None:
    p = Path(path)
    if not p.is_file():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
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
    run_id = report_dir.name

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
    market_init = Path(args.init_market_checkpoint) if args.init_market_checkpoint else None
    if market_init and market_init.is_file():
        ck = load_checkpoint(market_init, map_location=device)
        model.load_state_dict(ck["model"], strict=False)
        print(f"  loaded full market-state model from {market_init}")
    elif init_path.is_file():
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

    class_dist = {
        "train": label_distribution(bundle.raw_log_ret, train_samples, thr),
        "valid": label_distribution(bundle.raw_log_ret, valid_samples, thr),
        "test": label_distribution(bundle.raw_log_ret, test_samples, thr),
    }
    print(f"  thresholds: dir={thr.direction_threshold:.6f} risk_vol={thr.risk_vol_threshold:.6f}")
    print(f"  train direction dist: {class_dist['train']['direction']}")

    dir_class_w = risk_class_w = None
    if args.use_class_weights:
        dir_class_w, risk_class_w = compute_train_class_weights(train_loader, device)
        print(f"  class weights: dir={dir_class_w.cpu().tolist()} risk={risk_class_w.cpu().tolist()}")

    loss_kw = dict(
        return_weight=args.return_weight,
        direction_weight=args.direction_weight,
        volatility_weight=args.volatility_weight,
        risk_weight=args.risk_weight,
        cum_direction_weight=args.cum_direction_weight,
        direction_class_weight=dir_class_w,
        risk_class_weight=risk_class_w,
        risk_focal_loss=args.risk_focal_loss,
        focal_gamma=args.focal_gamma,
    )

    best = float("-inf")
    best_valid: dict[str, float] = {}
    history: list[dict[str, float]] = []
    stale = 0
    for ep in range(1, args.epochs + 1):
        tr = train_market_state_epoch(model, train_loader, opt, sched, device, grad_clip=args.grad_clip, **loss_kw)
        va = evaluate_market_state(model, valid_loader, device, **loss_kw)
        row = {"epoch": ep, "train_loss": tr.loss, **float_metrics(va)}
        history.append(row)
        score = composite_score(va)
        mark = ""
        eligible = va["risk_f1"] >= args.min_valid_risk_f1
        if eligible and score > best:
            best = score
            best_valid = float_metrics(va)
            stale = 0
            save_checkpoint(ckpt_dir / "market_state_best.pt", {"model": model.state_dict(), "args": vars(args)})
            mark = " *saved"
        else:
            stale += 1
            if not eligible:
                mark = " (skip: risk_f1)"
        if ep == 1 or ep % max(1, args.epochs // 6) == 0:
            print(
                f"  ep {ep:03d} tr={tr.loss:.4f} va={va['loss']:.4f} "
                f"dir={va['direction_acc']:.1%} macro_f1={va['direction_macro_f1']:.3f} "
                f"cum={va['cum_direction_acc']:.1%} ic={va['return_ic']:.3f} "
                f"risk_f1={va['risk_f1']:.3f} score={score:.4f}{mark}"
            )
        if args.early_stop_patience > 0 and stale >= args.early_stop_patience:
            print(f"  early stop at epoch {ep} (patience={args.early_stop_patience})")
            break

    ck = torch.load(ckpt_dir / "market_state_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    te_raw = evaluate_market_state(model, test_loader, device, **loss_kw, with_diagnostics=True)
    te = float_metrics(te_raw)
    diagnostics = {k: v for k, v in te_raw.items() if k.startswith("_") or k.startswith("direction_") or k.startswith("risk_") or k.startswith("return_ic_h")}
    print(
        f"[TEST] loss={te['loss']:.4f} dir={te['direction_acc']:.1%} "
        f"macro_f1={te['direction_macro_f1']:.3f} cum={te['cum_direction_acc']:.1%} "
        f"ic={te['return_ic']:.3f} ret_mae={te['return_mae']:.5f} "
        f"vol_mae={te['volatility_mae']:.5f} risk_f1={te['risk_f1']:.3f}"
    )

    baseline = load_baseline_metrics("reports/0052_market_state_class_weights/metrics.json")
    if baseline is None:
        baseline = load_baseline_metrics("reports/0050_market_state_formal/metrics.json")
    decision, reject_reasons, blocking_metric = acceptance_decision(te, diagnostics, vol_cap=0.10)
    plot_training_curves(history, report_dir / "01_training_curves.png", args.dpi)
    plot_test_metrics(te, baseline, report_dir / "02_test_metrics.png", args.dpi, baseline_label="0052 class_weights")
    write_report_md(
        report_dir / "REPORT.md",
        run_id=run_id,
        args=args,
        thresholds=thr,
        class_dist=class_dist,
        test_metrics=te,
        test_diagnostics=diagnostics,
        best_valid=best_valid,
        baseline_metrics=baseline,
        decision=decision,
        reject_reasons=reject_reasons,
        blocking_metric=blocking_metric,
        target_stage="usable",
    )

    payload = {
        "target_stage": "usable",
        "run_id": run_id,
        "args": vars(args),
        "thresholds": {
            "direction_threshold_quantile": args.direction_threshold_quantile,
            "risk_threshold_quantile": args.risk_threshold_quantile,
            "direction_threshold": thr.direction_threshold,
            "risk_vol_threshold": thr.risk_vol_threshold,
        },
        "class_distribution": class_dist,
        "history": history,
        "best_valid_metrics": best_valid,
        "best_composite_score": composite_score(best_valid) if best_valid else None,
        "test_metrics": te,
        "test_diagnostics": diagnostics,
        "baseline_0052": baseline,
        "decision": decision,
        "blocking_metric": blocking_metric,
        "reject_reasons": reject_reasons,
        "gates_passed": 5 - len(reject_reasons),
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
                f"decision={decision}",
                f"composite_score={composite_score(best_valid):.4f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"report saved: {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

