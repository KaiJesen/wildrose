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
    p.add_argument(
        "--target-stage",
        choices=["usable", "balanced_mature", "cum_return_recovery", "return_direction_branch", "step_return_recovery"],
        default="return_direction_branch",
    )
    p.add_argument("--report-dir", default="reports/0062c_market_state_cum_return_stabilized")
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument("--return-weight", type=float, default=0.35)
    p.add_argument("--direction-weight", type=float, default=0.30)
    p.add_argument("--volatility-weight", type=float, default=0.10)
    p.add_argument("--risk-weight", type=float, default=0.09)
    p.add_argument("--cum-return-weight", type=float, default=0.18)
    p.add_argument("--cum-direction-head-weight", type=float, default=0.03)
    p.add_argument("--return-consistency-weight", type=float, default=0.01)
    p.add_argument(
        "--return-horizon-weights",
        type=float,
        nargs="+",
        default=None,
        help="逐步 return Huber 损失逐 horizon 权重（默认全 1.0）",
    )
    p.add_argument("--use-cum-heads", action="store_true")
    p.add_argument("--use-horizon-return-head", action="store_true")
    p.add_argument("--detach-risk-vol-heads", action="store_true")
    p.add_argument("--return-direction-hidden-mult", type=float, default=1.0)
    p.add_argument("--direction-threshold-quantile", type=float, default=0.25)
    p.add_argument("--risk-threshold-quantile", type=float, default=0.70)
    p.add_argument("--use-class-weights", action="store_true", help="direction/risk CE 使用 train 类别权重（阶段 B）")
    p.add_argument(
        "--use-direction-class-weights",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="direction CE 类别权重（默认跟随 use_class_weights）",
    )
    p.add_argument(
        "--use-risk-class-weights",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="risk CE 类别权重（默认跟随 use_class_weights）",
    )
    p.add_argument(
        "--detach-risk-vol-after-epoch",
        type=int,
        default=0,
        help="从该 epoch 起启用 detach_risk_vol_heads（0=不启用）",
    )
    p.add_argument("--risk-focal-loss", action="store_true", help="risk 头使用 focal loss（0053 阶段 C）")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--cum-direction-weight", type=float, default=0.045, help="累计方向辅助损失权重")
    p.add_argument("--min-valid-risk-f1", type=float, default=0.45, help="选模时 valid risk_f1 下限")
    p.add_argument("--balanced-class-weights", action="store_true", help="down↑ flat↓ risk正类↓ 的 train 权重校准")
    p.add_argument("--early-stop-patience", type=int, default=12)
    p.add_argument("--bias-stop-return-ic", type=int, default=5, help="valid return_ic 连续<=0 则提前停止")
    p.add_argument("--bias-stop-cum-dir", type=float, default=0.53, help="valid cum_direction_acc 连续低于此值则停止")
    p.set_defaults(
        epochs=30,
        batch_size=64,
        d_model=128,
        n_heads=4,
        encoder_layers=2,
        lr=6e-5,
        encoder_lr_scale=0.0,
        checkpoint_dir="checkpoints/0062c_market_state_cum_return_stabilized",
        report_dir="reports/0062c_market_state_cum_return_stabilized",
        use_class_weights=True,
        balanced_class_weights=False,
        init_market_checkpoint="checkpoints/0060_market_state_cum_return_recovery/market_state_best.pt",
        cum_direction_weight=0.0,
        min_valid_risk_f1=0.45,
        use_cum_heads=True,
        use_horizon_return_head=True,
        detach_risk_vol_heads=False,
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


def composite_score_usable(metrics: dict[str, float]) -> float:
    """架构师-003 可用阶段选模分数 score_v1。"""
    return (
        0.45 * metrics["cum_direction_acc"]
        + 0.25 * metrics["direction_macro_f1"]
        + 0.15 * max(metrics["return_ic"], -0.05)
        + 0.10 * metrics["risk_f1"]
        - 0.05 * metrics["volatility_mae"]
    )


def distribution_health(diagnostics: dict[str, float]) -> float:
    """架构师-004 distribution_health。"""
    risk_true = diagnostics.get("risk_positive_rate_true", 0.23)
    risk_pred = diagnostics.get("risk_positive_rate_pred", 0.23)
    risk_ratio = risk_pred / max(risk_true, 1e-6)
    flat_true = diagnostics.get("direction_true_c1", 0.25)
    flat_pred = diagnostics.get("direction_pred_c1", 0.25)
    down_true = diagnostics.get("direction_true_c0", 0.35)
    down_pred = diagnostics.get("direction_pred_c0", 0.35)
    return (
        1.0
        - abs(risk_ratio - 1.0) * 0.3
        - abs(flat_pred - flat_true) * 1.0
        - max(0.0, down_true - down_pred) * 0.8
    )


def composite_score_balanced(metrics: dict[str, float], diagnostics: dict[str, float]) -> float:
    """架构师-004 score_0059。"""
    clipped_ic = max(-1.0, min(1.0, metrics["return_ic"] / 0.10))
    vol_q = 1.0 - min(1.0, max(0.0, metrics["volatility_mae"] / 0.12))
    dist_h = distribution_health(diagnostics)
    return (
        0.25 * metrics["cum_direction_acc"]
        + 0.25 * metrics["direction_macro_f1"]
        + 0.20 * clipped_ic
        + 0.15 * metrics["risk_f1"]
        + 0.10 * vol_q
        + 0.05 * dist_h
    )


def composite_score_recovery(metrics: dict[str, float], diagnostics: dict[str, float]) -> float:
    """架构师-005 score_0060。"""
    clipped_ic = max(-1.0, min(1.0, metrics["return_ic"] / 0.10))
    vol_q = 1.0 - min(1.0, max(0.0, metrics["volatility_mae"] / 0.12))
    dist_h = distribution_health(diagnostics)
    return (
        0.25 * metrics["cum_direction_acc"]
        + 0.22 * metrics["direction_macro_f1"]
        + 0.23 * clipped_ic
        + 0.12 * metrics["risk_f1"]
        + 0.10 * vol_q
        + 0.08 * dist_h
    )


def composite_score_step_return_recovery(metrics: dict[str, float], diagnostics: dict[str, float]) -> float:
    """架构师-010 score_0064。"""
    dist_h = distribution_health(diagnostics)
    clipped_return_ic = max(-1.0, min(1.0, metrics["return_ic"] / 0.04))
    clipped_cum_return_ic = max(-1.0, min(1.0, metrics.get("cum_return_ic", 0.0) / 0.12))
    return (
        0.28 * clipped_return_ic
        + 0.20 * clipped_cum_return_ic
        + 0.16 * metrics.get("cum_direction_from_return_acc", 0.0)
        + 0.14 * metrics["direction_macro_f1"]
        + 0.10 * metrics["risk_f1"]
        - 0.05 * metrics["volatility_mae"]
        + 0.07 * dist_h
    )


def composite_score_return_direction_branch(metrics: dict[str, float], diagnostics: dict[str, float]) -> float:
    """架构师-008 score_0062：优先 cum_return_ic，降低 cum_direction_head 权重。"""
    dist_h = distribution_health(diagnostics)
    return (
        0.22 * max(metrics.get("cum_return_ic", 0.0), -0.05)
        + 0.20 * metrics["direction_macro_f1"]
        + 0.15 * metrics.get("cum_direction_from_return_acc", 0.0)
        + 0.14 * max(metrics["return_ic"], -0.05)
        + 0.12 * metrics["risk_f1"]
        - 0.05 * metrics["volatility_mae"]
        + 0.07 * dist_h
        + 0.05 * metrics.get("cum_direction_head_acc", 0.0)
    )


def composite_score(metrics: dict[str, float], diagnostics: dict[str, float] | None = None, *, stage: str = "usable") -> float:
    if diagnostics is None:
        diagnostics = {}
    if stage == "step_return_recovery":
        return composite_score_step_return_recovery(metrics, diagnostics)
    if stage == "return_direction_branch":
        return composite_score_return_direction_branch(metrics, diagnostics)
    if stage == "cum_return_recovery":
        return composite_score_recovery(metrics, diagnostics)
    if stage == "balanced_mature":
        return composite_score_balanced(metrics, diagnostics)
    return composite_score_usable(metrics)


USABLE_GATES = {
    "cum_direction_acc>=56%": ("cum_direction_acc", lambda v: v >= 0.56),
    "direction_macro_f1>=0.30": ("direction_macro_f1", lambda v: v >= 0.30),
    "risk_f1>=0.48": ("risk_f1", lambda v: v >= 0.48),
    "return_ic>0": ("return_ic", lambda v: v > 0),
    "volatility_mae<=0.10": ("volatility_mae", lambda v: v <= 0.10),
}

BALANCED_GATES = {
    "cum_direction_acc>=58%": ("cum_direction_acc", lambda v: v >= 0.58),
    "direction_macro_f1>=0.33": ("direction_macro_f1", lambda v: v >= 0.33),
    "return_ic>=0.05": ("return_ic", lambda v: v >= 0.05),
    "volatility_mae<=0.085": ("volatility_mae", lambda v: v <= 0.085),
    "risk_f1>=0.52": ("risk_f1", lambda v: v >= 0.52),
}

RECOVERY_GATES = {
    "cum_direction_acc>=58%": ("cum_direction_acc", lambda v: v >= 0.58),
    "direction_macro_f1>=0.34": ("direction_macro_f1", lambda v: v >= 0.34),
    "return_ic>=0.05": ("return_ic", lambda v: v >= 0.05),
    "volatility_mae<=0.070": ("volatility_mae", lambda v: v <= 0.070),
    "risk_f1>=0.53": ("risk_f1", lambda v: v >= 0.53),
}

BRANCH_GATES = {
    "cum_return_ic>=0.08": ("cum_return_ic", lambda v: v >= 0.08),
    "return_ic>=0.035": ("return_ic", lambda v: v >= 0.035),
    "direction_macro_f1>=0.32": ("direction_macro_f1", lambda v: v >= 0.32),
    "risk_f1>=0.50": ("risk_f1", lambda v: v >= 0.50),
    "volatility_mae<=0.070": ("volatility_mae", lambda v: v <= 0.070),
    "cum_direction_from_return_acc>=54%": ("cum_direction_from_return_acc", lambda v: v >= 0.54),
}

STEP_RETURN_GATES = {
    "return_ic>=0.020": ("return_ic", lambda v: v >= 0.020),
    "cum_return_ic>=0.100": ("cum_return_ic", lambda v: v >= 0.100),
    "cum_direction_from_return_acc>=58%": ("cum_direction_from_return_acc", lambda v: v >= 0.58),
    "direction_macro_f1>=0.320": ("direction_macro_f1", lambda v: v >= 0.320),
    "risk_f1>=0.530": ("risk_f1", lambda v: v >= 0.530),
    "volatility_mae<=0.070": ("volatility_mae", lambda v: v <= 0.070),
}

ACCEPTANCE_TRACK = {
    "usable": "A",
    "balanced_mature": "A-ext",
    "cum_return_recovery": "A-ext",
    "return_direction_branch": "B",
    "step_return_recovery": "B",
}

ACCEPTANCE_TRACK_LABEL = {
    "A": "usable 主基线轨",
    "A-ext": "成熟/恢复过渡轨",
    "B": "新结构分支轨",
}


def acceptance_track_info(stage: str) -> dict[str, str]:
    track = ACCEPTANCE_TRACK.get(stage, stage)
    if stage == "step_return_recovery":
        doc = "document/010/架构师-010-0064收益指标恢复执行指导.md"
    elif stage == "return_direction_branch":
        doc = "document/011/架构师-011-0064训练复盘与下一阶段方向.md"
    elif track == "B":
        doc = "document/009/项目经理-009-双轨验收与基线说明.md"
    elif track == "A":
        doc = "document/003/架构师-003-理想模型指标目标指导.md"
    else:
        doc = "document/004/架构师-004-当前训练进度复盘与目标修正.md"
    return {
        "acceptance_track": track,
        "acceptance_track_label": ACCEPTANCE_TRACK_LABEL.get(track, track),
        "acceptance_doc": doc,
    }


def branch_metadata(stage: str) -> dict[str, str]:
    """架构师-011 分支类型与能力边界标注。"""
    if stage == "return_direction_branch":
        return {
            "branch_type": "cum_return_candidate",
            "known_limitation": "weak_step_return_ic",
            "branch_status": "recommended_stable_template",
        }
    if stage == "step_return_recovery":
        return {
            "branch_type": "step_return_recovery_experimental",
            "known_limitation": "conflicts_with_cum_return_branch",
            "branch_status": "experimental_not_recommended_by_default",
        }
    return {
        "branch_type": "usable_or_transition",
        "known_limitation": "",
        "branch_status": "standard",
    }


def collapse_gates(diagnostics: dict[str, float]) -> dict[str, bool]:
    """架构师-008 分类坍缩硬门槛（valid 选模 + 报告）。"""
    risk_true = diagnostics.get("risk_positive_rate_true", 0.0)
    risk_pred = diagnostics.get("risk_positive_rate_pred", 0.0)
    risk_ratio = risk_pred / max(risk_true, 1e-6) if risk_true > 0 else 999.0
    down_pred = diagnostics.get("direction_pred_c0", 0.0)
    flat_pred = diagnostics.get("direction_pred_c1", 0.0)
    up_pred = diagnostics.get("direction_pred_c2", 0.0)
    return {
        "direction_pred_down<=60%": down_pred <= 0.60,
        "direction_pred_flat>=8%": flat_pred >= 0.08,
        "direction_pred_up>=10%": up_pred >= 0.10,
        "risk_positive_rate_pred>=5%": risk_pred >= 0.05,
        "risk_ratio<=1.8": risk_ratio <= 1.8,
    }


def collapse_auto_reject(test: dict[str, float], diagnostics: dict[str, float]) -> list[str]:
    """架构师-008 测试集坍缩自动 reject。"""
    reasons: list[str] = []
    if diagnostics.get("direction_pred_c1", -1.0) == 0.0:
        reasons.append("direction_pred_flat==0")
    if diagnostics.get("risk_positive_rate_pred", -1.0) == 0.0:
        reasons.append("risk_positive_rate_pred==0")
    if test.get("direction_macro_f1", 1.0) < 0.25:
        reasons.append("direction_macro_f1<0.25")
    if test.get("cum_direction_head_acc", 1.0) < 0.45:
        reasons.append("cum_direction_head_acc<45%")
    return reasons


def _in_range(v: float, lo: float, hi: float) -> bool:
    return lo <= v <= hi


def distribution_gates(diagnostics: dict[str, float], *, stage: str = "balanced_mature") -> dict[str, bool]:
    risk_true = diagnostics.get("risk_positive_rate_true", 0.0)
    risk_pred = diagnostics.get("risk_positive_rate_pred", 0.0)
    risk_ratio = risk_pred / max(risk_true, 1e-6) if risk_true > 0 else 999.0
    down_pred = diagnostics.get("direction_pred_c0", 0.0)
    flat_pred = diagnostics.get("direction_pred_c1", 0.0)
    up_pred = diagnostics.get("direction_pred_c2", 0.0)
    if stage in ("return_direction_branch", "step_return_recovery"):
        return {
            "risk_ratio_in_[0.5,1.8]": _in_range(risk_ratio, 0.5, 1.8),
            "direction_pred_down_in_[25%,55%]": _in_range(down_pred, 0.25, 0.55),
            "direction_pred_flat_in_[10%,45%]": _in_range(flat_pred, 0.10, 0.45),
            "direction_pred_up_in_[15%,45%]": _in_range(up_pred, 0.15, 0.45),
        }
    if stage == "cum_return_recovery":
        return {
            "risk_ratio_in_[0.9,1.5]": _in_range(risk_ratio, 0.9, 1.5),
            "direction_pred_down_in_[32%,43%]": _in_range(down_pred, 0.32, 0.43),
            "direction_pred_flat_in_[25%,38%]": _in_range(flat_pred, 0.25, 0.38),
            "direction_pred_up_in_[26%,38%]": _in_range(up_pred, 0.26, 0.38),
        }
    return {
        "risk_ratio<=1.7": risk_ratio <= 1.7,
        "direction_pred_flat<=42%": flat_pred <= 0.42,
        "direction_pred_down>=25%": down_pred >= 0.25,
    }


def step_return_auto_reject(test: dict[str, float], diagnostics: dict[str, float]) -> list[str]:
    """架构师-010 0064 自动 reject。"""
    reasons: list[str] = []
    if test.get("return_ic", 1.0) < 0.015:
        reasons.append("return_ic<0.015")
    if test.get("cum_return_ic", 1.0) < 0.090:
        reasons.append("cum_return_ic<0.090")
    if test.get("cum_direction_from_return_acc", 1.0) < 0.56:
        reasons.append("cum_direction_from_return_acc<56%")
    if not all(collapse_gates(diagnostics).values()):
        reasons.append("collapse_gates_failed")
    reasons.extend(collapse_auto_reject(test, diagnostics))
    return list(dict.fromkeys(reasons))


def valid_hard_gates(metrics: dict[str, float], diagnostics: dict[str, float], *, stage: str) -> tuple[bool, str]:
    """valid 选模硬门槛。"""
    risk_true = diagnostics.get("risk_positive_rate_true", 0.0)
    risk_pred = diagnostics.get("risk_positive_rate_pred", 0.0)
    if stage == "step_return_recovery":
        if metrics.get("return_ic", 0.0) <= 0.015:
            return False, "return_ic<=0.015"
        if metrics.get("cum_return_ic", 0.0) < 0.09:
            return False, "cum_return_ic<0.09"
        if metrics.get("cum_direction_from_return_acc", 0.0) < 0.56:
            return False, "cum_dir_from_ret<56%"
        if metrics.get("direction_macro_f1", 0.0) < 0.31:
            return False, "macro_f1<0.31"
        if metrics.get("risk_f1", 0.0) < 0.50:
            return False, "risk_f1<0.50"
        if not all(collapse_gates(diagnostics).values()):
            return False, "collapse_gates"
        return True, ""
    if stage == "return_direction_branch":
        cum_ret_ic = metrics.get("cum_return_ic", 0.0)
        ret_ic = metrics.get("return_ic", 0.0)
        if ret_ic <= 0.02 and not (cum_ret_ic >= 0.04 and ret_ic >= -0.02):
            return False, "return_ic<=0.02"
        if cum_ret_ic <= 0.02:
            return False, "cum_return_ic<=0.02"
        if metrics.get("direction_macro_f1", 0.0) < 0.30:
            return False, "macro_f1<0.30"
        down_pred = diagnostics.get("direction_pred_c0", 0.0)
        flat_pred = diagnostics.get("direction_pred_c1", 0.0)
        up_pred = diagnostics.get("direction_pred_c2", 0.0)
        if down_pred > 0.60:
            return False, "direction_down_collapse"
        if flat_pred < 0.08:
            return False, "direction_flat_collapse"
        if up_pred < 0.10:
            return False, "direction_up_collapse"
        risk_pred = diagnostics.get("risk_positive_rate_pred", 0.0)
        if risk_pred < 0.05:
            return False, "risk_pred_collapse"
        if risk_true > 0:
            risk_ratio = risk_pred / max(risk_true, 1e-6)
            if risk_ratio > 1.8:
                return False, "risk_ratio>1.8"
        return True, ""
    if risk_true > 0 and risk_pred / risk_true > 1.6:
        return False, "risk_ratio>1.6"
    if metrics["direction_macro_f1"] < 0.33:
        return False, "macro_f1<0.33"
    # 累计方向恢复：cum 达标时允许 return_ic 轻微为负
    if metrics["cum_direction_acc"] >= 0.575 and metrics["return_ic"] >= -0.01:
        return True, ""
    if metrics["cum_direction_acc"] < 0.55:
        return False, "cum_dir<55%"
    if metrics["return_ic"] < 0.0:
        return False, "return_ic<0"
    return True, ""


def selection_eligible(
    metrics: dict[str, float],
    diagnostics: dict[str, float],
    *,
    min_risk_f1: float,
    stage: str,
) -> tuple[bool, str]:
    if metrics["risk_f1"] < min_risk_f1:
        return False, "risk_f1"
    if stage == "step_return_recovery":
        ok, reason = valid_hard_gates(metrics, diagnostics, stage=stage)
        if not ok:
            return False, reason
        return True, ""
    if stage == "return_direction_branch":
        ok, reason = valid_hard_gates(metrics, diagnostics, stage=stage)
        if not ok:
            return False, reason
        return True, ""
    if stage == "cum_return_recovery":
        ok, reason = valid_hard_gates(metrics, diagnostics, stage=stage)
        if not ok:
            return False, reason
        return True, ""
    if metrics["return_ic"] <= 0:
        return False, "return_ic"
    risk_true = diagnostics.get("risk_positive_rate_true", 0.0)
    risk_pred = diagnostics.get("risk_positive_rate_pred", 0.0)
    if risk_true > 0 and risk_pred / risk_true > 1.8:
        return False, "risk_ratio"
    if diagnostics.get("direction_pred_c1", 0.0) > 0.45:
        return False, "flat_bias"
    if stage == "balanced_mature" and metrics["direction_macro_f1"] < 0.30:
        return False, "macro_f1"
    return True, ""


def training_bias_detected(
    metrics: dict[str, float],
    diagnostics: dict[str, float],
    *,
    neg_return_ic_streak: int,
    low_cum_dir_streak: int,
    stage: str,
    bias_stop_return_ic: int = 5,
) -> str:
    if stage == "cum_return_recovery":
        if neg_return_ic_streak >= 5:
            return "return_ic_nonpositive_streak"
        if low_cum_dir_streak >= 5:
            return "cum_direction_low_streak"
        risk_true = diagnostics.get("risk_positive_rate_true", 0.0)
        risk_pred = diagnostics.get("risk_positive_rate_pred", 0.0)
        if risk_true > 0 and risk_pred / risk_true > 1.8:
            return "risk_overpredict"
        return ""
    if stage in ("return_direction_branch", "step_return_recovery"):
        if neg_return_ic_streak >= bias_stop_return_ic and metrics.get("cum_return_ic", 0.0) < 0.03:
            return "return_ic_negative_streak"
        if diagnostics.get("risk_positive_rate_pred", 1.0) == 0.0:
            return "risk_pred_collapsed"
        if diagnostics.get("direction_pred_c0", 0.0) > 0.75:
            return "direction_down_collapse"
        if diagnostics.get("risk_positive_rate_pred", 0.0) > 0.50:
            return "risk_overpredict"
        if diagnostics.get("direction_pred_c1", 0.0) > 0.55:
            return "flat_overpredict"
        return ""
    if neg_return_ic_streak >= 5:
        return "return_ic_negative_streak"
    if diagnostics.get("risk_positive_rate_pred", 0.0) > 0.50:
        return "risk_overpredict"
    if diagnostics.get("direction_pred_c1", 0.0) > 0.50:
        return "flat_overpredict"
    return ""


def float_metrics(metrics: dict) -> dict[str, float]:
    return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and not k.startswith("_")}


def compute_train_class_weights(
    loader: DataLoader,
    device: torch.device,
    *,
    balanced: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    dir_counts = torch.zeros(3, dtype=torch.float32)
    risk_counts = torch.zeros(2, dtype=torch.float32)
    for batch in loader:
        dir_counts += torch.bincount(batch["target_direction"].reshape(-1), minlength=3).float()
        risk_counts += torch.bincount(batch["target_risk"].reshape(-1).long(), minlength=2).float()
    dir_w = dir_counts.sum() / (3.0 * dir_counts.clamp(min=1.0))
    risk_w = risk_counts.sum() / (2.0 * risk_counts.clamp(min=1.0))
    if balanced:
        dir_w[0] *= 1.20  # down
        dir_w[1] *= 0.80  # flat
        dir_w = dir_w / dir_w.mean()
        risk_w[1] = min(float(risk_w[1].item()), 1.4) * 0.75
        risk_w = risk_w / risk_w.mean()
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
    stage: str = "usable",
    vol_cap: float = 0.10,
) -> tuple[str, list[str], str]:
    if stage == "usable":
        gate_map = USABLE_GATES
    elif stage == "cum_return_recovery":
        gate_map = RECOVERY_GATES
    elif stage == "step_return_recovery":
        gate_map = STEP_RETURN_GATES
    elif stage == "return_direction_branch":
        gate_map = BRANCH_GATES
    else:
        gate_map = BALANCED_GATES
    checks = {k: fn(test[key]) for k, (key, fn) in gate_map.items()}
    if stage == "usable":
        checks["volatility_mae<=cap"] = test["volatility_mae"] <= vol_cap
    if diagnostics:
        checks.update(distribution_gates(diagnostics, stage=stage))
        risk_pred = diagnostics.get("risk_positive_rate_pred")
        if risk_pred is not None and (risk_pred < 0.05 or risk_pred > 0.95):
            checks["risk_prediction_collapsed"] = False
    reasons = [k for k, ok in checks.items() if not ok]
    if stage == "step_return_recovery":
        collapse_reasons = step_return_auto_reject(test, diagnostics or {})
    elif stage == "return_direction_branch":
        collapse_reasons = collapse_auto_reject(test, diagnostics or {})
    else:
        collapse_reasons = []
    reasons.extend(collapse_reasons)
    blocking = reasons[0] if reasons else ""
    if stage == "step_return_recovery" and collapse_reasons:
        decision = "reject"
    elif test["return_ic"] <= 0:
        decision = "reject"
    elif "risk_prediction_collapsed" in reasons:
        decision = "reject"
    elif stage == "step_return_recovery":
        metric_passed = sum(fn(test[key]) for _, (key, fn) in STEP_RETURN_GATES.items())
        dist_passed = sum(distribution_gates(diagnostics or {}, stage=stage).values())
        collapse_passed = all(collapse_gates(diagnostics or {}).values())
        if metric_passed >= 6 and dist_passed >= 3 and collapse_passed:
            decision = "accept"
        elif (
            test["return_ic"] >= 0.020
            and test.get("cum_return_ic", 0.0) >= 0.100
            and collapse_passed
        ):
            decision = "conditional"
        else:
            decision = "reject"
    elif stage == "return_direction_branch":
        metric_passed = sum(fn(test[key]) for _, (key, fn) in BRANCH_GATES.items())
        dist_passed = sum(distribution_gates(diagnostics or {}, stage=stage).values())
        if collapse_reasons:
            decision = "reject"
        elif metric_passed >= 5 and dist_passed >= 3:
            decision = "accept"
        elif metric_passed >= 4 and dist_passed >= 2:
            decision = "conditional"
        else:
            decision = "reject"
    elif stage == "cum_return_recovery":
        metric_passed = sum(fn(test[key]) for _, (key, fn) in RECOVERY_GATES.items())
        dist_passed = sum(distribution_gates(diagnostics or {}, stage=stage).values())
        if metric_passed >= 5 and dist_passed >= 3:
            decision = "accept"
        elif metric_passed >= 4 and dist_passed >= 2:
            decision = "conditional"
        else:
            decision = "reject" if metric_passed < 3 else "conditional"
    elif stage == "balanced_mature":
        metric_passed = sum(fn(test[key]) for _, (key, fn) in BALANCED_GATES.items())
        dist_passed = sum(distribution_gates(diagnostics or {}, stage=stage).values())
        if metric_passed >= 4 and dist_passed >= 2:
            decision = "accept"
        elif metric_passed >= 3:
            decision = "conditional"
        else:
            decision = "reject"
    elif sum(checks.values()) >= 4 and test.get("direction_macro_f1", 0) >= 0.27:
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
    best_valid_diag: dict[str, float] = {},
    baseline_metrics: dict[str, float] | None,
    decision: str,
    reject_reasons: list[str],
    blocking_metric: str = "",
    target_stage: str = "usable",
    total_gates: int = 5,
    baseline_label: str = "baseline",
    best_selection_mode: str = "hard_gated",
    no_valid_checkpoint: bool = False,
    collapse_gate_valid: dict[str, bool] | None = None,
    collapse_gate_test: dict[str, bool] | None = None,
    acceptance_track: str = "A",
    acceptance_track_label: str = "usable 主基线轨",
    acceptance_doc: str = "document/003/架构师-003-理想模型指标目标指导.md",
    branch_type: str = "",
    known_limitation: str = "",
    branch_status: str = "",
) -> None:
    valid_summary = best_valid or test_metrics
    valid_diag_summary = best_valid_diag or test_diagnostics
    stage_label = {
        "usable": "可用",
        "balanced_mature": "稳定可用→成熟过渡",
        "cum_return_recovery": "累计方向+收益排序恢复",
        "return_direction_branch": "收益/累计方向分支解耦",
        "step_return_recovery": "step return_ic 恢复（0064）",
    }.get(target_stage, target_stage)
    score_name = {
        "usable": "v1",
        "balanced_mature": "balanced_0059",
        "cum_return_recovery": "recovery_0060",
        "return_direction_branch": "branch_0062",
        "step_return_recovery": "recovery_0064",
    }.get(target_stage, target_stage)
    lines = [
        f"# {run_id} 多任务市场状态模型训练报告",
        "",
        "## 实验依据",
        "",
        f"- `{acceptance_doc}`",
        "- `document/009/项目经理-009-双轨验收与基线说明.md`",
        "- `document/008/架构师-008-0061训练复盘与指导修正.md`",
        "- `document/007/架构师-007-0061新结构训练目标指导.md`",
        "",
        "## 验收轨道",
        "",
        f"- acceptance_track: **{acceptance_track}**（{acceptance_track_label}）",
        f"- 轨道说明: `{acceptance_doc}`",
        "",
        "## 分支类型",
        "",
        f"- branch_type: **{branch_type or 'n/a'}**",
        f"- known_limitation: `{known_limitation or 'none'}`",
        f"- branch_status: `{branch_status or 'standard'}`",
        "",
        f"## 目标阶段: **{target_stage}**（{stage_label}）",
        "",
        "## 本轮训练配置",
        "",
        f"- `direction_threshold_quantile={args.direction_threshold_quantile}`",
        f"- `risk_threshold_quantile={args.risk_threshold_quantile}`",
        f"- return/direction/volatility/risk = {args.return_weight}/{args.direction_weight}/"
        f"{args.volatility_weight}/{args.risk_weight}",
        f"- cum_direction_weight={args.cum_direction_weight}",
        f"- cum_return_weight={args.cum_return_weight}",
        f"- cum_direction_head_weight={args.cum_direction_head_weight}",
        f"- return_consistency_weight={args.return_consistency_weight}",
        f"- return_horizon_weights={getattr(args, 'return_horizon_weights', None) or 'uniform(1.0)'}",
        f"- use_cum_heads={args.use_cum_heads}, use_horizon_return_head={args.use_horizon_return_head}, "
        f"detach_risk_vol_heads={args.detach_risk_vol_heads}",
        f"- class_weights={args.use_class_weights}, balanced_class_weights={args.balanced_class_weights}",
        f"- direction_class_weights={getattr(args, 'use_direction_class_weights', None)}, "
        f"risk_class_weights={getattr(args, 'use_risk_class_weights', None)}",
        f"- detach_risk_vol_after_epoch={getattr(args, 'detach_risk_vol_after_epoch', 0)}",
        f"- init_market_checkpoint=`{args.init_market_checkpoint or 'none'}`",
        f"- score={score_name}, epochs={args.epochs}, lr={args.lr}",
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
        f"| 指标 | {run_id} |" + (f" {baseline_label} |" if baseline_metrics else ""),
        f"|------|------|" + ("------|" if baseline_metrics else ""),
    ]
    metric_rows = [
        ("cum_direction_acc", "{:.1%}"),
        ("cum_direction_head_acc", "{:.1%}"),
        ("cum_direction_from_return_acc", "{:.1%}"),
        ("direction_acc", "{:.1%}"),
        ("direction_macro_f1", "{:.3f}"),
        ("return_ic", "{:.3f}"),
        ("cum_return_ic", "{:.3f}"),
        ("return_mae", "{:.6f}"),
        ("cum_return_mae", "{:.6f}"),
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
            f"- composite_score={composite_score(valid_summary, valid_diag_summary, stage=target_stage):.4f}",
            f"- cum_direction_acc={valid_summary.get('cum_direction_acc', 0.0):.1%}",
            f"- cum_direction_head_acc={valid_summary.get('cum_direction_head_acc', 0.0):.1%}",
            f"- cum_direction_from_return_acc={valid_summary.get('cum_direction_from_return_acc', 0.0):.1%}",
            f"- direction_macro_f1={valid_summary.get('direction_macro_f1', 0.0):.3f}",
            f"- return_ic={valid_summary.get('return_ic', 0.0):.3f}",
            f"- cum_return_ic={valid_summary.get('cum_return_ic', 0.0):.3f}",
            f"- risk_f1={valid_summary.get('risk_f1', 0.0):.3f}",
            f"- volatility_mae={valid_summary.get('volatility_mae', 0.0):.6f}",
            f"- best_selection_mode={best_selection_mode}",
            f"- no_valid_checkpoint={no_valid_checkpoint}",
            "",
            "## 验证集分布（最佳 checkpoint）",
            "",
            f"- direction_pred: `{ {k: round(v,3) for k,v in valid_diag_summary.items() if k.startswith('direction_pred_')} }`",
            f"- risk_positive_rate_true/pred: "
            f"{valid_diag_summary.get('risk_positive_rate_true', 0):.3f} / "
            f"{valid_diag_summary.get('risk_positive_rate_pred', 0):.3f}",
            "",
            "## 测试诊断",
            "",
            f"- direction_pred: `{ {k: round(v,3) for k,v in test_diagnostics.items() if k.startswith('direction_pred_')} }`",
            f"- risk_positive_rate_true/pred: "
            f"{test_diagnostics.get('risk_positive_rate_true', 0):.3f} / "
            f"{test_diagnostics.get('risk_positive_rate_pred', 0):.3f}",
            "",
            f"- risk_precision/recall: "
            f"{test_diagnostics.get('risk_precision', 0):.3f} / {test_diagnostics.get('risk_recall', 0):.3f}",
            f"- direction_recall down/flat/up: "
            f"{test_diagnostics.get('direction_recall_c0', 0):.3f} / "
            f"{test_diagnostics.get('direction_recall_c1', 0):.3f} / "
            f"{test_diagnostics.get('direction_recall_c2', 0):.3f}",
            f"- step_cum_return_gap_mae={test_metrics.get('step_cum_return_gap_mae', 0.0):.6f}",
            f"- return_ic_h1..h5: "
            + str([round(test_metrics.get(f"return_ic_h{i}", 0.0), 3) for i in range(1, 6)]),
            "",
            "## 坍缩门槛",
            "",
            f"- valid collapse gates: `{collapse_gate_valid or {}}`",
            f"- test collapse gates: `{collapse_gate_test or {}}`",
            "",
            f"## 验收结论（{stage_label}）",
            "",
            f"- target_stage: **{target_stage}**",
            f"- decision: **{decision}**",
            f"- gates_passed: {total_gates - len(reject_reasons)}/{total_gates}",
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
            use_cum_heads=args.use_cum_heads,
            use_horizon_return_head=args.use_horizon_return_head,
            detach_risk_vol_heads=args.detach_risk_vol_heads,
            return_direction_hidden_mult=args.return_direction_hidden_mult,
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
    use_dir_cw = (
        args.use_direction_class_weights
        if args.use_direction_class_weights is not None
        else args.use_class_weights
    )
    use_risk_cw = (
        args.use_risk_class_weights
        if args.use_risk_class_weights is not None
        else args.use_class_weights
    )
    if use_dir_cw or use_risk_cw:
        dir_w, risk_w = compute_train_class_weights(
            train_loader, device, balanced=args.balanced_class_weights
        )
        dir_class_w = dir_w if use_dir_cw else None
        risk_class_w = risk_w if use_risk_cw else None
        print(
            f"  class weights: dir={dir_class_w.cpu().tolist() if dir_class_w is not None else 'off'} "
            f"risk={risk_class_w.cpu().tolist() if risk_class_w is not None else 'off'}"
        )

    horizon_w = None
    if args.return_horizon_weights:
        horizon_w = torch.tensor(args.return_horizon_weights, dtype=torch.float32, device=device)
        print(f"  return_horizon_weights={horizon_w.cpu().tolist()}")

    loss_kw = dict(
        return_weight=args.return_weight,
        direction_weight=args.direction_weight,
        volatility_weight=args.volatility_weight,
        risk_weight=args.risk_weight,
        cum_direction_weight=args.cum_direction_weight,
        cum_return_weight=args.cum_return_weight,
        cum_direction_head_weight=args.cum_direction_head_weight,
        return_consistency_weight=args.return_consistency_weight,
        return_horizon_weights=horizon_w,
        direction_class_weight=dir_class_w,
        risk_class_weight=risk_class_w,
        risk_focal_loss=args.risk_focal_loss,
        focal_gamma=args.focal_gamma,
    )

    best = float("-inf")
    best_valid: dict[str, float] = {}
    best_valid_diag: dict[str, float] = {}
    fallback_score = float("-inf")
    fallback_valid: dict[str, float] = {}
    fallback_valid_diag: dict[str, float] = {}
    history: list[dict[str, float]] = []
    stale = 0
    neg_ic_streak = 0
    low_cum_dir_streak = 0
    bias_stop_reason = ""
    last_valid: dict[str, float] = {}
    last_valid_diag: dict[str, float] = {}
    best_selection_mode = "hard_gated"
    no_valid_checkpoint = False
    for ep in range(1, args.epochs + 1):
        if (
            args.detach_risk_vol_after_epoch > 0
            and ep >= args.detach_risk_vol_after_epoch
            and model.market_state_head is not None
            and not model.market_state_head.detach_risk_vol_heads
        ):
            model.market_state_head.detach_risk_vol_heads = True
            print(f"  ep {ep:03d}: enabled detach_risk_vol_heads")
        tr = train_market_state_epoch(model, train_loader, opt, sched, device, grad_clip=args.grad_clip, **loss_kw)
        va_raw = evaluate_market_state(model, valid_loader, device, **loss_kw, with_diagnostics=True)
        va = float_metrics(va_raw)
        va_diag = {k: float(v) for k, v in va_raw.items() if isinstance(v, (int, float)) and not k.startswith("_")}
        last_valid = float_metrics(va)
        last_valid_diag = va_diag
        row = {"epoch": ep, "train_loss": tr.loss, **va, **{k: v for k, v in va_diag.items() if k.startswith("direction_") or k.startswith("risk_")}}
        history.append(row)
        score = composite_score(va, va_diag, stage=args.target_stage)
        mark = ""
        eligible, skip_reason = selection_eligible(
            va, va_diag, min_risk_f1=args.min_valid_risk_f1, stage=args.target_stage
        )
        if eligible and score > best:
            best = score
            best_valid = float_metrics(va)
            best_valid_diag = va_diag
            stale = 0
            save_checkpoint(ckpt_dir / "market_state_best.pt", {"model": model.state_dict(), "args": vars(args)})
            mark = " *saved"
        else:
            stale += 1
            if not eligible:
                mark = f" (skip: {skip_reason})"
            if (
                va["return_ic"] > 0
                and va["direction_macro_f1"] >= 0.33
                and score > fallback_score
            ):
                fallback_score = score
                fallback_valid = float_metrics(va)
                fallback_valid_diag = va_diag
                save_checkpoint(
                    ckpt_dir / "market_state_fallback.pt",
                    {"model": model.state_dict(), "args": vars(args)},
                )
        neg_ic_streak = neg_ic_streak + 1 if va["return_ic"] <= 0 else 0
        low_cum_dir_streak = low_cum_dir_streak + 1 if va["cum_direction_acc"] < args.bias_stop_cum_dir else 0
        bias_stop_reason = training_bias_detected(
            va, va_diag,
            neg_return_ic_streak=neg_ic_streak,
            low_cum_dir_streak=low_cum_dir_streak,
            stage=args.target_stage,
            bias_stop_return_ic=args.bias_stop_return_ic,
        )
        if ep == 1 or ep % max(1, args.epochs // 6) == 0:
            dist_h = distribution_health(va_diag)
            print(
                f"  ep {ep:03d} tr={tr.loss:.4f} va={va['loss']:.4f} "
                f"dir={va['direction_acc']:.1%} macro_f1={va['direction_macro_f1']:.3f} "
                f"cum={va['cum_direction_acc']:.1%}/{va.get('cum_direction_head_acc', 0.0):.1%} "
                f"ic={va['return_ic']:.3f}/{va.get('cum_return_ic', 0.0):.3f} "
                f"risk_f1={va['risk_f1']:.3f} flat_p={va_diag.get('direction_pred_c1', 0):.1%} "
                f"score={score:.4f} dist_h={dist_h:.3f}{mark}"
            )
        if bias_stop_reason and ep >= 8:
            print(f"  bias stop at epoch {ep}: {bias_stop_reason}")
            break
        if args.early_stop_patience > 0 and stale >= args.early_stop_patience:
            print(f"  early stop at epoch {ep} (patience={args.early_stop_patience})")
            break

    if not (ckpt_dir / "market_state_best.pt").is_file():
        no_valid_checkpoint = True
        save_checkpoint(
            ckpt_dir / "diagnostic_last.pt",
            {"model": model.state_dict(), "args": vars(args)},
        )
        best_selection_mode = "diagnostic_last"
        print("  warning: no eligible checkpoint; saved diagnostic_last.pt only (not a candidate model)")
    else:
        ck = torch.load(ckpt_dir / "market_state_best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
    if no_valid_checkpoint and fallback_valid:
        best_valid = fallback_valid
        best_valid_diag = fallback_valid_diag
    te_raw = evaluate_market_state(model, test_loader, device, **loss_kw, with_diagnostics=True)
    te = float_metrics(te_raw)
    diagnostics = {k: v for k, v in te_raw.items() if k.startswith("_") or k.startswith("direction_") or k.startswith("risk_") or k.startswith("return_ic_h")}
    print(
        f"[TEST] loss={te['loss']:.4f} dir={te['direction_acc']:.1%} "
        f"macro_f1={te['direction_macro_f1']:.3f} cum={te['cum_direction_acc']:.1%} "
        f"cum_head={te.get('cum_direction_head_acc', 0.0):.1%} "
        f"ic={te['return_ic']:.3f} cum_ic={te.get('cum_return_ic', 0.0):.3f} ret_mae={te['return_mae']:.5f} "
        f"vol_mae={te['volatility_mae']:.5f} risk_f1={te['risk_f1']:.3f}"
    )

    baseline = load_baseline_metrics("reports/0059c_market_state_balanced_mature/metrics.json")
    if baseline is None:
        baseline = load_baseline_metrics("reports/0058_market_state_usable/metrics.json")
    decision, reject_reasons, blocking_metric = acceptance_decision(
        te, diagnostics, stage=args.target_stage, vol_cap=0.10
    )
    if no_valid_checkpoint:
        decision = "reject"
        if "no_valid_checkpoint" not in reject_reasons:
            reject_reasons.insert(0, "no_valid_checkpoint")
        blocking_metric = blocking_metric or "no_valid_checkpoint"
    collapse_gate_valid = collapse_gates(best_valid_diag or last_valid_diag)
    collapse_gate_test = collapse_gates(diagnostics)
    track_info = acceptance_track_info(args.target_stage)
    branch_info = branch_metadata(args.target_stage)
    if args.target_stage == "cum_return_recovery":
        total_gates = len(RECOVERY_GATES) + len(distribution_gates(diagnostics, stage=args.target_stage))
    elif args.target_stage == "step_return_recovery":
        total_gates = len(STEP_RETURN_GATES) + len(distribution_gates(diagnostics, stage=args.target_stage))
    elif args.target_stage == "return_direction_branch":
        total_gates = len(BRANCH_GATES) + len(distribution_gates(diagnostics, stage=args.target_stage))
    elif args.target_stage == "balanced_mature":
        total_gates = len(BALANCED_GATES) + len(distribution_gates(diagnostics, stage=args.target_stage))
    else:
        total_gates = 5
    plot_training_curves(history, report_dir / "01_training_curves.png", args.dpi)
    plot_test_metrics(te, baseline, report_dir / "02_test_metrics.png", args.dpi, baseline_label="0059c balanced")
    write_report_md(
        report_dir / "REPORT.md",
        run_id=run_id,
        args=args,
        thresholds=thr,
        class_dist=class_dist,
        test_metrics=te,
        test_diagnostics=diagnostics,
        best_valid=best_valid,
        best_valid_diag=best_valid_diag,
        baseline_metrics=baseline,
        decision=decision,
        reject_reasons=reject_reasons,
        blocking_metric=blocking_metric,
        target_stage=args.target_stage,
        total_gates=total_gates,
        baseline_label="0059c balanced",
        best_selection_mode=best_selection_mode,
        no_valid_checkpoint=no_valid_checkpoint,
        collapse_gate_valid=collapse_gate_valid,
        collapse_gate_test=collapse_gate_test,
        acceptance_track=track_info["acceptance_track"],
        acceptance_track_label=track_info["acceptance_track_label"],
        acceptance_doc=track_info["acceptance_doc"],
        branch_type=branch_info["branch_type"],
        known_limitation=branch_info["known_limitation"],
        branch_status=branch_info["branch_status"],
    )

    payload = {
        "target_stage": args.target_stage,
        "branch_type": branch_info["branch_type"],
        "known_limitation": branch_info["known_limitation"],
        "branch_status": branch_info["branch_status"],
        "acceptance_track": track_info["acceptance_track"],
        "acceptance_track_label": track_info["acceptance_track_label"],
        "acceptance_doc": track_info["acceptance_doc"],
        "run_id": run_id,
        "bias_stop_reason": bias_stop_reason,
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
        "best_composite_score": (
            composite_score(best_valid or te, best_valid_diag or diagnostics, stage=args.target_stage)
            if (best_valid or te)
            else None
        ),
        "best_distribution_health": distribution_health(best_valid_diag or diagnostics) if (best_valid_diag or diagnostics) else None,
        "best_selection_mode": best_selection_mode,
        "no_valid_checkpoint": no_valid_checkpoint,
        "collapse_gates_valid": collapse_gate_valid,
        "collapse_gates_test": collapse_gate_test,
        "hard_gates_0064_pass": (
            all(collapse_gate_test.values())
            and te.get("return_ic", 0) > 0.015
            and te.get("cum_return_ic", 0) >= 0.09
            if args.target_stage == "step_return_recovery"
            else None
        ),
        "test_metrics": te,
        "test_diagnostics": diagnostics,
        "baseline_0059c": baseline,
        "decision": decision,
        "blocking_metric": blocking_metric,
        "reject_reasons": reject_reasons,
        "gates_passed": total_gates - len(reject_reasons),
        "distribution_gates": distribution_gates(diagnostics, stage=args.target_stage),
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
                f"composite_score={composite_score(best_valid or te, best_valid_diag or diagnostics, stage=args.target_stage):.4f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"report saved: {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

