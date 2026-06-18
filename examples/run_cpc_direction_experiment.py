#!/usr/bin/env python3
"""CPC pretrain encoder + lightweight direction decoder experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_EX = Path(__file__).resolve().parent
_ROOT = _EX.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_EX) not in sys.path:
    sys.path.insert(0, str(_EX))

from _train_common import (
    add_data_args,
    add_feature_args,
    add_train_args,
    apply_real_data_defaults,
    fetch_ohlcv_df,
    prepare_bar_series_from_args,
)
from transformer_kit.cpc import CPCEncoder, CPCEncoderConfig, CPCLossHead, DirectionDecoder
from transformer_kit.schedulers import build_adamw_with_warmup_cosine_restarts
from transformer_kit.segment_dataset import PatternSequenceDataset, SequenceSampleIndex, build_sequence_sample_indices
from transformer_kit.segment_features import feat_dim


class CPCWindowDataset(Dataset):
    def __init__(
        self,
        bars: np.ndarray,
        indices: np.ndarray,
        *,
        window: int,
        samples_per_epoch: int,
        seed: int,
    ) -> None:
        self.bars = bars
        self.indices = np.asarray(indices, dtype=np.int64)
        self.window = window
        self.samples_per_epoch = samples_per_epoch
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        valid = self.indices[self.indices + self.window <= self.bars.shape[0]]
        if valid.size == 0:
            raise ValueError("no valid cpc window")
        start = int(self.rng.choice(valid))
        x = self.bars[start : start + self.window].astype(np.float32)
        return {"x": torch.from_numpy(x)}


class MultiCPCWindowDataset(Dataset):
    """Sample CPC windows from multiple symbol bundles."""

    def __init__(
        self,
        pools: list[tuple[np.ndarray, np.ndarray]],
        *,
        window: int,
        samples_per_epoch: int,
        seed: int,
    ) -> None:
        self.window = window
        self.samples_per_epoch = samples_per_epoch
        self.rng = np.random.default_rng(seed)
        self.pools: list[tuple[np.ndarray, np.ndarray]] = []
        for bars, idx in pools:
            valid = np.asarray(idx)[np.asarray(idx) + window <= bars.shape[0]]
            if valid.size > 0:
                self.pools.append((bars, valid))
        if not self.pools:
            raise ValueError("no valid pools for multi-symbol CPC dataset")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pi = int(self.rng.integers(0, len(self.pools)))
        bars, valid = self.pools[pi]
        start = int(self.rng.choice(valid))
        x = bars[start : start + self.window].astype(np.float32)
        return {"x": torch.from_numpy(x)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CPC pretrain + direction decode")
    add_data_args(p)
    add_feature_args(p)
    add_train_args(p)
    p.add_argument("--context-bars", type=int, default=128)
    p.add_argument("--pred-horizon", type=int, default=5)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--cpc-epochs", type=int, default=12)
    p.add_argument("--decoder-epochs", type=int, default=24)
    p.add_argument("--cpc-pred-steps", type=int, default=5)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--encoder-layers", type=int, default=2)
    p.add_argument("--decoder-layers", type=int, default=1)
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--balanced-bce", action="store_true", help="按训练集每个 horizon 的涨跌比例设置 BCE pos_weight")
    p.add_argument("--all5-loss-weight", type=float, default=0.0, help="序列级 all5 surrogate 损失权重")
    p.add_argument("--all5-softmin-temp", type=float, default=0.5, help="all5 surrogate 的 softmin 温度")
    p.add_argument(
        "--pretrain-symbols",
        default="BTCUSDT,ETHUSDT",
        help="逗号分隔：CPC 预训练使用的多标的符号列表",
    )
    p.add_argument(
        "--decode-symbol",
        default="",
        help="方向解码训练目标标的（默认使用 --symbol）",
    )
    p.add_argument("--output-dir", default="reports/0044_cpc_direction")
    p.add_argument("--samples-per-epoch", type=int, default=1800)
    p.set_defaults(samples_per_epoch=1800, batch_size=64, lr=3e-4)
    return p.parse_args()


def split_samples(bundle, args) -> tuple[list[SequenceSampleIndex], list[SequenceSampleIndex], list[SequenceSampleIndex]]:
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


def _bundle_for_symbol(args: argparse.Namespace, symbol: str):
    local_args = SimpleNamespace(**vars(args))
    local_args.symbol = symbol
    df = fetch_ohlcv_df(local_args)
    return prepare_bar_series_from_args(df, local_args)


def direction_metrics(logits: torch.Tensor, future: torch.Tensor) -> dict[str, float]:
    labels = (future[..., 0] > 0).float()
    pred = (logits > 0).float()
    step_match = (pred == labels).float()
    step = float(step_match.mean().item())
    all5 = float(step_match.all(dim=1).float().mean().item())
    cum = float(((pred.sum(dim=1) > (pred.shape[1] / 2)).float() == (labels.sum(dim=1) > (labels.shape[1] / 2)).float()).float().mean().item())
    step_per_h = step_match.mean(dim=0)
    pred_up = pred.mean(dim=0)
    true_up = labels.mean(dim=0)
    h = pred.shape[1]
    fair_all5 = float(0.5 ** h)
    emp_true_all5 = float(torch.prod(true_up * true_up + (1.0 - true_up) * (1.0 - true_up)).item())
    biased_all5 = float(torch.prod(pred_up * true_up + (1.0 - pred_up) * (1.0 - true_up)).item())
    return {
        "step_dir": step,
        "all5_dir": all5,
        "cum_dir": cum,
        "n_samples": float(pred.shape[0]),
        "step_dir_h1": float(step_per_h[0].item()),
        "step_dir_h2": float(step_per_h[1].item()),
        "step_dir_h3": float(step_per_h[2].item()),
        "step_dir_h4": float(step_per_h[3].item()),
        "step_dir_h5": float(step_per_h[4].item()),
        "pred_up_h1": float(pred_up[0].item()),
        "pred_up_h2": float(pred_up[1].item()),
        "pred_up_h3": float(pred_up[2].item()),
        "pred_up_h4": float(pred_up[3].item()),
        "pred_up_h5": float(pred_up[4].item()),
        "true_up_h1": float(true_up[0].item()),
        "true_up_h2": float(true_up[1].item()),
        "true_up_h3": float(true_up[2].item()),
        "true_up_h4": float(true_up[3].item()),
        "true_up_h5": float(true_up[4].item()),
        "baseline_fair_all5": fair_all5,
        "baseline_emp_true_all5": emp_true_all5,
        "baseline_biased_guess_all5": biased_all5,
    }


def all5_surrogate_loss(logits: torch.Tensor, labels: torch.Tensor, *, temp: float = 0.5) -> torch.Tensor:
    """Smooth surrogate for all-steps-correct event."""
    signed = (labels * 2.0 - 1.0) * logits
    t = max(temp, 1e-3)
    softmin = -t * torch.logsumexp(-signed / t, dim=1)
    return F.softplus(-softmin).mean()


def evaluate_decoder(model: DirectionDecoder, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    logits_all: list[torch.Tensor] = []
    fut_all: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["ctx_bars"].to(device)
            y = batch["future_bars"].to(device)
            logits = model(x)
            labels = (y[..., 0] > 0).float()
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            losses.append(float(loss.item()))
            logits_all.append(logits.cpu())
            fut_all.append(y.cpu())
    logits = torch.cat(logits_all, dim=0)
    fut = torch.cat(fut_all, dim=0)
    out = {"loss": float(np.mean(losses))}
    out.update(direction_metrics(logits, fut))
    return out


def estimate_horizon_pos_weight(loader: DataLoader, device: torch.device) -> torch.Tensor:
    total = None
    n = 0
    for batch in loader:
        y = (batch["future_bars"][..., 0] > 0).float().to(device)
        if total is None:
            total = y.sum(dim=0)
        else:
            total = total + y.sum(dim=0)
        n += y.size(0)
    if total is None or n == 0:
        return torch.ones(5, device=device)
    p = (total / float(n)).clamp(min=1e-3, max=1 - 1e-3)
    w = ((1.0 - p) / p).clamp(min=0.5, max=2.0)
    return w


def main() -> int:
    args = parse_args()
    apply_real_data_defaults(args)
    out_dir = Path(args.output_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    pretrain_symbols = [s.strip() for s in args.pretrain_symbols.split(",") if s.strip()]
    if not pretrain_symbols:
        pretrain_symbols = [args.symbol]
    decode_symbol = args.decode_symbol.strip() or args.symbol
    if decode_symbol not in pretrain_symbols:
        pretrain_symbols.append(decode_symbol)

    bundles = {sym: _bundle_for_symbol(args, sym) for sym in pretrain_symbols}
    bundle = bundles[decode_symbol]
    in_dim = feat_dim(use_trend_features=args.trend_features, windows=tuple(args.trend_windows))

    # ---------- Stage A: CPC pretrain ----------
    enc = CPCEncoder(
        CPCEncoderConfig(
            feat_dim=in_dim,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.encoder_layers,
            max_ctx_len=args.context_bars,
        )
    ).to(device)
    cpc_head = CPCLossHead(args.d_model, pred_steps=args.cpc_pred_steps).to(device)
    cpc_train = DataLoader(
        MultiCPCWindowDataset(
            [(b.bars, b.train_idx) for b in bundles.values()],
            window=args.context_bars,
            samples_per_epoch=args.samples_per_epoch,
            seed=args.seed,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    cpc_valid = DataLoader(
        MultiCPCWindowDataset(
            [(b.bars, b.valid_idx) for b in bundles.values()],
            window=args.context_bars,
            samples_per_epoch=400,
            seed=args.seed + 1,
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    cpc_params = list(enc.parameters()) + list(cpc_head.parameters())
    opt, sched = build_adamw_with_warmup_cosine_restarts(
        cpc_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        t0=args.cosine_t0,
        t_mult=args.cosine_t_mult,
        eta_min=args.eta_min,
    )
    best_cpc = float("inf")
    cpc_hist: list[dict[str, float]] = []
    for ep in range(1, args.cpc_epochs + 1):
        enc.train()
        cpc_head.train()
        tr_losses = []
        tr_accs = []
        for batch in cpc_train:
            x = batch["x"].to(device)
            opt.zero_grad(set_to_none=True)
            z = enc(x)
            loss, extra = cpc_head(z)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cpc_params, args.grad_clip)
            opt.step()
            if sched is not None:
                sched.step()
            tr_losses.append(float(loss.item()))
            tr_accs.append(extra["cpc_top1"])
        enc.eval()
        cpc_head.eval()
        va_losses = []
        va_accs = []
        with torch.no_grad():
            for batch in cpc_valid:
                z = enc(batch["x"].to(device))
                loss, extra = cpc_head(z)
                va_losses.append(float(loss.item()))
                va_accs.append(extra["cpc_top1"])
        row = {
            "epoch": ep,
            "train_loss": float(np.mean(tr_losses)),
            "valid_loss": float(np.mean(va_losses)),
            "train_top1": float(np.mean(tr_accs)),
            "valid_top1": float(np.mean(va_accs)),
        }
        cpc_hist.append(row)
        if row["valid_loss"] < best_cpc:
            best_cpc = row["valid_loss"]
            torch.save({"encoder": enc.state_dict(), "cpc_head": cpc_head.state_dict(), "args": vars(args)}, ckpt_dir / "cpc_encoder.pt")
        if ep == 1 or ep % max(1, args.cpc_epochs // 4) == 0:
            print(f"[CPC] ep={ep:03d} tr={row['train_loss']:.4f} va={row['valid_loss']:.4f} top1={row['valid_top1']:.1%}")

    ck = torch.load(ckpt_dir / "cpc_encoder.pt", map_location=device, weights_only=False)
    enc.load_state_dict(ck["encoder"])

    # ---------- Stage B: direction decode ----------
    train_idx, valid_idx, test_idx = split_samples(bundle, args)
    train_loader = DataLoader(
        PatternSequenceDataset(bundle.bars, train_idx, bundle.raw_log_ret, zscore_window=bundle.zscore_window),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        PatternSequenceDataset(bundle.bars, valid_idx, bundle.raw_log_ret, zscore_window=bundle.zscore_window),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        PatternSequenceDataset(bundle.bars, test_idx, bundle.raw_log_ret, zscore_window=bundle.zscore_window),
        batch_size=args.batch_size,
        shuffle=False,
    )

    dec = DirectionDecoder(enc, pred_horizon=args.pred_horizon, decoder_layers=args.decoder_layers).to(device)
    if args.freeze_encoder:
        for p in dec.encoder.parameters():
            p.requires_grad = False
    dec_params = [p for p in dec.parameters() if p.requires_grad]
    opt2, sched2 = build_adamw_with_warmup_cosine_restarts(
        dec_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        t0=args.cosine_t0,
        t_mult=args.cosine_t_mult,
        eta_min=args.eta_min,
    )
    best_all5 = -1.0
    best_metrics: dict[str, float] = {}
    dec_hist: list[dict[str, float]] = []
    pos_weight = estimate_horizon_pos_weight(train_loader, device) if args.balanced_bce else None
    for ep in range(1, args.decoder_epochs + 1):
        dec.train()
        for batch in train_loader:
            x = batch["ctx_bars"].to(device)
            y = batch["future_bars"].to(device)
            labels = (y[..., 0] > 0).float()
            opt2.zero_grad(set_to_none=True)
            logits = dec(x)
            step_loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            seq_loss = all5_surrogate_loss(logits, labels, temp=args.all5_softmin_temp)
            loss = step_loss + args.all5_loss_weight * seq_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dec_params, args.grad_clip)
            opt2.step()
            if sched2 is not None:
                sched2.step()
        va = evaluate_decoder(dec, valid_loader, device)
        va["epoch"] = ep
        dec_hist.append(va)
        if va["all5_dir"] > best_all5:
            best_all5 = va["all5_dir"]
            torch.save({"decoder": dec.state_dict(), "args": vars(args)}, ckpt_dir / "direction_decoder.pt")
        if ep == 1 or ep % max(1, args.decoder_epochs // 4) == 0:
            print(
                f"[DEC] ep={ep:03d} va_loss={va['loss']:.4f} "
                f"step={va['step_dir']:.1%} all5={va['all5_dir']:.1%} cum={va['cum_dir']:.1%}"
            )

    dec.load_state_dict(torch.load(ckpt_dir / "direction_decoder.pt", map_location=device, weights_only=False)["decoder"])
    test = evaluate_decoder(dec, test_loader, device)
    best_metrics = {"test_step_dir": test["step_dir"], "test_all5_dir": test["all5_dir"], "test_cum_dir": test["cum_dir"], "test_loss": test["loss"]}
    print(
        f"[TEST] loss={test['loss']:.4f} "
        f"step={test['step_dir']:.1%} all5={test['all5_dir']:.1%} cum={test['cum_dir']:.1%}"
    )

    payload = {
        "args": vars(args),
        "cpc_history": cpc_hist,
        "decoder_history": dec_hist,
        "metrics": best_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "metrics.txt").write_text(
        "\n".join(
            [
                "=== CPC + Direction Decoder ===",
                f"source={args.source} pretrain_symbols={','.join(pretrain_symbols)} decode_symbol={decode_symbol}",
                f"interval={args.interval} days={args.days}",
                f"trend_features={args.trend_features}",
                f"all5_loss_weight={args.all5_loss_weight} softmin_temp={args.all5_softmin_temp}",
                f"balanced_bce={args.balanced_bce}",
                f"test_step_direction_acc={test['step_dir']:.1%}",
                f"test_all5_direction_acc={test['all5_dir']:.1%}",
                f"test_cum_direction_acc={test['cum_dir']:.1%}",
                f"baseline_fair_all5={test['baseline_fair_all5']:.3%}",
                f"baseline_emp_true_all5={test['baseline_emp_true_all5']:.3%}",
                f"baseline_biased_guess_all5={test['baseline_biased_guess_all5']:.3%}",
                f"pred_up_rate_h1..h5="
                f"{test['pred_up_h1']:.3f},{test['pred_up_h2']:.3f},{test['pred_up_h3']:.3f},{test['pred_up_h4']:.3f},{test['pred_up_h5']:.3f}",
                f"true_up_rate_h1..h5="
                f"{test['true_up_h1']:.3f},{test['true_up_h2']:.3f},{test['true_up_h3']:.3f},{test['true_up_h4']:.3f},{test['true_up_h5']:.3f}",
                f"test_loss={test['loss']:.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

