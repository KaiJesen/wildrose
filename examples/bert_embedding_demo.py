#!/usr/bin/env python3
"""BERT 风格 K 线 Embedding 演示。

Pipeline:
  1. 通过 market_data 抓 K 线（默认 AkShare → 东方财富）
  2. 因果特征工程（log_ret + causal z-score + minute/dow）
  3. 切滑动窗口 → Tensor[B, T, feat_dim]
  4. 喂入 KlineBertEmbedding，打印输出形状、统计量、参数量

安装:
  pip install -e ".[all,transformer]"

最小运行示例:
  python examples/bert_embedding_demo.py --symbol 600519 --interval 60m --days 90
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from market_data import get_kline_provider
from transformer_kit.embeddings import KlineBertEmbedding, KlineBertEmbeddingConfig
from transformer_kit.features import build_feature_frame, make_sliding_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BERT-style embedding demo on K-line data")
    parser.add_argument("--source", default="akshare_em", help="market_data 数据源 id")
    parser.add_argument("--symbol", default="600519")
    parser.add_argument("--interval", default="60m")
    parser.add_argument("--days", type=int, default=90, help="向前取多少自然日")
    parser.add_argument("--window", type=int, default=64, help="序列窗口长度 T")
    parser.add_argument("--stride", type=int, default=1, help="滑窗步长")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--zscore-window", type=int, default=60)
    parser.add_argument("--batch-limit", type=int, default=8, help="演示用 batch 数上限")
    parser.add_argument("--value-proj", choices=["linear", "mlp"], default="linear")
    parser.add_argument(
        "--position-type",
        choices=["learned", "sincos"],
        default="learned",
    )
    parser.add_argument("--no-minute", action="store_true", help="禁用 minute embedding")
    parser.add_argument("--no-dow", action="store_true", help="禁用 day-of-week embedding")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    end = datetime.now()
    start = end - timedelta(days=args.days)

    provider = get_kline_provider(args.source)
    print(f"[1/4] fetch {args.symbol} via {provider.id} ({args.interval}, {args.days}d)")
    df = provider.fetch_kline(args.symbol, args.interval, start, end)
    if df.empty:
        print("  → 空数据；换更长 --days 或更小 --interval 重试。")
        return 1
    if len(df) < args.window + 5:
        print(f"  → 仅取到 {len(df)} 根，少于 window+5；增大 --days。")
        return 1
    print(f"  rows={len(df)}, columns={list(df.columns)[:8]}...")

    print(f"[2/4] build causal features (z-score window={args.zscore_window})")
    feat_df, feat_cols = build_feature_frame(df, zscore_window=args.zscore_window)
    feat_array = feat_df[feat_cols].to_numpy(dtype=np.float32)
    minute_arr = feat_df["minute_of_day"].to_numpy(dtype=np.int64)
    dow_arr = feat_df["dow"].to_numpy(dtype=np.int64)
    print(f"  feature columns: {feat_cols}")
    print(f"  feat_array shape: {feat_array.shape}")

    print(f"[3/4] slice sliding windows (T={args.window}, stride={args.stride})")
    feats_np, minute_np, dow_np = make_sliding_windows(
        feat_array, minute_arr, dow_arr, window=args.window, stride=args.stride
    )
    if feats_np.shape[0] > args.batch_limit:
        feats_np = feats_np[-args.batch_limit :]
        minute_np = minute_np[-args.batch_limit :]
        dow_np = dow_np[-args.batch_limit :]
    feats_t = torch.from_numpy(feats_np)
    minute_t = torch.from_numpy(minute_np)
    dow_t = torch.from_numpy(dow_np)
    print(f"  batch: feats={tuple(feats_t.shape)}, minute={tuple(minute_t.shape)}")

    print("[4/4] build KlineBertEmbedding and forward")
    cfg = KlineBertEmbeddingConfig(
        feat_dim=feats_t.shape[-1],
        d_model=args.d_model,
        max_len=args.window,
        n_assets=0,
        value_proj=args.value_proj,
        position_type=args.position_type,
        use_time_minute=not args.no_minute,
        use_time_dow=not args.no_dow,
    )
    model = KlineBertEmbedding(cfg)
    model.eval()
    print(f"  model: {model}")

    with torch.no_grad():
        x = model(
            feats_t,
            minute_ids=minute_t if cfg.use_time_minute else None,
            dow_ids=dow_t if cfg.use_time_dow else None,
        )

    print()
    print("=== Embedding 输出 ===")
    print(f"  input feats : {tuple(feats_t.shape)}  dtype={feats_t.dtype}")
    print(f"  output x    : {tuple(x.shape)}  dtype={x.dtype}")
    print(f"  x.mean / std: {x.mean().item():+.4f} / {x.std().item():.4f}")
    print(f"  trainable   : {model.num_parameters():,} params")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
