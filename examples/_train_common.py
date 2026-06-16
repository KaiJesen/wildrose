"""训练脚本公共：数据加载与参数。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--synthetic", action="store_true", help="使用合成 K 线")
    p.add_argument("--source", default="akshare_em")
    p.add_argument("--symbol", default="600519")
    p.add_argument("--interval", default="60m")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--cache-dir", default="data/cache/kline")
    p.add_argument("--csv", default="")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--seed", type=int, default=42)


def add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--cosine-t0", type=int, default=200)
    p.add_argument("--cosine-t-mult", type=int, default=2)
    p.add_argument("--eta-min", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--checkpoint-dir", default="checkpoints/pattern")


def add_segment_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--min-seg-len", type=int, default=4)
    p.add_argument("--max-seg-len", type=int, default=32, help="单段最大 bar 数")
    p.add_argument("--max-segments", type=int, default=16, help="自动切分最大段数")
    p.add_argument("--context-bars", type=int, default=128, help="输入 K 线窗口长度")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--encoder-layers", type=int, default=3)
    p.add_argument("--samples-per-epoch", type=int, default=3000)


def add_vq_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--num-codes", type=int, default=16)
    p.add_argument("--vq-beta", type=float, default=1.0)
    p.add_argument("--vq-dead-threshold", type=float, default=0.1, help="EMA 计数低于此值的码视为 dead")
    p.add_argument(
        "--vq-max-code-frac",
        type=float,
        default=0.18,
        help="单码 epoch 占比超过此值则重置（防 code 0 塌缩）",
    )
    p.add_argument(
        "--vq-kmeans-frac",
        type=float,
        default=0.45,
        help="epoch 最大码占比超过此值时做 k-means 码本重初始化",
    )
    p.add_argument("--contrastive-weight", type=float, default=0.1)
    p.add_argument("--augment-noise", type=float, default=0.08)


def add_stage3_loss_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mse-weight", type=float, default=0.7, help="Stage3 逐步 MSE 权重")
    p.add_argument("--step-corr-weight", type=float, default=0.25, help="Stage3 逐步 (1-corr) 权重")
    p.add_argument("--cum-corr-weight", type=float, default=0.35, help="Stage3 累计收益 (1-corr) 权重")
    p.add_argument("--sign-weight", type=float, default=0.15, help="Stage3 符号一致损失权重")
    p.add_argument("--corr-weight", type=float, default=-1.0, help="兼容旧参数，>0 时覆盖 step-corr-weight")
    p.add_argument("--rank-weight", type=float, default=0.0, help="Stage3 batch 成对排序损失权重")
    p.add_argument("--direction-weight", type=float, default=0.0, help="未来累计涨跌方向分类损失权重")
    p.add_argument("--pool-mode", choices=("attn", "mean", "last"), default="attn", help="Stage3 token 汇聚")
    p.add_argument("--no-ic-loss", action="store_true", help="Stage3 仅用 MSE（旧行为）")
    p.add_argument("--no-learnable-scale", action="store_true", help="关闭可学习输出幅度缩放")


def add_break_vol_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--break-vol-weight", type=float, default=0.12, help="波动率伪标签监督切分")
    p.add_argument("--break-vol-window", type=int, default=12)
    p.add_argument("--break-vol-top-frac", type=float, default=0.12)
    p.add_argument("--diversity-weight", type=float, default=0.25, help="Stage1/2 VQ perplexity 奖励")
    p.add_argument(
        "--usage-balance-weight",
        type=float,
        default=0.35,
        help="Stage1/2 可微码分布熵惩罚（越大越强制均匀用码）",
    )
    p.add_argument(
        "--z-spread-weight",
        type=float,
        default=0.15,
        help="Stage1/2 segment 向量方差惩罚，防止 encoder 输出塌缩",
    )


def fetch_ohlcv_df(args: argparse.Namespace):
    from transformer_kit.segment_dataset import make_synthetic_ohlcv

    if args.synthetic:
        from transformer_kit.data_cache import load_kline_csv, save_kline_csv

        if (
            args.csv
            and not args.no_cache
            and Path(args.csv).is_file()
            and not args.force_download
        ):
            print(f"  load cached csv: {args.csv}")
            return load_kline_csv(args.csv)
        df = make_synthetic_ohlcv(n=2000, seed=args.seed)
        if args.csv and not args.no_cache:
            path = save_kline_csv(df, args.csv)
            print(f"  saved synthetic csv: {path}")
        return df

    from market_data import get_kline_provider
    from transformer_kit.data_cache import (
        load_kline_csv,
        resolve_kline_csv_path,
        save_kline_csv,
    )

    end = datetime.now()
    cache_path = None
    if not args.no_cache:
        cache_path = resolve_kline_csv_path(
            source=args.source,
            symbol=args.symbol,
            interval=args.interval,
            days=args.days,
            cache_dir=args.cache_dir,
            csv_path=args.csv or None,
            end=end,
        )
        if cache_path.is_file() and not args.force_download:
            print(f"  load cached csv: {cache_path}")
            return load_kline_csv(cache_path)

    start = end - timedelta(days=args.days)
    print(f"  download {args.symbol} via {args.source} ({args.interval}, {args.days}d)")
    provider = get_kline_provider(args.source)
    df = provider.fetch_kline(args.symbol, args.interval, start, end)
    if df.empty:
        raise RuntimeError("empty kline; try --synthetic")
    if not args.no_cache and cache_path is not None:
        save_kline_csv(df, cache_path)
    return df
