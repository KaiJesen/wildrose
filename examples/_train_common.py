"""训练脚本公共：数据加载与参数。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


REAL_DATA_DEFAULTS: dict[str, object] = {
    "source": "binance_vision",
    "symbol": "BTCUSDT",
    "interval": "1h",
    "days": 365,
}


def add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="使用合成 K 线（仅调试；正式实验请用真实数据）",
    )
    p.add_argument("--source", default=str(REAL_DATA_DEFAULTS["source"]))
    p.add_argument("--symbol", default=str(REAL_DATA_DEFAULTS["symbol"]))
    p.add_argument("--interval", default=str(REAL_DATA_DEFAULTS["interval"]))
    p.add_argument("--days", type=int, default=int(REAL_DATA_DEFAULTS["days"]))
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
    p.add_argument("--no-segment-cnn", action="store_true", help="关闭 segment 并行 CNN 分支")
    p.add_argument("--segment-cnn-weight", type=float, default=1.0, help="VQ token 与 CNN token 融合权重")
    p.add_argument("--samples-per-epoch", type=int, default=3000)


def add_vq_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--num-codes", type=int, default=16)
    p.add_argument("--vq-beta", type=float, default=1.0)
    p.add_argument("--vq-dead-threshold", type=float, default=0.1, help="EMA 计数低于此值的码视为 dead")
    p.add_argument(
        "--vq-max-code-frac",
        type=float,
        default=0.15,
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
    p.add_argument(
        "--vq-inverse-freq-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="EMA 码本按逆频率加权更新，缓解主导码垄断",
    )


def add_stage3_loss_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mse-weight", type=float, default=0.7, help="Stage3 逐步 MSE 权重")
    p.add_argument("--step-corr-weight", type=float, default=0.25, help="Stage3 逐步 (1-corr) 权重")
    p.add_argument("--cum-corr-weight", type=float, default=0.35, help="Stage3 累计收益 (1-corr) 权重")
    p.add_argument("--sign-weight", type=float, default=0.15, help="Stage3 符号一致损失权重")
    p.add_argument("--corr-weight", type=float, default=-1.0, help="兼容旧参数，>0 时覆盖 step-corr-weight")
    p.add_argument("--rank-weight", type=float, default=0.0, help="Stage3 batch 成对排序损失权重")
    p.add_argument("--direction-weight", type=float, default=0.0, help="未来累计涨跌方向分类损失权重")
    p.add_argument("--shape-weight", type=float, default=0.0, help="未来 K 线形态维度 MSE 权重")
    p.add_argument("--path-shape-weight", type=float, default=0.0, help="未来累计收益路径相对形状损失权重")
    p.add_argument("--cum-magnitude-weight", type=float, default=0.0, help="累计 log_ret 幅度 MSE 权重")
    p.add_argument("--relative-magnitude-weight", type=float, default=0.0, help="价格变动相对幅度损失权重")
    p.add_argument("--raw-mse-weight", type=float, default=0.0, help="原始尺度 log_ret MSE 权重")
    p.add_argument("--magnitude-tolerance", type=float, default=0.2, help="幅度评估容差（相对误差）")
    p.add_argument("--magnitude-min-move", type=float, default=1e-4, help="幅度评估最小真实变动阈值")
    p.add_argument("--vol-focus-weight", type=float, default=0.0, help="高波动样本损失加权强度")
    p.add_argument("--vol-focus-top-frac", type=float, default=0.3, help="按未来波动强度加权的 top 比例")
    p.add_argument("--move-focus-weight", type=float, default=0.0, help="按未来累计 |log_ret| 连续加权强度")
    p.add_argument("--move-focus-scale", type=float, default=3.0, help="move-focus 归一化上限")
    p.add_argument("--break-focus-weight", type=float, default=0.0, help="按上下文末尾切分概率加权强度")
    p.add_argument("--break-focus-tail", type=int, default=16, help="break-focus 统计的末尾 bar 窗口")
    p.add_argument(
        "--code-supervision-weight",
        type=float,
        default=0.0,
        help="末段形态 token 预测未来累计方向的辅助损失权重",
    )
    p.add_argument("--anti-lag-weight", type=float, default=0.0, help="惩罚预测复刻刚过去走势的反滞后损失权重")
    p.add_argument("--anti-lag-margin", type=float, default=0.05, help="未来相关性应高于过去相关性的 margin")
    p.add_argument("--pool-mode", choices=("attn", "mean", "last"), default="attn", help="Stage3 token 汇聚")
    p.add_argument("--horizon-head", action="store_true", help="使用逐 horizon cross-attention 预测头，降低多步预测滞后")
    p.add_argument("--no-ic-loss", action="store_true", help="Stage3 仅用 MSE（旧行为）")
    p.add_argument("--no-learnable-scale", action="store_true", help="关闭可学习输出幅度缩放")
    p.add_argument("--direction-seq-only", action="store_true", help="Stage3 改为逐步涨跌二分类（未来5步）")
    p.add_argument("--direction-seq-weight", type=float, default=1.0, help="逐步方向 BCE 损失权重")
    p.add_argument("--direction-seq-crf", action="store_true", help="方向分类使用线性链 CRF + Viterbi 解码")


def add_feature_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--trend-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用多尺度线性趋势特征（斜率/残差离散/R²/强度）",
    )
    p.add_argument(
        "--trend-windows",
        type=int,
        nargs="+",
        default=[20, 60, 120],
        help="趋势回归窗口（bar 数）",
    )


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
    p.add_argument(
        "--break-aware-vq-balance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="VQ 熵惩罚对切分后段加权、背景段降权",
    )
    p.add_argument("--break-seg-vq-weight", type=float, default=2.0, help="切分后 segment 的 VQ 平衡权重")
    p.add_argument("--background-seg-vq-weight", type=float, default=0.35, help="首段/背景 segment 的 VQ 平衡权重")


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


def apply_real_data_defaults(args: argparse.Namespace) -> None:
    """无 CLI 参数时启用真实 K 线默认配置（实验不使用随机合成数据）。"""
    if len(sys.argv) > 1:
        return
    args.synthetic = False
    for key, value in REAL_DATA_DEFAULTS.items():
        setattr(args, key, value)


def prepare_bar_series_from_args(df, args):
    from transformer_kit.segment_dataset import prepare_bar_series

    return prepare_bar_series(
        df,
        use_trend_features=getattr(args, "trend_features", True),
        trend_windows=tuple(getattr(args, "trend_windows", [20, 60, 120])),
    )
