from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch

from market_data.schema import COL_CLOSE, COL_HIGH, COL_LOW, COL_TIME
from transformer_kit.causal_transformer import CausalTransformerConfig
from transformer_kit.pattern_encoder import pattern_config_from_args
from transformer_kit.pattern_model import KlinePatternPredictor, MarketStateOutput, PatternPredictorConfig
from transformer_kit.train_utils import load_checkpoint
from trading_system.config import TradingSystemConfig
from trading_system.signal import TradingSignal


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = np.empty_like(tr)
    atr[:period] = tr[:period].mean()
    alpha = 1.0 / max(1, period)
    for i in range(period, len(tr)):
        atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1]
    return atr


@dataclass
class ModelSignalProvider:
    model: KlinePatternPredictor
    bars: np.ndarray
    times: np.ndarray
    close: np.ndarray
    atr: np.ndarray
    context_bars: int
    cfg: TradingSystemConfig
    device: torch.device

    @classmethod
    def from_checkpoint(
        cls,
        *,
        checkpoint: str,
        bars: np.ndarray,
        df,
        context_bars: int,
        d_model: int,
        n_heads: int,
        trunk_layers: int,
        trend_features: bool,
        trend_windows: tuple[int, ...],
        max_seg_len: int,
        max_segments: int,
        min_seg_len: int,
        num_codes: int,
        vq_beta: float,
        vq_inverse_freq_ema: bool,
        cfg: TradingSystemConfig,
        device: str,
    ) -> "ModelSignalProvider":
        ck = load_checkpoint(checkpoint, map_location=device)
        ck_args = ck.get("args", {}) if isinstance(ck.get("args"), dict) else {}
        d_model = int(ck_args.get("d_model", d_model))
        n_heads = int(ck_args.get("n_heads", n_heads))
        context_bars = int(ck_args.get("context_bars", context_bars))
        max_seg_len = int(ck_args.get("max_seg_len", max_seg_len))
        max_segments = int(ck_args.get("max_segments", max_segments))
        min_seg_len = int(ck_args.get("min_seg_len", min_seg_len))
        num_codes = int(ck_args.get("num_codes", num_codes))
        vq_beta = float(ck_args.get("vq_beta", vq_beta))
        vq_inverse_freq_ema = bool(ck_args.get("vq_inverse_freq_ema", vq_inverse_freq_ema))
        trend_features = bool(ck_args.get("trend_features", trend_features))
        trend_windows = tuple(ck_args.get("trend_windows", trend_windows))
        trunk_layers = int(ck_args.get("trunk_layers", trunk_layers))
        ns = SimpleNamespace(
            d_model=d_model,
            n_heads=n_heads,
            encoder_layers=2,
            context_bars=context_bars,
            max_seg_len=max_seg_len,
            max_segments=max_segments,
            min_seg_len=min_seg_len,
            num_codes=num_codes,
            vq_beta=vq_beta,
            vq_inverse_freq_ema=vq_inverse_freq_ema,
            trend_features=trend_features,
            trend_windows=list(trend_windows),
        )
        auto_cfg = pattern_config_from_args(ns)
        model = KlinePatternPredictor(
            PatternPredictorConfig(
                auto_segment=auto_cfg,
                trunk=CausalTransformerConfig(d_model=d_model, n_heads=n_heads, n_layers=trunk_layers),
                pred_horizon=5,
                pred_feat_dim=1,
                pool_mode="attn",
                learnable_scale=True,
                use_market_state_head=True,
                direction_classes=3,
                risk_classes=2,
                use_cum_heads=True,
                use_horizon_return_head=True,
            )
        ).to(torch.device(device))
        model.load_state_dict(ck["model"], strict=False)
        model.eval()
        close = df[COL_CLOSE].to_numpy(dtype=np.float64)
        atr = compute_atr(
            df[COL_HIGH].to_numpy(dtype=np.float64),
            df[COL_LOW].to_numpy(dtype=np.float64),
            close,
            cfg.execution.atr_period,
        )
        return cls(
            model=model,
            bars=bars,
            times=df[COL_TIME].to_numpy(),
            close=close,
            atr=atr,
            context_bars=context_bars,
            cfg=cfg,
            device=torch.device(device),
        )

    @torch.no_grad()
    def signal_at(self, idx: int) -> TradingSignal:
        ctx = torch.from_numpy(self.bars[idx - self.context_bars : idx].astype(np.float32)).unsqueeze(0).to(self.device)
        ctx_len = torch.tensor([self.context_bars], dtype=torch.long, device=self.device)
        out = self.model(ctx, ctx_len)
        if not isinstance(out, MarketStateOutput):
            raise RuntimeError("market state provider expects MarketStateOutput")
        dir_prob = torch.softmax(out.direction_logits[0, 0], dim=-1).cpu().numpy()
        risk_prob = torch.softmax(out.risk_logits[0, 0], dim=-1).cpu().numpy()
        pred = out.return_pred[0].detach().cpu().numpy()
        cum = float(out.cum_return_pred[0].item()) if out.cum_return_pred is not None else float(pred.sum())
        ts = self.times[idx]
        if isinstance(ts, np.datetime64):
            ts = ts.astype("datetime64[ns]").tolist()
        sig = TradingSignal(
            ts=ts,
            price=float(self.close[idx]),
            atr=float(self.atr[idx]),
            p_up=float(dir_prob[2]),
            p_down=float(dir_prob[0]),
            p_flat=float(dir_prob[1]),
            p_risk=float(risk_prob[1]),
            pred_ret_1=float(pred[0]) if len(pred) > 0 else 0.0,
            pred_ret_2=float(pred[1]) if len(pred) > 1 else 0.0,
            pred_ret_3=float(pred[2]) if len(pred) > 2 else 0.0,
            pred_ret_4=float(pred[3]) if len(pred) > 3 else 0.0,
            pred_ret_5=float(pred[4]) if len(pred) > 4 else 0.0,
            pred_cum_ret_5=cum,
            source="market_state_model",
            raw={},
        )
        return sig.finalize(self.cfg.rule.risk_open_max)

