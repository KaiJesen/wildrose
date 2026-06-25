from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
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
from trading_system.teq_edge import (
    TeqEdgeCalibrator,
    apply_teq_calibration,
    compute_teq_edge_raw,
)


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


def _leg_align_horizons_from_ckpt(ck_args: dict, state: dict) -> tuple[int, ...]:
    if "leg_align_horizons" in ck_args:
        raw = ck_args["leg_align_horizons"]
        return tuple(int(h) for h in raw)
    variant = str(ck_args.get("variant", "0"))
    if variant in {"1", "2"}:
        return (12, 24)
    has_hz = any(k.startswith("market_state_head.hz_return_heads.") for k in state)
    return (12, 24) if has_hz else ()


def _use_participation_heads_from_ckpt(ck_args: dict, state: dict) -> bool:
    if ck_args.get("variant") in {"0", "1", "2"}:
        return True
    return any(k.startswith("market_state_head.participation_logit_") for k in state)


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
    teq_calibrator: TeqEdgeCalibrator | None = None

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
        state = ck.get("model", {})
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
        leg_align_horizons = _leg_align_horizons_from_ckpt(ck_args, state)
        use_participation_heads = _use_participation_heads_from_ckpt(ck_args, state)
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
                use_participation_heads=use_participation_heads,
                leg_align_horizons=leg_align_horizons,
            )
        ).to(torch.device(device))
        model.load_state_dict(state, strict=False)
        model.eval()
        close = df[COL_CLOSE].to_numpy(dtype=np.float64)
        atr = compute_atr(
            df[COL_HIGH].to_numpy(dtype=np.float64),
            df[COL_LOW].to_numpy(dtype=np.float64),
            close,
            cfg.execution.atr_period,
        )
        calibrator = None
        teq_cfg = cfg.teq_edge
        if teq_cfg.enabled and teq_cfg.use_calibrated and teq_cfg.calibration_path:
            cal_path = Path(teq_cfg.calibration_path)
            if cal_path.is_file():
                calibrator = TeqEdgeCalibrator.load(cal_path)
        return cls(
            model=model,
            bars=bars,
            times=df[COL_TIME].to_numpy(),
            close=close,
            atr=atr,
            context_bars=context_bars,
            cfg=cfg,
            device=torch.device(device),
            teq_calibrator=calibrator,
        )

    def _attach_teq_fields(self, sig: TradingSignal, out: MarketStateOutput) -> None:
        teq_cfg = self.cfg.teq_edge
        if out.participation_logit_long is not None:
            part_long = float(torch.sigmoid(out.participation_logit_long[0]).item())
        else:
            part_long = 0.5
        if out.participation_logit_short is not None:
            part_short = float(torch.sigmoid(out.participation_logit_short[0]).item())
        else:
            part_short = 0.5
        if self.teq_calibrator is not None and self.cfg.teq_edge.use_calibrated:
            part_long = self.teq_calibrator.apply_part_long(part_long)
            part_short = self.teq_calibrator.apply_part_short(part_short)
        sig.participate_score_long = part_long
        sig.participate_score_short = part_short

        edge_5 = float(sig.pred_cum_ret_5 or 0.0)
        edge_24 = edge_5
        if out.hz_return_pred:
            if 24 in out.hz_return_pred:
                edge_24 = float(out.hz_return_pred[24][0].item())
            elif out.hz_return_pred:
                first_hz = next(iter(out.hz_return_pred.values()))
                edge_24 = float(first_hz[0].item())
        sig.edge_long_hz = edge_24
        sig.edge_short_hz = -edge_24

        if not teq_cfg.enabled:
            sig.teq_edge_long = sig.edge
            sig.teq_edge_short = -sig.edge
            return

        teq_long_raw, teq_short_raw = compute_teq_edge_raw(
            edge_5=edge_5,
            edge_24=edge_24,
            participate_score_long=part_long,
            participate_score_short=part_short,
            cfg=teq_cfg,
        )
        teq_long, teq_short = apply_teq_calibration(
            teq_long_raw,
            teq_short_raw,
            calibrator=self.teq_calibrator,
            cfg=teq_cfg,
        )
        sig.teq_edge_long = teq_long
        sig.teq_edge_short = teq_short

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
        sig = sig.finalize(self.cfg.rule.risk_open_max)
        self._attach_teq_fields(sig, out)
        return sig
