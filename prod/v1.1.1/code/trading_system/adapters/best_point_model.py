from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from best_point.features import compute_causal_features
from best_point.model import BestPointModelConfig, BestPointSignalModel


@dataclass
class BestPointSignal:
    p_long_entry_zone: float = 0.0
    p_short_entry_zone: float = 0.0
    p_hold_long: float = 0.0
    p_hold_short: float = 0.0
    p_exit_long: float = 0.0
    p_exit_short: float = 0.0
    expected_opportunity_roi: float = 0.0


class BestPointSignalProvider:
    def __init__(self, rows: list[BestPointSignal]) -> None:
        self.rows = rows

    @classmethod
    def from_checkpoint(
        cls,
        *,
        checkpoint: str,
        df,
        context_bars: int | None = None,
        device: str = "cpu",
    ) -> "BestPointSignalProvider":
        ck = torch.load(checkpoint, map_location=device)
        feat = compute_causal_features(df)
        feat = feat.loc[:, ck["feature_columns"]].to_numpy(dtype=np.float32)
        mean = np.asarray(ck["feature_mean"], dtype=np.float32)
        std = np.clip(np.asarray(ck["feature_std"], dtype=np.float32), 1e-6, None)
        ctx = int(context_bars or ck.get("context_bars", 96))
        model = BestPointSignalModel(BestPointModelConfig(**ck["config"])).to(device)
        model.load_state_dict(ck["model"])
        model.eval()

        rows: list[BestPointSignal] = []
        for i in range(len(feat)):
            if i < ctx:
                rows.append(BestPointSignal())
                continue
            x = (feat[i - ctx : i] - mean) / std
            with torch.no_grad():
                out = model(torch.from_numpy(x).unsqueeze(0).to(device))
            e = torch.softmax(out["entry_logits"][0], dim=-1).cpu().numpy()
            h = torch.softmax(out["hold_logits"][0], dim=-1).cpu().numpy()
            ex = torch.softmax(out["exit_logits"][0], dim=-1).cpu().numpy()
            opp = float(out["opportunity_pred"][0, 0].detach().cpu().item())
            rows.append(
                BestPointSignal(
                    p_long_entry_zone=float(e[1]),
                    p_short_entry_zone=float(e[2]),
                    p_hold_long=float(h[1]),
                    p_hold_short=float(h[2]),
                    p_exit_long=float(ex[1]),
                    p_exit_short=float(ex[2]),
                    expected_opportunity_roi=opp,
                )
            )
        return cls(rows)

    def signal_at(self, idx: int) -> BestPointSignal:
        if idx < 0 or idx >= len(self.rows):
            return BestPointSignal()
        return self.rows[idx]

