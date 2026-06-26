"""025: channel-specific edge mapping and calibration (parallel to teq_edge)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from trading_system.config import ChannelEdgeMixConfig, SlowUpParticipationGateConfig


@dataclass
class ChannelEdgeCalibrator:
    """Per-channel affine calibration fitted on valid (no test leakage)."""

    long_shift: float = 0.0
    short_shift: float = 0.0
    method: str = "shift_anchor"

    def apply_long(self, raw: float) -> float:
        return float(raw + self.long_shift)

    def apply_short(self, raw: float) -> float:
        return float(raw + self.short_shift)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ChannelEdgeCalibrator":
        return cls(
            long_shift=float(data.get("long_shift", 0.0)),
            short_shift=float(data.get("short_shift", 0.0)),
            method=str(data.get("method", "shift_anchor")),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ChannelEdgeCalibrator":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def compute_channel_edge_raw(
    *,
    legacy_edge: float,
    teq_edge_long_raw: float,
    teq_edge_short_raw: float,
    participate_score_long: float,
    participate_score_short: float,
    cfg: ChannelEdgeMixConfig,
) -> tuple[float, float]:
    """Map legacy + raw TEQ + participation to channel raw edges (pre-calibration)."""
    part_long = 2.0 * participate_score_long - 1.0
    part_short = 2.0 * participate_score_short - 1.0
    long_raw = (
        cfg.weight_legacy * legacy_edge
        + cfg.weight_teq * teq_edge_long_raw
        + cfg.weight_part * part_long
    )
    short_raw = (
        cfg.weight_legacy * (-legacy_edge)
        + cfg.weight_teq * teq_edge_short_raw
        + cfg.weight_part * part_short
    )
    return float(long_raw), float(short_raw)


def apply_channel_calibration(
    raw_long: float,
    raw_short: float,
    *,
    calibrator: ChannelEdgeCalibrator | None,
    use_calibrated: bool,
) -> tuple[float, float]:
    if not use_calibrated or calibrator is None:
        return raw_long, raw_short
    return calibrator.apply_long(raw_long), calibrator.apply_short(raw_short)
