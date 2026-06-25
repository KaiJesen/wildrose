"""024 Phase 2: TEQ-specific edge mapping and calibration (scheme B)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from trading_system.config import TeqEdgeConfig


@dataclass
class TeqEdgeCalibrator:
    """Calibration fitted on valid (no test leakage)."""

    long_scale: float = 1.0
    long_bias: float = 0.0
    short_scale: float = 1.0
    short_bias: float = 0.0
    long_shift: float = 0.0
    short_shift: float = 0.0
    part_long_x: list[float] = field(default_factory=list)
    part_long_y: list[float] = field(default_factory=list)
    part_short_x: list[float] = field(default_factory=list)
    part_short_y: list[float] = field(default_factory=list)
    method: str = "isotonic_participation_anchor"

    def apply_part_long(self, score: float) -> float:
        return _interp_iso(score, self.part_long_x, self.part_long_y, default=score)

    def apply_part_short(self, score: float) -> float:
        return _interp_iso(score, self.part_short_x, self.part_short_y, default=score)

    def apply_long(self, raw: float) -> float:
        if self.method == "affine_valid":
            return float(raw * self.long_scale + self.long_bias)
        return float(raw + self.long_shift)

    def apply_short(self, raw: float) -> float:
        if self.method == "affine_valid":
            return float(raw * self.short_scale + self.short_bias)
        return float(raw + self.short_shift)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TeqEdgeCalibrator":
        return cls(
            long_scale=float(data.get("long_scale", 1.0)),
            long_bias=float(data.get("long_bias", 0.0)),
            short_scale=float(data.get("short_scale", 1.0)),
            short_bias=float(data.get("short_bias", 0.0)),
            long_shift=float(data.get("long_shift", 0.0)),
            short_shift=float(data.get("short_shift", 0.0)),
            part_long_x=[float(v) for v in data.get("part_long_x", [])],
            part_long_y=[float(v) for v in data.get("part_long_y", [])],
            part_short_x=[float(v) for v in data.get("part_short_x", [])],
            part_short_y=[float(v) for v in data.get("part_short_y", [])],
            method=str(data.get("method", "isotonic_participation_anchor")),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TeqEdgeCalibrator":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _interp_iso(x: float, knots_x: list[float], knots_y: list[float], *, default: float) -> float:
    if not knots_x or not knots_y or len(knots_x) != len(knots_y):
        return default
    return float(np.interp(x, np.asarray(knots_x, dtype=np.float64), np.asarray(knots_y, dtype=np.float64)))


def _pav_isotonic(x: np.ndarray, y: np.ndarray, *, increasing: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators isotonic regression (numpy only)."""
    order = np.argsort(x)
    xs = x[order].astype(np.float64)
    ys = y[order].astype(np.float64)
    if xs.size == 0:
        return xs, ys
    block_x: list[list[float]] = [[float(xs[0])]]
    block_y: list[list[float]] = [[float(ys[0])]]
    block_w: list[float] = [1.0]
    for i in range(1, xs.size):
        block_x.append([float(xs[i])])
        block_y.append([float(ys[i])])
        block_w.append(1.0)
        while len(block_y) >= 2:
            avg_prev = sum(block_y[-2]) / block_w[-2]
            avg_curr = sum(block_y[-1]) / block_w[-1]
            bad = avg_prev > avg_curr if increasing else avg_prev < avg_curr
            if not bad:
                break
            merged_y = block_y[-2] + block_y[-1]
            merged_x = block_x[-2] + block_x[-1]
            block_y = block_y[:-2] + [merged_y]
            block_x = block_x[:-2] + [merged_x]
            block_w = block_w[:-2] + [block_w[-2] + block_w[-1]]
    knot_x = np.asarray([sum(xb) / len(xb) for xb in block_x], dtype=np.float64)
    knot_y = np.asarray([sum(yb) / w for yb, w in zip(block_y, block_w)], dtype=np.float64)
    return knot_x, knot_y


def fit_isotonic_knots(scores: np.ndarray, targets: np.ndarray) -> tuple[list[float], list[float]]:
    if scores.size < 2:
        return [], []
    xs, ys = _pav_isotonic(scores.astype(np.float64), targets.astype(np.float64), increasing=True)
    return xs.tolist(), ys.tolist()


def compute_teq_edge_raw(
    *,
    edge_5: float,
    edge_24: float,
    participate_score_long: float,
    participate_score_short: float,
    cfg: TeqEdgeConfig,
) -> tuple[float, float]:
    """Map model heads to raw TEQ edges (pre-calibration)."""
    part_long = 2.0 * participate_score_long - 1.0
    part_short = 2.0 * participate_score_short - 1.0
    teq_long = (
        cfg.weight_edge_5 * edge_5
        + cfg.weight_edge_24 * edge_24
        + cfg.weight_participation * part_long
    )
    teq_short = (
        cfg.weight_edge_5 * (-edge_5)
        + cfg.weight_edge_24 * (-edge_24)
        + cfg.weight_participation * part_short
    )
    return float(teq_long), float(teq_short)


def apply_teq_calibration(
    teq_long_raw: float,
    teq_short_raw: float,
    *,
    calibrator: TeqEdgeCalibrator | None,
    cfg: TeqEdgeConfig,
) -> tuple[float, float]:
    if not cfg.use_calibrated or calibrator is None:
        return teq_long_raw, teq_short_raw
    return calibrator.apply_long(teq_long_raw), calibrator.apply_short(teq_short_raw)


def _fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size < 2 or np.std(x) < 1e-8:
        return 1.0, 0.0
    scale = float(np.cov(x, y, bias=True)[0, 1] / (np.var(x) + 1e-8))
    bias = float(y.mean() - scale * x.mean())
    return scale, bias


def fit_teq_edge_calibrator(
    raw_long: np.ndarray,
    ideal_long: np.ndarray,
    raw_short: np.ndarray,
    ideal_short: np.ndarray,
    *,
    part_long: np.ndarray | None = None,
    part_short: np.ndarray | None = None,
    legacy_edge_long: np.ndarray | None = None,
    legacy_edge_short: np.ndarray | None = None,
) -> TeqEdgeCalibrator:
    """Fit isotonic participation + edge anchor on valid labels."""
    part_long = np.asarray([] if part_long is None else part_long, dtype=np.float64)
    part_short = np.asarray([] if part_short is None else part_short, dtype=np.float64)
    legacy_edge_long = np.asarray([] if legacy_edge_long is None else legacy_edge_long, dtype=np.float64)
    legacy_edge_short = np.asarray([] if legacy_edge_short is None else legacy_edge_short, dtype=np.float64)

    pl_x, pl_y = fit_isotonic_knots(part_long, ideal_long.astype(np.float64)) if part_long.size else ([], [])
    ps_x, ps_y = fit_isotonic_knots(part_short, ideal_short.astype(np.float64)) if part_short.size else ([], [])

    def _recompute_raw(part_scores: np.ndarray, raw: np.ndarray, knots_x: list[float], knots_y: list[float]) -> np.ndarray:
        if not knots_x:
            return raw
        calibrated = np.asarray([_interp_iso(float(s), knots_x, knots_y, default=float(s)) for s in part_scores])
        part_term = 2.0 * calibrated - 1.0
        base = raw - 0.0  # raw already includes participation; rebuild from structure is hard
        # Approximate: replace participation component by delta
        delta = 0.20 * (part_term - (2.0 * part_scores - 1.0))
        return raw + delta

    adj_long = _recompute_raw(part_long, raw_long.astype(np.float64), pl_x, pl_y) if part_long.size else raw_long.astype(np.float64)
    adj_short = _recompute_raw(part_short, raw_short.astype(np.float64), ps_x, ps_y) if part_short.size else raw_short.astype(np.float64)

    long_shift = 0.0
    short_shift = 0.0
    ideal_l = ideal_long.astype(np.float64) >= 0.5
    ideal_s = ideal_short.astype(np.float64) >= 0.5
    if legacy_edge_long.size and np.any(ideal_l):
        target = float(np.median(legacy_edge_long[ideal_l]))
        current = float(np.median(adj_long[ideal_l]))
        long_shift = target - current
    elif adj_long.size:
        long_shift = float(np.median(legacy_edge_long) - np.median(adj_long)) if legacy_edge_long.size else 0.0

    if legacy_edge_short.size and np.any(ideal_s):
        target = float(np.median(-legacy_edge_short[ideal_s]))
        current = float(np.median(adj_short[ideal_s]))
        short_shift = target - current
    elif adj_short.size and legacy_edge_short.size:
        short_shift = float(np.median(-legacy_edge_short) - np.median(adj_short))

    return TeqEdgeCalibrator(
        part_long_x=pl_x,
        part_long_y=pl_y,
        part_short_x=ps_x,
        part_short_y=ps_y,
        long_shift=long_shift,
        short_shift=short_shift,
        method="isotonic_participation_anchor",
    )
