"""Trend rail / close-line search with vectorized NumPy and optional GPU backend."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np

try:
    import torch

    _TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH = False


class Backend(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"


@dataclass(frozen=True)
class TrendLine:
    """y = slope * x + intercept (x is global bar index)."""

    i0: int
    i1: int
    slope: float
    intercept: float

    def value_at(self, x: float) -> float:
        return self.slope * x + self.intercept

    def values_at(self, xs: np.ndarray) -> np.ndarray:
        return self.slope * xs + self.intercept


def _cuda_ready() -> bool:
    if not _TORCH:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_backend(backend: Backend) -> Literal["cpu", "gpu"]:
    if backend == Backend.CPU:
        return "cpu"
    if backend == Backend.GPU:
        if not _cuda_ready():
            raise RuntimeError("backend=gpu requested but CUDA is not available")
        return "gpu"
    return "gpu" if _cuda_ready() else "cpu"


def _pair_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    i, j = np.triu_indices(n, k=1)
    return i, j


def _select_pair(
    slopes: np.ndarray,
    valid: np.ndarray,
    *,
    pick: Literal["flattest", "steepest"],
) -> int | None:
    if not np.any(valid):
        return None
    masked = np.where(valid, slopes, np.nan)
    if pick == "flattest":
        return int(np.nanargmin(np.abs(masked)))
    return int(np.nanargmax(np.abs(masked)))


def _find_line_cpu(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    bound: Literal["below", "above"],
    pick: Literal["flattest", "steepest"],
) -> TrendLine | None:
    n = len(xs)
    if n < 2:
        return None
    i_loc, j_loc = _pair_indices(n)
    if len(i_loc) == 0:
        return None

    x0 = xs[i_loc]
    x1 = xs[j_loc]
    y0 = ys[i_loc]
    y1 = ys[j_loc]
    dx = x1 - x0
    ok = np.abs(dx) > 1e-12
    if not np.any(ok):
        return None

    i_loc = i_loc[ok]
    j_loc = j_loc[ok]
    x0, x1, y0, y1, dx = x0[ok], x1[ok], y0[ok], y1[ok], dx[ok]
    slopes = (y1 - y0) / dx
    intercepts = y0 - slopes * x0

    valid = np.ones(len(slopes), dtype=bool)
    for k in range(n):
        mask = (i_loc != k) & (j_loc != k)
        if not np.any(mask):
            continue
        y_line = slopes[mask] * xs[k] + intercepts[mask]
        if bound == "below":
            valid[mask] &= ys[k] <= y_line + 1e-8
        else:
            valid[mask] &= ys[k] >= y_line - 1e-8

    idx = _select_pair(slopes, valid, pick=pick)
    if idx is None:
        return None
    gi = int(i_loc[idx])
    gj = int(j_loc[idx])
    return TrendLine(gi, gj, float(slopes[idx]), float(intercepts[idx]))


def _find_line_gpu(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    bound: Literal["below", "above"],
    pick: Literal["flattest", "steepest"],
) -> TrendLine | None:
    device = torch.device("cuda")
    xs_t = torch.as_tensor(xs, dtype=torch.float64, device=device)
    ys_t = torch.as_tensor(ys, dtype=torch.float64, device=device)
    n = xs_t.numel()
    if n < 2:
        return None
    i_loc, j_loc = _pair_indices(n)
    if len(i_loc) == 0:
        return None
    i_t = torch.as_tensor(i_loc, device=device)
    j_t = torch.as_tensor(j_loc, device=device)
    x0 = xs_t[i_t]
    x1 = xs_t[j_t]
    y0 = ys_t[i_t]
    y1 = ys_t[j_t]
    dx = x1 - x0
    ok = dx.abs() > 1e-12
    if not torch.any(ok):
        return None
    i_t, j_t = i_t[ok], j_t[ok]
    x0, x1, y0, y1, dx = x0[ok], x1[ok], y0[ok], y1[ok], dx[ok]
    slopes = (y1 - y0) / dx
    intercepts = y0 - slopes * x0
    valid = torch.ones(slopes.shape[0], dtype=torch.bool, device=device)
    for k in range(n):
        mask = (i_t != k) & (j_t != k)
        if not torch.any(mask):
            continue
        y_line = slopes[mask] * xs_t[k] + intercepts[mask]
        if bound == "below":
            valid[mask] &= ys_t[k] <= y_line + 1e-8
        else:
            valid[mask] &= ys_t[k] >= y_line - 1e-8
    slopes_np = slopes.detach().cpu().numpy()
    valid_np = valid.detach().cpu().numpy()
    idx = _select_pair(slopes_np, valid_np, pick=pick)
    if idx is None:
        return None
    gi = int(i_t[idx].item())
    gj = int(j_t[idx].item())
    return TrendLine(gi, gj, float(slopes_np[idx]), float(intercepts[idx].item()))


def find_trend_line(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    bound: Literal["below", "above"],
    pick: Literal["flattest", "steepest"],
    backend: Backend = Backend.AUTO,
) -> TrendLine | None:
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    use = resolve_backend(backend)
    if use == "gpu":
        try:
            return _find_line_gpu(xs, ys, bound=bound, pick=pick)
        except Exception:
            pass
    return _find_line_cpu(xs, ys, bound=bound, pick=pick)


def segment_bounds(highs: np.ndarray, lows: np.ndarray, a: int, t: int) -> tuple[int, int]:
    """Return (a, b) where b is argmin low (for upper-rail) or argmax high (for lower-rail) in [a,t]."""
    if t < a:
        return a, a
    seg_low = lows[a : t + 1]
    seg_high = highs[a : t + 1]
    b_low = a + int(np.argmin(seg_low))
    b_high = a + int(np.argmax(seg_high))
    return b_low, b_high


def find_upper_rail(
    highs: np.ndarray,
    lows: np.ndarray,
    a: int,
    t: int,
    *,
    backend: Backend = Backend.AUTO,
) -> TrendLine | None:
    """Upper rail on highs in [a, b] with b = argmin(low) in [a,t]."""
    b, _ = segment_bounds(highs, lows, a, t)
    if b <= a:
        b = min(a + 1, t)
    xs = np.arange(a, b + 1, dtype=np.float64)
    ys = highs[a : b + 1].astype(np.float64)
    return find_trend_line(xs, ys, bound="below", pick="flattest", backend=backend)


def find_lower_rail(
    highs: np.ndarray,
    lows: np.ndarray,
    a: int,
    t: int,
    *,
    backend: Backend = Backend.AUTO,
) -> TrendLine | None:
    """Lower rail on lows in [a, b] with b = argmax(high) in [a,t]."""
    _, b = segment_bounds(highs, lows, a, t)
    if b <= a:
        b = min(a + 1, t)
    xs = np.arange(a, b + 1, dtype=np.float64)
    ys = lows[a : b + 1].astype(np.float64)
    return find_trend_line(xs, ys, bound="above", pick="flattest", backend=backend)


def find_long_close_line(
    highs: np.ndarray,
    lows: np.ndarray,
    a: int,
    t: int,
    *,
    backend: Backend = Backend.AUTO,
) -> TrendLine | None:
    """Long close line on lows in [a, b] with b = argmax(high) in [a,t], steepest slope."""
    _, b = segment_bounds(highs, lows, a, t)
    if b <= a:
        b = min(a + 1, t)
    xs = np.arange(a, b + 1, dtype=np.float64)
    ys = lows[a : b + 1].astype(np.float64)
    return find_trend_line(xs, ys, bound="above", pick="steepest", backend=backend)


def find_short_close_line(
    highs: np.ndarray,
    lows: np.ndarray,
    a: int,
    t: int,
    *,
    backend: Backend = Backend.AUTO,
) -> TrendLine | None:
    """Short close line on highs in [a, b] with b = argmin(low) in [a,t], steepest slope."""
    b, _ = segment_bounds(highs, lows, a, t)
    if b <= a:
        b = min(a + 1, t)
    xs = np.arange(a, b + 1, dtype=np.float64)
    ys = highs[a : b + 1].astype(np.float64)
    return find_trend_line(xs, ys, bound="below", pick="steepest", backend=backend)
