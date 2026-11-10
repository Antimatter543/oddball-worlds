"""Shared light-curve plot helpers (sanity plots, ML quick look, the labelling gallery).

All of them draw on an existing matplotlib axis and never create figures, so callers own the
layout. Times are BTJD on the global 30-min grid (``store.load_lightcurves`` ``time``).
"""

from __future__ import annotations

import numpy as np

from oddball.tess.lightcurve import DEFAULT_QUALITY_BITMASK

DUMP_BIT = 32  # QUALITY "Desat": reaction-wheel momentum dump
SCATTERED_BITS = 2048 | 4096  # "Straylight", "Straylight2"
# Pointing / safe-mode / manual-exclude / outlier / calibration bits, without the dump bit,
# which is drawn separately as red ticks.
OTHER_FLAG_BITS = DEFAULT_QUALITY_BITMASK & ~DUMP_BIT


def surprise_time(origin: int, n: int, bin_minutes: float = 30.0) -> np.ndarray:
    """BTJD bin centres of a time-resolved map written on the ML fixed grid starting at ``origin``."""
    return (origin + np.arange(n) + 0.5) * bin_minutes / 1440.0


def mark_systematics(ax, t: np.ndarray, qflags: np.ndarray, dump_times=None, *,
                     other_flags: bool = False) -> None:  # fmt: skip
    """Momentum dumps (red ticks), scattered-light bins (orange band), optionally other flags.

    ``dump_times``: sector-wide dump times to draw instead of the light curve's own bit-32 bins
    (dumps are spacecraft events, so every star shares them).
    """
    q = np.asarray(qflags).astype(np.int64)
    dumps = t[(q & DUMP_BIT) != 0] if dump_times is None else np.asarray(dump_times)
    for tt in dumps:
        ax.axvline(tt, color="tab:red", lw=0.6, alpha=0.6, zorder=0)
    sl = (q & SCATTERED_BITS) != 0
    if sl.any():
        ax.fill_between(t, 0, 1, where=sl, transform=ax.get_xaxis_transform(),
                        color="tab:orange", alpha=0.15, lw=0, zorder=0)  # fmt: skip
    if other_flags:
        of = (q & OTHER_FLAG_BITS) != 0
        if of.any():
            ax.fill_between(t, 0, 1, where=of, transform=ax.get_xaxis_transform(),
                            color="0.5", alpha=0.18, lw=0, zorder=0)  # fmt: skip


def overlay_surprise(ax, t: np.ndarray, s: np.ndarray, color: str = "tab:red",
                     alpha: float = 0.25):  # fmt: skip
    """Time-resolved anomaly as a filled curve on a twin axis (no ticks). Returns the twin."""
    ax2 = ax.twinx()
    ax2.fill_between(t, 0, np.nan_to_num(np.clip(s, 0, None)), color=color, alpha=alpha, lw=0)
    ax2.set_yticks([])
    ax2.set_ylim(bottom=0)
    return ax2


def robust_ylim(ax, *series, lo: float = 0.5, hi: float = 99.5, pad: float = 0.15) -> None:
    """Y limits from percentiles of the finite values of all ``series`` (ignores outliers)."""
    v = np.concatenate([np.asarray(s, np.float64).ravel() for s in series])
    v = v[np.isfinite(v)]
    if not len(v):
        return
    a, b = np.percentile(v, [lo, hi])
    p = pad * (b - a) + 1e-6
    ax.set_ylim(a - p, b + p)
