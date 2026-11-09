"""Calibration strata shared by every scorer and the evaluator (docs/ML_DESIGN.md §4, §6).

A stratum is ``Tmag bin x camera/CCD``. Faint stars are noisier and some CCDs carry more
scattered light, so a score only means "weird" relative to stars like it. The scorer turns raw
scores into percentiles within a stratum, and the evaluator ranks injections against real stars
of the same stratum. Both must use this one definition.
"""

from __future__ import annotations

import numpy as np

# Bin edges in TESS magnitude. Five bins: <=10, 10-11, 11-12, 12-13, >13.
# (S32: ~9% / 14% / 32% / 41% / 4% of light curves; the >13 bin holds the crowded
# faint stars whose PDCSAP amplitudes are unreliable.)
TMAG_EDGES = (10.0, 11.0, 12.0, 13.0)
TMAG_LABELS = ("<=10", "10-11", "11-12", "12-13", ">13")


def tmag_bin(tmag) -> np.ndarray:
    """Integer Tmag bin 0..4. NaN Tmag goes to the faintest bin (least trusted)."""
    t = np.asarray(tmag, dtype=np.float64)
    t = np.where(np.isfinite(t), t, 99.0)
    return np.searchsorted(np.asarray(TMAG_EDGES), t, side="left").astype(np.int64)


def ccd_index(camera, ccd) -> np.ndarray:
    """0..15 for camera 1-4 x CCD 1-4."""
    return (np.asarray(camera, np.int64) - 1) * 4 + (np.asarray(ccd, np.int64) - 1)


def stratum_id(tmag, camera, ccd) -> np.ndarray:
    """Integer stratum id = tmag_bin * 16 + ccd_index (0..79)."""
    return tmag_bin(tmag) * 16 + ccd_index(camera, ccd)


def stratum_label(sid: int) -> str:
    sid = int(sid)
    c = sid % 16
    return f"T{TMAG_LABELS[sid // 16]}/cam{c // 4 + 1}ccd{c % 4 + 1}"
