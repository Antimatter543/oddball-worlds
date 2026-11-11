"""Gate 2 preprocessing: optional rotation removal before the gate-1 pipeline, and the two
period ranges.

Rotation removal "harm" (per sector, before the gate-1 preprocessing):
  1. Lomb-Scargle of the raw normalised flux (per-segment median removed, segment edges and
     >3 sigma upward points excluded) over P_rot in [ROT_PMIN, ROT_PMAX] d.
  2. If the peak's standard-normalised power >= ROT_POWER, fit by robust least squares
         f - 1 = sum_{h=1..ROT_H} [a_h(t) cos(2 pi h t / P) + b_h(t) sin(2 pi h t / P)] + c(t)
     where a_h, b_h, c are linear hat-function splines with knots every ROT_KNOT_D days within
     each segment (spot evolution), clipping |resid| > 3 sigma (both signs) from the FIT only,
     3 passes. Then divide the flux by (1 + model) at every point.
  3. Run the unchanged gate-1 preprocessing (trim, upward flare clip, 0.5 d biweight, sigma).
"none" is exactly the gate-1 preprocessing.

Search ranges: SHORT = (1, 10) d and LONG = (10, 60) d, searched independently (separate signal
slots), with the gate-1 engine (search.search_star with prange).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BIN_D
from search import GAP_D, TRIM_D, preprocess, robust_sigma

SHORT = (1.0, 10.0)
LONG = (10.0, 60.0)
ROT_PMIN, ROT_PMAX = 0.1, 5.0
ROT_POWER = 0.02
ROT_H = 3
ROT_KNOT_D = 3.0


def _segments(t):
    br = np.flatnonzero(np.diff(t) > GAP_D) + 1
    return np.split(np.arange(len(t)), br)


def _hat_basis(t, seg_list, knot_d):
    """Linear hat functions with knots every knot_d days inside each segment."""
    cols = []
    for seg in seg_list:
        ts = t[seg]
        lo, hi = ts[0], ts[-1]
        nk = max(2, int(np.ceil((hi - lo) / knot_d)) + 1)
        knots = np.linspace(lo, hi, nk)
        h = knots[1] - knots[0]
        for k in knots:
            c = np.zeros(len(t))
            c[seg] = np.clip(1 - np.abs(ts - k) / h, 0, None)
            cols.append(c)
    return np.array(cols).T


def rot_sector(t, f):
    """Rotation model (relative) for one sector, and info. f is the raw normalised flux."""
    from astropy.timeseries import LombScargle

    segs = _segments(t)
    y = f - 1.0
    use = np.isfinite(y)
    for seg in segs:
        ts = t[seg]
        use[seg[(ts < ts[0] + TRIM_D) | (ts > ts[-1] - TRIM_D)]] = False
    yy = y.copy()
    for seg in segs:
        m = use[seg]
        if m.sum() > 10:
            yy[seg] -= np.median(y[seg][m])
    sig = robust_sigma(yy[use])
    use &= yy < 3 * sig + np.median(yy[use]) if np.isfinite(sig) else use
    info = {"P": None, "power": 0.0, "applied": False}
    if use.sum() < 200:
        return np.zeros_like(y), info
    freq = np.linspace(1 / ROT_PMAX, 1 / ROT_PMIN, 20000)
    pw = LombScargle(t[use], yy[use]).power(freq, method="fast")
    j = int(np.argmax(pw))
    P = float(1 / freq[j])
    info.update(P=P, power=float(pw[j]))
    if pw[j] < ROT_POWER:
        return np.zeros_like(y), info
    hb = _hat_basis(t, segs, ROT_KNOT_D)
    cols = [hb]
    for h in range(1, ROT_H + 1):
        ph = 2 * np.pi * h * t / P
        cols += [hb * np.cos(ph)[:, None], hb * np.sin(ph)[:, None]]
    A = np.hstack(cols)
    fit = use.copy()
    model = np.zeros_like(y)
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(A[fit], y[fit], rcond=None)
        model = A @ coef
        r = y - model
        s = robust_sigma(r[fit])
        fit = use & (np.abs(r) < 3 * s)
    # report only the rotational part's amplitude (baseline c(t) is removed by the biweight too)
    rot_part = A[:, hb.shape[1]:] @ coef[hb.shape[1]:]
    info.update(applied=True, amp=float(np.percentile(rot_part[use], 95)
                                        - np.percentile(rot_part[use], 5)) / 2,
                sigma=float(sig))
    return model, info


def remove_rotation(lc):
    t = (lc["idx"] + 0.5) * BIN_D
    f = lc["flux"].astype(np.float64)
    out = f.copy()
    infos = {}
    for s in np.unique(lc["sector"]):
        m = np.flatnonzero(lc["sector"] == s)
        model, info = rot_sector(t[m], f[m])
        out[m] = f[m] / (1.0 + model)
        infos[int(s)] = info
    lc2 = dict(lc)
    lc2["flux"] = out
    return lc2, infos


def rot_summary(infos):
    ap = [v for v in infos.values() if v["applied"]]
    Ps = [v["P"] for v in ap]
    return {"n_sec_applied": len(ap), "n_sec": len(infos),
            "prot": float(np.median(Ps)) if Ps else None,
            "amp": float(np.median([v["amp"] for v in ap])) if ap else 0.0,
            "applied": bool(ap)}


def preprocess2(lc, rot="harm"):
    if rot == "none":
        return preprocess(lc), {"applied": False}
    lc2, infos = remove_rotation(lc)
    info = rot_summary(infos)
    info["sectors"] = {k: {kk: vv for kk, vv in v.items()} for k, v in infos.items()}
    return preprocess(lc2), info
