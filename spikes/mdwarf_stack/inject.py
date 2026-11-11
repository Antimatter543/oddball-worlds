"""Small-planet limb-darkened transit injection into raw binned flux, and transit-time matching."""

from __future__ import annotations

import math

import numpy as np

from common import BIN_D
from search import G_AU3_MSUN_D2, RSUN_AU

REARTH_RSUN = 0.0091577
U1, U2 = 0.25, 0.35
NSUB = 10


def transit_model(t, P, t0, rp_re, b, mass, radius):
    """Relative flux (1 - dip) at times t; circular orbit, small-planet LD, linear limb ramp."""
    k = rp_re * REARTH_RSUN / radius
    a_rs = (G_AU3_MSUN_D2 * mass * P**2 / (4 * math.pi**2)) ** (1 / 3) / (radius * RSUN_AU)
    ph = 2 * math.pi * (((t - t0) / P + 0.5) % 1.0 - 0.5)
    x = a_rs * np.sin(ph)
    y = b * np.cos(ph)
    z = np.sqrt(x**2 + y**2)
    front = np.cos(ph) > 0
    r = np.clip(z, 0, 1)
    mu = np.sqrt(1 - r**2)
    intens = 1 - U1 * (1 - mu) - U2 * (1 - mu) ** 2
    norm = 1 - U1 / 3 - U2 / 6
    frac = np.clip((1 + k - z) / (2 * k), 0, 1)
    dip = np.where(front, k**2 * intens / norm * frac, 0.0)
    return 1 - dip


def binned_model(idx, **kw):
    off = (np.arange(NSUB) + 0.5) / NSUB
    tt = (idx[:, None] + off[None, :]) * BIN_D
    return transit_model(tt, **kw).mean(axis=1)


def duration_14(P, rp_re, b, mass, radius):
    k = rp_re * REARTH_RSUN / radius
    a_rs = (G_AU3_MSUN_D2 * mass * P**2 / (4 * math.pi**2)) ** (1 / 3) / (radius * RSUN_AU)
    arg = math.sqrt(max((1 + k) ** 2 - b**2, 0)) / (a_rs * math.sin(math.acos(b / a_rs)))
    return P / math.pi * math.asin(min(arg, 1.0))


def epochs_in_data(idx, P, t0, dur, min_cover=0.5):
    """Mid-times of epochs where >= min_cover of the transit window has data."""
    t = (idx + 0.5) * BIN_D
    n = np.round((t - t0) / P)
    inw = np.abs(t - t0 - n * P) < dur / 2
    nb_full = max(dur / BIN_D, 1.0)
    ep, c = np.unique(n[inw], return_counts=True)
    keep = c >= min_cover * nb_full
    return t0 + ep[keep] * P


def match(sig: dict, P_true, t0_true, dur_true, idx, nmax=6, frac=0.8):
    """Pre-registered transit-time match. Returns (matched, alias_n, frac_det, frac_known)."""
    Pd = sig["P"]
    ratio = Pd / P_true
    n = None
    for cand in (ratio, 1 / ratio):
        r = round(cand)
        if 1 <= r <= nmax and abs(cand - r) <= 0.01 * r:
            n = r if cand is ratio else -r
            break
    det = np.array([e[0] for e in sig["epochs"]])
    known = epochs_in_data(idx, P_true, t0_true, dur_true)
    tol = max(dur_true, sig["dur_h"] / 24)
    if len(det) == 0 or len(known) == 0:
        return False, n, 0.0, 0.0
    fd = float(np.mean([np.min(np.abs(known - x)) < tol for x in det]))
    fk = float(np.mean([np.min(np.abs(det - x)) < tol for x in known]))
    ok = n is not None and (fd >= frac or fk >= frac)
    return bool(ok), n, fd, fk
