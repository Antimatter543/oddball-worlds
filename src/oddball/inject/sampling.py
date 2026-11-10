"""Parameter draws for each injection kind (docs/ML_DESIGN.md §6).

Common to all kinds: ``depth`` log-uniform over 0.1-30 %, ``duration_d`` log-uniform over
2 h - 5 d, and event times placed where the host has data (``pick_time``). What ``duration_d``
means per kind:

* single_dip, exocomet, outburst: FWHM of the event.
* aperiodic_dipper, quasiperiodic_dipper: the typical (median) FWHM of the individual dips.
* boyajian: the width of the window the overlapping dips are spread over.

``t0_btjd`` is always the time of the deepest point (peak |m-1|) of the full model.
"""

from __future__ import annotations

import numpy as np

from oddball.inject.profiles import LN2, normalise

KINDS = (
    "aperiodic_dipper",
    "quasiperiodic_dipper",
    "exocomet",
    "single_dip",
    "boyajian",
    "outburst",
)
NULL_KINDS = ("null_ccd", "null_dump")
# Extra kinds for the EB check set (``eb1``; docs/ML_DESIGN.md §14). ``null_eb`` is a per-host
# null (``is_null=True``): an eclipsing binary that should *not* rank high. ``dusty_tail`` is an
# astrophysical target (a disintegrating planet) that must not be mistaken for one.
EB_KINDS = ("null_eb", "dusty_tail")
PER_HOST_NULL_KINDS = ("null_eb",)
EB_DEPTH_RANGE = (0.003, 0.5)
EB_PERIOD_RANGE = (0.4, 40.0)
DUSTY_PERIOD_RANGE = (0.3, 3.0)
DUSTY_DEPTH_RANGE = (0.002, 0.1)
DUSTY_DURATION_RANGE = (1.0 / 24.0, 4.0 / 24.0)
RSUN_AU = 1.0 / 215.032
DEPTH_RANGE = (1e-3, 0.30)
DURATION_RANGE = (2.0 / 24.0, 5.0)
MIN_FWHM = 2.0 / 24.0


def loguniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def pick_time(
    rng: np.random.Generator,
    t: np.ndarray,
    valid: np.ndarray,
    lo_off: float,
    hi_off: float,
    min_frac: float = 0.75,
    tries: int = 300,
) -> float:
    """Random time ``tc`` whose window ``[tc+lo_off, tc+hi_off]`` is >= ``min_frac`` covered
    by valid bins and lies inside the light curve. Falls back to the best-covered of the tries
    (long windows on gappy hosts) rather than failing.
    """
    tv = t[valid]
    if tv.size == 0:
        raise ValueError("host has no valid bins")
    t_lo, t_hi = tv[0] - lo_off, tv[-1] - hi_off
    if t_hi <= t_lo:  # window longer than the light curve: centre it
        return float(0.5 * (tv[0] + tv[-1]) - 0.5 * (lo_off + hi_off))
    dt = float(np.median(np.diff(t)))
    best, best_frac = None, -1.0
    for _ in range(tries):
        tc = rng.uniform(t_lo, t_hi)
        w = (t >= tc + lo_off - dt / 2) & (t <= tc + hi_off + dt / 2)
        frac = float(valid[w].mean()) if w.any() else 0.0
        # the peak bin itself must hold data, otherwise the event is invisible
        k = int(np.argmin(np.abs(t - tc)))
        if not valid[k]:
            frac = 0.0
        if frac >= min_frac:
            return float(tc)
        if frac > best_frac:
            best, best_frac = tc, frac
    return float(best)


def _finish(kind, depth, duration, comps, sign, t, extra=None) -> dict:
    norm, t_peak = normalise(comps, float(t[0]) - 1.0, float(t[-1]) + 1.0)
    p = {
        "kind": kind,
        "depth": float(depth),
        "duration_d": float(duration),
        "sign": int(sign),
        "norm": norm,
        "t0_btjd": t_peak,
        "components": comps,
    }
    if extra:
        p.update(extra)
    return p


def _agauss(tc, fwhm, rng, spread=0.4):
    r = float(np.exp(rng.normal(0.0, spread)))  # egress/ingress width ratio
    fwhm = max(fwhm, MIN_FWHM)
    return {"shape": "agauss", "tc": float(tc), "fwhm_in": fwhm * 2 / (1 + r),
            "fwhm_eg": fwhm * 2 * r / (1 + r)}  # fmt: skip


def draw_params(kind: str, rng: np.random.Generator, t: np.ndarray, valid: np.ndarray,
                host: dict | None = None) -> dict:  # fmt: skip
    """Draw one injection of ``kind`` for a host with bin centres ``t`` and validity ``valid``.

    ``host`` (``radius``, R_sun) is used by ``null_eb`` for physical eclipse durations.
    """
    if kind == "null_eb":
        return draw_eb(rng, t, valid, (host or {}).get("radius", np.nan))
    if kind == "dusty_tail":
        return draw_dusty_tail(rng, t, valid)
    depth = loguniform(rng, *DEPTH_RANGE)
    dur = loguniform(rng, *DURATION_RANGE)

    if kind == "single_dip":
        tc = pick_time(rng, t, valid, -dur / 2, dur / 2)
        return _finish(kind, depth, dur, [{"shape": "gauss", "tc": tc, "fwhm": dur}], -1, t)

    if kind == "exocomet":
        rise = dur * rng.uniform(0.03, 0.15)  # half-max -> peak; full ramp is 2*rise
        tau = (dur - rise) / LN2
        tc = pick_time(rng, t, valid, -2 * rise, dur)
        comp = {"shape": "fred", "tc": tc, "rise": rise, "tau": tau, "rise_kind": "linear"}
        return _finish(kind, depth, dur, [comp], -1, t)

    if kind == "outburst":
        rise = dur * rng.uniform(0.1, 0.3)
        tau = (dur - rise) / LN2
        tc = pick_time(rng, t, valid, -2 * rise, dur)
        comp = {"shape": "fred", "tc": tc, "rise": rise, "tau": tau, "rise_kind": "gauss"}
        return _finish(kind, depth, dur, [comp], +1, t)

    if kind == "aperiodic_dipper":
        n = int(rng.integers(3, 11))
        comps = []
        for i in range(n):
            w = float(np.clip(dur * np.exp(rng.normal(0.0, 0.3)), MIN_FWHM, 2 * dur))
            tc = pick_time(rng, t, valid, -w / 2, w / 2)
            c = _agauss(tc, w, rng)
            c["amp"] = 1.0 if i == 0 else float(rng.uniform(0.3, 1.0))
            comps.append(c)
        return _finish(kind, depth, dur, comps, -1, t, {"n_dips": n})

    if kind == "quasiperiodic_dipper":
        period = float(np.clip(dur * rng.uniform(3.0, 6.0), 0.5, 13.0))
        anchor = pick_time(rng, t, valid, -dur / 2, dur / 2)
        ks = np.arange(-int((anchor - t[0]) / period) - 1, int((t[-1] - anchor) / period) + 2)
        comps = []
        for k in ks:
            jit = 0.0 if k == 0 else rng.normal(0.0, 0.08 * period)
            tc = anchor + k * period + jit
            if tc < t[0] - dur or tc > t[-1] + dur:
                continue
            w = dur * rng.uniform(0.7, 1.3) if k else dur
            c = _agauss(tc, w, rng, spread=0.3)
            c["amp"] = 1.0 if k == 0 else float(rng.uniform(0.4, 1.0))
            comps.append(c)
        return _finish(kind, depth, dur, comps, -1, t, {"period_d": period})

    if kind == "boyajian":
        n = int(rng.integers(3, 8))
        c0 = pick_time(rng, t, valid, -dur / 2, dur / 2)
        comps = []
        for i in range(n):
            tc = c0 + (0.0 if i == 0 else rng.uniform(-dur / 2, dur / 2))
            w = float(max(dur * rng.uniform(0.15, 0.5), MIN_FWHM))
            c = _agauss(tc, w, rng, spread=0.8)  # strongly asymmetric components
            c["amp"] = 1.0 if i == 0 else float(rng.uniform(0.3, 1.0))
            comps.append(c)
        return _finish(kind, depth, dur, comps, -1, t, {"n_dips": n})

    raise ValueError(f"unknown kind {kind!r}")


def draw_null_event(
    kind: str, rng: np.random.Generator, t: np.ndarray, valid_frac: np.ndarray, depth: float,
    dump_times: np.ndarray | None = None,
) -> dict:
    """One instrument-like event, applied *identically* to every host it is given to.

    ``valid_frac``: per-bin fraction of the event's hosts that have data (the CCD's common
    coverage). ``null_ccd`` picks a time where >= 90 % of hosts have data and a generic
    instrumental shape (Gaussian, sharp-edged box, or linear ramp with instant recovery).
    ``null_dump`` starts at a momentum dump (QUALITY bit 32) with a sharp drop and an
    exponential recovery of 2-8 h: the exocomet-like shape that pointing jitter makes.
    """
    valid = valid_frac >= 0.9
    if kind == "null_ccd":
        dur = loguniform(rng, 2.0 / 24.0, 2.0)
        shape = str(rng.choice(["gauss", "box", "ramp"]))
        if shape == "gauss":
            tc = pick_time(rng, t, valid, -dur / 2, dur / 2, min_frac=0.9)
            comp = {"shape": "gauss", "tc": tc, "fwhm": dur}
        elif shape == "box":
            edge = dur * 0.05
            tc = pick_time(rng, t, valid, -dur / 2, dur / 2, min_frac=0.9)
            comp = {"shape": "box", "tc": tc, "width": dur - edge, "edge": edge}
        else:
            w = 2 * dur  # FWHM of a ramp is half its length
            tc = pick_time(rng, t, valid, -w, 0.0, min_frac=0.9)
            comp = {"shape": "ramp", "tc": tc, "width": w}
        p = _finish(kind, depth, dur, [comp], -1, t, {"null_shape": shape})
    elif kind == "null_dump":
        if dump_times is None or len(dump_times) == 0:
            raise ValueError("null_dump needs momentum-dump times")
        td = float(rng.choice(dump_times))
        dur = loguniform(rng, 2.0 / 24.0, 8.0 / 24.0)
        rise = 5.0 / 1440.0
        tau = (dur - rise) / LN2
        comp = {"shape": "fred", "tc": td + 2 * rise, "rise": rise, "tau": tau,
                "rise_kind": "linear"}  # fmt: skip
        p = _finish(kind, depth, dur, [comp], -1, t, {"null_shape": "dump_fred",
                                                      "dump_btjd": td})
    else:
        raise ValueError(kind)
    p["is_null"] = True
    return p


def _epochs(rng, t, valid, period, lo_off, hi_off):
    """Epoch times ``t_ref + k P`` covering the light curve (+- one period), with ``t_ref`` put
    where the host has data so that at least one eclipse is visible."""
    t_ref = pick_time(rng, t, valid, lo_off, hi_off)
    k = np.arange(-int((t_ref - t[0]) / period) - 2, int((t[-1] - t_ref) / period) + 3)
    return t_ref + k * period


def draw_eb(rng: np.random.Generator, t: np.ndarray, valid: np.ndarray, radius: float) -> dict:
    """A detached or grazing eclipsing binary on the host (a per-host null, ``is_null=True``).

    Period log-uniform 0.4-40 d (so some show a single eclipse per sector); primary depth
    log-uniform 0.3-50%; total duration from the host radius (TIC; 1 R_sun if missing):
    ``T14 = P/pi asin(k sqrt(1 - b^2))``, ``k = (R1 + R2)/a`` with R2/R1 ~ U(0.3, 1.2), a from
    Kepler's law (M = 2 x clip(R, 0.3, 2) M_sun), impact parameter b ~ U(0, 0.95): flat-bottomed
    (trapezoid) unless grazing (b > 0.8: V-shaped). Secondary depth ratio 0 (25%) or U(0.05, 1);
    at phase 0.5 (70%) or U(0.3, 0.7) (eccentric, with its own width x U(0.6, 1.5)).
    """
    R = radius if np.isfinite(radius) and radius > 0 else 1.0
    R = float(np.clip(R, 0.3, 10.0))
    m = 2.0 * float(np.clip(R, 0.3, 2.0))
    P = loguniform(rng, *EB_PERIOD_RANGE)
    a_rs = (m * (P / 365.25) ** 2) ** (1.0 / 3.0) / RSUN_AU  # semi-major axis in R_sun
    q = float(rng.uniform(0.3, 1.2))
    k = min(0.95, R * (1.0 + q) / a_rs)
    b = float(rng.uniform(0.0, 0.95))
    T14 = max(P / np.pi * np.arcsin(k * np.sqrt(1.0 - b * b)), 1.0 / 24.0)
    grazing = b > 0.8
    edge = 0.5 * T14 if grazing else T14 * float(rng.uniform(0.1, 0.3))
    flat = max(T14 - 2.0 * edge, 0.0)
    depth = loguniform(rng, *EB_DEPTH_RANGE)
    sec = 0.0 if rng.uniform() < 0.25 else float(rng.uniform(0.05, 1.0))
    ecc = rng.uniform() >= 0.7
    phase = float(rng.uniform(0.3, 0.7)) if ecc else 0.5
    wsec = float(rng.uniform(0.6, 1.5)) if ecc else 1.0
    comps = []
    for tc in _epochs(rng, t, valid, P, -T14 / 2, T14 / 2):
        comps.append({"shape": "box", "tc": float(tc), "width": flat, "edge": edge, "amp": 1.0})
        if sec > 0:
            comps.append({"shape": "box", "tc": float(tc + phase * P), "width": flat * wsec,
                          "edge": edge * wsec, "amp": sec})  # fmt: skip
    p = _finish("null_eb", depth, T14, comps, -1, t,
                {"period_d": P, "sec_ratio": sec, "sec_phase": phase, "impact": b,
                 "grazing": bool(grazing), "host_radius": R})  # fmt: skip
    p["is_null"] = True
    return p


def draw_dusty_tail(rng: np.random.Generator, t: np.ndarray, valid: np.ndarray) -> dict:
    """A disintegrating planet: strictly periodic transits whose depth changes transit to transit.

    Period log-uniform 0.3-3 d (KIC 12557548 0.65 d, K2-22 0.38 d, BD+05 4868 Ab 1.27 d); total
    transit duration 1-4 h; a sharp ingress and an exponential egress from the trailing dust tail
    (``fred``, rise 10%, tau 40% of the duration); each transit's amplitude U(0.05, 1) of the
    maximum depth, which is log-uniform 0.2-10%.
    """
    P = loguniform(rng, *DUSTY_PERIOD_RANGE)
    dur = loguniform(rng, *DUSTY_DURATION_RANGE)
    depth = loguniform(rng, *DUSTY_DEPTH_RANGE)
    comps = []
    for tc in _epochs(rng, t, valid, P, -0.2 * dur, dur):
        comps.append({"shape": "fred", "tc": float(tc), "rise": 0.1 * dur, "tau": 0.4 * dur,
                      "rise_kind": "linear", "amp": float(rng.uniform(0.05, 1.0))})  # fmt: skip
    # the deepest transit defines ``depth``: renormalise amplitudes so the max is 1
    amax = max(c["amp"] for c in comps)
    for c in comps:
        c["amp"] = c["amp"] / amax
    return _finish("dusty_tail", depth, dur, comps, -1, t, {"period_d": P})
