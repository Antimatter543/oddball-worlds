"""Eclipsing-binary tests from the light curve alone (docs/VETTING.md §3, E1-E4, E6, E7).

The tests ask whether a light curve's dips behave like eclipses: strictly periodic, constant in
depth, width and timing, symmetric, physically possible in duration for the star, with no other
dips left unexplained. Dippers, Boyajian-like stars, exocomets and disintegrating planets fail
at least one of these by construction (aperiodic or quasi-periodic timing, varying depth,
asymmetric profiles, extra dips), so they score low. This is the property the explain-away needs:
``kc_v1``'s typicality called 15 of 18 known dippers "typical EB or rotator" (ML_DESIGN §11).

Per light curve (PDCSAP only; 30-min bins on the global grid):

1. **Residual and events** (vectorised over a batch). ``r = x - baseline`` with a running median
   of 4-h blocks over a 1.5-d window (eclipses up to ~16 h total survive). ``sigma_r`` is the
   robust (MAD) scatter of ``r``, so red noise counts as noise. An *event* is a run of bins whose
   centred 2-h box SNR exceeds ``EV_SNR`` (gaps of up to 2 bins closed). Per event: peak depth on
   the 3-bin smoothed residual, FWHM (bins below half depth), half-depth asymmetry, centroid.
2. **Anchored period search** (E1). Anchored on each of the deepest ``N_ANCHOR`` events (box SNR
   >= ``ANCHOR_SNR``), every trial period ``P`` predicts epochs ``t_a + nP``. At each covered
   epoch the *window depth* is the mean residual over the whole eclipse window (1.5 FWHM + 1 bin
   each side), which is insensitive to where an unresolved eclipse falls inside a 30-min bin. An
   epoch is a hit when that depth is at least half the anchor's and at least 2.5 window-sigma
   (``sigma_win``: the robust scatter of window means across the light curve, red noise
   included). Score ``hits - 2 misses``; ties go to the longer period (the smaller claim). The
   grid is log-spaced so the farthest epoch drifts by at most ``max(0.5, FWHM/4)`` bins.
3. **At the best period:** re-centre each hit on its local centroid, fit a linear ephemeris, and
   measure
   * ``hit_frac`` (covered epochs that hold an eclipse; a dipper's "period" leaves misses);
   * ``depth_fex``: excess fractional scatter of the hit depths beyond ``sigma_win``, also within
     odd and even epochs separately (E2; an EB folded at half its period alternates depths);
   * ``time_rms``: timing residuals of the hits (>= 3 hits), against ``max(0.5 bin, FWHM/10)``;
   * ``width_fex``: scatter of per-hit FWHM (FWHM >= 4 bins only; coarser widths are quantised);
   * ``asym`` from the stacked (sub-bin) profile of the hits (E4);
   * a **secondary eclipse** (E3): the most significant window at any other phase, kept when
     SNR >= 6 and >= 75% of its covered epochs hold it;
   * ``n_extra``: events at least 0.3x the anchor's depth explained by neither the primary nor
     the secondary ephemeris (orbit-edge events excepted);
   * ``dur_ratio`` (E6): FWHM over the longest total duration an eclipse between two stars of
     the TIC radius can have, ``T14max = P/pi asin(min(1, 2R/a))`` with Kepler's law for a total
     mass of 2 M (M from R; 1.5 Msun above 3 Rsun).
4. ``eb_periodic`` multiplies soft factors in [0, 1] (``FACTORS``), so one strong violation is
   enough to rule out an EB. A lone event gets ``eb_single_raw = s_snr * c_shape * c_extra``
   (symmetric, isolated, significant): weak evidence on its own, weighted by ``w_single`` when
   combined. ``combine_eb`` also takes ``eb_xsec`` from other sectors (E7, ``cross_sector``).

All thresholds are physical or set from synthetic light curves and val injections; none was
looked up on the known positives, the test split or the eyeballed candidates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from oddball.knownclass.features import close_gaps
from oddball.score.classical import (
    _fill_nearest,
    centred_box_snr,
    nanquantile_last,
    running_baseline,
    smooth3,
)
from oddball.tess.arrays import Grid, iter_batches

EB_BLOCK = 8  # 4-h blocks for the baseline
EB_WIN = 9  # blocks: 1.5-d running-median window
EV_SNR = 5.0  # 2-h box SNR for an event bin
ANCHOR_SNR = 7.0  # an anchor event must reach this box SNR
N_ANCHOR = 2
MAX_EVENTS = 40
P_MIN_D = 0.15
HIT_FRAC_DEPTH = 0.5
HIT_SIG = 2.5
MIN_COVER = 0.75
EXTRA_DEPTH = 0.3
EXTRA_SNR = 6.0
EDGE_D = 0.5  # events this close to a >= 1-d gap or the data ends are ignored by n_extra
SEC_SNR = 6.0
SEC_HITFRAC = 0.75
FEX_TOL = 0.10  # tolerated excess fractional depth scatter (EB: ~2-5%; dippers: >= 25%)
WIDTH_TOL = 0.15
ASYM_TOL = 0.3
EXTRA_FACTOR = 0.6
RSUN_AU = 1.0 / 215.032
FACTORS = ("s_snr", "c_hit", "c_depth", "c_time", "c_width", "c_shape", "c_phys", "c_extra",
           "c_n")  # fmt: skip


# ---------------------------------------------------------------------------------------------
# vectorised part


def baseline(x: np.ndarray, block: int = EB_BLOCK, win: int = EB_WIN) -> np.ndarray:
    """Running median of block medians (as ``classical.running_baseline``, shorter window)."""
    n, L = x.shape
    nb = -(-L // block)
    xp = np.full((n, nb * block), np.nan)
    xp[:, :L] = x
    blocks = xp.reshape(n, nb, block)
    bmed = nanquantile_last(blocks, 0.5)[0]
    bmed[np.isfinite(blocks).sum(-1) < block // 3] = np.nan
    half = win // 2
    padded = np.pad(bmed, ((0, 0), (half, half)), constant_values=np.nan)
    w = np.lib.stride_tricks.sliding_window_view(padded, win, axis=1)
    base_b = nanquantile_last(w, 0.5)[0]
    base_b[np.isfinite(w).sum(-1) < 3] = np.nan
    base_b = _fill_nearest(base_b)
    pos = np.clip((np.arange(L) - (block - 1) / 2) / block, 0, nb - 1)
    il = np.floor(pos).astype(np.int64)
    ir = np.minimum(il + 1, nb - 1)
    f = pos - il
    return base_b[:, il] * (1 - f) + base_b[:, ir] * f


def residuals(flux: np.ndarray):
    """(r_long, r, y, sigma_r, snr2h, mask) for a batch of raw PDCSAP fluxes (n, L).

    ``r`` is the residual from the 1.5-d baseline used for detection and depths; ``r_long`` the
    residual from the classical 3.25-d baseline, used only to measure widths, so a multi-day
    dip that the short baseline cuts down to ~0.5 d is not mistaken for a short eclipse.
    """
    f = flux.astype(np.float64)
    med = nanquantile_last(f, 0.5)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        x = f / med[:, None] - 1.0
    x[~np.isfinite(x)] = np.nan
    r = x - baseline(x)
    q50 = nanquantile_last(r, 0.5)[0]
    sig = 1.4826 * nanquantile_last(np.abs(r - q50[:, None]), 0.5)[0]
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, np.nan)
    snr = centred_box_snr(r, np.nan_to_num(sig, nan=1.0), W=4)
    snr[~np.isfinite(sig)] = np.nan
    mask = close_gaps(np.nan_to_num(snr, nan=0.0) > EV_SNR, 2)
    return x - running_baseline(x), r, smooth3(r), sig, snr, mask


# ---------------------------------------------------------------------------------------------
# per light curve


def _events(r, y, snr, mask, t0g: float):
    """Events of one light curve as a dict of arrays, deepest first."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    L = len(r)
    out = {k: [] for k in ("imin", "depth", "fwhm", "asym", "cen", "snr", "lo", "hi")}
    for run in runs:
        yy = y[run]
        if not np.isfinite(yy).any():
            continue
        k = int(run[np.nanargmin(yy)])
        d = -y[k]
        s = float(np.nanmax(snr[run]))
        if not (d > 0) or s < EV_SNR:
            continue
        lo, hi = k, k
        while lo - 1 >= 0 and np.isfinite(y[lo - 1]) and y[lo - 1] < -0.5 * d:
            lo -= 1
        while hi + 1 < L and np.isfinite(y[hi + 1]) and y[hi + 1] < -0.5 * d:
            hi += 1
        ing, eg = k - lo + 0.5, hi - k + 0.5
        seg = np.arange(lo, hi + 1)
        w = np.clip(-np.nan_to_num(r[seg], nan=0.0), 0.0, None)
        cen = float((seg * w).sum() / w.sum()) if w.sum() > 0 else float(k)
        for key, v in (("imin", k), ("depth", d), ("fwhm", hi - lo + 1.0),
                       ("asym", (eg - ing) / (eg + ing)), ("cen", cen), ("snr", s),
                       ("lo", lo), ("hi", hi)):  # fmt: skip
            out[key].append(v)
    if not out["depth"]:
        return None
    ev = {k: np.asarray(v, np.float64) for k, v in out.items()}
    o = np.argsort(-ev["depth"], kind="stable")[:MAX_EVENTS]
    return {k: v[o] for k, v in ev.items()}


class _Win:
    """Window-mean depths of a residual series via cumulative sums."""

    def __init__(self, r: np.ndarray):
        v = np.isfinite(r)
        self.v = v
        self.L = len(r)
        self.cs = np.concatenate([[0.0], np.cumsum(np.where(v, r, 0.0))])
        self.cn = np.concatenate([[0], np.cumsum(v)])

    def depth(self, c: np.ndarray, hw: float):
        """(-mean r, covered fraction, centre-bin valid) in windows ``[c - hw, c + hw]`` (bins)."""
        c = np.asarray(c, np.float64)
        lo = np.rint(c - hw).astype(np.int64)
        hi = np.rint(c + hw).astype(np.int64)
        nwin = hi - lo + 1
        inside = (lo >= 0) & (hi < self.L)
        lo_c, hi_c = np.clip(lo, 0, self.L - 1), np.clip(hi, 0, self.L - 1)
        s = self.cs[hi_c + 1] - self.cs[lo_c]
        n = self.cn[hi_c + 1] - self.cn[lo_c]
        ci = np.clip(np.rint(c).astype(np.int64), 0, self.L - 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.where(n > 0, -s / np.maximum(n, 1), np.nan)
        cov = np.where(inside, n / nwin, 0.0)
        return d, cov, inside & self.v[ci]

    def sigma(self, hw: float, exclude: np.ndarray) -> float:
        """Robust scatter of window means over the light curve, event bins excluded."""
        w = int(np.rint(hw)) * 2 + 1
        if w >= self.L:
            return np.nan
        ok = self.v & ~exclude
        vals = np.where(ok, np.diff(self.cs), 0.0)
        cs = np.concatenate([[0.0], np.cumsum(vals)])
        cn = np.concatenate([[0], np.cumsum(ok)])
        s = cs[w:] - cs[:-w]
        n = cn[w:] - cn[:-w]
        good = n >= 0.75 * w
        if good.sum() < 20:
            return np.nan
        m = s[good] / n[good]
        med = np.median(m)
        return float(1.4826 * np.median(np.abs(m - med)))


def _period_search(win: _Win, ca: float, da: float, hw: float, sig_w: float, first: int,
                   last: int, tol: float) -> tuple[float, int, int, int] | None:  # fmt: skip
    """Best trial period (bins) anchored at ``ca``: (P, hits, misses, n_trials) or None."""
    span = float(last - first)
    pmin = max(P_MIN_D / win.dt, 2.0 * hw + 1.0)
    if span <= pmin:
        return None
    ratio = 1.0 + tol / span
    n_tr = int(np.ceil(np.log(span / pmin) / np.log(ratio))) + 1
    P = pmin * ratio ** np.arange(n_tr)
    P = P[P <= span]
    nlo = np.ceil((first - ca) / P).astype(np.int64)
    nhi = np.floor((last - ca) / P).astype(np.int64)
    cnt = nhi - nlo + 1
    tid = np.repeat(np.arange(len(P)), cnt)
    off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    n = nlo[tid] + off
    keep = n != 0
    tid, n = tid[keep], n[keep]
    d, cov, cen_ok = win.depth(ca + n * P[tid], hw)
    covered = (cov >= MIN_COVER) & cen_ok
    hit = covered & (np.nan_to_num(d, nan=-np.inf) >= max(HIT_FRAC_DEPTH * da, HIT_SIG * sig_w))
    miss = covered & ~hit
    h = np.bincount(tid, weights=hit, minlength=len(P))
    m = np.bincount(tid, weights=miss, minlength=len(P))
    score = h - 2.0 * m
    best = np.flatnonzero(score == score.max())[-1]  # ties: the longest period
    if h[best] < 1 or score[best] <= 0:
        return None
    return float(P[best]), int(h[best]), int(m[best]), len(P)


def _half_width(prof: np.ndarray, off: np.ndarray):
    """FWHM and asymmetry of a dip profile sampled at offsets ``off`` (bins).

    Asymmetry compares the edge times: ingress ``t_in`` from the 20% to the 80% depth level on
    the way in, egress ``t_eg`` from 80% back to 20%, both found by walking outward from the
    deepest point (so a flat bottom does not matter). ``A = (t_eg - t_in) / (t_eg + t_in + 2)``;
    the 2-bin regulariser keeps 30-min quantisation of sharp eclipse edges from looking
    asymmetric. A box, V or Gaussian gives ~0; an exocomet (sharp ingress, exponential egress)
    ~0.8.
    """
    ok = np.isfinite(prof)
    if ok.sum() < 3:
        return np.nan, np.nan
    p, o = prof[ok], off[ok]
    k = int(np.argmin(p))
    d = -p[k]
    if not (d > 0):
        return np.nan, np.nan

    def cross(level, step):
        lv = -level * d
        j = k
        while 0 <= j + step < len(p) and p[j + step] < lv:
            j += step
        if not (0 <= j + step < len(p)):
            return o[j]
        a, b = p[j], p[j + step]  # j below the level, j + step above it
        f = (lv - a) / (b - a) if b != a else 0.5
        return o[j] + f * (o[j + step] - o[j])

    l50, r50 = cross(0.5, -1), cross(0.5, +1)
    l80, r80 = cross(0.8, -1), cross(0.8, +1)
    l20, r20 = cross(0.2, -1), cross(0.2, +1)
    fwhm = r50 - l50
    if fwhm <= 0:
        return np.nan, np.nan
    t_in, t_eg = max(l80 - l20, 0.0), max(r20 - r80, 0.0)
    return float(fwhm), float((t_eg - t_in) / (t_eg + t_in + 2.0))


def _segments(r: np.ndarray, centres: np.ndarray, reach: float):
    """(values, offsets) of ``r`` around each centre: rows of bins within +-``reach`` bins."""
    c = np.asarray(centres, np.float64)
    R = int(np.ceil(reach)) + 1
    J = np.rint(c).astype(np.int64)[:, None] + np.arange(-R, R + 1)[None, :]
    inside = (J >= 0) & (J < len(r))
    vals = np.where(inside, r[np.clip(J, 0, len(r) - 1)], np.nan)
    off = J - c[:, None]
    vals[np.abs(off) > reach] = np.nan
    return vals, off


def _stack(r: np.ndarray, centres: np.ndarray, reach: float, step: float = 0.5):
    """Mean residual profile of several events, aligned on ``centres`` (sub-bin, ``step`` bins)."""
    vals, off = _segments(r, centres, reach)
    edges = np.arange(-reach, reach + step, step)
    ok = np.isfinite(vals)
    b = np.searchsorted(edges, off[ok], side="right") - 1
    good = (b >= 0) & (b < len(edges) - 1)
    sums = np.bincount(b[good], weights=vals[ok][good], minlength=len(edges) - 1)
    cnts = np.bincount(b[good], minlength=len(edges) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    return prof, 0.5 * (edges[:-1] + edges[1:])


def max_duration_d(period_d: float, radius: float) -> float:
    """Longest total eclipse duration (d) for two stars of the TIC radius on a circular orbit."""
    R = radius if np.isfinite(radius) and radius > 0 else 1.5
    m = float(np.clip(R, 0.2, 2.0)) if R <= 3.0 else 1.5
    a_au = (2.0 * m * (period_d / 365.25) ** 2) ** (1.0 / 3.0)
    k = min(1.0, 2.0 * R * RSUN_AU / a_au)
    return period_d / np.pi * np.arcsin(k)


def _analyse_anchor(r, y, win, ev, ia, sig_r, first, last, edge_ok, radius, dt, rl=None):
    """EB evidence for one anchor event; returns a dict of diagnostics.

    ``rl``: the long-baseline residual for width and shape (defaults to ``r``).
    """
    rl = r if rl is None else rl
    ca, fw = float(ev["cen"][ia]), float(max(ev["fwhm"][ia], 1.0))
    hw = 0.6 * fw + 1.0
    da_arr, _, _ = win.depth(np.array([ca]), hw)
    da = float(da_arr[0])
    excl = np.zeros(win.L, bool)
    for lo, hi in zip(ev["lo"], ev["hi"], strict=True):
        excl[max(0, int(lo) - 2) : int(hi) + 3] = True
    sig_w = win.sigma(hw, excl)
    if not np.isfinite(sig_w) or sig_w <= 0:
        sig_w = sig_r / np.sqrt(2 * np.rint(hw) + 1)
    out = {"anchor_t": ca, "anchor_depth": float(ev["depth"][ia]), "anchor_fwhm": fw,
           "anchor_snr": float(ev["snr"][ia]), "anchor_asym": float(ev["asym"][ia]),
           "win_depth": da, "sig_win": sig_w}  # fmt: skip
    tol = max(0.5, 0.25 * fw)
    ps = _period_search(win, ca, da, hw, sig_w, first, last, tol) if da > 0 else None
    out["n_trials"] = ps[3] if ps else 0
    D = float(ev["depth"][ia])

    def extra(prim_c, sec_c, sec_w=None):
        """Events >= EXTRA_DEPTH x the anchor explained by neither ephemeris. ``sec_w``: the
        candidate secondary's epochs when it is not confirmed (one covered epoch); an event
        there is explained only if its FWHM matches the primary's within 25%."""
        n_ex = 0
        for k in range(len(ev["depth"])):
            if k == ia or ev["depth"][k] < EXTRA_DEPTH * D or ev["snr"][k] < EXTRA_SNR:
                continue
            c = ev["cen"][k]
            if not edge_ok[int(np.clip(np.rint(c), 0, win.L - 1))]:
                continue
            near = [np.abs(np.asarray(cc) - c).min() <= hw + 1.0 for cc in (prim_c, sec_c)
                    if len(cc)]  # fmt: skip
            if sec_w is not None and len(sec_w) and np.abs(sec_w - c).min() <= hw + 1.0:
                near.append(abs(ev["fwhm"][k] / fw - 1.0) <= 0.25)
            if not any(near):
                n_ex += 1
        return n_ex

    if ps is None:  # a lone event
        prof, off = _stack(rl, [ca], max(2.0 * hw + 2.0, 3.0 * fw + 4.0), step=1.0)
        fwhm_s, asym = _half_width(prof, off)
        out.update(is_periodic=False, P=np.nan, n_cov=1, n_hit=1, n_extra=extra([ca], []),
                   asym=asym if np.isfinite(asym) else float(ev["asym"][ia]),
                   fwhm=fwhm_s if np.isfinite(fwhm_s) else fw, snr_ep=da / sig_w)  # fmt: skip
        return out

    P = ps[0]
    nlo, nhi = int(np.ceil((first - ca) / P)), int(np.floor((last - ca) / P))
    n = np.arange(nlo, nhi + 1)
    d, cov, cen_ok = win.depth(ca + n * P, hw)
    covered = (cov >= MIN_COVER) & cen_ok
    hit = covered & (np.nan_to_num(d, nan=-np.inf) >= max(HIT_FRAC_DEPTH * da, HIT_SIG * sig_w))
    hit[n == 0] = True
    covered[n == 0] = True
    # re-centre hits on their local centroid; linear ephemeris
    cen = np.full(len(n), np.nan)
    for j in np.flatnonzero(hit):
        c0 = ca + n[j] * P
        lo, hi = int(max(0, np.floor(c0 - hw))), int(min(win.L - 1, np.ceil(c0 + hw)))
        seg = np.arange(lo, hi + 1)
        yy = y[seg]
        if not np.isfinite(yy).any():
            continue
        # deficit centroid over the eclipse window, re-centred once (a flat bottom has no
        # well-defined minimum, so the window is centred on the prediction, then the centroid)
        cj = c0
        for _ in range(2):
            s2 = np.arange(max(0, int(np.floor(cj - hw))), min(win.L, int(np.ceil(cj + hw)) + 1))
            w = np.clip(-np.nan_to_num(r[s2], nan=0.0), 0.0, None)
            if w.sum() <= 0:
                break
            cj = float((s2 * w).sum() / w.sum())
        cen[j] = cj
    hj = np.flatnonzero(hit & np.isfinite(cen))
    t0, Pr = ca, P
    time_rms = np.nan
    if len(hj) >= 2:
        A = np.column_stack([np.ones(len(hj)), n[hj]])
        coef, *_ = np.linalg.lstsq(A, cen[hj], rcond=None)
        t0, Pr = float(coef[0]), float(coef[1])
        if len(hj) >= 3:
            res = cen[hj] - A @ coef
            time_rms = float(np.sqrt((res**2).sum() / (len(hj) - 2)))
    pred = t0 + n * Pr
    cc = np.where(np.isfinite(cen) & hit, cen, pred)
    d2, _cov2, _ok2 = win.depth(cc, hw)
    dh = d2[hit]
    nh = int(hit.sum())
    dbar = float(np.mean(dh))
    fex = np.nan
    fex_oe = np.nan
    oe_sig = np.nan
    if nh >= 2:
        fex = float(np.sqrt(max(0.0, np.var(dh, ddof=1) - sig_w**2)) / max(dbar, 1e-12))
        par = (n[hit] % 2).astype(bool)
        if par.sum() >= 2 and (~par).sum() >= 2:
            do, de = dh[par], dh[~par]
            pooled = (np.var(do, ddof=1) * (len(do) - 1) + np.var(de, ddof=1) * (len(de) - 1)) / (
                nh - 2
            )
            fex_oe = float(np.sqrt(max(0.0, pooled - sig_w**2)) / max(dbar, 1e-12))
            oe_sig = float(
                abs(do.mean() - de.mean()) / (sig_w * np.sqrt(1 / len(do) + 1 / len(de)))
            )
    # per-hit widths (resolved eclipses only)
    width_fex = np.nan
    if fw >= 4 and nh >= 2:
        vals, off = _segments(rl, cc[hit], max(2.0 * hw + 2.0, 3.0 * fw + 4.0))
        ws = [
            w
            for w in (_half_width(v, o)[0] for v, o in zip(vals, off, strict=True))
            if np.isfinite(w)
        ]
        if len(ws) >= 2:
            ws = np.asarray(ws)
            width_fex = float(np.sqrt(max(0.0, np.var(ws, ddof=1) - 1.0 / 6.0)) / ws.mean())
    prof, off = _stack(
        rl, cc[hit], max(2.0 * hw + 2.0, 3.0 * fw + 4.0), step=0.5 if nh >= 3 else 1.0
    )
    fwhm_s, asym = _half_width(prof, off)
    # secondary eclipse at any other phase
    sec = {"sec_phase": np.nan, "sec_depth": np.nan, "sec_snr": np.nan, "sec_hitfrac": np.nan,
           "sec_ok": False}  # fmt: skip
    sec_c = np.array([])
    sec_w = None
    edge = (2.0 * hw + 1.0) / Pr
    ph = np.arange(edge, 1.0 - edge + 1e-9, max(0.5, 0.25 * fw) / Pr) if edge < 0.5 else []
    if len(ph):
        ns = np.arange(nlo - 1, nhi + 2)
        cs_ = t0 + (ns[None, :] + ph[:, None]) * Pr
        ds, cv, ok = (a.reshape(cs_.shape) for a in win.depth(cs_.ravel(), hw))
        cvr = (cv >= MIN_COVER) & ok
        ncv = cvr.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_d = np.where(ncv > 0, np.where(cvr, ds, 0.0).sum(1) / np.maximum(ncv, 1), np.nan)
            snr_s = mean_d / (sig_w / np.sqrt(np.maximum(ncv, 1)))
        if np.isfinite(snr_s).any():
            b = int(np.nanargmax(snr_s))
            hits2 = cvr[b] & (np.nan_to_num(ds[b], nan=-np.inf) >= 0.5 * mean_d[b])
            hf2 = hits2.sum() / max(ncv[b], 1)
            sec = {
                "sec_phase": float(ph[b]),
                "sec_depth": float(mean_d[b]),
                "sec_snr": float(snr_s[b]),
                "sec_hitfrac": float(hf2),
                "sec_ok": bool(snr_s[b] >= SEC_SNR and ncv[b] >= 2 and hf2 >= SEC_HITFRAC),
            }
            if sec["sec_ok"]:
                sec_c = t0 + (ns + ph[b]) * Pr
            elif snr_s[b] >= SEC_SNR and hf2 >= SEC_HITFRAC:
                sec_w = t0 + (ns + ph[b]) * Pr
    fwhm_d = (fwhm_s if np.isfinite(fwhm_s) else fw) * dt
    out.update(is_periodic=nh >= 2, P=Pr * dt, t0=t0, n_cov=int(covered.sum()), n_hit=nh,
               hit_frac=nh / max(int(covered.sum()), 1), depth_fex=fex, depth_fex_oe=fex_oe,
               oe_sig=oe_sig, time_rms=time_rms, time_tol=max(0.5, 0.1 * fw),
               width_fex=width_fex, asym=asym, fwhm=fwhm_s if np.isfinite(fwhm_s) else fw,
               snr_ep=dbar / sig_w, mean_depth=dbar,
               n_extra=extra(pred, sec_c, sec_w),
               dur_ratio=fwhm_d / max(max_duration_d(Pr * dt, radius), 1e-9), **sec)  # fmt: skip
    return out


def informative(snr_ep: float, n_hit: int) -> float:
    """How well the depth-constancy test can see a dipper-like depth change, in [0, 1].

    ``f_min = sqrt(2 sqrt(2 / (N - 1))) / snr_ep`` is the smallest fractional depth scatter whose
    excess variance stands out at ~2 sigma among ``N`` eclipses of per-eclipse SNR ``snr_ep``.
    1 when f_min <= 0.2, 0 when f_min >= 0.35. A lone event (N = 1) uses its SNR directly
    (1 at SNR >= 8, 0 at SNR <= 4).
    """
    snr = float(np.nan_to_num(snr_ep, nan=0.0))
    if n_hit < 2:
        return float(np.clip((snr - 4.0) / 4.0, 0.0, 1.0))
    if snr <= 0:
        return 0.0
    f_min = np.sqrt(2.0 * np.sqrt(2.0 / (n_hit - 1))) / snr
    return float(np.clip((0.35 - f_min) / 0.15, 0.0, 1.0))


def factors(o: dict) -> dict:
    """Soft factors in [0, 1] for one anchor's diagnostics (see module docstring)."""

    def g(x, tol):
        return float(np.exp(-0.5 * (x / tol) ** 2)) if np.isfinite(x) else 1.0

    f = {"s_snr": informative(o.get("snr_ep", 0.0), o.get("n_hit", 1) if o.get("is_periodic")
                              else 1),
         "c_shape": g(o.get("asym", np.nan), ASYM_TOL),
         "c_extra": EXTRA_FACTOR ** int(o.get("n_extra", 0))}  # fmt: skip
    if not o.get("is_periodic"):
        f.update(c_hit=0.0, c_depth=1.0, c_time=1.0, c_width=1.0, c_phys=1.0, c_n=0.0)
        return f
    fex = o["depth_fex"]
    if np.isfinite(o.get("depth_fex_oe", np.nan)):
        fex = min(fex, o["depth_fex_oe"])
    rho = o.get("dur_ratio", np.nan)
    c_phys = 1.0 if not np.isfinite(rho) or rho <= 1.5 else float(np.clip((3.0 - rho) / 1.5, 0, 1))
    wf = o.get("width_fex", np.nan)
    f.update(c_hit=float(o["hit_frac"]) ** 2, c_depth=g(fex, FEX_TOL),
             c_time=g(o["time_rms"], o["time_tol"]) if o["n_hit"] >= 3 else 1.0,
             c_width=g(max(0.0, wf - 0.1), WIDTH_TOL) if np.isfinite(wf) else 1.0,
             c_phys=c_phys,
             c_n=1.0 if o["n_hit"] >= 3 else (0.9 if o.get("sec_ok") else 0.75))  # fmt: skip
    return f


def analyse(r, y, snr, mask, sig_r, dt: float, radius: float = np.nan, rl=None) -> dict:
    """EB diagnostics of one light curve (residual ``r``, smoothed ``y``, event ``mask``;
    ``rl`` the long-baseline residual for widths, see ``residuals``)."""
    base = {"eb_periodic": 0.0, "eb_single_raw": 0.0, "n_events": 0}
    if not np.isfinite(sig_r):
        return base
    ev = _events(r, y, snr, mask, 0.0)
    if ev is None:
        return base
    base["n_events"] = len(ev["depth"])
    valid = np.isfinite(r)
    vi = np.flatnonzero(valid)
    first, last = int(vi[0]), int(vi[-1])
    # orbit-edge zone: within EDGE_D of the data ends or of a gap >= 1 d
    edge_ok = np.ones(len(r), bool)
    gaps = np.flatnonzero(np.diff(vi) * dt >= 1.0)
    e_bins = int(np.ceil(EDGE_D / dt))
    for a, b in [(vi[0], vi[0])] + [(vi[g], vi[g + 1]) for g in gaps] + [(vi[-1], vi[-1])]:
        edge_ok[max(0, a - e_bins) : a + 1] = False
        edge_ok[b : b + e_bins + 1] = False
    win = _Win(r)
    win.dt = dt
    best, best_eb, best_f = None, -1.0, None
    for ia in range(min(N_ANCHOR, len(ev["depth"]))):
        if ev["snr"][ia] < ANCHOR_SNR or (
            ia > 0 and ev["depth"][ia] < EXTRA_DEPTH * ev["depth"][0]
        ):
            continue
        o = _analyse_anchor(r, y, win, ev, ia, sig_r, first, last, edge_ok, radius, dt, rl)
        f = factors(o)
        eb_p = float(np.prod([f[k] for k in FACTORS]))
        eb_s = f["s_snr"] * f["c_shape"] * f["c_extra"]
        key = max(eb_p, 0.5 * eb_s)
        if key > best_eb:
            best, best_eb, best_f = o, key, f
            best["eb_periodic"], best["eb_single_raw"] = eb_p, eb_s
    if best is None:
        return base
    best.update({k: v for k, v in best_f.items()})
    best["n_events"] = len(ev["depth"])
    return best


OUT_COLS = ("eb_periodic", "eb_single_raw", "n_events", "is_periodic", "P", "t0", "n_cov", "n_hit",
            "hit_frac", "depth_fex", "depth_fex_oe", "oe_sig", "time_rms", "time_tol",
            "width_fex", "asym", "fwhm", "snr_ep", "mean_depth", "n_extra", "dur_ratio",
            "sec_phase", "sec_depth", "sec_snr", "sec_hitfrac", "sec_ok", "anchor_t",
            "anchor_depth", "anchor_fwhm", "anchor_snr", "anchor_asym", "win_depth", "sig_win",
            *FACTORS)  # fmt: skip


def batch_eb(b: dict, mm: pd.DataFrame, grid: Grid) -> pd.DataFrame:
    """EB diagnostics for one ``iter_batches`` batch (``mm``: aligned metadata rows)."""
    rl, r, y, sig, snr, mask = residuals(b["flux"])
    radius = mm.radius.to_numpy(np.float64)
    rows = []
    for i in range(r.shape[0]):
        if not mask[i].any() or not (np.nanmax(np.nan_to_num(snr[i], nan=0.0)) >= ANCHOR_SNR):
            rows.append({"eb_periodic": 0.0, "eb_single_raw": 0.0,
                         "n_events": int(mask[i].any())})  # fmt: skip
            continue
        rows.append(analyse(r[i], y[i], snr[i], mask[i], sig[i], grid.bin_days, radius[i], rl[i]))
    df = pd.DataFrame(rows).reindex(columns=list(OUT_COLS))
    df["is_periodic"] = df.is_periodic.eq(True)
    df["sec_ok"] = df.sec_ok.eq(True)
    for c in ("anchor_t", "t0"):  # bin index -> BTJD
        df[c] = (grid.g0 + df[c].astype(float) + 0.5) * grid.bin_days
    for c in ("fwhm", "anchor_fwhm", "time_rms", "time_tol"):  # bins -> days
        df[c] = df[c].astype(float) * grid.bin_days
    df = df.rename(columns={c: f"eb_{c}" for c in OUT_COLS if not c.startswith("eb_")})
    key = pd.DataFrame(
        {"tic": b["tic"], "inj_id": b["inj_id"] if b["inj_id"] is not None else None}
    )
    return pd.concat([key, df.reset_index(drop=True)], axis=1)


def eb_dir(d, grid: Grid, meta: pd.DataFrame, sector: int, batch_rows: int = 1000,
           log=None, files=None) -> pd.DataFrame:  # fmt: skip
    """EB diagnostics for every light curve under shard directory ``d`` (or ``files`` of it)."""
    key = "inj_id" if "inj_id" in meta.columns else "tic"
    m = meta.set_index(key)
    parts, n = [], 0
    for b in iter_batches(d, grid, batch_rows=batch_rows, columns=("flux",), files=files):
        ids = b["inj_id"] if key == "inj_id" else b["tic"]
        parts.append(batch_eb(b, m.loc[ids], grid))
        n += len(ids)
        if log:
            log(f"    {b['shard_file']}: {n} light curves")
    out = pd.concat(parts, ignore_index=True)
    out.insert(1, "sector", np.int16(sector))
    return out


# ---------------------------------------------------------------------------------------------
# cross-sector (E7) and the final score

XS_DEPTH = 1.5  # depth ratio tolerance between sectors (PDCSAP is crowding-corrected per sector)
XS_WIDTH = 1.5
XS_PERIOD = 0.02
# a lone event matched by a lone event elsewhere counts fully only when depth and width agree
# this well (89% of catalogued-EB lone-event pairs in the S32 train split, S33/S38 partners;
# 25% for other catalogued classes); otherwise (within XS_DEPTH/XS_WIDTH) at half weight
XS_DEPTH_TIGHT = 1.15
XS_WIDTH_TIGHT = 1.25


def cross_sector(this: pd.DataFrame, others: pd.DataFrame) -> np.ndarray:
    """``eb_xsec`` per row of ``this`` from the same star's rows in ``others`` (other sectors).

    Support needs a matching eclipse in another sector: same depth and width within a factor
    1.5, and either (a) that sector shows a periodic EB (``eb_periodic``) whose period agrees
    with this sector's (within 2%, also at 1/2 and 2x) when this sector has one, or (b) for a
    lone event here, that sector's EB evidence, or its own lone event if it is isolated and
    symmetric (``eb_single_raw``; full weight when depth and width agree within 1.15 and 1.25,
    half weight otherwise). Absence in other sectors is never evidence against an EB: a
    long-period EB can miss them, and the injected EB nulls have no other sectors at all.
    ``this`` must carry ``host_tic`` (= tic for real stars).
    """
    out = np.zeros(len(this))
    if others is None or others.empty:
        return out
    o = others[others.eb_anchor_depth.notna()]
    by = {t: g for t, g in o.groupby("tic")}
    for i, row in enumerate(this.itertuples(index=False)):
        g = by.get(int(row.host_tic))
        if g is None or not np.isfinite(row.eb_anchor_depth):
            continue
        dr = g.eb_anchor_depth.to_numpy() / row.eb_anchor_depth
        wr = g.eb_fwhm.to_numpy() / row.eb_fwhm
        ok = (dr <= XS_DEPTH) & (dr >= 1 / XS_DEPTH) & (wr <= XS_WIDTH) & (wr >= 1 / XS_WIDTH)
        if not ok.any():
            continue
        gp = g.eb_periodic.to_numpy()
        if bool(row.eb_periodic > 0) and np.isfinite(row.eb_P):
            rat = g.eb_P.to_numpy() / row.eb_P
            pm = np.zeros(len(g), bool)
            for k in (0.5, 1.0, 2.0):
                pm |= np.abs(rat - k) <= XS_PERIOD * k
            sup = np.where(ok & pm, gp, 0.0)
        else:
            tight = ((np.abs(np.log(dr)) <= np.log(XS_DEPTH_TIGHT))
                     & (np.abs(np.log(wr)) <= np.log(XS_WIDTH_TIGHT)))  # fmt: skip
            single = g.eb_single_raw.to_numpy() * np.where(tight, 1.0, 0.5)
            sup = np.where(ok, np.maximum(gp, single), 0.0)
        out[i] = float(np.nanmax(sup)) if len(sup) else 0.0
    return out


def combine_eb(df: pd.DataFrame, w_single: float = 0.0, w_xsec: float = 1.0) -> np.ndarray:
    """``eb_like = max(eb_periodic, w_single * eb_single_raw, w_xsec * eb_xsec * shape * extra)``.

    The cross-sector term is gated by this sector's own event being symmetric and isolated, so
    an asymmetric or crowded (dipper-like) event here is not explained by an EB elsewhere.
    """
    p = np.nan_to_num(df.eb_periodic.to_numpy(np.float64))
    s = np.nan_to_num(df.eb_single_raw.to_numpy(np.float64))
    x = np.nan_to_num(df["eb_xsec"].to_numpy(np.float64)) if "eb_xsec" in df else 0.0
    gate = (np.nan_to_num(df.eb_c_shape.to_numpy(np.float64))
            * np.nan_to_num(df.eb_c_extra.to_numpy(np.float64)))  # fmt: skip
    return np.clip(np.maximum.reduce([p, w_single * s, w_xsec * x * gate]), 0.0, 1.0)
