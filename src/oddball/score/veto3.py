"""Per-event veto ``cmv3``: ``cmv2`` plus background, PDC/SAP, gap-edge, QUALITY, dump-edge,
asteroid-track and same-column labels (docs/VETTING.md §2 F1, F2, F4, F5, F8, F12; §4 build
order items 1 and 3; docs/ML_DESIGN.md §13).

``cmv2`` (``oddball.score.veto``) finds events per (channel, sign, box width) series and labels
each one systematic when it is common-mode on its CCD or 512-px cell, sits in SPOC's
scattered-light windows, or peaks just after a momentum dump (stage 1). Stage 2 then relabels
events that overlap a systematic one in the same channel. ``cmv3`` adds new **stage-1** labels
before stage 2, so a relabelled event still drags its wider-box views and detrending rebounds
with it, and a star keeps any real event the new labels do not touch.

Per-event labels (stage 1). The *event window* is the box at the event's peak, ``W`` bins wide.
All fluxes are in units of the star's median SAP flux (``bkg`` is stored that way at ingest:
SAP_BKG, the background summed over the aperture and already subtracted from SAP_FLUX, divided
by the SAP median), detrended by the classical running median (3.25 d).

* ``bkg`` **(F1, background explains the event).** In the window, SAP moves significantly in
  the event's direction (box SNR >= ``SAP_MIN_SNR``), the background moves the other way by at
  least ``BKG_FRAC`` of the SAP change, and the raw aperture flux ``sap_flux + bkg`` does not
  move significantly (box SNR < ``RAW_SNR``). The star did not fade; the background estimate
  rose (an asteroid or glint in the background pixels).
* ``pdc`` **(F12, PDCSAP-only event).** A PDCSAP event whose crowding-corrected SAP depth
  ``delta_SAP / CROWDSAP`` is below ``PDC_RATIO`` of the PDCSAP depth, at >= ``PDC_SIG`` sigma.
  The reverse case (SAP deeper, e.g. PDS 110) is never labelled.
* ``gap`` **(F4b, gap or orbit edge with a moving background).** The window lies within
  ``GAP_DUR`` box widths of an orbit start/end or a >= 0.3-d gap of the star's own light curve,
  and the mean ``|d bkg / dt|`` over the window +- W is at or above the CCD's ``SLOPE_Q``
  quantile of per-bin ``|d bkg / dt|``.
* ``qbits`` **(F4a).** At least half of the event's SNR^2 lies in bins carrying QUALITY bits
  16, 128, 2048 or 32768, on the star itself or on >= 20% of its CCD's light curves (widened by
  one bin). In S32 only bits 128 and 32768 occur (orbit starts/ends, dumps).
* ``dumpedge`` **(F5).** The event run starts within [-1, +2] bins of a momentum dump or ends
  within [-2, +1] bins of one: a flux step at the dump, which the running median turns into a
  long dip or bump that begins or ends exactly there (``cmv2`` only checks the peak).
* ``track`` **(F2, asteroid track).** Short background bumps (``bkg`` box SNR >=
  ``POINT_SNR``, 2-h box) and SAP bumps (2-h, not already systematic) are points
  ``(col, row, t)`` on each CCD. Short means at most ``POINT_MAXLEN`` bins at or above half the
  bump's peak SNR (a bright crossing's run above SNR 3 is much longer than its core). Two points of different stars, 1-``TMAX`` bins apart,
  define a line at a speed of ``VMIN``-``VMAX`` (5-300 arcsec/h, 0.12-7.1 px per 30-min bin);
  it is a track when at least one more star's point lies within ``TOL`` px (+ one bin of
  motion) of the line. Bins where more than ``CROWD_FRAC`` of the CCD's stars have a point are
  dropped (CCD-wide events are not movers). Every star on the CCD whose position the track
  passes within ``TRACK_R`` px (the 11 x 11 stamp's background pixels) gets a time window; its
  events overlapping that window are labelled.
* ``flank`` **(F2, asteroid crossing the aperture, one star).** A bump event with a
  ``bkg``-labelled dip of the same star whose peak lies 1-``FLANK`` bins away and whose run does
  not contain the bump's peak: the dip-spike-dip of a body crossing the background pixels, then
  the aperture (the spike), then the background pixels again. Needs no other star, so it works
  where the track test lacks supporting points. A flare that leaks into the background pixels
  raises SAP, so it makes no ``bkg`` dip.
* ``column`` **(F8, single-column event).** For an event of one series, the number of other
  stars on the CCD within +-``COL_HALF`` columns with an event of the same series at the event's
  peak bin, against the rate in control columns 10-60 px away (scaled by star counts). Labelled
  when that count is >= ``COL_MIN`` and its Poisson tail probability is < ``COL_P``.

Thresholds are the VETTING.md priors; which labels to enable and the star-level flag
(``sys_snr >= t_sys and res_snr < t_res``) are chosen on val (``scripts/tune_veto3.py``).

Cross-star labels (``track``, ``column``) need every star's events, so a build is three steps:
:func:`scan_dir` (stream: events, points, per-CCD statistics), :func:`cross_labels` (tables
only) and :func:`classify_dir` (stream again: all labels, stage 2 per configuration, per-star
summaries). For an injection set the cross-star tests run on the set's events together with the
real sector's (the hosts' own real events left out), so an injected track or column is judged
among the real stars around it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import poisson

from oddball.common.strata import ccd_index
from oddball.score import veto as V
from oddball.score.classical import centred_box_snr, nanquantile_last, running_baseline
from oddball.tess.arrays import Grid, iter_batches

LABELS = ("bkg", "pdc", "gap", "qbits", "dumpedge", "track", "flank", "column")
CAUSES = V.CAUSES + LABELS  # cause codes: cmv2's 0-6, then 7.. for the new labels
LABEL_CODE = {lab: len(V.CAUSES) + i for i, lab in enumerate(LABELS)}

# F1 background
SAP_MIN_SNR = 3.0
BKG_FRAC = 0.5
RAW_SNR = 3.0
# F12 PDC-only
PDC_RATIO = 0.5
PDC_SIG = 3.0
# F4b gap edge
GAP_BINS = 15  # >= 0.3 d of missing data
GAP_DUR = 2  # within 2 box widths
SLOPE_Q = 0.9
SLOPE_EDGES = np.linspace(-9.0, 1.0, 501)  # log10 |d bkg / d bin| histogram edges
# F4a QUALITY bits
QBITS = 16 | 128 | 2048 | 32768
QB_FRAC = 0.2
# F2 asteroid track (pixel units; 1 bin = 30 min; 1 px = 21 arcsec)
POINT_SNR = 5.0
POINT_MAXLEN = 8
TMAX = 24  # bins between the two defining points (12 h)
VMIN = 5.0 / 21.0 / 2.0  # px per bin (5 arcsec/h)
VMAX = 300.0 / 21.0 / 2.0  # px per bin (300 arcsec/h)
TOL = 3.0  # px
SUPPORT_WIN = 24  # bins before/after the defining pair searched for supporting points
CROWD_FRAC = 0.01
TRACK_R = 6.0  # px
TRACK_MARGIN = 6  # bins added to each end of a track's time span
# F2 single-star crossing (dip-spike-dip)
FLANK = 24  # bins (12 h): a body at >= 4.4 arcsec/h reaches the background pixels (2.5 px)
# F8 same column
COL_HALF = 2
COL_CTRL = (10, 60)
COL_MIN = 3
COL_P = 1e-4
NCOL = 2200  # columns 0..2199 cover tess-point's 45..2092 science columns

# Configurations evaluated in one classify pass: cmv2 alone (a sanity check: identical to the
# cmv2 build), all labels, and leave-one-out / only-one ablations.
CONFIGS = {"cmv2": (), "all": LABELS,
           **{f"no_{lab}": tuple(x for x in LABELS if x != lab) for lab in LABELS},
           **{f"only_{lab}": (lab,) for lab in LABELS}}  # fmt: skip


# --------------------------------------------------------------------------------------------
# Per-batch signal


def _resid_series(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detrended residual and white-noise sigma (point-to-point MAD) of a series ``x``."""
    d = np.diff(x, axis=1)
    dmed = nanquantile_last(d, 0.5)[0]
    sigma = 1.4826 * nanquantile_last(np.abs(d - dmed[:, None]), 0.5)[0] / np.sqrt(2.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan)
    return x - running_baseline(x), sigma


@dataclass
class BatchSignal:
    snr: dict  # cmv2's 12 series
    r: dict  # residual per name: pdc, sap, bkg, raw (n, L) float64
    sig: dict  # white sigma per name (n,)
    bkg_bump: np.ndarray  # (n, L) 2-h box bump SNR of the background, red-noise rescaled
    slope: np.ndarray  # (n, L) |d bkg / d bin| of the 4-bin smoothed background (NaN no data)
    edges: np.ndarray  # (n, L+1) cumulative count of gap/orbit-edge bins


def batch_signal(b: dict) -> BatchSignal:
    snr, r, sig = {}, {}, {}
    for c, (key, name) in enumerate((("flux", "pdc"), ("sap_flux", "sap"))):
        res: dict = {}
        for (s, w), v in V.channel_snr(b[key], res).items():
            snr[(c, s, w)] = v
        r[name], sig[name] = res["r"], res["sigma"]
    sap = b["sap_flux"].astype(np.float64)
    bkg = b["bkg"].astype(np.float64)
    bkg[~np.isfinite(sap)] = np.nan
    r["bkg"], sig["bkg"] = _resid_series(bkg)
    r["raw"], sig["raw"] = _resid_series(sap + bkg)
    sb = np.where(np.isfinite(sig["bkg"]), sig["bkg"], 1.0)
    bb = -centred_box_snr(r["bkg"], sb, 4, min_count=3)
    bb[~np.isfinite(sig["bkg"])] = np.nan
    med = nanquantile_last(bb, 0.5)[0]
    rs = 1.4826 * nanquantile_last(np.abs(bb - med[:, None]), 0.5)[0]
    bb = bb / np.maximum(np.nan_to_num(rs, nan=1.0), 1.0)[:, None]
    # smoothed background slope
    v = np.isfinite(bkg)
    cs = np.concatenate([np.zeros((len(bkg), 1)), np.cumsum(np.where(v, bkg, 0.0), 1)], 1)
    cn = np.concatenate([np.zeros((len(bkg), 1)), np.cumsum(v, 1)], 1)
    k = 4
    sm = np.full(bkg.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        cnt = cn[:, k:] - cn[:, :-k]
        sm[:, k // 2 : k // 2 + cnt.shape[1]] = np.where(cnt >= 2, (cs[:, k:] - cs[:, :-k])
                                                         / np.maximum(cnt, 1), np.nan)  # fmt: skip
    slope = np.full(bkg.shape, np.nan)
    slope[:, 1:] = np.abs(np.diff(sm, axis=1))
    return BatchSignal(snr, r, sig, bb.astype(np.float32), slope, edge_counts(b["flux"]))


def edge_counts(flux: np.ndarray, gap_bins: int = GAP_BINS) -> np.ndarray:
    """(n, L+1) prefix counts of *edge* bins: valid bins next to a run of >= ``gap_bins``
    missing bins or next to the padding before/after the light curve (orbit and sector ends)."""
    n, L = flux.shape
    miss = ~np.isfinite(flux)
    idx, _, starts, rows = V._segments(miss)
    E = np.zeros((n, L), bool)
    if idx.size:
        ends = np.concatenate([starts[1:], [idx.size]]) - 1
        c0, c1 = idx[starts] % L, idx[ends] % L
        long_ = ((c1 - c0 + 1) >= gap_bins) | (c0 == 0) | (c1 == L - 1)
        for cc, off in ((c0, -1), (c1, 1)):
            j = cc + off
            ok = long_ & (j >= 0) & (j < L)
            E[rows[ok], j[ok]] = True
        E &= ~miss
    v = ~miss
    anyv = v.any(1)
    r = np.flatnonzero(anyv)
    E[r, np.argmax(v[r], 1)] = True  # first and last valid bin
    E[r, L - 1 - np.argmax(v[r, ::-1], 1)] = True
    return np.concatenate([np.zeros((n, 1), np.int32), np.cumsum(E, 1, dtype=np.int32)], 1)


def _prefix(a: np.ndarray):
    v = np.isfinite(a)
    cs = np.concatenate([np.zeros((len(a), 1)), np.cumsum(np.where(v, a, 0.0), 1)], 1)
    cn = np.concatenate([np.zeros((len(a), 1), np.int32), np.cumsum(v, 1, dtype=np.int32)], 1)
    return cs, cn


def _win_mean(pref, rows, lo, hi):
    cs, cn = pref
    n = cn[rows, hi] - cn[rows, lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs[rows, hi] - cs[rows, lo]) / np.maximum(n, 1), np.nan), n


# --------------------------------------------------------------------------------------------
# Per-event measurements and stage-1 labels


def event_measures(ev: pd.DataFrame, sig: BatchSignal, crowd: np.ndarray) -> pd.DataFrame:
    """Per-event window measurements used by the ``bkg``, ``pdc`` and ``gap`` labels."""
    L = sig.slope.shape[1]
    rows = ev.row.to_numpy(np.int64)
    W = np.asarray(V.WIDTHS)[ev.w.to_numpy(np.int64)]
    lo = np.clip(ev.peak_col.to_numpy(np.int64) - W // 2, 0, L)
    hi = np.clip(lo + W, 0, L)
    sgn = np.where(ev.sign.to_numpy() == 0, 1.0, -1.0)  # +1 dip events, -1 bump events
    out = {}
    for name in ("pdc", "sap", "bkg", "raw"):
        m, nn = _win_mean(_prefix(sig.r[name]), rows, lo, hi)
        out[f"m_{name}"] = m
        out[f"n_{name}"] = nn
    sq = {k: sig.sig[k][rows] for k in ("pdc", "sap", "raw")}
    dep_sap = -sgn * out["m_sap"]
    dep_pdc = -sgn * out["m_pdc"]
    dep_raw = -sgn * out["m_raw"]
    bkg_rise = sgn * out["m_bkg"]
    with np.errstate(invalid="ignore", divide="ignore"):
        snr_sap = dep_sap / (sq["sap"] / np.sqrt(np.maximum(out["n_sap"], 1)))
        snr_raw = dep_raw / (sq["raw"] / np.sqrt(np.maximum(out["n_raw"], 1)))
        c = np.where(np.isfinite(crowd) & (crowd > 0) & (crowd <= 1), crowd, 1.0)[rows]
        dep_sapc = dep_sap / c
        s_comb = np.sqrt(sq["pdc"] ** 2 + (sq["sap"] / c) ** 2) / np.sqrt(
            np.maximum(np.minimum(out["n_pdc"], out["n_sap"]), 1))
        pdc_sig = (dep_pdc - dep_sapc) / s_comb
    sl, _ = _win_mean(_prefix(sig.slope), rows, np.clip(lo - W, 0, L), np.clip(hi + W, 0, L))
    e = sig.edges
    near = (e[rows, np.clip(hi + GAP_DUR * W, 0, L)] - e[rows, np.clip(lo - GAP_DUR * W, 0, L)]) > 0
    return pd.DataFrame({"dep_sap": dep_sap, "dep_pdc": dep_pdc, "dep_sapc": dep_sapc,
                         "bkg_rise": bkg_rise, "snr_sap": snr_sap, "snr_raw": snr_raw,
                         "pdc_sig": pdc_sig, "slope": sl, "near_edge": near})  # fmt: skip


def measure_labels(ev: pd.DataFrame, me: pd.DataFrame, slope_q: np.ndarray, ci: np.ndarray
                   ) -> dict[str, np.ndarray]:  # fmt: skip
    """``bkg``, ``pdc`` and ``gap`` labels from :func:`event_measures`."""
    rows = ev.row.to_numpy(np.int64)
    f = lambda a: np.nan_to_num(a, nan=-np.inf)  # NaN never passes a ">=" test
    bkg = ((f(me.snr_sap.to_numpy()) >= SAP_MIN_SNR)
           & (f(me.bkg_rise.to_numpy()) >= BKG_FRAC * me.dep_sap.to_numpy())
           & (np.nan_to_num(me.snr_raw.to_numpy(), nan=np.inf) < RAW_SNR))  # fmt: skip
    pdc = ((ev.ch.to_numpy() == 0) & (f(me.dep_pdc.to_numpy()) > 0)
           & (np.nan_to_num(me.dep_sapc.to_numpy(), nan=np.inf)
              < PDC_RATIO * me.dep_pdc.to_numpy())
           & (f(me.pdc_sig.to_numpy()) >= PDC_SIG))  # fmt: skip
    gap = me.near_edge.to_numpy() & (f(me.slope.to_numpy()) >= slope_q[ci[rows]])
    return {"bkg": bkg, "pdc": pdc, "gap": gap}


def qbits_share(members: list, q: np.ndarray, qwin: np.ndarray | None, ci: np.ndarray):
    """Per series: share of each event's SNR^2 in bins with QUALITY bits ``QBITS`` (star's own
    flags, or the CCD window ``qwin`` (16, L))."""
    out = []
    for c, rr, colx, w2, starts, tot, _ in members:
        hit = (q[rr, colx] & QBITS) > 0
        if qwin is not None:
            hit |= qwin[ci[rr], colx]
        out.append(np.add.reduceat(w2 * hit, starts) / tot)
    return out


def half_max_width(snr: np.ndarray, ev: pd.DataFrame) -> np.ndarray:
    """Per event (row, start, end, peak) of the series ``snr`` (n, L): the number of bins of its
    run at or above half its peak SNR."""
    if len(ev) == 0:
        return np.zeros(0, np.int64)
    s = ev.start.to_numpy(np.int64)
    ln = ev.end.to_numpy(np.int64) - s + 1
    k = np.repeat(np.arange(len(ev)), ln)
    col = np.repeat(s, ln) + np.arange(ln.sum()) - np.repeat(np.cumsum(ln) - ln, ln)
    v = snr[ev.row.to_numpy(np.int64)[k], col]
    hit = np.nan_to_num(v, nan=-np.inf) >= 0.5 * ev.peak.to_numpy(np.float64)[k]
    return np.bincount(k, weights=hit, minlength=len(ev)).astype(np.int64)


def flank_labels(parts: list, labs: list, win: int = FLANK) -> list:
    """Per series: the ``flank`` label (module docstring) of each event, from every series'
    ``bkg`` labels of the same light curves."""
    dips = [p.loc[lab["bkg"] & (p.sign.to_numpy() == 0), ["row", "start", "end", "peak_col"]]
            for p, lab in zip(parts, labs, strict=True)]  # fmt: skip
    D = pd.concat(dips, ignore_index=True) if dips else pd.DataFrame()
    out = []
    for p in parts:
        hit = np.zeros(len(p), bool)
        bump = p.sign.to_numpy() == 1
        if len(D) and bump.any():
            k = p.loc[bump, ["row", "peak_col"]].assign(i=np.flatnonzero(bump))
            j = k.merge(D, on="row", suffixes=("", "_d"))
            pc, pd_ = j.peak_col.to_numpy(np.int64), j.peak_col_d.to_numpy(np.int64)
            dt = np.abs(pd_ - pc)
            inside = (j.start.to_numpy() <= pc) & (j.end.to_numpy() >= pc)
            ok = (dt >= 1) & (dt <= win) & ~inside
            hit[j.i.to_numpy()[ok]] = True
        out.append(hit)
    return out


def dumpedge_label(ev: pd.DataFrame, dump_bins: np.ndarray) -> np.ndarray:
    """Event run starts or ends within ``1 + W/2`` bins of a momentum dump (W: box width;
    the box smears a step's edge by up to W/2)."""
    s = ev.start.to_numpy(np.int64)
    e = ev.end.to_numpy(np.int64)
    h = 1 + np.asarray(V.WIDTHS)[ev.w.to_numpy(np.int64)] // 2
    hit = np.zeros(len(ev), bool)
    for db in dump_bins:
        hit |= (np.abs(s - db) <= h) | (np.abs(e - db) <= h)
    return hit


# --------------------------------------------------------------------------------------------
# Pass A: scan (events, points, per-CCD statistics)


@dataclass
class ScanStats:
    slope_hist: np.ndarray  # (16, len(SLOPE_EDGES)-1)
    qbit: np.ndarray  # (16, L) light curves with QBITS set per CCD and bin
    n_ccd: np.ndarray  # (16,)

    def slope_q(self, q: float = SLOPE_Q) -> np.ndarray:
        c = np.cumsum(self.slope_hist, 1)
        tot = np.maximum(c[:, -1:], 1)
        k = np.argmax(c / tot >= q, axis=1)
        out = 10.0 ** SLOPE_EDGES[k + 1]
        out[c[:, -1] == 0] = np.inf
        return out

    def qwin(self) -> np.ndarray:
        w = self.qbit / np.maximum(self.n_ccd[:, None], 1) >= QB_FRAC
        w[:, 1:] |= w[:, :-1].copy()
        w[:, :-1] |= w[:, 1:].copy()
        return w


def _lc_table(b: dict, key: str, mm: pd.DataFrame, n0: int, ci, cell) -> pd.DataFrame:
    return pd.DataFrame({"lc": n0 + np.arange(len(ci)), "tic": b["tic"],
                         "inj_id": b["inj_id"] if b["inj_id"] is not None else None,
                         "ccd": ci.astype(np.int8), "cell": cell.astype(np.int8),
                         "crowdsap": mm.crowdsap.to_numpy(np.float64)})  # fmt: skip


def scan_dir(d, grid: Grid, meta: pd.DataFrame, cells: pd.DataFrame, masks: V.SysMasks,
             batch_rows: int = 2000, log=None):  # fmt: skip
    """Pass A. Returns ``(lcs, events, points, stats)``.

    ``events``: every stage-1 event (lc, ch, sign, w, start, end, peak_col, peak, sys1).
    ``points``: short background and SAP bumps for the track test (lc, t, snr, kind).
    """
    key = "inj_id" if "inj_id" in meta.columns else "tic"
    m = V._lookup(meta, cells, key)
    L = grid.length
    hist = np.zeros((16, len(SLOPE_EDGES) - 1), np.int64)
    qbit = np.zeros((16, L), np.float32)
    n_ccd = np.zeros(16, np.int64)
    lcs, evs, pts = [], [], []
    n0 = 0
    for b in iter_batches(d, grid, batch_rows=batch_rows,
                          columns=("flux", "sap_flux", "bkg", "qflags")):  # fmt: skip
        mm = m.loc[V._ids(b, key)]
        ci = ccd_index(mm.camera.to_numpy(), mm.ccd.to_numpy())
        cell = mm.cell.to_numpy()
        sig = batch_signal(b)
        parts, _ = V.events_stage1(sig.snr, ci, cell, masks)
        ev = (pd.concat(parts, ignore_index=True) if parts
              else pd.DataFrame(columns=V.EVENT_COLS))  # fmt: skip
        evs.append(pd.DataFrame({
            "lc": (n0 + ev.row.to_numpy(np.int64)), "ch": ev.ch.to_numpy(np.int8),
            "sign": ev.sign.to_numpy(np.int8), "w": ev.w.to_numpy(np.int8),
            "start": ev.start.to_numpy(np.int16), "end": ev.end.to_numpy(np.int16),
            "peak_col": ev.peak_col.to_numpy(np.int16), "peak": ev.peak.to_numpy(np.float32),
            "sys1": ev.sys.to_numpy(bool)}))  # fmt: skip
        # points: short SAP bumps (2-h box, not systematic) and short background bumps
        sb = ev[(ev.ch == 1) & (ev.sign == 1) & (ev.w == 0) & ~ev.sys.astype(bool)
                & (ev.peak >= POINT_SNR)]  # fmt: skip
        sb = sb[half_max_width(sig.snr[(1, 1, 0)], sb) <= POINT_MAXLEN]
        pts.append(pd.DataFrame({"lc": n0 + sb.row.to_numpy(np.int64),
                                 "t": sb.peak_col.to_numpy(np.int16),
                                 "snr": sb.peak.to_numpy(np.float32), "kind": np.int8(0)}))
        a = np.nan_to_num(sig.bkg_bump, nan=0.0) > POINT_SNR
        idx, seg, starts, rows = V._segments(a)
        if idx.size:
            val = sig.bkg_bump.ravel()[idx]
            pk = np.maximum.reduceat(val, starts)
            ln = np.add.reduceat((val >= 0.5 * pk[seg]).astype(np.int64), starts)  # half max
            big = np.iinfo(np.int64).max
            pcol = np.minimum.reduceat(np.where(val == pk[seg], idx % L, big), starts)
            ok = ln <= POINT_MAXLEN
            pts.append(pd.DataFrame({"lc": n0 + rows[ok], "t": pcol[ok].astype(np.int16),
                                     "snr": pk[ok].astype(np.float32), "kind": np.int8(1)}))
        # per-CCD statistics
        lg = np.log10(np.where(sig.slope > 0, sig.slope, np.nan))
        for c in np.unique(ci):
            v = lg[ci == c]
            hist[c] += np.histogram(v[np.isfinite(v)], bins=SLOPE_EDGES)[0]
        qbit += V.group_sum(((b["qflags"] & QBITS) > 0).astype(np.float32), ci, 16)
        n_ccd += np.bincount(ci, minlength=16)
        lcs.append(_lc_table(b, key, mm, n0, ci, cell))
        n0 += len(ci)
        if log:
            log(f"    scan {b['shard_file']}: {n0} light curves")
    lcs = pd.concat(lcs, ignore_index=True)
    return (lcs, pd.concat(evs, ignore_index=True), pd.concat(pts, ignore_index=True),
            ScanStats(hist, qbit, n_ccd))


# --------------------------------------------------------------------------------------------
# Pass B: cross-star labels


def find_tracks(x, y, t, star, tmax=TMAX, vmin=VMIN, vmax=VMAX, tol=TOL, win=SUPPORT_WIN,
                chunk=2_000_000) -> pd.DataFrame:  # fmt: skip
    """Straight, steady tracks through points ``(x, y, t)`` (px, px, bin) of one CCD.

    A pair of points of different stars ``i, j`` (``1 <= t_j - t_i <= tmax``, speed in
    ``[vmin, vmax]`` px/bin) is a track when a third star's point lies within ``tol + |v|`` px of
    the line through them, within ``win`` bins of the pair. Returns one row per accepted pair:
    ``x0, y0, t0`` (point i), ``vx, vy`` (px/bin), ``t_lo, t_hi`` (span of all supporting
    points), ``n_star`` (distinct stars supporting, incl. i and j).
    """
    o = np.argsort(t, kind="stable")
    x, y, t, star = (np.asarray(a)[o] for a in (x, y, t, star))
    t = t.astype(np.int64)
    n = len(t)
    cols = ["x0", "y0", "t0", "vx", "vy", "t_lo", "t_hi", "n_star"]
    if n < 3:
        return pd.DataFrame(columns=cols)
    a = np.searchsorted(t, t + 1, side="left")
    b = np.searchsorted(t, t + tmax, side="right")
    cnt = b - a
    out = []
    # process point i in groups so the (pair) and (pair, k) arrays stay bounded
    for g0 in range(0, n, 2000):
        gi = np.arange(g0, min(n, g0 + 2000))
        ii = np.repeat(gi, cnt[gi])
        if ii.size == 0:
            continue
        jj = np.concatenate([np.arange(a[k], b[k]) for k in gi if cnt[k] > 0])
        dt = (t[jj] - t[ii]).astype(np.float64)
        dx, dy = x[jj] - x[ii], y[jj] - y[ii]
        v = np.hypot(dx, dy) / dt
        ok = (star[ii] != star[jj]) & (v >= vmin) & (v <= vmax)
        ii, jj, dt, dx, dy, v = ii[ok], jj[ok], dt[ok], dx[ok], dy[ok], v[ok]
        if ii.size == 0:
            continue
        vx, vy = dx / dt, dy / dt
        ka = np.searchsorted(t, t[ii] - win, side="left")
        kb = np.searchsorted(t, t[jj] + win, side="right")
        kc = kb - ka
        # bounded (pair, k) expansion in sub-chunks
        step = max(1, chunk // max(int(kc.mean()), 1))
        for s0 in range(0, ii.size, step):
            sl = slice(s0, s0 + step)
            pi, pj, pvx, pvy, pv = ii[sl], jj[sl], vx[sl], vy[sl], v[sl]
            pc = kc[sl]
            if pc.sum() == 0:
                continue
            pid = np.repeat(np.arange(pi.size), pc)
            kk = (np.repeat(ka[sl], pc)
                  + np.arange(pc.sum()) - np.repeat(np.cumsum(pc) - pc, pc))  # fmt: skip
            tt = (t[kk] - t[pi[pid]]).astype(np.float64)
            px = x[pi[pid]] + pvx[pid] * tt
            py = y[pi[pid]] + pvy[pid] * tt
            d = np.hypot(x[kk] - px, y[kk] - py)
            sup = ((d <= tol + pv[pid]) & (star[kk] != star[pi[pid]])
                   & (star[kk] != star[pj[pid]]))  # fmt: skip
            if not sup.any():
                continue
            sp, sk = pid[sup], kk[sup]
            df = pd.DataFrame({"p": sp, "s": star[sk], "t": t[sk]}).drop_duplicates(["p", "s"])
            g = df.groupby("p")
            nst = g.size()
            tlo, thi = g.t.min(), g.t.max()
            p = nst.index.to_numpy()
            out.append(pd.DataFrame({
                "x0": x[pi[p]], "y0": y[pi[p]], "t0": t[pi[p]], "vx": pvx[p], "vy": pvy[p],
                "t_lo": np.minimum(tlo.to_numpy(), t[pi[p]]),
                "t_hi": np.maximum(thi.to_numpy(), t[pj[p]]),
                "n_star": nst.to_numpy() + 2}))  # fmt: skip
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


def track_windows(tracks: pd.DataFrame, sx, sy, slc, r=TRACK_R, margin=TRACK_MARGIN,
                  ) -> pd.DataFrame:  # fmt: skip
    """Per star (``slc`` ids at pixel positions ``sx, sy``) and track, the bins
    ``[lo, hi]`` during which the track passes within ``r`` px."""
    if len(tracks) == 0 or len(sx) == 0:
        return pd.DataFrame(columns=["lc", "lo", "hi"])
    from scipy.spatial import cKDTree

    tree = cKDTree(np.column_stack([sx, sy]))
    out = []
    for tr in tracks.itertuples(index=False):
        v2 = tr.vx**2 + tr.vy**2
        t_a, t_b = tr.t_lo - margin, tr.t_hi + margin
        xa, ya = tr.x0 + tr.vx * (t_a - tr.t0), tr.y0 + tr.vy * (t_a - tr.t0)
        xb, yb = tr.x0 + tr.vx * (t_b - tr.t0), tr.y0 + tr.vy * (t_b - tr.t0)
        # candidates: stars within r of the segment (query around its midpoint)
        mx, my = (xa + xb) / 2, (ya + yb) / 2
        half = np.hypot(xb - xa, yb - ya) / 2 + r
        idx = np.asarray(tree.query_ball_point([mx, my], half), dtype=np.int64)
        if idx.size == 0:
            continue
        ex, ey = sx[idx] - tr.x0, sy[idx] - tr.y0
        tc = tr.t0 + (ex * tr.vx + ey * tr.vy) / v2  # time of closest approach
        dperp = np.abs(ex * tr.vy - ey * tr.vx) / np.sqrt(v2)
        ok = dperp <= r
        if not ok.any():
            continue
        hw = np.sqrt(np.maximum(r**2 - dperp[ok] ** 2, 0.0)) / np.sqrt(v2)
        lo = np.maximum(np.floor(tc[ok] - hw) - 1, t_a)
        hi = np.minimum(np.ceil(tc[ok] + hw) + 1, t_b)
        keep = hi >= lo
        out.append(pd.DataFrame({"lc": slc[idx[ok]][keep], "lo": lo[keep].astype(np.int64),
                                 "hi": hi[keep].astype(np.int64)}))  # fmt: skip
    if not out:
        return pd.DataFrame(columns=["lc", "lo", "hi"])
    return pd.concat(out, ignore_index=True).drop_duplicates()


def _crowded_bins(pts: pd.DataFrame, n_ccd: np.ndarray, frac=CROWD_FRAC) -> pd.Series:
    """Bool per point: its CCD has points from more than ``frac`` of its stars in that bin."""
    c = pts.groupby(["ccd", "t"]).lc.nunique().rename("k").reset_index()
    j = pts[["ccd", "t"]].merge(c, on=["ccd", "t"], how="left")
    lim = np.maximum(5, frac * n_ccd[j.ccd.to_numpy(np.int64)])
    return pd.Series(j.k.to_numpy() > lim, index=pts.index)


def track_labels(points: pd.DataFrame, stars: pd.DataFrame, n_ccd: np.ndarray,
                 query_lcs: np.ndarray | None = None, shuffle_seed: int | None = None, log=None):
    """Track test over every CCD.

    ``points``: lc, t, ccd, col, row, star (TIC: one star counts once per track).
    ``stars``: lc, ccd, col, row: every light curve whose windows are wanted.
    ``query_lcs``: only tracks supported by at least one of these light curves' points, or
    passing them, matter (an injection set); None = all.
    ``shuffle_seed``: permute point times within each CCD first (the chance-track null).
    Returns ``(windows: lc, lo, hi; tracks: per CCD)``.
    """
    pts = points.copy()
    pts = pts[~_crowded_bins(pts, n_ccd).to_numpy()]
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    wins, trs = [], []
    for c, g in pts.groupby("ccd"):
        if rng is not None:
            g = g.assign(t=rng.permutation(g.t.to_numpy()))
        tr = find_tracks(g.col.to_numpy(np.float64), g.row.to_numpy(np.float64),
                         g.t.to_numpy(np.int64), g.star.to_numpy(np.int64))  # fmt: skip
        if len(tr) == 0:
            continue
        tr = tr.assign(ccd=int(c))
        st = stars[stars.ccd == c]
        w = track_windows(tr, st.col.to_numpy(np.float64), st.row.to_numpy(np.float64),
                          st.lc.to_numpy(np.int64))  # fmt: skip
        if query_lcs is not None and len(w):
            w = w[np.isin(w.lc.to_numpy(), query_lcs)]
        trs.append(tr)
        wins.append(w)
        if log:
            log(f"    tracks ccd {c}: {len(g)} points, {len(tr)} track pairs, "
                f"{w.lc.nunique() if len(w) else 0} light curves in windows")
    W = (pd.concat(wins, ignore_index=True) if wins
         else pd.DataFrame(columns=["lc", "lo", "hi"]))  # fmt: skip
    T = pd.concat(trs, ignore_index=True) if trs else pd.DataFrame()
    return W, T


def events_in_windows(ev: pd.DataFrame, win: pd.DataFrame) -> np.ndarray:
    """Bool per event of ``ev`` (lc, start, end): overlaps any window of its light curve."""
    if len(win) == 0 or len(ev) == 0:
        return np.zeros(len(ev), bool)
    e = ev[["lc", "start", "end"]].reset_index(drop=True).reset_index()
    j = e.merge(win, on="lc", how="inner")
    hit = (j.start.to_numpy() <= j.hi.to_numpy()) & (j.end.to_numpy() >= j.lo.to_numpy())
    out = np.zeros(len(ev), bool)
    out[j["index"].to_numpy()[hit]] = True
    return out


def column_labels(ev: pd.DataFrame, ref: pd.DataFrame, star_cols: pd.DataFrame, L: int,
                  query: np.ndarray | None = None) -> np.ndarray:  # fmt: skip
    """Same-column test (see module docstring).

    ``ev`` / ``ref``: events with lc, ccd, col (pixel column), ch, sign, w, start, end,
    peak_col. ``ref`` is the population counted (it includes ``ev``'s own events: each event
    discounts itself). ``star_cols``: ccd, col of every star (density). Returns a bool per row
    of ``ev`` (only rows with ``query`` True are tested, if given).
    """
    out = np.zeros(len(ev), bool)
    if len(ev) == 0:
        return out
    q = np.ones(len(ev), bool) if query is None else np.asarray(query, bool)
    lo_c, hi_c = COL_CTRL
    for c in np.unique(ev.ccd.to_numpy()[q]):
        dens = np.bincount(np.clip(star_cols.col[star_cols.ccd == c].round().astype(np.int64),
                                   0, NCOL - 1), minlength=NCOL).astype(np.float64)  # fmt: skip
        cd = np.concatenate([[0.0], np.cumsum(dens)])
        rc = ref[ref.ccd == c]
        for (ch, s, w), rg in rc.groupby(["ch", "sign", "w"]):
            sel = np.flatnonzero(q & (ev.ccd.to_numpy() == c) & (ev.ch.to_numpy() == ch)
                                 & (ev.sign.to_numpy() == s) & (ev.w.to_numpy() == w))  # fmt: skip
            if sel.size == 0:
                continue
            C = np.zeros((NCOL, L + 1), np.int32)
            col = np.clip(rg.col.round().to_numpy(np.int64), 0, NCOL - 1)
            np.add.at(C, (col, rg.start.to_numpy(np.int64)), 1)
            np.add.at(C, (col, rg.end.to_numpy(np.int64) + 1), -1)
            C = np.cumsum(C, axis=1)[:, :L]
            CC = np.concatenate([np.zeros((1, L), np.int32), np.cumsum(C, axis=0)], 0)
            e = ev.iloc[sel]
            c0 = np.clip(e.col.round().to_numpy(np.int64), 0, NCOL - 1)
            tp = e.peak_col.to_numpy(np.int64)

            def band(a, b, c0=c0, tp=tp, CC=CC, cd=cd):
                a = np.clip(c0 + a, 0, NCOL)
                b = np.clip(c0 + b, 0, NCOL)
                return CC[b, tp] - CC[a, tp], cd[b] - cd[a]

            n_s, d_s = band(-COL_HALF, COL_HALF + 1)
            n1, d1 = band(-hi_c, -lo_c + 1)
            n2, d2 = band(lo_c, hi_c + 1)
            n_same = n_s - 1  # the event itself
            d_s = np.maximum(d_s - 1, 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                lam = (n1 + n2) * d_s / np.maximum(d1 + d2, 1.0)
            lam = np.maximum(lam, 1e-3)
            p = poisson.sf(n_same - 1, lam)
            out[sel] = (n_same >= COL_MIN) & (p < COL_P)
    return out


def _with_pos(df: pd.DataFrame, lcs: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    pos = lcs[["lc", "tic", "ccd"]].merge(cells[["tic", "col", "row"]], on="tic", how="left")
    pos["col"] = pos.col.fillna(1024.0)
    pos["row"] = pos.row.fillna(1024.0)
    return df.merge(pos, on="lc", how="left")


def cross_labels(lcs: pd.DataFrame, events: pd.DataFrame, points: pd.DataFrame,
                 cells: pd.DataFrame, n_ccd: np.ndarray, L: int, ref: tuple | None = None,
                 shuffle_seed: int | None = None, log=None):  # fmt: skip
    """Pass B: ``track`` and ``column`` labels for the events of one scan.

    ``ref``: ``(lcs, events, points)`` of the real sector's scan when this scan is an injection
    set. The real light curves of the set's hosts are left out of the reference (the injected
    copies replace them). Returns ``(labels, windows, tracks)``: ``labels`` holds lc, ch, sign,
    w, start, track, column for every labelled event of this scan.
    """
    ev = _with_pos(events, lcs, cells)
    pts = _with_pos(points, lcs, cells).rename(columns={"tic": "star"})
    stars = _with_pos(lcs[["lc"]], lcs, cells)
    query = None
    ref_ev = ev
    if ref is not None:
        rl, re_, rp = ref
        hosts = set(lcs.tic.tolist())
        keep = rl[~rl.tic.isin(hosts)]
        off = int(lcs.lc.max()) + 1  # distinct ids for the reference light curves
        kl = keep.assign(lc=keep.lc + off)
        rev = _with_pos(re_[re_.lc.isin(keep.lc)].assign(lc=lambda x: x.lc + off), kl, cells)
        rpt = _with_pos(rp[rp.lc.isin(keep.lc)].assign(lc=lambda x: x.lc + off), kl,
                        cells).rename(columns={"tic": "star"})  # fmt: skip
        pts = pd.concat([pts, rpt], ignore_index=True)
        stars = pd.concat([stars, _with_pos(kl[["lc"]], kl, cells)], ignore_index=True)
        ref_ev = pd.concat([ev, rev], ignore_index=True)
        query = lcs.lc.to_numpy()
    win, tracks = track_labels(pts, stars, n_ccd, query_lcs=query, shuffle_seed=shuffle_seed,
                               log=log)  # fmt: skip
    tr = events_in_windows(ev, win)
    co = column_labels(ev, ref_ev, stars[["ccd", "col"]], L)
    lab = ev.loc[tr | co, ["lc", "ch", "sign", "w", "start"]].copy()
    lab["track"] = tr[tr | co]
    lab["column"] = co[tr | co]
    return lab.reset_index(drop=True), win, tracks


# --------------------------------------------------------------------------------------------
# Pass C: classify


def summarise(ev: pd.DataFrame, n: int, t: np.ndarray, prefix: str) -> dict:
    """Per light curve: strongest residual / systematic event, overall and PDCSAP-only, with
    the time and label of each (the ``cmv2`` summary, with ``cmv3`` cause names)."""
    out = {}
    sys_ = ev.sys.to_numpy(bool)
    for lab, sel in (("", np.ones(len(ev), bool)), ("_pdc", (ev.ch == 0).to_numpy())):
        for kind, m in (("res", ~sys_), ("sys", sys_)):
            ee = ev[sel & m]
            pk = np.zeros(n)
            np.maximum.at(pk, ee.row.to_numpy(np.int64), ee.peak.to_numpy(np.float64))
            out[f"{prefix}_{kind}_snr{lab}"] = pk
    for kind, m in (("res", ~sys_), ("sys", sys_)):
        ee = ev[m].sort_values("peak", ascending=False).drop_duplicates("row")
        tt = np.full(n, np.nan)
        lab = np.full(n, "", dtype=object)
        tt[ee.row.to_numpy(np.int64)] = t[ee.peak_col.to_numpy(np.int64)]
        lab[ee.row.to_numpy(np.int64)] = [
            f"{V.CHANNELS[c]}/{V.SIGNS[s]}/{V.WIDTHS[w] // 2}h/{CAUSES[k]}"
            for c, s, w, k in zip(ee.ch, ee.sign, ee.w, ee.cause, strict=True)]
        out[f"{prefix}_{kind}_t"] = tt
        out[f"{prefix}_{kind}_ev"] = lab
    return out


def classify_dir(d, grid: Grid, meta: pd.DataFrame, cells: pd.DataFrame, masks: V.SysMasks,
                 stats: ScanStats, cross: pd.DataFrame, configs: dict = CONFIGS,
                 batch_rows: int = 2000, log=None, keep_events=None):  # fmt: skip
    """Pass C. Per light curve, for every configuration ``name -> labels``: ``<name>_res_snr``,
    ``<name>_sys_snr`` (+ ``_pdc``), and the strongest events' times and labels. Also, for the
    ``all`` configuration, per new label the number of events it labels and the strongest
    PDCSAP event it labels.

    ``cross``: lc, ch, sign, w, start, track, column (from :func:`cross_labels`).
    """
    key = "inj_id" if "inj_id" in meta.columns else "tic"
    m = V._lookup(meta, cells, key)
    slope_q = stats.slope_q()
    qwin = stats.qwin()
    cross = cross.set_index("lc").sort_index() if len(cross) else cross
    frames, kept = [], []
    n0 = 0
    for b in iter_batches(d, grid, batch_rows=batch_rows,
                          columns=("flux", "sap_flux", "bkg", "qflags")):  # fmt: skip
        ids = V._ids(b, key)
        mm = m.loc[ids]
        ci = ccd_index(mm.camera.to_numpy(), mm.ccd.to_numpy())
        cell = mm.cell.to_numpy()
        n = len(ci)
        sig = batch_signal(b)
        parts, members = V.events_stage1(sig.snr, ci, cell, masks)
        qsh = qbits_share(members, b["qflags"], qwin, ci)
        labs_parts = []
        crowd = mm.crowdsap.to_numpy(np.float64)
        cr = cross.loc[n0 : n0 + n - 1] if len(cross) else None
        for p, qs in zip(parts, qsh, strict=True):
            me = event_measures(p, sig, crowd)
            lab = measure_labels(p, me, slope_q, ci)
            lab["qbits"] = qs >= V.SYS_SHARE
            lab["dumpedge"] = dumpedge_label(p, masks.dump_bins)
            tr = np.zeros(len(p), bool)
            co = np.zeros(len(p), bool)
            if cr is not None and len(cr):
                k = p[["row", "ch", "sign", "w", "start"]].copy()
                k["lc"] = n0 + k.row.to_numpy(np.int64)
                k["i"] = np.arange(len(k))
                j = k.merge(cr.reset_index(), on=["lc", "ch", "sign", "w", "start"], how="inner")
                tr[j.i.to_numpy()] = j.track.to_numpy(bool)
                co[j.i.to_numpy()] = j.column.to_numpy(bool)
            lab["track"], lab["column"] = tr, co
            labs_parts.append(lab)
        for lab, fl in zip(labs_parts, flank_labels(parts, labs_parts), strict=True):
            lab["flank"] = fl
        df = {"tic": b["tic"], "inj_id": b["inj_id"] if b["inj_id"] is not None else None}
        for name, enabled in configs.items():
            pp = []
            for p, lab in zip(parts, labs_parts, strict=True):
                q = p.copy()
                sys0 = q.sys.to_numpy(bool)
                cause = q.cause.to_numpy(np.int8).copy()
                new = np.zeros(len(q), bool)
                for lb in reversed(enabled):  # first enabled label wins the cause
                    hit = lab[lb] & ~sys0
                    cause[hit] = LABEL_CODE[lb]
                    new |= hit
                q["sys"] = sys0 | new
                q["cause"] = cause
                pp.append(q)
            ev = V.events_stage2(pp, members, n)
            df.update(summarise(ev, n, grid.time, name))
            if name == "all":
                ev_all = ev
                lab_cat = {lb: np.concatenate([x[lb] for x in labs_parts]) if labs_parts
                           else np.zeros(0, bool) for lb in LABELS}  # fmt: skip
                rows = ev.row.to_numpy(np.int64)
                pk = ev.peak.to_numpy(np.float64)
                pdc = (ev.ch == 0).to_numpy()
                for lb in LABELS:
                    hit = lab_cat[lb]
                    df[f"n_{lb}"] = np.bincount(rows[hit], minlength=n)
                    mx = np.zeros(n)
                    np.maximum.at(mx, rows[hit & pdc], pk[hit & pdc])
                    df[f"pk_{lb}_pdc"] = mx
        frames.append(pd.DataFrame(df))
        if keep_events:
            ids_arr = np.asarray(ids, dtype=object)
            want = np.array([i in keep_events for i in ids_arr])
            if want.any():
                e = ev_all.copy()
                for lb in LABELS:
                    e[f"lab_{lb}"] = lab_cat[lb]
                e = e[want[e.row.to_numpy()]]
                e["id"] = ids_arr[e.row.to_numpy()]
                e["t"] = grid.time[e.peak_col.to_numpy(np.int64)]
                kept.append(e)
        n0 += n
        if log:
            log(f"    classify {b['shard_file']}: {n0} light curves")
    out = pd.concat(frames, ignore_index=True)
    return out, (pd.concat(kept, ignore_index=True) if kept else None)


def veto_flag(df: pd.DataFrame, config: str, t_sys: float, t_res: float,
              pdc_only: bool = False) -> np.ndarray:  # fmt: skip
    sfx = "_pdc" if pdc_only else ""
    return ((df[f"{config}_sys_snr{sfx}"].to_numpy() >= t_sys)
            & (df[f"{config}_res_snr{sfx}"].to_numpy() < t_res))  # fmt: skip
