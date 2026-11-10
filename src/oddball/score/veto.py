"""Per-event systematics veto ``cmv2`` (docs/ML_DESIGN.md §5, §12).

The v0 flag (``classical.systematics_features``) is per *star*: ``sys_flag`` fires when at least
half of a light curve's dip chi^2 sits in common-mode bins. SAP carries scattered light in
10-24% of every CCD's bins (orbit starts and ends, the run-up to each momentum dump), so on S32
the SAP ramp outweighs everything else on most quiet stars and v0 flags 77.5% of real stars. A
weak real dip next to a strong shared ramp is then thrown away with the ramp.

``cmv2`` decides per **event** and then asks what is left:

1. **Events.** Per channel (PDCSAP, SAP), the flux is median-normalised and detrended by the
   classical running median (3.25 d). A centred box of 2 h, 8 h or 32 h (``WIDTHS``) gives a dip
   SNR and a bump SNR per bin (white-noise sigma from the point-to-point scatter), each series
   divided by its own robust scatter when that exceeds 1 (red noise: an event's SNR says how
   far it stands out from the rest of *this* light curve). For each of the 12 (channel, sign,
   width) series, a run of consecutive bins above ``ANOM_SNR`` is one event, with its peak SNR
   and its SNR^2 profile.
2. **Common-mode maps.** For each series and time bin, the fraction of light curves on the same
   CCD with that bin anomalous, and the same fraction in the light curve's own 512 x 512 pixel
   cell of the CCD (``NCELL`` x ``NCELL`` cells from ``tess-point`` positions: localised
   scattered light, glints). A (series, CCD [, cell], bin) is common-mode when its fraction is
   above the series' median + ``K_MAD`` robust sigma (floor ``FLOOR``) with at least
   ``MIN_COUNT`` light curves (the v0 rule, now per sign, width and cell). For an injection set,
   whose few hosts per CCD are not the CCD, the real sector's maps are OR-ed with the set's own
   CCD map under a binomial tail test (``classical.common_mode_bins_binomial``), so an event
   shared by the set's hosts on one CCD is visible while scattered single injections are not.
3. **Known windows.** Momentum dumps are the bins where most light curves carry QUALITY bit 32
   (S32: BTJD 2179.48, 2193.48). Scattered-light windows are the bins where at least 20% of a
   CCD's light curves carry SPOC's scattered-light bit (4096), widened by one bin.
4. **Per-event decision.** An event is *systematic* when at least half of its SNR^2 lies in
   common-mode or scattered-light bins, or when its peak falls within ``1 + W/2`` bins after a
   momentum dump (``W`` its box width; from one bin before). Then, per channel, an event that
   mostly lies inside a systematic event's span, or contains the peak of an equally strong
   systematic event seen through the same or a narrower box, is systematic too (the same instrument event seen at another
   width or sign; ``events_batch``).
5. **Per light curve.** ``cmv2_res_snr`` is the peak SNR of its strongest *non-systematic* event
   (the astrophysical residual), ``cmv2_sys_snr`` that of its strongest systematic event, each
   also per channel. The flag is ``cmv2_flag = sys_snr >= t_sys and res_snr < t_res``: a strong
   systematic event and nothing astrophysical left. A star with one real dip and a shared ramp
   keeps its real dip and is not flagged. ``t_sys`` and ``t_res`` are tuned on val
   (``scripts/tune_veto.py``).

Everything streams in batches; a pass over S32 is a few minutes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from oddball.common.strata import ccd_index
from oddball.score.classical import (
    centred_box_snr,
    common_mode_bins_binomial,
    nanquantile_last,
    running_baseline,
)
from oddball.tess.arrays import Grid, iter_batches

CHANNELS = ("pdc", "sap")
SIGNS = ("dip", "bump")
WIDTHS = (4, 16, 64)  # bins of 30 min: 2 h, 8 h, 32 h
SERIES = tuple((c, s, w) for c in range(2) for s in range(2) for w in range(len(WIDTHS)))
ANOM_SNR = 3.0
NCELL = 4  # cells per CCD side (512 px)
K_MAD = 8.0
FLOOR = 0.01
MIN_COUNT = 5
MIN_DEN = 20
SYS_SHARE = 0.5
SCAT_BIT = 4096
DUMP_BIT = 32
SCAT_FRAC = 0.2
DUMP_FRAC = 0.5
DIR_CM_PMAX = 1e-7  # binomial tail for an injection set's own common-mode cells (as classical)
CAUSES = ("none", "cm_ccd", "cm_cell", "scattered", "dump", "set", "overlap")


# --------------------------------------------------------------------------------------------
# Per-batch signal


def channel_snr(f: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    """{(sign, width index): (n, L) float32 box SNR} of one channel's detrended residual.

    Sign 0 = dips (positive SNR for a dimming), 1 = bumps. NaN where the box lacks data.
    """
    f = f.astype(np.float64)
    med = nanquantile_last(f, 0.5)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        x = f / med[:, None] - 1.0
    x[~np.isfinite(x)] = np.nan
    d = np.diff(x, axis=1)
    dmed = nanquantile_last(d, 0.5)[0]
    sigma = 1.4826 * nanquantile_last(np.abs(d - dmed[:, None]), 0.5)[0] / np.sqrt(2.0)
    bad = ~(np.isfinite(sigma) & (sigma > 0))
    sig = np.where(bad, 1.0, sigma)
    r = x - running_baseline(x)
    out = {}
    for wi, W in enumerate(WIDTHS):
        s = centred_box_snr(r, sig, W, min_count=max(3, int(np.ceil(0.6 * W))))
        s[bad] = np.nan
        # red noise: rescale by the series' own robust scatter (never below white noise), so an
        # event's SNR says how much it stands out from the rest of *this* light curve
        med = nanquantile_last(s, 0.5)[0]
        rs = 1.4826 * nanquantile_last(np.abs(s - med[:, None]), 0.5)[0]
        s = s / np.maximum(np.nan_to_num(rs, nan=1.0), 1.0)[:, None]
        out[(0, wi)] = s.astype(np.float32)
        out[(1, wi)] = (-s).astype(np.float32)
    return out


def batch_snr(b: dict) -> dict[tuple[int, int, int], np.ndarray]:
    out = {}
    for c, key in enumerate(("flux", "sap_flux")):
        for (s, w), v in channel_snr(b[key]).items():
            out[(c, s, w)] = v
    return out


# --------------------------------------------------------------------------------------------
# Positions on the CCD


def pixel_cells(meta: pd.DataFrame, sector: int, ncell: int = NCELL) -> pd.DataFrame:
    """tic -> (col, row, cell) from tess-point; cell = ncell x ncell grid over the 2048 px CCD.

    Stars tess-point puts on another camera/CCD than the light curve's (edge cases) get the
    cell from their position clipped to the CCD.
    """
    from tess_stars2px import tess_stars2px_function_entry as s2p

    m = meta[["tic", "ra", "dec", "camera", "ccd"]].drop_duplicates("tic")
    parts = []
    step = 20000
    for i in range(0, len(m), step):
        mm = m.iloc[i : i + step]
        tid, _, _, sec, cam, ccd, col, row, _ = s2p(mm.tic.to_numpy(), mm.ra.to_numpy(),
                                                    mm.dec.to_numpy(), trySector=sector)
        parts.append(pd.DataFrame({"tic": tid, "sec": sec, "cam_tp": cam, "ccd_tp": ccd,
                                   "col": col, "row": row}))  # fmt: skip
    tp = pd.concat(parts, ignore_index=True)
    tp = tp[tp.sec == sector].drop(columns="sec")
    j = m.merge(tp, on="tic", how="left")
    # keep the solution on the light curve's own CCD when tess-point returns several
    j["own"] = (j.cam_tp == j.camera) & (j.ccd_tp == j.ccd)
    j = j.sort_values("own", ascending=False).drop_duplicates("tic")
    col = np.clip(np.nan_to_num(j.col.to_numpy(np.float64), nan=1024.0), 45, 2092) - 45
    row = np.clip(np.nan_to_num(j.row.to_numpy(np.float64), nan=1024.0), 1, 2048) - 1
    side = 2048.0 / ncell
    cell = (np.minimum((col // side), ncell - 1) * ncell + np.minimum(row // side, ncell - 1))
    return pd.DataFrame({"tic": j.tic.to_numpy(np.int64), "col": j.col.to_numpy(),
                         "row": j.row.to_numpy(), "own_ccd": j.own.to_numpy(),
                         "cell": cell.astype(np.int64)})  # fmt: skip


# --------------------------------------------------------------------------------------------
# Pass 1: maps


@dataclass
class Maps:
    num: np.ndarray  # (12, 16, L) anomalous counts per series, CCD, bin
    den: np.ndarray  # (12, 16, L) valid counts
    num_cell: np.ndarray  # (12, 16, NCELL^2, L)
    den_cell: np.ndarray
    scat: np.ndarray  # (16, L) count of LCs with the scattered-light bit
    dump: np.ndarray  # (L,) count of LCs with the momentum-dump bit
    n_ccd: np.ndarray  # (16,) light curves per CCD
    n: int = 0


def group_sum(a: np.ndarray, keys: np.ndarray, nkeys: int) -> np.ndarray:
    """(nkeys, L) row sums of ``a`` (n, L) grouped by ``keys`` (n,) in [0, nkeys)."""
    out = np.zeros((nkeys, a.shape[1]), np.float64)
    if len(keys) == 0:
        return out
    o = np.argsort(keys, kind="stable")
    k = keys[o]
    starts = np.flatnonzero(np.concatenate([[True], k[1:] != k[:-1]]))
    out[k[starts]] = np.add.reduceat(a[o], starts, axis=0)
    return out


def _ids(b: dict, key: str):
    return b["inj_id"] if key == "inj_id" else b["tic"]


def _lookup(meta: pd.DataFrame, cells: pd.DataFrame | None, key: str):
    m = meta.copy()
    tic_col = "host_tic" if (key == "inj_id" and "host_tic" in m) else "tic"
    if cells is not None:
        m = m.merge(cells[["tic", "cell"]].rename(columns={"tic": tic_col}), on=tic_col,
                    how="left")  # fmt: skip
        m["cell"] = m["cell"].fillna(0).astype(np.int64)
    else:
        m["cell"] = 0
    return m.set_index(key)


def scan_maps(d, grid: Grid, meta: pd.DataFrame, cells: pd.DataFrame | None,
              batch_rows: int = 2000, log=None) -> Maps:  # fmt: skip
    key = "inj_id" if "inj_id" in meta.columns else "tic"
    m = _lookup(meta, cells, key)
    L, nc = grid.length, NCELL * NCELL
    num = np.zeros((len(SERIES), 16, L), np.float32)
    den = np.zeros_like(num)
    num_c = np.zeros((len(SERIES), 16, nc, L), np.float32)
    den_c = np.zeros_like(num_c)
    scat = np.zeros((16, L), np.float32)
    dump = np.zeros(L, np.float32)
    n_ccd = np.zeros(16, np.int64)
    n = 0
    for b in iter_batches(d, grid, batch_rows=batch_rows,
                          columns=("flux", "sap_flux", "qflags")):  # fmt: skip
        mm = m.loc[_ids(b, key)]
        ci = ccd_index(mm.camera.to_numpy(), mm.ccd.to_numpy())
        cell = mm.cell.to_numpy()
        flat = ci * nc + cell
        q = b["qflags"]
        scat += group_sum(((q & SCAT_BIT) > 0).astype(np.float32), ci, 16)
        dump += ((q & DUMP_BIT) > 0).sum(0)
        n_ccd += np.bincount(ci, minlength=16)
        snr = batch_snr(b)
        for si, (c, s, w) in enumerate(SERIES):
            v = snr[(c, s, w)]
            a = (np.nan_to_num(v, nan=0.0) > ANOM_SNR).astype(np.float32)
            ok = np.isfinite(v).astype(np.float32)
            num[si] += group_sum(a, ci, 16)
            den[si] += group_sum(ok, ci, 16)
            num_c[si] += group_sum(a, flat, 16 * nc).reshape(16, nc, L)
            den_c[si] += group_sum(ok, flat, 16 * nc).reshape(16, nc, L)
        n += len(ci)
        if log:
            log(f"    maps {b['shard_file']}: {n} light curves")
    return Maps(num, den, num_c, den_c, scat, dump, n_ccd, n)


def threshold_cm(num: np.ndarray, den: np.ndarray, k: float = K_MAD, floor: float = FLOOR,
                 min_count: int = MIN_COUNT, min_den: int = MIN_DEN) -> np.ndarray:
    """Common-mode cells along the last (time) axis: fraction > median + k * sigma, sigma the
    robust (MAD) scatter of the series but never below binomial noise, and >= ``min_count``."""
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(den >= min_den, num / np.maximum(den, 1), np.nan)
    ok = np.isfinite(frac).sum(-1) > 0
    fr = np.where(ok[..., None], frac, 0.0)
    med = np.nanmedian(np.where(ok[..., None], frac, 0.0), axis=-1)
    mad = 1.4826 * np.nanmedian(np.abs(fr - med[..., None]), axis=-1)
    # never tighter than binomial noise at the series' median population
    n_typ = np.maximum(np.median(den, axis=-1), 1.0)
    sig = np.maximum(mad, np.sqrt(np.clip(med, 1e-4, 1.0) * (1.0 - np.clip(med, 0.0, 0.999))
                                  / n_typ))  # fmt: skip
    th = np.maximum(floor, med + k * sig)
    return (np.nan_to_num(frac, nan=0.0) > th[..., None]) & (num >= min_count) & (den >= min_den)


@dataclass
class SysMasks:
    cm_ccd: np.ndarray  # (12, 16, L) bool
    cm_cell: np.ndarray  # (12, 16, NCELL^2, L) bool
    scat: np.ndarray  # (16, L) bool
    dump_bins: np.ndarray  # (k,) int grid columns of momentum dumps
    cm_set: np.ndarray | None = None  # (12, 16, L) bool, injection set's own map
    info: dict = field(default_factory=dict)


def real_masks(maps: Maps, k: float = K_MAD) -> SysMasks:
    cm_ccd = threshold_cm(maps.num, maps.den, k)
    cm_cell = threshold_cm(maps.num_cell, maps.den_cell, k)
    with np.errstate(invalid="ignore", divide="ignore"):
        sf = maps.scat / np.maximum(maps.n_ccd[:, None], 1)
    scat = sf >= SCAT_FRAC
    scat[:, 1:] |= scat[:, :-1].copy()
    scat[:, :-1] |= scat[:, 1:].copy()
    dump_bins = np.nonzero(maps.dump >= DUMP_FRAC * maps.n)[0]
    # collapse adjacent dump bins to their first bin (a dump's flag can straddle two bins)
    if len(dump_bins):
        dump_bins = dump_bins[np.concatenate([[True], np.diff(dump_bins) > 1])]
    info = {"cm_ccd_bins_per_ccd": cm_ccd.sum(-1).mean(1).round(1).tolist(),
            "cm_cell_bins_per_cell": cm_cell.sum(-1).mean((1, 2)).round(1).tolist(),
            "scattered_bins_per_ccd": scat.sum(1).tolist(),
            "dump_bins": dump_bins.tolist(), "k_mad": k}  # fmt: skip
    return SysMasks(cm_ccd, cm_cell, scat, dump_bins, None, info)


def set_masks(real: SysMasks, own: Maps, p_max: float = DIR_CM_PMAX) -> SysMasks:
    """Masks for an injection set: the real sector's, plus the set's own binomial CCD map."""
    cm_set = np.zeros_like(real.cm_ccd)
    for si in range(len(SERIES)):
        cm_set[si] = common_mode_bins_binomial(own.num[si][None], own.den[si][None],
                                               p_max=p_max)[0]  # fmt: skip
    info = {**real.info, "set_extra_cells": int((cm_set & ~real.cm_ccd).sum())}
    return SysMasks(real.cm_ccd, real.cm_cell, real.scat, real.dump_bins, cm_set, info)


# --------------------------------------------------------------------------------------------
# Pass 2: events


def _segments(a: np.ndarray):
    """Runs of True along rows of (n, L) bool: (flat index of members, segment id per member,
    segment starts into members, row per segment)."""
    L = a.shape[1]
    idx = np.flatnonzero(a)
    if idx.size == 0:
        return idx, idx, idx, idx
    col = idx % L
    new = np.ones(idx.size, bool)
    new[1:] = (idx[1:] != idx[:-1] + 1) | (col[1:] == 0)
    seg = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    return idx, seg, starts, idx[starts] // L


def events_batch(snr: dict, ci: np.ndarray, cell: np.ndarray, masks: SysMasks) -> pd.DataFrame:
    """One row per event: row, channel, sign, width, start/end/peak column, peak SNR, chi2,
    systematic share, ``sys`` and cause.

    Stage 1 classifies each event on its own (common-mode / scattered-light share, dump timing).
    Stage 2 marks a remaining event systematic when, in the same channel (any sign; the same or
    a narrower box width), at least half of its SNR^2 lies inside the time span of stage-1
    systematic events, or it contains the peak of such an event at least as strong as itself:
    the same instrument event seen through a wider box, or the detrending rebound next to it.
    A long real dip that merely contains a weaker systematic blip is kept, and so is a short
    real dip whose wide-box view runs into a nearby systematic window.
    """
    parts, members = [], []
    n = len(ci)
    big = np.iinfo(np.int64).max
    for si, (c, s, w) in enumerate(SERIES):
        v = snr[(c, s, w)]
        L = v.shape[1]
        a = np.nan_to_num(v, nan=0.0) > ANOM_SNR
        idx, seg, starts, erow = _segments(a)
        if idx.size == 0:
            continue
        val = v.ravel()[idx].astype(np.float64)
        w2 = val**2
        colx = idx % L
        rr = idx // L
        m_ccd = masks.cm_ccd[si][ci[rr], colx]
        m_cell = masks.cm_cell[si][ci[rr], cell[rr], colx]
        m_scat = masks.scat[ci[rr], colx]
        m_set = (masks.cm_set[si][ci[rr], colx] if masks.cm_set is not None
                 else np.zeros_like(m_ccd))  # fmt: skip
        tot = np.add.reduceat(w2, starts)
        stack = np.vstack([np.add.reduceat(w2 * mk, starts) / tot
                           for mk in (m_ccd, m_cell, m_scat, m_set)])  # fmt: skip
        share = np.add.reduceat(w2 * (m_ccd | m_cell | m_scat | m_set), starts) / tot
        peak = np.maximum.reduceat(val, starts)
        pcol = np.minimum.reduceat(np.where(val == peak[seg], colx, big), starts)
        ends = np.concatenate([starts[1:], [idx.size]]) - 1
        W = WIDTHS[w]
        dump_hit = np.zeros(len(starts), bool)
        for db in masks.dump_bins:
            dump_hit |= (pcol >= db - 1) & (pcol <= db + 1 + W // 2)
        sys_ev = (share >= SYS_SHARE) | dump_hit
        cause = np.where(dump_hit, 4,
                         np.where(sys_ev, np.array([1, 2, 3, 5])[stack.argmax(0)], 0))
        parts.append(pd.DataFrame({
            "row": erow.astype(np.int32), "ch": np.int8(c), "sign": np.int8(s),
            "w": np.int8(w), "start": colx[starts].astype(np.int16),
            "end": colx[ends].astype(np.int16), "peak_col": pcol.astype(np.int16),
            "peak": peak.astype(np.float32), "chi2": tot.astype(np.float32),
            "share": share.astype(np.float32), "sys": sys_ev, "cause": cause.astype(np.int8),
        }))  # fmt: skip
        members.append((c, rr, colx, w2, starts, tot, L))
    cols = ["row", "ch", "sign", "w", "start", "end", "peak_col", "peak", "chi2", "share",
            "sys", "cause"]  # fmt: skip
    if not parts:
        return pd.DataFrame(columns=cols)
    # stage 2: per-channel time spans of the stage-1 systematic events
    L = members[0][6]
    span = np.zeros((2, len(WIDTHS), n, L + 1), np.int32)
    # peak SNR of stage-1 systematic events at their peak bin, per channel and box width
    syspk = np.zeros((2, len(WIDTHS), n, L), np.float32)
    for p in parts:
        e = p[p.sys]
        if len(e):
            ch, wi = int(e.ch.iloc[0]), int(e.w.iloc[0])
            r_, pc = e.row.to_numpy(), e.peak_col.to_numpy(np.int64)
            np.add.at(span[ch, wi], (r_, e.start.to_numpy(np.int64)), 1)
            np.add.at(span[ch, wi], (r_, e.end.to_numpy(np.int64) + 1), -1)
            np.maximum.at(syspk[ch, wi], (r_, pc), e.peak.to_numpy(np.float32))
    # [:, w] = union / max over box widths <= w: only a same-or-narrower systematic event can
    # relabel an event (a wide box's run and peak smear a short real dip into a nearby window)
    in_span = np.logical_or.accumulate(np.cumsum(span, axis=3)[..., :L] > 0, axis=1)
    syspk = np.maximum.accumulate(syspk, axis=1)
    for p, (c, rr, colx, w2, starts, tot, _) in zip(parts, members, strict=True):
        wi = int(p.w.iloc[0])
        ov = np.add.reduceat(w2 * in_span[c, wi][rr, colx], starts) / tot
        inner = np.maximum.reduceat(syspk[c, wi][rr, colx], starts) >= p.peak.to_numpy()
        hit = (~p.sys.to_numpy()) & ((ov >= SYS_SHARE) | inner)
        p.loc[hit, "sys"] = True
        p.loc[hit, "cause"] = np.int8(6)
    ev = pd.concat(parts, ignore_index=True)
    assert ev.row.max() < n
    return ev


def summarise_events(ev: pd.DataFrame, n: int, t: np.ndarray) -> pd.DataFrame:
    """Per light curve aggregates (see module docstring)."""
    out = {}
    for lab, sel in (("", np.ones(len(ev), bool)), ("_pdc", (ev.ch == 0).to_numpy())):
        e = ev[sel]
        for kind, m in (("res", ~e.sys.to_numpy()), ("sys", e.sys.to_numpy())):
            ee = e[m]
            pk = np.zeros(n)
            np.maximum.at(pk, ee.row.to_numpy(), ee.peak.to_numpy(np.float64))
            out[f"cmv2_{kind}_snr{lab}"] = pk
    # strongest events (both channels): time, channel/sign/width, cause
    for kind, m in (("res", ~ev.sys.to_numpy()), ("sys", ev.sys.to_numpy())):
        ee = ev[m].sort_values("peak", ascending=False).drop_duplicates("row")
        tt = np.full(n, np.nan)
        lab = np.full(n, "", dtype=object)
        tt[ee.row.to_numpy()] = t[ee.peak_col.to_numpy(np.int64)]
        lab[ee.row.to_numpy()] = [f"{CHANNELS[c]}/{SIGNS[s]}/{WIDTHS[w] // 2}h/{CAUSES[k]}"
                                  for c, s, w, k in zip(ee.ch, ee.sign, ee.w, ee.cause,
                                                        strict=True)]  # fmt: skip
        out[f"cmv2_{kind}_t"] = tt
        out[f"cmv2_{kind}_ev"] = lab
    w0 = ev[ev.w == 0]
    tot = np.bincount(w0.row, weights=w0.chi2.astype(np.float64), minlength=n)
    sysw = np.bincount(w0.row, weights=(w0.chi2 * w0.sys).astype(np.float64), minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["cmv2_sys_share"] = np.where(tot > 0, sysw / tot, 0.0)
    out["cmv2_n_ev"] = np.bincount(ev.row, minlength=n)
    out["cmv2_n_sys"] = np.bincount(ev.row, weights=ev.sys.astype(float), minlength=n).astype(
        np.int64)
    return pd.DataFrame(out)


def classify_dir(d, grid: Grid, meta: pd.DataFrame, cells: pd.DataFrame | None,
                 masks: SysMasks, batch_rows: int = 2000, log=None, keep_events=None):
    """Per light curve veto features for every light curve under ``d``.

    ``keep_events``: optional set of ids (tic or inj_id) whose full event lists are returned too.
    """  # fmt: skip
    key = "inj_id" if "inj_id" in meta.columns else "tic"
    m = _lookup(meta, cells, key)
    frames, kept = [], []
    n_done = 0
    for b in iter_batches(d, grid, batch_rows=batch_rows,
                          columns=("flux", "sap_flux")):  # fmt: skip
        ids = _ids(b, key)
        mm = m.loc[ids]
        ci = ccd_index(mm.camera.to_numpy(), mm.ccd.to_numpy())
        cell = mm.cell.to_numpy()
        ev = events_batch(batch_snr(b), ci, cell, masks)
        df = summarise_events(ev, len(ci), grid.time)
        df.insert(0, "tic", b["tic"])
        df.insert(1, "inj_id", b["inj_id"] if b["inj_id"] is not None else None)
        frames.append(df)
        if keep_events:
            ids_arr = np.asarray(ids, dtype=object)
            want = np.array([i in keep_events for i in ids_arr])
            if want.any():
                e = ev[want[ev.row.to_numpy()]].copy()
                e["id"] = ids_arr[e.row.to_numpy()]
                e["t"] = grid.time[e.peak_col.to_numpy(np.int64)]
                kept.append(e)
        n_done += len(ci)
        if log:
            log(f"    events {b['shard_file']}: {n_done} light curves")
    out = pd.concat(frames, ignore_index=True)
    return out, (pd.concat(kept, ignore_index=True) if kept else None)


def veto_flag(df: pd.DataFrame, t_sys: float, t_res: float, pdc_only: bool = False) -> np.ndarray:
    """``sys_snr >= t_sys and res_snr < t_res`` (both channels, or PDCSAP only)."""
    sfx = "_pdc" if pdc_only else ""
    return ((df[f"cmv2_sys_snr{sfx}"].to_numpy() >= t_sys)
            & (df[f"cmv2_res_snr{sfx}"].to_numpy() < t_res))  # fmt: skip
