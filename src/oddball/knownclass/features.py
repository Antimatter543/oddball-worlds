"""Per-(tic, sector) feature table for the known-class classifier.

**Contract (pluggable):** a feature table is any DataFrame with key columns ``tic, sector,
inj_id`` (``inj_id`` null for real light curves) plus numeric feature columns (NaN allowed).
The classifier uses every non-key column, so the ML embeddings drop in later as another table
joined on the keys (``join_tables``).

This module builds the first table, ``kcf_v1``, from the light curves in one pass:

* the classical features (``oddball.score.classical.channel_features``) for PDCSAP and SAP,
  without ``dip_t`` (a time, not a property), ``sigma_w`` as log10;
* periodogram features on the gap-zeroed PDCSAP flux (FFT, 0.04-23 c/d): peak period, peak
  power fraction and amplitude, high-/low-frequency power fractions, power at half and twice the
  peak frequency (eclipsing binaries put the peak at P_orb/2);
* the folded profile at P and 2P (peak period and twice it): variance explained, the depth
  ratio of the two deepest minima at 2P (primary vs secondary eclipse), and profile skewness
  measures (narrow eclipses vs sinusoids vs sawtooth pulsators);
* dip-event counts on the detrended residual: separate events with SNR > 5, events at least
  half as deep as the deepest, and the irregularity of their spacing (periodic eclipses vs
  single or aperiodic dips; robust to events lost in data gaps);
* the SAP low-frequency power fraction and peak period (PDC removes slow trends, LPVs keep them);
* stellar context: Tmag, log10 Teff, log10 radius (TIC).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from oddball.score.classical import _rowmean, _rowvar, channel_features, nanquantile_last, smooth3
from oddball.tess.arrays import Grid, iter_batches

KEYS = ("tic", "sector", "inj_id")
PG_FRANGE = (0.04, 23.0)  # cycles/day (P = 0.043 - 25 d); Nyquist of 30-min bins is 24 c/d
NFFT_PG = 8192
HF_CUT = 5.0  # c/d: delta Scuti territory
LF_CUT = 0.3  # c/d: slower than ~3.3 d
NPH = 40
CLASSICAL_DROP = ("dip_t",)


def _shift(m: np.ndarray, k: int, fill: bool = False) -> np.ndarray:
    """Shift along axis 1 by ``k`` (positive: to the right), filling with ``fill``."""
    out = np.full_like(m, fill)
    if k > 0:
        out[:, k:] = m[:, :-k]
    elif k < 0:
        out[:, :k] = m[:, -k:]
    else:
        out[:] = m
    return out


def close_gaps(m: np.ndarray, gap: int = 2) -> np.ndarray:
    """Binary closing along axis 1: fill runs of <= ``gap`` False bins between True bins."""
    r = (gap + 1) // 2
    dil = m.copy()
    for k in range(1, r + 1):
        dil |= _shift(m, k) | _shift(m, -k)
    ero = dil.copy()
    for k in range(1, r + 1):
        ero &= _shift(dil, k, True) & _shift(dil, -k, True)
    return ero | m


def event_stats(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Number of separate True runs per row, and the irregularity of their spacing.

    Irregularity: the mean distance of each interval between run starts from the nearest
    integer multiple of the row's median interval (0 = strictly periodic even when events fall
    in data gaps, ~0.25 = random). NaN with fewer than three events.
    """
    n = m.shape[0]
    starts = m & ~_shift(m, 1)
    count = starts.sum(1).astype(np.float64)
    r, c = np.nonzero(starts)  # row-major: sorted by row, then column
    same = r[1:] == r[:-1]
    d = (c[1:] - c[:-1]).astype(np.float64)[same]
    rr = r[1:][same]
    irr = np.full(n, np.nan)
    if len(d):
        g = pd.DataFrame({"r": rr, "d": d})
        med = g.groupby("r").d.transform("median").to_numpy()
        k = d / med
        g["dev"] = np.abs(k - np.round(k))
        agg = g.groupby("r").agg(dev=("dev", "mean"), n=("d", "size"))
        agg = agg[agg.n >= 2]
        irr[agg.index.to_numpy()] = agg.dev.to_numpy()
    return count, irr


def _fold(x0: np.ndarray, v: np.ndarray, t: np.ndarray, period: np.ndarray, nph: int):
    n = x0.shape[0]
    phase = np.mod((t[None, :] - t[0]) / period[:, None], 1.0)
    pb = np.minimum((phase * nph).astype(np.int64), nph - 1)
    key = (np.arange(n)[:, None] * nph + pb)[v]
    sums = np.bincount(key, weights=x0[v], minlength=n * nph).reshape(n, nph)
    cnts = np.bincount(key, minlength=n * nph).reshape(n, nph)
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    resid = np.where(v, x0 - np.take_along_axis(np.nan_to_num(prof), pb, axis=1), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = 1.0 - _rowvar(resid) / _rowvar(np.where(v, x0, np.nan))
    return prof, r2


def _profile_shape(prof: np.ndarray):
    """Second/first minimum depth ratio (>= 1/4 cycle apart), (max-med)/(max-min), low frac."""
    _n, k = prof.shape
    p = np.where(np.isfinite(prof), prof, np.nan)
    med = nanquantile_last(p, 0.5)[0]
    pmin = np.nanmin(np.where(np.isfinite(p), p, np.inf), axis=1)
    pmax = np.nanmax(np.where(np.isfinite(p), p, -np.inf), axis=1)
    i1 = np.argmin(np.where(np.isfinite(p), p, np.inf), axis=1)
    j = np.arange(k)[None, :]
    dist = np.abs(j - i1[:, None])
    dist = np.minimum(dist, k - dist)
    far = np.where((dist >= k // 4) & np.isfinite(p), p, np.inf)
    m2 = far.min(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (med - m2) / (med - pmin)
        skew = (pmax - med) / (pmax - pmin)
        low = (np.where(np.isfinite(p), p, np.inf) < ((pmax + pmin) / 2)[:, None]).sum(1) / (
            np.isfinite(p).sum(1)
        )
    bad = ~np.isfinite(pmin) | ~np.isfinite(pmax) | (pmax <= pmin)
    return (np.where(bad, np.nan, np.clip(ratio, -1, 1)), np.where(bad, np.nan, skew),
            np.where(bad, np.nan, low))  # fmt: skip


def periodogram_features(x: np.ndarray, t: np.ndarray, prefix: str = "pg",
                         full: bool = True) -> dict:  # fmt: skip
    """FFT power-spectrum + fold features of normalised flux ``x`` (n, L), NaN in gaps."""
    n, L = x.shape
    v = np.isfinite(x)
    nv = v.sum(1)
    mu = _rowmean(x)
    x0 = np.where(v, x - mu[:, None], 0.0)
    F = np.fft.rfft(x0.astype(np.float32), n=NFFT_PG, axis=1)
    p = (F.real**2 + F.imag**2).astype(np.float32)
    del F
    dt = float(t[1] - t[0])
    freqs = np.fft.rfftfreq(NFFT_PG, d=dt)
    sel = (freqs >= PG_FRANGE[0]) & (freqs <= PG_FRANGE[1])
    ps, fs = p[:, sel], freqs[sel]
    del p
    tot = ps.sum(1).astype(np.float64)
    k = np.argmax(ps, axis=1)
    fpk = fs[k]
    w = max(1, round(NFFT_PG / L))  # oversampling: bins per natural resolution element
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(ps, axis=1, dtype=np.float64)], axis=1)
    lo, hi = np.clip(k - w, 0, len(fs)), np.clip(k + w + 1, 0, len(fs))
    pk_pow = cs[np.arange(n), hi] - cs[np.arange(n), lo]
    pmax = ps[np.arange(n), k].astype(np.float64)

    def pow_near(f, spec=ps):
        i = np.searchsorted(fs, f)
        out = np.full(n, np.nan)
        ok = (f >= fs[0]) & (f <= fs[-1])
        for o in range(-w, w + 1):
            ii = np.clip(i + o, 0, len(fs) - 1)
            out = np.fmax(out, np.where(ok, spec[np.arange(n), ii], np.nan))
        return out

    with np.errstate(invalid="ignore", divide="ignore"):
        out = {
            f"{prefix}_logP": np.log10(1.0 / fpk),
            f"{prefix}_peak_frac": pk_pow / tot,
            f"{prefix}_lf_frac": ps[:, fs < LF_CUT].sum(1) / tot,
        }
        if full:
            out[f"{prefix}_log_amp"] = np.log10(2.0 * np.sqrt(pmax) / np.maximum(nv, 1))
            out[f"{prefix}_hf_frac"] = ps[:, fs > HF_CUT].sum(1) / tot
            out[f"{prefix}_harm2"] = np.log10(pow_near(2 * fpk) / pmax)
            out[f"{prefix}_sub2"] = np.log10(pow_near(fpk / 2) / pmax)
    del ps, cs
    if full:
        x0n = np.where(v, x0, 0.0)
        _, r2_1 = _fold(x0n, v, t, 1.0 / fpk, NPH)
        prof2, r2_2 = _fold(x0n, v, t, 2.0 / fpk, NPH)
        ratio, skew, low = _profile_shape(prof2)
        out.update({f"{prefix}_r2_p": r2_1, f"{prefix}_r2_2p": r2_2, f"{prefix}_min2_ratio": ratio,
                    f"{prefix}_prof_skew": skew, f"{prefix}_prof_low": low})  # fmt: skip
    bad = nv < 50
    return {kk: np.where(bad, np.nan, vv).astype(np.float64) for kk, vv in out.items()}


def dip_event_features(resid: np.ndarray, snr4: np.ndarray, dip_depth: np.ndarray) -> dict:
    sn = np.nan_to_num(snr4, nan=0.0)
    n5, irr5 = event_stats(close_gaps(sn > 5.0))
    y = smooth3(resid)
    half = np.nan_to_num(y, nan=np.inf) < (-0.5 * np.nan_to_num(dip_depth, nan=np.inf))[:, None]
    nh, irrh = event_stats(close_gaps(half & (sn > 3.0)))
    return {"ev_n_snr5": n5, "ev_n_half": nh, "ev_irreg_half": irrh, "ev_irreg_snr5": irr5}


def batch_features(b: dict, mm: pd.DataFrame, grid: Grid) -> pd.DataFrame:
    """Features for one ``iter_batches`` batch; ``mm``: its metadata rows (aligned)."""
    t = grid.time
    crowd = mm.crowdsap.to_numpy(np.float64)
    crowd = np.where(np.isfinite(crowd) & (crowd > 0) & (crowd <= 1), crowd, 1.0)
    pdc = channel_features(b["flux"], b["flux_err"], t, grid.bin_days)
    sap = channel_features(b["sap_flux"], b["flux_err"] * crowd[:, None], t, grid.bin_days)
    cols = {}
    for ch, o in (("pdc", pdc), ("sap", sap)):
        for k, v in o.feats.items():
            if k in CLASSICAL_DROP:
                continue
            if k == "sigma_w":
                with np.errstate(invalid="ignore", divide="ignore"):
                    v = np.log10(v)
            cols[f"{ch}_{k}"] = v
    with np.errstate(invalid="ignore", divide="ignore"):
        q05, q95 = nanquantile_last(pdc.x, [0.05, 0.95])
        cols["pdc_log_amp9005"] = np.log10(q95 - q05)
    cols.update(periodogram_features(pdc.x, t, "pg"))
    cols.update(periodogram_features(sap.x, t, "pgsap", full=False))
    cols.update(dip_event_features(pdc.resid, pdc.snr4, pdc.feats["dip_depth"]))
    with np.errstate(invalid="ignore", divide="ignore"):
        cols["ctx_tmag"] = mm.tmag.to_numpy(np.float64)
        cols["ctx_logteff"] = np.log10(mm.teff.to_numpy(np.float64))
        cols["ctx_logradius"] = np.log10(mm.radius.to_numpy(np.float64))
    df = pd.DataFrame({"tic": b["tic"], "inj_id": b["inj_id"] if b["inj_id"] is not None else None})
    feat = pd.DataFrame(cols).replace([np.inf, -np.inf], np.nan).astype(np.float32)
    return pd.concat([df, feat], axis=1)


def featurize_dir(d, grid: Grid, meta: pd.DataFrame, sector: int, batch_rows: int = 1000,
                  log=None) -> pd.DataFrame:  # fmt: skip
    """The ``kcf_v1`` feature table for every light curve under shard directory ``d``."""
    key = "inj_id" if "inj_id" in meta.columns else "tic"
    m = meta.set_index(key)
    parts, n = [], 0
    for b in iter_batches(d, grid, batch_rows=batch_rows,
                          columns=("flux", "flux_err", "sap_flux")):  # fmt: skip
        ids = b["inj_id"] if key == "inj_id" else b["tic"]
        parts.append(batch_features(b, m.loc[ids], grid))
        n += len(ids)
        if log:
            log(f"    {b['shard_file']}: {n} light curves")
    out = pd.concat(parts, ignore_index=True)
    out.insert(1, "sector", np.int16(sector))
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in KEYS and c not in ("split", "label", "y")]


def join_tables(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Join several feature tables on (tic, sector, inj_id); column names must not collide."""
    keys = list(KEYS)
    out = tables[0].copy()
    out["inj_id"] = out["inj_id"].astype(object)
    for t in tables[1:]:
        t = t.copy()
        t["inj_id"] = t["inj_id"].astype(object)
        clash = set(feature_columns(out)) & set(feature_columns(t))
        if clash:
            raise ValueError(f"feature column clash: {sorted(clash)[:5]}")
        out = out.merge(t, on=keys, how="left", validate="one_to_one")
    return out
