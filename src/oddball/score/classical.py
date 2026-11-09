"""Classical baseline anomaly scorer (docs/ML_DESIGN.md §4.4, §5; PLAN.md 1d, 1f).

Everything is vectorised over a batch of light curves on the global bin grid
(``oddball.tess.arrays``), so a sector streams through in ~2k-row batches.

Per channel (PDCSAP ``pdc`` and SAP ``sap``), on the median-normalised flux ``x = f/med - 1``:

* ``skew``      robust (quantile) skewness ``(q99 + q01 - 2 q50) / (q99 - q01)``.
* ``M``         Cody et al. (2014) flux asymmetry ``(<d10%> - median) / rms``, where ``<d10%>``
                is the mean of the top and bottom deciles. Flux units, so dippers are
                *negative* (Cody's magnitudes flip the sign).
* ``Q``         quasi-periodicity, simplified from Cody+2014: the dominant period is the
                peak of a gap-zeroed FFT power spectrum over 0.25-10 d (not a Lomb-Scargle,
                and no smoothing before folding); the light curve is folded on it and a
                20-phase-bin mean profile subtracted. ``Q = (var(resid) - s2) /
                (var(x) - s2)`` with ``s2`` the white-noise variance from the point-to-point
                scatter, clipped to [0, 1]. 0 = periodic, 1 = aperiodic. Only meaningful for
                variable stars, so its sub-score is ``Q * var_frac`` where ``var_frac`` is the
                fraction of the variance above white noise.
* ``dip_*``     the largest single dip. ``x`` is detrended by a running median (6-h block
                medians, 13-block = 3.25-d window), then a box search over widths
                2 h - 2.7 d gives the most significant dip (``dip_snr``, against the
                white-noise sigma). At that dip, on a 3-bin smoothed residual: ``dip_depth``
                (deepest point), ``dip_dur_d`` (width of the run below half depth) and
                ``dip_asym = (egress - ingress) / (egress + ingress)`` measured from the deepest
                bin to the half-depth crossings (> 0: sharp ingress, slow egress).
                Detrending caveat (failure catalogue #6): dips longer than ~2 d are partly
                absorbed by the baseline; ``M``, ``skew`` and ``xs_var`` see them undetrended.
* ``bump_snr``  the same box search for brightenings (outbursts).
* ``xs_var``    log10 of variance over the mean per-bin error^2 (SAP errors are the PDCSAP
                errors times CROWDSAP, since normalised SAP = CROWDSAP x normalised PDCSAP).
* ``frac_neg3`` fraction of bins with detrended residual below -3 sigma.

Sub-scores (higher = weirder) are listed in ``SUBSCORES``. Each is turned into a percentile
**within its stratum** (Tmag bin x camera/CCD) against the real light curves of the sector;
``score`` is the percentile, again within stratum, of the mean of those sub-percentiles
(rank-averaging).

Common-mode systematics (v0): a bin of a light curve is *anomalous* when the centred 2-h box
of the detrended residual is below -3 sigma. For each channel x CCD x time bin, the map is the
fraction of stars with that bin anomalous. A bin is *common-mode* when that fraction is far
above the CCD's typical level (``cm_threshold``). Per light curve, ``cm_frac`` is the share
of its anomalous bins that are common-mode, ``dump_frac`` the share within one bin of its own
momentum-dump flags (QUALITY bit 32), and ``sys_frac`` the share that is either one. Shares are
weighted by each bin's SNR^2, so a few strong dips outweigh many marginal systematic bins.
``sys_flag = sys_frac >= 0.5 and n_anom >= 4``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from oddball.common.strata import ccd_index, stratum_id
from oddball.tess.arrays import Grid, iter_batches

CHANNELS = ("pdc", "sap")
BOX_WIDTHS = (4, 8, 16, 32, 64, 128)  # bins (30 min): 2 h .. 2.7 d
BLOCK = 12  # 6-h blocks for the running-median baseline
BASE_WIN = 13  # blocks (3.25 d)
NPHASE = 20
FREQ_RANGE = (0.1, 4.0)  # cycles/day: periods 0.25-10 d
NFFT = 4096
ANOM_SNR = 3.0
FEATURES = ("skew", "M", "Q", "var_frac", "sigma_w", "dip_snr", "dip_depth", "dip_dur_d",
            "dip_asym", "dip_t", "bump_snr", "xs_var", "frac_neg3")  # fmt: skip
SUBSCORES = ("skew_abs", "m_abs", "q_aper", "dip_snr", "dip_depth", "dip_asym", "dip_dur",
             "bump_snr", "xs_var", "frac_neg3")  # fmt: skip


# --------------------------------------------------------------------------------------------
# NaN-aware helpers on the last axis (np.nan* reductions loop per row and are slow)


def nanquantile_last(a: np.ndarray, qs) -> list[np.ndarray]:
    s = np.sort(a, axis=-1)  # NaN sorts last
    n = np.isfinite(a).sum(axis=-1)
    out = []
    for q in np.atleast_1d(qs):
        pos = np.clip((n - 1) * q, 0, None)
        lo = np.floor(pos).astype(np.int64)
        hi = np.minimum(lo + 1, np.maximum(n - 1, 0))
        w = pos - lo
        vlo = np.take_along_axis(s, lo[..., None], axis=-1)[..., 0]
        vhi = np.take_along_axis(s, hi[..., None], axis=-1)[..., 0]
        out.append(np.where(n > 0, vlo * (1 - w) + vhi * w, np.nan))
    return out


def _fill_nearest(b: np.ndarray) -> np.ndarray:
    """Fill NaNs along axis 1 with the previous valid value, then the next one."""
    k = b.shape[1]
    ok = np.isfinite(b)
    idx = np.where(ok, np.arange(k)[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    f = np.take_along_axis(b, idx, axis=1)
    f[~np.isfinite(f)] = np.nan
    first_ok = np.where(ok.any(1), ok.argmax(1), 0)
    lead = np.arange(k)[None, :] < first_ok[:, None]
    f[lead] = np.take_along_axis(b, first_ok[:, None], axis=1).repeat(k, 1)[lead]
    return f


def running_baseline(x: np.ndarray) -> np.ndarray:
    n, L = x.shape
    nb = -(-L // BLOCK)
    xp = np.full((n, nb * BLOCK), np.nan)
    xp[:, :L] = x
    blocks = xp.reshape(n, nb, BLOCK)
    bmed = nanquantile_last(blocks, 0.5)[0]
    bmed[np.isfinite(blocks).sum(-1) < BLOCK // 3] = np.nan
    half = BASE_WIN // 2
    padded = np.pad(bmed, ((0, 0), (half, half)), constant_values=np.nan)
    win = np.lib.stride_tricks.sliding_window_view(padded, BASE_WIN, axis=1)
    base_b = nanquantile_last(win, 0.5)[0]
    base_b[np.isfinite(win).sum(-1) < 3] = np.nan
    base_b = _fill_nearest(base_b)
    pos = np.clip((np.arange(L) - (BLOCK - 1) / 2) / BLOCK, 0, nb - 1)
    il = np.floor(pos).astype(np.int64)
    ir = np.minimum(il + 1, nb - 1)
    w = pos - il
    return base_b[:, il] * (1 - w) + base_b[:, ir] * w


def _box_sums(r: np.ndarray, W: int):
    v = np.isfinite(r)
    cs = np.concatenate([np.zeros((r.shape[0], 1)), np.cumsum(np.where(v, r, 0.0), 1)], 1)
    cn = np.concatenate([np.zeros((r.shape[0], 1)), np.cumsum(v, 1)], 1)
    return cs[:, W:] - cs[:, :-W], cn[:, W:] - cn[:, :-W]


def box_search(r: np.ndarray, sigma: np.ndarray, sign: int):
    """Best box over BOX_WIDTHS: returns (snr, start, width) per row; sign -1 = dips."""
    n = r.shape[0]
    best = np.full(n, -np.inf)
    bstart = np.zeros(n, np.int64)
    bw = np.full(n, BOX_WIDTHS[0], np.int64)
    for W in BOX_WIDTHS:
        if W >= r.shape[1]:
            break
        s, c = _box_sums(r, W)
        with np.errstate(invalid="ignore", divide="ignore"):
            snr = sign * s / np.sqrt(np.maximum(c, 1)) / sigma[:, None]
        snr[c < np.ceil(0.6 * W)] = -np.inf
        i = np.argmax(snr, axis=1)
        v = snr[np.arange(n), i]
        better = v > best
        best[better], bstart[better], bw[better] = v[better], i[better], W
    best[~np.isfinite(best)] = np.nan
    return best, bstart, bw


def centred_box_snr(r: np.ndarray, sigma: np.ndarray, W: int = 4, min_count: int = 3):
    s, c = _box_sums(r, W)
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = -s / np.sqrt(np.maximum(c, 1)) / sigma[:, None]
    snr[c < min_count] = np.nan
    out = np.full(r.shape, np.nan)
    off = W // 2
    out[:, off : off + snr.shape[1]] = snr
    return out


def smooth3(r: np.ndarray) -> np.ndarray:
    s, c = _box_sums(r, 3)
    out = np.full(r.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[:, 1:-1] = np.where(c >= 2, s / np.maximum(c, 1), np.nan)
    return out


def dip_shape(r: np.ndarray, start: np.ndarray, width: np.ndarray, bin_days: float):
    """Deepest point, half-depth duration and asymmetry around each row's best box."""
    n, L = r.shape
    y = smooth3(r)
    j = np.arange(L)[None, :]
    lo = (start - width // 2)[:, None]
    hi = (start + width + width // 2)[:, None]
    yw = np.where((j >= lo) & (j < hi) & np.isfinite(y), y, np.inf)
    imin = np.argmin(yw, axis=1)
    depth = -yw[np.arange(n), imin]
    depth[~np.isfinite(depth)] = np.nan
    below = np.nan_to_num(y, nan=np.inf) < (-0.5 * depth)[:, None]
    starts = below & ~np.concatenate([np.zeros((n, 1), bool), below[:, :-1]], axis=1)
    label = np.cumsum(starts, axis=1)
    run = below & (label == label[np.arange(n), imin][:, None])
    has = run.any(1)
    left = np.where(has, run.argmax(1), imin)
    right = np.where(has, L - 1 - run[:, ::-1].argmax(1), imin)
    ingress = imin - left + 0.5
    egress = right - imin + 0.5
    dur = (right - left + 1) * bin_days
    asym = (egress - ingress) / (egress + ingress)
    bad = ~np.isfinite(depth) | (depth <= 0)
    return (np.where(bad, np.nan, depth), np.where(bad, np.nan, dur),
            np.where(bad, np.nan, asym), imin)  # fmt: skip


def periodic_q(x: np.ndarray, sig2: np.ndarray, t: np.ndarray):
    n = x.shape[0]
    v = np.isfinite(x)
    mu = _rowmean(x)
    x0 = np.where(v, x - mu[:, None], 0.0)
    p = np.abs(np.fft.rfft(x0, n=NFFT, axis=1)) ** 2
    dt = float(t[1] - t[0])
    freqs = np.fft.rfftfreq(NFFT, d=dt)
    sel = (freqs >= FREQ_RANGE[0]) & (freqs <= FREQ_RANGE[1])
    fstar = freqs[sel][np.argmax(p[:, sel], axis=1)]
    phase = np.mod((t[None, :] - t[0]) * fstar[:, None], 1.0)
    pb = np.minimum((phase * NPHASE).astype(np.int64), NPHASE - 1)
    key = (np.arange(n)[:, None] * NPHASE + pb)[v]
    sums = np.bincount(key, weights=x0[v], minlength=n * NPHASE).reshape(n, NPHASE)
    cnts = np.bincount(key, minlength=n * NPHASE).reshape(n, NPHASE)
    prof = sums / np.maximum(cnts, 1)
    resid = np.where(v, x0 - np.take_along_axis(prof, pb, axis=1), np.nan)
    var_x = _rowvar(x)
    var_r = _rowvar(resid)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = (var_r - sig2) / (var_x - sig2)
        var_frac = np.clip(1.0 - sig2 / var_x, 0.0, 1.0)
    q = np.where(var_x > sig2, np.clip(q, 0.0, 1.0), np.nan)
    return q, var_frac, 1.0 / fstar


def _rowmean(x):
    v = np.isfinite(x)
    return np.where(v, x, 0).sum(1) / np.maximum(v.sum(1), 1)


def _rowvar(x):
    v = np.isfinite(x)
    m = _rowmean(x)
    return (np.where(v, x - m[:, None], 0) ** 2).sum(1) / np.maximum(v.sum(1) - 1, 1)


@dataclass
class ChannelOut:
    feats: dict
    anom: np.ndarray  # (n, L) bool: anomalous bins
    valid: np.ndarray  # (n, L) bool
    snr4: np.ndarray  # (n, L) 2-h box dip SNR of the detrended residual


def channel_features(f: np.ndarray, err: np.ndarray, t: np.ndarray, bin_days: float):
    f = f.astype(np.float64)
    med = nanquantile_last(f, 0.5)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        x = f / med[:, None] - 1.0
        e = err.astype(np.float64) / med[:, None]
    x[~np.isfinite(x)] = np.nan
    valid = np.isfinite(x)

    d = np.diff(x, axis=1)
    dmed = nanquantile_last(d, 0.5)[0]
    sigma = 1.4826 * nanquantile_last(np.abs(d - dmed[:, None]), 0.5)[0] / np.sqrt(2.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan)
    sig_safe = np.nan_to_num(sigma, nan=1.0)

    q01, q10, q50, q90, q99 = nanquantile_last(x, [0.01, 0.1, 0.5, 0.9, 0.99])
    with np.errstate(invalid="ignore", divide="ignore"):
        skew = (q99 + q01 - 2 * q50) / (q99 - q01)
        tails = valid & ((x <= q10[:, None]) | (x >= q90[:, None]))
        d10 = np.where(tails, x, 0).sum(1) / np.maximum(tails.sum(1), 1)
        rms = np.sqrt(_rowvar(x))
        M = (d10 - q50) / rms

    Q, var_frac, _period = periodic_q(x, sigma**2, t)

    r = x - running_baseline(x)
    dip_snr, dstart, dwidth = box_search(r, sig_safe, -1)
    bump_snr, _, _ = box_search(r, sig_safe, +1)
    dip_depth, dip_dur, dip_asym, imin = dip_shape(r, dstart, dwidth, bin_days)
    with np.errstate(invalid="ignore", divide="ignore"):
        xs_var = np.log10(_rowvar(x) / _rowmean(np.where(valid, e, np.nan) ** 2))
        frac_neg3 = (np.nan_to_num(r, nan=0) < -3 * sig_safe[:, None]).sum(1) / np.maximum(
            valid.sum(1), 1
        )
    snr4 = centred_box_snr(r, sig_safe)
    anom = np.nan_to_num(snr4, nan=0.0) > ANOM_SNR
    nan_row = ~np.isfinite(sigma)
    anom[nan_row] = False
    feats = {
        "skew": skew, "M": M, "Q": Q, "var_frac": var_frac, "sigma_w": sigma,
        "dip_snr": dip_snr, "dip_depth": dip_depth, "dip_dur_d": dip_dur, "dip_asym": dip_asym,
        "dip_t": t[imin], "bump_snr": bump_snr, "xs_var": xs_var, "frac_neg3": frac_neg3,
    }  # fmt: skip
    feats = {k: np.where(nan_row, np.nan, v).astype(np.float64) for k, v in feats.items()}
    return ChannelOut(feats, anom, valid, snr4)


def subscores(feat: pd.DataFrame) -> pd.DataFrame:
    """Higher = weirder. NaN means 'no evidence' and is ranked lowest by the calibrator."""
    out = {}
    for ch in CHANNELS:
        def g(k, ch=ch):
            return feat[f"{ch}_{k}"].to_numpy()

        gate = np.clip(np.nan_to_num(g("dip_snr")) / 10.0, 0.0, 1.0)
        out[f"{ch}_skew_abs"] = np.abs(g("skew"))
        out[f"{ch}_m_abs"] = np.abs(g("M"))
        out[f"{ch}_q_aper"] = np.nan_to_num(g("Q"), nan=0.0) * np.nan_to_num(g("var_frac"))
        out[f"{ch}_dip_snr"] = g("dip_snr")
        out[f"{ch}_dip_depth"] = g("dip_depth")
        out[f"{ch}_dip_asym"] = np.abs(g("dip_asym")) * gate
        out[f"{ch}_dip_dur"] = np.log10(g("dip_dur_d")) + 2.0 * gate  # long *and* significant
        out[f"{ch}_bump_snr"] = g("bump_snr")
        out[f"{ch}_xs_var"] = g("xs_var")
        out[f"{ch}_frac_neg3"] = g("frac_neg3")
    return pd.DataFrame(out, index=feat.index)


# --------------------------------------------------------------------------------------------
# Directory pass


@dataclass
class DirResult:
    feats: pd.DataFrame  # one row per light curve: tic, inj_id, features
    # channel -> sparse anomalous bins: rows (LC index), cols (grid bin), w (SNR^2),
    # dump (bin within 1 bin of the LC's own QUALITY bit-32 momentum-dump flag)
    anom: dict
    ccd: np.ndarray  # (n,) ccd index 0..15
    map_num: np.ndarray  # (2, 16, L) anomalous counts
    map_den: np.ndarray  # (2, 16, L) valid counts


def featurize_dir(
    d, grid: Grid, meta: pd.DataFrame, batch_rows: int = 2000, log=None,
    mask: np.ndarray | None = None,
):
    """Features + common-mode counts for every light curve under shard directory ``d``.

    ``meta``: metadata with tic, camera, ccd, crowdsap (and inj_id for injection sets).
    ``mask``: optional (2, 16, L) bool of (channel, CCD, bin) to remove before computing
    features (the systematics filter: common-mode bins); bins within one bin of each light
    curve's own momentum-dump flag are removed too when a mask is given.
    """  # fmt: skip
    key = "inj_id" if "inj_id" in meta.columns else "tic"
    m = meta.set_index(key)
    t = grid.time
    L = grid.length
    rows, ccds = [], []
    sparse = {c: {"rows": [], "cols": [], "w": [], "dump": []} for c in CHANNELS}
    num = np.zeros((2, 16, L))
    den = np.zeros((2, 16, L))
    nb = 0
    for b in iter_batches(d, grid, batch_rows=batch_rows,
                          columns=("flux", "flux_err", "qflags", "sap_flux")):  # fmt: skip
        ids = b["inj_id"] if (key == "inj_id") else b["tic"]
        mm = m.loc[ids]
        crowd = mm.crowdsap.to_numpy(np.float64)
        crowd = np.where(np.isfinite(crowd) & (crowd > 0) & (crowd <= 1), crowd, 1.0)
        ci = ccd_index(mm.camera.to_numpy(), mm.ccd.to_numpy())
        dump = (b["qflags"] & 32) > 0
        dadj = dump.copy()
        dadj[:, 1:] |= dump[:, :-1]
        dadj[:, :-1] |= dump[:, 1:]
        fp, fs = b["flux"], b["sap_flux"]
        if mask is not None:
            fp = np.where(mask[0][ci] | dadj, np.nan, fp)
            fs = np.where(mask[1][ci] | dadj, np.nan, fs)
        pdc = channel_features(fp, b["flux_err"], t, grid.bin_days)
        sap = channel_features(fs, b["flux_err"] * crowd[:, None], t, grid.bin_days)
        df = pd.DataFrame({"tic": b["tic"]})
        df["inj_id"] = b["inj_id"] if b["inj_id"] is not None else None
        for ch, o in (("pdc", pdc), ("sap", sap)):
            for k, v in o.feats.items():
                df[f"{ch}_{k}"] = v
        df["n_valid"] = pdc.valid.sum(1)
        rows.append(df)
        for c, (ch, o) in enumerate((("pdc", pdc), ("sap", sap))):
            rr, cc = np.nonzero(o.anom)
            sp = sparse[ch]
            sp["rows"].append((rr + nb).astype(np.int32))
            sp["cols"].append(cc.astype(np.int16))
            sp["w"].append((o.snr4[rr, cc] ** 2).astype(np.float32))
            sp["dump"].append(dadj[rr, cc])
            for k in np.unique(ci):
                s = ci == k
                num[c, k] += o.anom[s].sum(0)
                den[c, k] += o.valid[s].sum(0)
        ccds.append(ci)
        nb += len(df)
        if log:
            log(f"    {b['shard_file']}: {nb} light curves")
    anom = {c: {k: np.concatenate(v) for k, v in sparse[c].items()} for c in CHANNELS}
    return DirResult(pd.concat(rows, ignore_index=True), anom, np.concatenate(ccds), num, den)


def cm_threshold(frac: np.ndarray, den: np.ndarray, k: float = 8.0, floor: float = 0.01):
    """Per (channel, CCD) threshold on the anomalous fraction: median + k * robust sigma."""
    th = np.zeros(frac.shape[:2])
    for c in range(frac.shape[0]):
        for i in range(frac.shape[1]):
            v = frac[c, i][den[c, i] >= 20]
            if v.size == 0:
                th[c, i] = 1.0
                continue
            med = np.median(v)
            mad = 1.4826 * np.median(np.abs(v - med))
            th[c, i] = max(floor, med + k * mad)
    return th


def common_mode_bins(num, den, thresh, min_count: int = 5, min_frac: float = 0.0):
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(den > 0, num / den, 0.0)
    return (frac > np.maximum(thresh, min_frac)[:, :, None]) & (num >= min_count)


def systematics_features(res: DirResult, cm: np.ndarray) -> pd.DataFrame:
    """Per light curve: the share of its dip evidence that sits in systematic bins.

    Each anomalous bin is weighted by its SNR^2 (its contribution to the dip chi^2), so a
    strong astrophysical dip is not outvoted by a host's many weak systematic bins.
    ``cm_frac``: share in common-mode bins (``cm``: (2, 16, L) bool); ``dump_frac``: share
    within one bin of the light curve's own momentum-dump flags; ``sys_frac``: either.
    """
    n = len(res.ccd)
    tot, w_cm, w_dp, w_sys, n_a = (np.zeros(n) for _ in range(5))
    for c, ch in enumerate(CHANNELS):
        sp = res.anom[ch]
        r, w = sp["rows"], sp["w"].astype(np.float64)
        is_cm = cm[c][res.ccd[r], sp["cols"]]
        tot += np.bincount(r, weights=w, minlength=n)
        w_cm += np.bincount(r, weights=w * is_cm, minlength=n)
        w_dp += np.bincount(r, weights=w * sp["dump"], minlength=n)
        w_sys += np.bincount(r, weights=w * (is_cm | sp["dump"]), minlength=n)
        n_a += np.bincount(r, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = pd.DataFrame({
            "n_anom": n_a.astype(np.int64),
            "cm_frac": np.where(tot > 0, w_cm / tot, 0.0),
            "dump_frac": np.where(tot > 0, w_dp / tot, 0.0),
            "sys_frac": np.where(tot > 0, w_sys / tot, 0.0),
        })  # fmt: skip
    out["sys_flag"] = (out.sys_frac >= 0.5) & (out.n_anom >= 4)
    return out


# --------------------------------------------------------------------------------------------
# Calibration: percentiles within strata against the real light curves


class StratumECDF:
    """Mid-rank percentiles of values against a reference set, per stratum and column."""

    def fit(self, values: np.ndarray, strata: np.ndarray) -> StratumECDF:
        v = np.where(np.isfinite(values), values, -np.inf)
        self.ref = {int(s): np.sort(v[strata == s], axis=0) for s in np.unique(strata)}
        return self

    def transform(self, values: np.ndarray, strata: np.ndarray) -> np.ndarray:
        v = np.where(np.isfinite(values), values, -np.inf)
        out = np.full(v.shape, np.nan)
        for s in np.unique(strata):
            sel = strata == s
            ref = self.ref.get(int(s))
            if ref is None:
                raise KeyError(f"stratum {s} has no real reference light curves")
            for j in range(v.shape[1]):
                lo = np.searchsorted(ref[:, j], v[sel, j], side="left")
                hi = np.searchsorted(ref[:, j], v[sel, j], side="right")
                out[sel, j] = (lo + hi) / 2.0 / ref.shape[0]
        return out


def combine(pct: np.ndarray, how: str = "mean") -> np.ndarray:
    if how == "mean":
        return pct.mean(axis=1)
    if how == "tail":  # rank-based, but the most extreme sub-scores dominate
        return (-np.log(np.clip(1.0 - pct, 1e-6, 1.0))).mean(axis=1)
    raise ValueError(how)


def strata_for(meta: pd.DataFrame) -> np.ndarray:
    return stratum_id(meta.tmag.to_numpy(), meta.camera.to_numpy(), meta.ccd.to_numpy())
