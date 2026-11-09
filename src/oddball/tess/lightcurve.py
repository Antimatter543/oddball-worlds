"""Pure light-curve operations: quality mask, normalisation, binning, FITS -> compact record.

Design notes
------------
* **No detrending here.** Flattening at ingest would bake in a choice that can erase long
  exocomet tails (failure catalogue #6). Detrending belongs to the detector stage, where it can
  be tested against injections. We only divide by the median.
* **One global bin grid.** Bin ``k`` covers ``[k*w, (k+1)*w)`` in BTJD for every target, sector
  and camera. A light curve is stored as ``bin0`` (global index of its first bin) plus
  contiguous arrays, with NaN in empty interior bins (gaps). Time is recovered exactly as
  ``(bin0 + i + 0.5) * w``, and light curves on the same CCD line up bin-for-bin, which the
  neighbour-coincidence systematics filter needs.
* **Quality flags survive the FITS deletion.** Each bin stores the bitwise OR of the QUALITY
  word of every cadence that fell in it (masked or not), so momentum dumps and scattered-light
  windows can still be vetoed later.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np

# Exocomet dips in TESS last hours to ~a day (beta Pic events: ~0.5-2 d, Zieba+2019;
# Norazman & Kennedy 2025 search durations of ~0.1-2 d). 30 min puts >=8 bins across even a
# 4 h event, which preserves the sharp-ingress / slow-egress asymmetry, while cutting the
# point count 3x (10-min cadence, S27-S55) or 9x (200-s cadence, S56+) and the noise per point
# by sqrt(3)-sqrt(9). It is an integer multiple of both native FFI cadences, so every full
# bin holds the same number of cadences. Caveat: ~1.5 h dusty-tail transits
# (KIC 12557548 b-like) get only ~3 bins; pass a smaller width if that class needs it.
DEFAULT_BIN_MINUTES = 30.0

# TESS QUALITY bits (lightkurve TessQualityFlags naming).
QUALITY_BITS = {
    1: "AttitudeTweak",
    2: "SafeMode",
    4: "CoarsePoint",
    8: "EarthPoint",
    16: "Argabrightening",
    32: "Desat",  # reaction-wheel momentum dump
    64: "ApertureCosmic",
    128: "ManualExclude",
    256: "Discontinuity",
    512: "ImpulsiveOutlier",
    1024: "CollateralCosmic",
    2048: "Straylight",
    4096: "Straylight2",
    8192: "PlanetSearchExclude",
    16384: "BadCalibrationExclude",
    32768: "InsufficientTargets",
}

# Same as lightkurve's "default" bitmask (17087): pointing problems, momentum dumps,
# manual excludes, impulsive outliers, bad calibration. Scattered-light bits (2048/4096) are
# NOT in it: dropping them would delete whole orbit edges in some sectors, and they are kept
# in the per-bin flag word instead so later stages can decide. (In practice SPOC already sets
# PDCSAP_FLUX to NaN on most flagged cadences; the finite check below catches that too.)
DEFAULT_QUALITY_BITMASK = 1 | 2 | 4 | 8 | 16 | 32 | 128 | 512 | 16384
assert DEFAULT_QUALITY_BITMASK == 17087


def quality_mask(
    quality: np.ndarray, *arrays: np.ndarray, bitmask: int = DEFAULT_QUALITY_BITMASK
) -> np.ndarray:
    """True where a cadence is usable: no bad quality bits and every given array finite."""
    good = (np.asarray(quality).astype(np.int64) & bitmask) == 0
    for a in arrays:
        good &= np.isfinite(a)
    return good


def normalise(flux: np.ndarray, err: np.ndarray, good: np.ndarray):
    """Divide flux and error by the median of the good flux.

    Returns ``(flux_n, err_n, median)``. ``median`` is NaN (and the outputs all-NaN) when
    there are no good points or the median is not positive, since a ratio is meaningless then.
    """
    flux = np.asarray(flux, dtype=np.float64)
    err = np.asarray(err, dtype=np.float64)
    if not np.any(good):
        nan = np.full_like(flux, np.nan)
        return nan, nan.copy(), float("nan")
    med = float(np.median(flux[good]))
    if not np.isfinite(med) or med <= 0:
        nan = np.full_like(flux, np.nan)
        return nan, nan.copy(), float("nan")
    return flux / med, err / med, med


@dataclass
class Binned:
    bin0: int  # global index of first bin
    flux: np.ndarray  # mean of good normalised flux per bin (NaN if empty)
    flux_err: np.ndarray  # sqrt(sum sigma^2) / n  (NaN if empty)
    npts: np.ndarray  # good cadences per bin
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def nbins(self) -> int:
        return len(self.flux)

    def time(self, bin_width_days: float) -> np.ndarray:
        return bin_centres(self.bin0, self.nbins, bin_width_days)


def bin_centres(bin0: int, nbins: int, bin_width_days: float) -> np.ndarray:
    return (bin0 + np.arange(nbins) + 0.5) * bin_width_days


def bin_index(time: np.ndarray, bin_width_days: float) -> np.ndarray:
    return np.floor(np.asarray(time, dtype=np.float64) / bin_width_days).astype(np.int64)


def _bin_mean(local: np.ndarray, values: np.ndarray, mask: np.ndarray, nbins: int):
    n = np.bincount(local[mask], minlength=nbins)
    s = np.bincount(local[mask], weights=values[mask], minlength=nbins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return mean, n


def bin_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    good: np.ndarray,
    bin_width_days: float,
    quality: np.ndarray | None = None,
    extras: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> Binned | None:
    """Bin onto the global grid.

    The bin range runs from the first to the last bin containing a good flux point; interior
    bins with no good point (gaps, masked stretches) are NaN with ``npts == 0``.

    ``quality``: if given, ``extras['qflags']`` holds the bitwise OR of QUALITY over *all*
    cadences with finite time in each bin, including masked ones.
    ``extras``: ``name -> (values, mask)``; each is binned as a plain mean over its own mask.

    Returns None when there are no good points.
    """
    time = np.asarray(time, dtype=np.float64)
    good = np.asarray(good, dtype=bool) & np.isfinite(time) & np.isfinite(flux) & np.isfinite(err)
    if not np.any(good):
        return None
    idx = bin_index(np.where(np.isfinite(time), time, 0.0), bin_width_days)
    lo, hi = int(idx[good].min()), int(idx[good].max())
    nbins = hi - lo + 1
    local = idx - lo
    inrange = np.isfinite(time) & (local >= 0) & (local < nbins)

    fmean, n = _bin_mean(local, np.asarray(flux, np.float64), good, nbins)
    var = np.bincount(local[good], weights=np.asarray(err, np.float64)[good] ** 2, minlength=nbins)
    with np.errstate(invalid="ignore", divide="ignore"):
        ferr = np.where(n > 0, np.sqrt(var) / np.maximum(n, 1), np.nan)

    out_extras: dict[str, np.ndarray] = {}
    if quality is not None:
        q = np.zeros(nbins, dtype=np.int32)
        np.bitwise_or.at(q, local[inrange], np.asarray(quality).astype(np.int32)[inrange])
        out_extras["qflags"] = q
    for name, (vals, m) in (extras or {}).items():
        vals = np.asarray(vals, np.float64)
        m = np.asarray(m, bool) & inrange & np.isfinite(vals)
        out_extras[name], _ = _bin_mean(local, vals, m, nbins)

    return Binned(bin0=lo, flux=fmean, flux_err=ferr, npts=n, extras=out_extras)


def robust_std(x: np.ndarray) -> float:
    """1.4826 * MAD of the finite values (NaN if fewer than 3)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


# --------------------------------------------------------------------------------------------
# FITS bytes -> record


class LightCurveError(ValueError):
    """A light curve that cannot be converted (recorded as a failure, not retried blindly)."""


_HDR0 = {
    "TICID": "tic",
    "SECTOR": "sector",
    "CAMERA": "camera",
    "CCD": "ccd",
    "TESSMAG": "tmag",
    "RA_OBJ": "ra",
    "DEC_OBJ": "dec",
    "TEFF": "teff",
    "RADIUS": "radius",
    "DATA_REL": "data_rel",
    "PROCVER": "procver",
}
_HDR1 = {"CROWDSAP": "crowdsap", "FLFRCSAP": "flfrcsap", "PDCMETHD": "pdcmethod"}


def convert_fits_bytes(
    data: bytes,
    bin_minutes: float = DEFAULT_BIN_MINUTES,
    bitmask: int = DEFAULT_QUALITY_BITMASK,
) -> tuple[dict, Binned]:
    """Parse one TESS-SPOC (SPOC-format) light-curve FITS held in memory.

    Returns ``(meta, binned)``. Flux is PDCSAP; SAP flux (normalised by its own median) and
    SAP background (divided by the SAP median, i.e. as a fraction of the star's raw flux) ride
    along as extras so later stages can check what PDC did and spot scattered light.
    """
    from astropy.io import fits  # local import keeps the pure functions astropy-free

    w = bin_minutes / (24 * 60)
    with fits.open(io.BytesIO(data), memmap=False, lazy_load_hdus=False) as hdul:
        h0, h1 = hdul[0].header, hdul[1].header
        d = hdul[1].data
        time = np.array(d["TIME"], dtype=np.float64)
        pdc = np.array(d["PDCSAP_FLUX"], dtype=np.float64)
        pdc_e = np.array(d["PDCSAP_FLUX_ERR"], dtype=np.float64)
        sap = np.array(d["SAP_FLUX"], dtype=np.float64)
        bkg = np.array(d["SAP_BKG"], dtype=np.float64)
        quality = np.array(d["QUALITY"], dtype=np.int32)
        meta = {v: h0.get(k) for k, v in _HDR0.items()}
        meta.update({v: h1.get(k) for k, v in _HDR1.items()})
        timedel = h1.get("TIMEDEL")

    if meta["tic"] is None:
        raise LightCurveError("no TICID in header")
    meta["cadence_s"] = float(timedel) * 86400.0 if timedel else float("nan")
    meta["n_cadences"] = len(time)

    good = quality_mask(quality, time, pdc, pdc_e, bitmask=bitmask)
    flux_n, err_n, med = normalise(pdc, pdc_e, good)
    meta["median_flux"] = med
    meta["n_good"] = int(good.sum())
    if not np.isfinite(med):
        raise LightCurveError(f"no usable PDCSAP flux (n_good={meta['n_good']})")

    sap_good = quality_mask(quality, time, sap, bitmask=bitmask)
    sap_med = float(np.median(sap[sap_good])) if sap_good.any() else float("nan")
    if not (np.isfinite(sap_med) and sap_med > 0):
        sap_med = float("nan")

    binned = bin_lightcurve(
        time,
        flux_n,
        err_n,
        good,
        w,
        quality=quality,
        extras={"sap_flux": (sap / sap_med, sap_good), "bkg": (bkg / sap_med, sap_good)},
    )
    assert binned is not None  # good.any() guaranteed by finite median

    meta["n_bins"] = binned.nbins
    meta["n_bins_filled"] = int((binned.npts > 0).sum())
    meta["rms_bin"] = robust_std(binned.flux)  # robust scatter of the binned, normalised flux
    meta["p2p_native"] = robust_std(np.diff(flux_n[good])) / np.sqrt(2)
    meta["bin_minutes"] = float(bin_minutes)
    meta["t_start"] = float(np.nanmin(time[good]))
    meta["t_stop"] = float(np.nanmax(time[good]))
    return meta, binned
