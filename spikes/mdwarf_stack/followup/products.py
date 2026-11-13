"""Alternative light-curve products for one star and sector, as the survey's raw-lc dict
(idx on the 10-min grid, flux normalised per sector by its median, sector), so the frozen
preprocessing can run on them unchanged.

  qlp      MIT Quick-Look Pipeline HLSP (SAP_FLUX, QUALITY == 0). Bucket s3://stpubdata
           mast/hlsp/qlp/ (S1-S98 seen 2026-11-13), MAST download for later sectors.
  tglc     TGLC HLSP (aperture flux, TESS_flags == 0 and TGLC_flags == 0); MAST URI.
  eleanor  GSFC-eleanor-lite HLSP (CORR_FLUX, QUALITY == 0); MAST URI.
  t16      T16 HLSP (image subtraction, IFL2 = instrumental flux, aperture 2); MAST URI.
  tars     TESS All-sky Rotation Survey HLSP (unpopular CPM-detrended FLUX); MAST URI.
  tesscut  our own aperture photometry on an 11x11 TESScut FFI cutout (native FFI cadence):
           circular aperture of radius 1.5 px on the target's WCS position, local background =
           median of the 3.5-5.5 px annulus after dropping its brightest 30% of pixels. No
           crowding correction (unlike PDCSAP), so depths are diluted by neighbours' flux.

FITS bytes are parsed in memory; cutout zips go to a temp dir and are deleted at once.
"""

from __future__ import annotations

import io
import math
import shutil
import tempfile
import time
import zipfile

import fu
import numpy as np
import requests

from oddball.common.http import ANON_USER_AGENT

S3 = "https://stpubdata.s3.amazonaws.com/"
MAST_DL = "https://mast.stsci.edu/api/v0.1/Download/file?uri="
TESSCUT = "https://mast.stsci.edu/tesscut/api/v0.1/astrocut"
_S = requests.Session()
_S.headers["User-Agent"] = ANON_USER_AGENT
HARD_FLAGS = 17087  # the survey's quality bitmask (oddball.tess.lightcurve default)


def _get(url, params=None, timeout=180, tries=3):
    err = None
    for k in range(tries):
        try:
            r = _S.get(url, params=params, timeout=timeout)
            if r.status_code in (403, 404):
                raise FileNotFoundError(f"{r.status_code} {url}")
            r.raise_for_status()
            return r.content
        except FileNotFoundError:
            raise
        except requests.RequestException as e:
            err = e
            time.sleep(3 * (k + 1))
    raise RuntimeError(f"download failed: {err!r}"[:300])


def _bin(t, f, sector):
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    med = np.median(f)
    if not (np.isfinite(med) and med > 0):
        raise ValueError("non-positive median flux")
    f = f / med
    idx = np.floor(t / fu.BIN_D).astype(np.int64)
    u, inv = np.unique(idx, return_inverse=True)
    fb = np.bincount(inv, weights=f) / np.bincount(inv)
    return {"idx": u, "flux": fb.astype(np.float64), "sector": np.full(len(u), sector, np.int16)}


def qlp_key(tic: int, sector: int) -> str:
    t = f"{tic:016d}"
    return (f"mast/hlsp/qlp/s{sector:04d}/{t[0:4]}/{t[4:8]}/{t[8:12]}/{t[12:16]}/"
            f"hlsp_qlp_tess_ffi_s{sector:04d}-{t}_tess_v01_llc.fits")


def _fits_table(data: bytes):
    from astropy.io import fits

    with fits.open(io.BytesIO(data), memmap=False) as h:
        d = h[1].data
        cols = {c: np.array(d[c]) for c in d.columns.names}
        hdr = dict(h[0].header)
        hdr1 = dict(h[1].header)
    return cols, hdr, hdr1


def qlp(tic: int, sector: int, mast_uri: str | None = None):
    src = "s3"
    try:
        data = _get(S3 + qlp_key(tic, sector))
    except FileNotFoundError:
        if mast_uri is None:
            raise
        data = _get(MAST_DL + mast_uri)
        src = "mast"
    cols, hdr, _ = _fits_table(data)
    t = cols["TIME"].astype(float)
    f = cols["SAP_FLUX"].astype(float)
    q = cols["QUALITY"].astype(int)
    m = q == 0
    lc = _bin(t[m], f[m], sector)
    return lc, {"product": "qlp", "source": src, "column": "SAP_FLUX", "n_cad": int(m.sum()),
                "tic_hdr": hdr.get("TICID") or hdr.get("OBJECT")}


def generic_hlsp(uri: str, sector: int, product: str):
    data = _get(MAST_DL + uri)
    cols, hdr, hdr1 = _fits_table(data)
    names = {c.lower(): c for c in cols}
    tcol = names.get("time") or names.get("tmid_bjd")
    if product == "t16":
        fcol = names.get("ifl2")
        good = np.ones(len(cols[tcol]), bool)
    elif product == "tglc":
        fcol = names.get("cal_aper_flux") or names.get("aperture_flux")
        good = np.ones(len(cols[tcol]), bool)
        for fl in ("tess_flags", "tglc_flags"):
            if fl in names:
                good &= cols[names[fl]].astype(int) == 0
    elif product == "eleanor":
        fcol = names.get("corr_flux") or names.get("raw_flux")
        good = cols[names["quality"]].astype(int) == 0 if "quality" in names else True
    else:
        cand = [c for c in ("sap_flux", "flux", "aper_flux", "flux_aper", "raw_flux")
                if c in names]
        fcol = names[cand[0]] if cand else None
        good = cols[names["quality"]].astype(int) == 0 if "quality" in names else True
    if fcol is None or tcol is None:
        raise KeyError(f"{product}: no time/flux column in {list(cols)[:20]}")
    t = cols[tcol].astype(float)
    f = cols[fcol].astype(float)
    if product == "tars" and abs(np.nanmedian(f)) < 0.5:  # CPM residual flux centred on 0
        f = f + 1.0
    good = np.asarray(good) & np.isfinite(t) & np.isfinite(f)
    if np.nanmedian(t) > 2_400_000:
        t = t - 2457000.0
    lc = _bin(t[good], f[good], sector)
    return lc, {"product": product, "source": "mast", "column": fcol, "n_cad": int(good.sum()),
                "uri": uri}


def tesscut_cube(ra: float, dec: float, sector: int, size: int = 11):
    from astropy.io import fits
    from astropy.wcs import WCS

    content = _get(TESSCUT, params={"ra": f"{ra:.6f}", "dec": f"{dec:.6f}", "x": size, "y": size,
                                    "units": "px", "sector": int(sector)}, timeout=600)
    if content[:2] != b"PK":
        raise RuntimeError("TESScut: not a zip")
    tmp = fu.FU_DATA / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    d = tempfile.mkdtemp(dir=tmp)
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            names = [n for n in z.namelist() if n.endswith(".fits")]
            if not names:
                raise RuntimeError("TESScut zip holds no FITS")
            z.extract(names[0], d)
        with fits.open(f"{d}/{names[0]}", memmap=False) as h:
            tb = h[1].data
            t = np.asarray(tb["TIME"], float)
            fl = np.asarray(tb["FLUX"], np.float32)
            q = np.asarray(tb["QUALITY"], int)
            w = WCS(h[2].header)
            x, y = w.all_world2pix([[ra, dec]], 0)[0]
            hdr0 = dict(h[0].header)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return t, fl, q, float(x), float(y), {"camera": hdr0.get("CAMERA"), "ccd": hdr0.get("CCD")}


def tesscut(ra: float, dec: float, sector: int):
    t, fl, q, x, y, info = tesscut_cube(ra, dec, sector)
    good = np.isfinite(t) & ((q & HARD_FLAGS) == 0) & np.isfinite(fl).all(axis=(1, 2))
    t, fl = t[good], fl[good]
    ny, nx = fl.shape[1:]
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - x, yy - y)
    ann = (r >= 3.5) & (r <= 5.5)
    av = fl[:, ann]
    thr = np.nanpercentile(np.nanmedian(av, axis=0), 70)
    bsel = np.nanmedian(av, axis=0) <= thr
    bkg = np.nanmedian(av[:, bsel], axis=1)
    # Fixed 1.5 px radius (about the 3x3 core). Choosing the radius by the lowest relative
    # scatter was tried first and dropped before any window was read: relative scatter falls as
    # neighbours' flux is added, so that rule selects the most diluted aperture.
    rad = APERTURE_PX
    ap = r <= rad
    f = fl[:, ap].sum(axis=1) - bkg * ap.sum()
    lc = _bin(t, f, sector)
    edge = min(x, y, nx - 1 - x, ny - 1 - y)
    return lc, {"product": "tesscut", "source": "tesscut", "aperture_px": rad,
                "n_pix": int(ap.sum()), "x": x, "y": y, "edge_px": float(edge),
                "n_cad": len(t), "bkg_median": float(np.median(bkg)),
                "ap_median": float(np.median(f)), **info}


APERTURE_PX = 1.5


def ok_float(x):
    return None if x is None or not math.isfinite(x) else float(x)


def tglc_key(gaia_id: int, sector: int, cam: int, ccd: int) -> str:
    g = "00" + str(int(gaia_id))  # bucket layout: "00" + id, split in 4-digit groups
    return (f"mast/hlsp/tglc/s{sector:04d}/cam{cam}-ccd{ccd}/{g[0:4]}/{g[4:8]}/{g[8:12]}/"
            f"{g[12:16]}/hlsp_tglc_tess_ffi_gaiaid-{int(gaia_id)}-s{sector:04d}-cam{cam}-ccd{ccd}"
            f"_tess_v1_llc.fits")


def cam_ccd(ra: float, dec: float, sector: int):
    import tess_stars2px as t2p

    out = t2p.tess_stars2px_function_entry(1, ra, dec, trySector=sector)
    for s, c, d in zip(np.atleast_1d(out[3]), np.atleast_1d(out[4]), np.atleast_1d(out[5])):
        if int(s) == sector:
            return int(c), int(d)
    return None


def tglc_s3(gaia_id: int, ra: float, dec: float, sector: int, camccd=None):
    """TGLC S1-S55 from the bucket (MAST's API lists S12-S55 only via bulk scripts)."""
    cc = camccd or cam_ccd(ra, dec, sector)
    if cc is None:
        raise FileNotFoundError(f"tess-point: not on silicon in S{sector}")
    data = _get(S3 + tglc_key(gaia_id, sector, *cc))
    cols, hdr, _ = _fits_table(data)
    names = {c.lower(): c for c in cols}
    t = cols[names["time"]].astype(float)
    f = cols[names["cal_aper_flux"]].astype(float)
    good = np.isfinite(t) & np.isfinite(f)
    for fl in ("tess_flags", "tglc_flags"):
        if fl in names:
            good &= cols[names[fl]].astype(int) == 0
    lc = _bin(t[good], f[good], sector)
    return lc, {"product": "tglc", "source": "s3", "column": "cal_aper_flux",
                "n_cad": int(good.sum()), "gaia_id": int(gaia_id), "cam_ccd": cc}
