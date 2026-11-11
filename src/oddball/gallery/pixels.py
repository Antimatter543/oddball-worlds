"""Pixel-level vetting from TESScut FFI cutouts (docs/VETTING.md F10c, F11).

For one star-sector and one event:

* **Difference image** (out-of-event minus in-event for a dip, the reverse for a bump), in units
  of its own noise, and the centroid of the significant part against the catalogue position.
  A dip that belongs to the target puts the difference on the target's pixel; a blend puts it on
  the neighbour; an asteroid in the background pixels puts a *negative* (brightening) blob away
  from the star.
* **Signal-locus map**: each pixel's flux regressed on [1, t, background, the star's own PDCSAP
  light curve]. The PDCSAP coefficient map traces where the star's variability lives (works for
  rotators, pulsators and EBs, not only for the strongest event). Its centroid is compared with
  the target too.
* A per-pixel light-curve grid (7x7 around the target, common scale) around the event.

The cutout is downloaded to a temporary directory, measured, and deleted (CLAUDE.md: never keep
FITS). What is kept is a 30-min binned, background-subtracted float32 cube (``.npz``, ~0.6 MB),
so cards can be re-rendered without a new download, plus the numbers and a PNG.

FFI cutouts are not background-subtracted (``FLUX_BKG`` is 0), so a local per-frame background
(median of the faintest 25% of pixels in the median image) is subtracted here. Absolute
centroids are uncertain at the ~0.1-0.3 px level; offsets above ~1 px (21") are what matter.
"""

from __future__ import annotations

import io
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

from oddball.common.http import ANON_USER_AGENT

TESSCUT = "https://mast.stsci.edu/tesscut/api/v0.1/astrocut"
PX = 21.0  # arcsec per TESS pixel
BIN_D = 1.0 / 48.0
# Frames dropped: attitude tweak, safe mode, coarse point, Earth point, desaturation, manual
# exclude (bits 1, 2, 4, 8, 32, 128). Stray-light frames (2048, 4096) are KEPT and marked: that
# is where scattered-light "events" live, and the pixels must be seen there.
HARD_FLAGS = 1 | 2 | 4 | 8 | 32 | 128
STRAY_FLAGS = 2048 | 4096


@dataclass
class Cube:
    time: np.ndarray  # (n,)
    flux: np.ndarray  # (n, ny, nx) background-subtracted, e-/s
    bkg: np.ndarray  # (n,) local background per pixel, e-/s
    wcs_header: str  # APERTURE extension header (WCS), as text
    sector: int
    size: int
    stray: np.ndarray | None = None  # (n,) fraction of frames in the bin with stray-light bits


def fetch_cube(ra: float, dec: float, sector: int, size: int = 11,
               sess: requests.Session | None = None, timeout: float = 180.0) -> Cube:  # fmt: skip
    """Download an FFI cutout, bin it to 30 min, subtract a local background, delete the FITS."""
    from astropy.io import fits

    sess = sess or requests.Session()
    sess.headers["User-Agent"] = ANON_USER_AGENT
    params = {"ra": f"{ra:.6f}", "dec": f"{dec:.6f}", "x": size, "y": size, "units": "px",
              "sector": int(sector)}  # fmt: skip
    err = None
    for attempt in range(3):
        try:
            r = sess.get(TESSCUT, params=params, timeout=timeout)
            if r.status_code == 200 and r.content[:2] == b"PK":
                break
            err = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.RequestException as e:
            err = type(e).__name__
        time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(f"TESScut failed: {err}")
    tmp = Path(tempfile.mkdtemp(prefix="tesscut_"))
    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            names = [n for n in z.namelist() if n.endswith(".fits")]
            if not names:
                raise RuntimeError("TESScut zip holds no FITS")
            z.extract(names[0], tmp)
        with fits.open(tmp / names[0], memmap=False) as h:
            d = h[1].data
            t = np.asarray(d["TIME"], float)
            f = np.asarray(d["FLUX"], np.float32)
            q = np.asarray(d["QUALITY"], int)
            hdr = h[2].header.tostring()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    good = np.isfinite(t) & ((q & HARD_FLAGS) == 0) & np.isfinite(f).all(axis=(1, 2))
    t, f, q = t[good], f[good], q[good]
    med = np.nanmedian(f, axis=0)
    faint = med <= np.nanpercentile(med, 25)
    bkg = np.nanmedian(f[:, faint], axis=1)
    f = f - bkg[:, None, None]
    # 30-min bins
    k = np.floor((t - t[0]) / BIN_D).astype(int)
    nb = k.max() + 1
    cnt = np.bincount(k, minlength=nb)
    keep = cnt > 0
    tb = np.bincount(k, weights=t, minlength=nb)[keep] / cnt[keep]
    bb = np.bincount(k, weights=bkg, minlength=nb)[keep] / cnt[keep]
    st = np.bincount(k, weights=((q & STRAY_FLAGS) > 0).astype(float), minlength=nb)[keep] / cnt[keep]
    fb = np.stack([np.bincount(k, weights=f[:, i, j], minlength=nb)[keep] / cnt[keep]
                   for i in range(f.shape[1]) for j in range(f.shape[2])], axis=1)  # fmt: skip
    fb = fb.reshape(len(tb), f.shape[1], f.shape[2]).astype(np.float32)
    return Cube(tb, fb, bb.astype(np.float32), hdr, int(sector), int(size), st.astype(np.float32))


def save_cube(c: Cube, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, time=c.time, flux=c.flux, bkg=c.bkg, wcs=np.array(c.wcs_header),
                        sector=c.sector, size=c.size,
                        stray=c.stray if c.stray is not None else np.zeros(0))  # fmt: skip


def load_cube(path: Path) -> Cube:
    z = np.load(path, allow_pickle=False)
    st = z["stray"] if "stray" in z and len(z["stray"]) == len(z["time"]) else None
    return Cube(z["time"], z["flux"], z["bkg"], str(z["wcs"]), int(z["sector"]), int(z["size"]), st)


def wcs_of(c: Cube):
    from astropy.io import fits
    from astropy.wcs import WCS

    return WCS(fits.Header.fromstring(c.wcs_header))


def _centroid(img: np.ndarray, peak_rc: tuple[int, int], half: int = 1) -> tuple[float, float]:
    """Flux-weighted centroid (x=col, y=row) of the positive part in a box around the peak."""
    r0, c0 = peak_rc
    ny, nx = img.shape
    rs, cs = slice(max(0, r0 - half), min(ny, r0 + half + 1)), slice(max(0, c0 - half),
                                                                      min(nx, c0 + half + 1))  # fmt: skip
    sub = np.clip(img[rs, cs], 0, None)
    if not np.isfinite(sub).any() or sub.sum() <= 0:
        return float(c0), float(r0)
    yy, xx = np.mgrid[rs, cs]
    return float((xx * sub).sum() / sub.sum()), float((yy * sub).sum() / sub.sum())


def analyse(c: Cube, ra: float, dec: float, t_ev: float, w_days: float, kind: str,
            template_t: np.ndarray | None = None, template_f: np.ndarray | None = None) -> dict:  # fmt: skip
    """Difference image and signal-locus measurements. Returns numbers plus arrays for plots."""
    w = wcs_of(c)
    tx, ty = (float(v) for v in w.world_to_pixel_values(ra, dec))
    t, f = c.time, c.flux
    ny, nx = f.shape[1:]
    out = {"tx": tx, "ty": ty, "n_bins": len(t)}
    med = np.nanmedian(f, axis=0)
    out["median_img"] = med
    # --- difference image around the event
    hw = max(w_days / 2.0, BIN_D)
    dt_ = np.abs(t - t_ev)
    inn = dt_ <= hw
    outw = (dt_ > hw + 0.05) & (dt_ <= hw + 0.05 + max(2 * hw, 0.25))
    loc = (dt_ <= 3.0) & ~inn
    out.update({"n_in": int(inn.sum()), "n_out": int(outw.sum())})
    if c.stray is not None and inn.any():
        out["in_stray_frac"] = float(np.mean(c.stray[inn] > 0.5))
    if inn.sum() >= 1 and outw.sum() >= 3 and loc.sum() >= 10:
        sgn = 1.0 if kind != "bump" else -1.0
        diff = sgn * (f[outw].mean(0) - f[inn].mean(0))
        # per-pixel noise: point-to-point scatter of the local out-of-event bins
        sd = 1.4826 * np.nanmedian(np.abs(np.diff(f[loc], axis=0)), axis=0) / np.sqrt(2)
        noise = sd * np.sqrt(1.0 / inn.sum() + 1.0 / outw.sum())
        snr = diff / np.where(noise > 0, noise, np.nan)
        out["diff_img"], out["diff_snr"] = diff, snr
        pk = np.unravel_index(np.nanargmax(snr), snr.shape)
        out["diff_peak_snr"] = float(snr[pk])
        cx, cy = _centroid(diff, pk)
        out["diff_cx"], out["diff_cy"] = cx, cy
        out["diff_offset_px"] = float(np.hypot(cx - tx, cy - ty))
        # signal in the target's 3x3 vs the most significant pixel anywhere
        r0, c0 = round(ty), round(tx)
        box = diff[max(0, r0 - 1): r0 + 2, max(0, c0 - 1): c0 + 2]
        nbox = noise[max(0, r0 - 1): r0 + 2, max(0, c0 - 1): c0 + 2]
        out["diff_target3x3_snr"] = float(box.sum() / np.sqrt((nbox ** 2).sum()))
        # the opposite sign: a brightening in background pixels (asteroid, glint)
        out["diff_neg_peak_snr"] = float(-np.nanmin(snr))
        nk = np.unravel_index(np.nanargmin(snr), snr.shape)
        out["diff_neg_offset_px"] = float(np.hypot(nk[1] - tx, nk[0] - ty))
    # --- signal-locus map from the star's own light curve
    if template_t is not None and len(template_t) > 20:
        m = np.isfinite(template_f)
        tf = np.interp(t, template_t[m], template_f[m], left=np.nan, right=np.nan)
        ok = np.isfinite(tf)
        if ok.sum() > 50 and np.nanstd(tf[ok]) > 0:
            tt = (t[ok] - t[ok].mean()) / max(np.ptp(t[ok]), 1e-6)
            X = np.column_stack([np.ones(ok.sum()), tt, c.bkg[ok] - c.bkg[ok].mean(),
                                 tf[ok] - np.nanmean(tf[ok])])  # fmt: skip
            Y = f[ok].reshape(ok.sum(), -1)
            beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
            resid = Y - X @ beta
            cov = np.linalg.pinv(X.T @ X)
            se = np.sqrt(np.clip(cov[3, 3] * (resid ** 2).sum(0) / max(ok.sum() - 4, 1), 0,
                                 None))  # fmt: skip
            amp = beta[3].reshape(ny, nx)
            asnr = (beta[3] / np.where(se > 0, se, np.nan)).reshape(ny, nx)
            out["locus_img"], out["locus_snr"] = amp, asnr
            pk = np.unravel_index(np.nanargmax(asnr), asnr.shape)
            out["locus_peak_snr"] = float(asnr[pk])
            cx, cy = _centroid(amp, pk)
            out["locus_cx"], out["locus_cy"] = cx, cy
            out["locus_offset_px"] = float(np.hypot(cx - tx, cy - ty))
    return out


def summary(res: dict) -> dict:
    """Scalar part of :func:`analyse` (for a table)."""
    return {k: v for k, v in res.items() if not isinstance(v, np.ndarray)}


def verdict(res: dict) -> str:
    """One plain-language line. Thresholds: offsets > 1 px with a significant signal."""
    bits = []
    if "diff_peak_snr" in res:
        if res["diff_peak_snr"] < 4:
            bits.append(f"event not detected in pixels (peak SNR {res['diff_peak_snr']:.1f})")
        elif res["diff_offset_px"] > 1.0:
            bits.append(f"event OFF target: difference centroid {res['diff_offset_px']:.1f} px "
                        f"from the star (peak SNR {res['diff_peak_snr']:.0f})")  # fmt: skip
        else:
            bits.append(f"event on target ({res['diff_offset_px']:.1f} px, peak SNR "
                        f"{res['diff_peak_snr']:.0f})")  # fmt: skip
        if res.get("diff_neg_peak_snr", 0) >= 5 and res.get("diff_neg_offset_px", 0) > 1.5:
            bits.append(f"opposite-sign blob {res['diff_neg_offset_px']:.1f} px away "
                        f"(SNR {res['diff_neg_peak_snr']:.0f}: moving object or background?)")  # fmt: skip
    if "locus_peak_snr" in res:
        if res["locus_peak_snr"] < 5:
            bits.append("variability too weak to locate")
        elif res["locus_offset_px"] > 1.0:
            bits.append(f"variability OFF target ({res['locus_offset_px']:.1f} px)")
        else:
            bits.append(f"variability on target ({res['locus_offset_px']:.1f} px)")
    if res.get("n_in", 0) == 0:
        bits.insert(0, "no FFI data in the event window (gap or flagged)")
    if res.get("in_stray_frac", 0) > 0.5:
        bits.append(f"{100 * res['in_stray_frac']:.0f}% of the event lies in stray-light-flagged frames")
    return "; ".join(bits) or "no pixel measurement"


def render(res: dict, c: Cube, path: Path, title: str, t_ev: float, w_days: float,
           neighbours=None, dpi: int = 70) -> None:  # fmt: skip
    """Pixel card: median image + Gaia neighbours, difference SNR, locus SNR, 7x7 pixel grid."""
    # Object-oriented API only: pyplot's global "current figure" is not thread-safe, and the
    # first run (4 render threads) attached colorbars to other cards' figures.
    from matplotlib.figure import Figure

    fig = Figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.5], wspace=0.25, left=0.03, right=0.99,
                          top=0.86, bottom=0.08)  # fmt: skip
    tx, ty = res["tx"], res["ty"]

    def mark(ax):
        ax.plot(tx, ty, "+", color="red", ms=16, mew=2)
        if neighbours is not None:
            for n in neighbours:
                x, y, dg = n
                if -0.5 <= x <= c.size - 0.5 and -0.5 <= y <= c.size - 0.5:
                    ax.plot(x, y, "o", mfc="none", mec="cyan", ms=max(3, 11 - 1.6 * dg), mew=1.2)
                    if dg < 3:
                        ax.text(x + 0.25, y + 0.25, f"{dg:+.1f}", color="cyan", fontsize=7)
        ax.set_xlim(-0.5, c.size - 0.5)
        ax.set_ylim(-0.5, c.size - 0.5)
        ax.tick_params(labelsize=6)

    ax = fig.add_subplot(gs[0, 0])
    med = res["median_img"]
    lo = np.nanpercentile(med, 5)
    ax.imshow(np.sqrt(np.clip(med - lo, 0, None)), origin="lower", cmap="gray",
              vmax=np.sqrt(max(np.nanpercentile(med, 99.5) - lo, 1e-6)))  # fmt: skip
    mark(ax)
    ax.set_title("median image (sqrt); red + = target,\ncyan = Gaia dG < 5 (max 25 brightest; label = dG)",
                 fontsize=8)  # fmt: skip
    for j, (key, lab, cx, cy) in enumerate(
        [("diff_snr", "difference image / noise (red = the event's sign:\nlight lost in a dip, gained in a bump; x = centroid)",
          "diff_cx", "diff_cy"),
         ("locus_snr", "signal locus: PDCSAP coefficient / se\n(where the star's variability lives)",
          "locus_cx", "locus_cy")]):  # fmt: skip
        ax = fig.add_subplot(gs[0, 1 + j])
        if key in res:
            im = res[key]
            v = np.nanmax(np.abs(im[np.isfinite(im)])) if np.isfinite(im).any() else 1
            h = ax.imshow(im, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
            fig.colorbar(h, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
            ax.plot(res[cx], res[cy], "x", color="k", ms=12, mew=2)
        else:
            ax.text(0.5, 0.5, "not measured", ha="center", transform=ax.transAxes)
        mark(ax)
        ax.set_title(lab, fontsize=8)
    # 7x7 per-pixel light curves around the event (common y scale, e-/s minus the local median)
    sg = gs[0, 3].subgridspec(7, 7, hspace=0.0, wspace=0.0)
    r0, c0 = round(ty), round(tx)
    span = max(3 * w_days, 1.0)
    m = np.abs(c.time - t_ev) <= span
    tt = c.time[m]
    loc = c.flux[m]
    loc = loc - np.nanmedian(loc, axis=0)
    rr = [r for r in range(r0 + 3, r0 - 4, -1)]
    cc = [x for x in range(c0 - 3, c0 + 4)]
    vals = [loc[:, r, x] for r in rr for x in cc if 0 <= r < c.size and 0 <= x < c.size]
    allv = np.concatenate(vals) if vals else np.array([])
    lim = np.nanpercentile(np.abs(allv), 99.5) if np.isfinite(allv).any() else 1.0
    lim = lim if np.isfinite(lim) and lim > 0 else 1.0
    for i, r in enumerate(rr):
        for k, x in enumerate(cc):
            a = fig.add_subplot(sg[i, k])
            a.set_xticks([])
            a.set_yticks([])
            if 0 <= r < c.size and 0 <= x < c.size:
                a.plot(tt, loc[:, r, x], ".", ms=1, color="k")
                a.axvspan(t_ev - w_days / 2, t_ev + w_days / 2, color="tab:orange", alpha=0.25, lw=0)
                a.set_ylim(-lim, lim)
                if (r, x) == (r0, c0):
                    for s_ in a.spines.values():
                        s_.set_color("red")
                        s_.set_linewidth(2)
            else:
                a.set_facecolor("0.9")
            if i == 0 and k == 0:
                a.set_title(f"pixels around target (red box), BTJD {t_ev - span:.1f}-{t_ev + span:.1f},"
                            f" common scale +-{lim:.0f} e/s; orange = event", fontsize=7, loc="left")  # fmt: skip
    fig.suptitle(title + "\n" + verdict(res), fontsize=9, x=0.01, ha="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
