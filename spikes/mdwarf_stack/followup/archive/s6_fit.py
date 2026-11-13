"""Background-source test at the 2026.9 position of a high-proper-motion star on archival
plates/images. Gaussian-PSF scene model of every Gaia DR3 source (propagated to the plate epoch,
global shift and FWHM fitted), plus an extra point source at the 2026.9 position; the extra
amplitude is compared with the same fit at null positions at the same distance from the nearest
bright blending star, and calibrated to Gaia G with isolated field stars.
Usage: s6_fit.py DLDIR OUTDIR TMAG DEPTH"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.table import Table  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

warnings.filterwarnings("ignore")
dl, out = Path(sys.argv[1]), Path(sys.argv[2])
TMAG, DEPTH = float(sys.argv[3]), float(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
TGT = 180614374211036416
EPOCH_NOW = 2026.9
EP = {"dss_poss1_red": 1955.81, "dss_poss1_blue": 1955.81, "dss_poss2ukstu_red": 1989.89,
      "dss_poss2ukstu_blue": 1996.77, "dss_poss2ukstu_ir": 1996.71, "2mass_J": 1998.09,
      "2mass_H": 1998.09, "2mass_K": 1998.09, "ps1_g": 2012.0, "ps1_r": 2012.0,
      "ps1_i": 2012.0, "ps1_z": 2012.0, "ps1_y": 2012.0}
# PS1 stacks combine 2010-2014 exposures; 2012.0 adopted (star moves 0.3"/yr, i.e. +-0.6")

g = Table.read(dl / "gaia_field.csv", format="csv")
g = g[np.isfinite(np.asarray(g["Gmag"], float))]
pmra = np.nan_to_num(np.asarray(g["pmRA"], float))
pmde = np.nan_to_num(np.asarray(g["pmDE"], float))
ra16, de16 = np.asarray(g["RA_ICRS"], float), np.asarray(g["DE_ICRS"], float)
G = np.asarray(g["Gmag"], float)
src = np.asarray(g["Source"], np.int64)
it = int(np.where(src == TGT)[0][0])


def radec(ep):
    dt = ep - 2016.0
    return (ra16 + pmra * dt / 3.6e6 / np.cos(np.radians(de16)), de16 + pmde * dt / 3.6e6)


ra_now, de_now = radec(EPOCH_NOW)
P_NOW = (ra_now[it], de_now[it])
# required brightness of a background source at the current position
dmag = 2.5 * np.log10(1 / DEPTH)
T_REQ = TMAG + dmag


def g_minus_t(c):  # TIC v8 relation T(G, BP-RP), inverted
    return -(-0.00522555 * c**3 + 0.0891337 * c**2 - 0.633923 * c + 0.0324473)


G_REQ = {"BP-RP=0.5": T_REQ + g_minus_t(0.5), "BP-RP=1.0": T_REQ + g_minus_t(1.0),
         "BP-RP=2.5": T_REQ + g_minus_t(2.5)}


def design(xx, yy, xs, ys, fw, tgt_i, tgt_w, wing=None):
    """core Gaussian per source (first column of each source); a 2x-wide wing for sources
    flagged in `wing` (G < 18; photographic/saturated profiles); extra halo terms for target."""
    s = fw / 2.3548
    cols = []
    for j, (x, y) in enumerate(zip(xs, ys)):
        r2 = (xx - x) ** 2 + (yy - y) ** 2
        cols.append(np.exp(-r2 / (2 * s * s)))
        if j == tgt_i:
            for m in tgt_w:
                cols.append(np.exp(-r2 / (2 * (m * s) ** 2)))
        elif wing is not None and wing[j]:
            cols.append(np.exp(-r2 / (2 * (2.0 * s) ** 2)))
    cols += [np.ones_like(xx), xx - xx.mean(), yy - yy.mean()]
    return np.vstack(cols).T


def scene_fit(d, w, sc, ep, centre, fw, shift, extra=None, rad=12.0, mask_r=None,
              return_model=False):
    """Fit all Gaia sources within rad+8" of centre (pixel coords) on pixels within rad"."""
    ra, de = radec(ep)
    xy = np.array(w.all_world2pix(np.c_[ra, de], 0)) + np.array(shift) / sc
    cx, cy = centre
    ny, nx = d.shape
    r_pix = rad / sc
    y0, y1 = int(max(0, cy - r_pix - 2)), int(min(ny, cy + r_pix + 3))
    x0, x1 = int(max(0, cx - r_pix - 2)), int(min(nx, cx + r_pix + 3))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r_pix**2
    dist_t = np.hypot(xx - xy[it, 0], yy - xy[it, 1]) * sc
    if mask_r is not None:
        m &= dist_t > mask_r
    m &= np.isfinite(d[y0:y1, x0:x1])
    near = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) * sc < rad + 8
    idx = np.where(near)[0]
    xs, ys = list(xy[idx, 0]), list(xy[idx, 1])
    ti = int(np.where(idx == it)[0][0]) if it in idx else -1
    wing = list(G[idx] < 18.0)
    if extra is not None:
        xs.append(extra[0])
        ys.append(extra[1])
        wing.append(False)
    X = design(xx[m].astype(float), yy[m].astype(float), xs, ys, fw / sc, ti, (2.0, 4.0), wing)
    z = d[y0:y1, x0:x1][m]
    coef, *_ = np.linalg.lstsq(X, z, rcond=None)
    res = z - X @ coef
    chi = float(np.sum(res**2))
    o = {"chi2": chi, "coef": coef, "idx": idx, "npix": int(m.sum()), "rms": float(np.std(res))}
    if extra is not None:
        k = X.shape[1] - 4  # extra source core column (last source; 3 background columns)
        cov = np.linalg.pinv(X.T @ X) * (np.var(res))
        o["amp"] = float(coef[k])
        o["amp_err_formal"] = float(np.sqrt(cov[k, k]))
    if return_model:
        full = np.full(d[y0:y1, x0:x1].shape, np.nan)
        mod = np.full_like(full, np.nan)
        Xf = design(xx.ravel().astype(float), yy.ravel().astype(float), xs, ys, fw / sc, ti,
                    (2.0, 4.0), wing)
        mod = (Xf @ coef).reshape(xx.shape)
        o["cut"] = (d[y0:y1, x0:x1], mod, m, (x0, y0))
    return o


def local_amp(d, x, y, fw_pix):
    r = 2.0 * fw_pix
    y0, y1 = int(y - r - 1), int(y + r + 2)
    x0, x1 = int(x - r - 1), int(x + r + 2)
    if y0 < 0 or x0 < 0 or y1 > d.shape[0] or x1 > d.shape[1]:
        return np.nan
    yy, xx = np.mgrid[y0:y1, x0:x1]
    s = fw_pix / 2.3548
    p = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s)).ravel()
    z = d[y0:y1, x0:x1].ravel()
    ok = np.isfinite(z)
    if ok.sum() < 0.8 * len(z):
        return np.nan
    X = np.c_[p, np.ones_like(p)]
    c, *_ = np.linalg.lstsq(X[ok], z[ok], rcond=None)
    return c[0]


rows = []
for key, ep in EP.items():
    f = dl / f"{key}.fits"
    if not f.exists():
        rows.append({"image": key, "status": "not checked: file missing"})
        continue
    h = fits.getheader(f)
    d = fits.getdata(f).astype(float)
    w = WCS(h)
    sc = float(np.sqrt(abs(np.linalg.det(w.pixel_scale_matrix))) * 3600)
    ra_e, de_e = radec(ep)
    sep_move = float(np.hypot((ra_e[it] - P_NOW[0]) * np.cos(np.radians(P_NOW[1])),
                              de_e[it] - P_NOW[1]) * 3600)
    c_now = w.all_world2pix([[P_NOW[0], P_NOW[1]]], 0)[0]
    mask_r = min(0.6 * sep_move, 8.0)
    # grid over shift and FWHM
    best = None
    for fw in np.arange(1.0, 5.01, 0.25) if not key.startswith("ps1") else np.arange(0.8, 2.01, 0.15):
        for dx in np.arange(-1.5, 1.51, 0.5):
            for dy in np.arange(-1.5, 1.51, 0.5):
                o = scene_fit(d, w, sc, ep, c_now, fw, (dx, dy), mask_r=mask_r, rad=14.0)
                if best is None or o["chi2"] < best[0]:
                    best = (o["chi2"], fw, dx, dy)
    _, fw, dx, dy = best
    for ddx in np.arange(-0.4, 0.41, 0.2):  # refine shift
        for ddy in np.arange(-0.4, 0.41, 0.2):
            o = scene_fit(d, w, sc, ep, c_now, fw, (dx + ddx, dy + ddy), mask_r=mask_r, rad=14.0)
            if o["chi2"] < best[0]:
                best = (o["chi2"], fw, dx + ddx, dy + ddy)
    _, fw, dx, dy = best
    shift = (dx, dy)
    obs = scene_fit(d, w, sc, ep, c_now, fw, shift, extra=c_now, mask_r=mask_r,
                    return_model=True)
    # nearest bright neighbour (other than the target) to the current position -> null ring
    xy_e = np.array(w.all_world2pix(np.c_[ra_e, de_e], 0)) + np.array(shift) / sc
    dists = np.hypot(xy_e[:, 0] - c_now[0], xy_e[:, 1] - c_now[1]) * sc
    cand = np.where((np.arange(len(G)) != it) & (G < 17.0))[0]
    jn = cand[np.argmin(dists[cand])]
    rr = dists[jn]
    nulls = []
    for pa in np.arange(0, 360, 15):
        px = xy_e[jn, 0] + rr / sc * np.cos(np.radians(pa))
        py = xy_e[jn, 1] + rr / sc * np.sin(np.radians(pa))
        dd = np.hypot(xy_e[:, 0] - px, xy_e[:, 1] - py) * sc
        dd[jn] = 99
        if np.min(dd[G < 20.5]) < 1.5 * fw or np.hypot(px - c_now[0], py - c_now[1]) * sc < 1.5 * fw:
            continue
        if np.hypot(px - xy_e[it, 0], py - xy_e[it, 1]) * sc < sep_move:  # closer to target
            continue
        try:
            on = scene_fit(d, w, sc, ep, (px, py), fw, shift, extra=(px, py), mask_r=mask_r)
            nulls.append(on["amp"])
        except Exception:  # noqa: BLE001
            pass
    # blank-sky nulls (positions > 3 FWHM from every Gaia source and the target)
    rng = np.random.default_rng(1)
    blank = []
    tries = 0
    while len(blank) < 40 and tries < 2000:
        tries += 1
        px, py = rng.uniform(15 / sc, d.shape[1] - 15 / sc), rng.uniform(15 / sc, d.shape[0] - 15 / sc)
        dd = np.hypot(xy_e[:, 0] - px, xy_e[:, 1] - py) * sc
        if np.min(dd) < 3 * fw or dd[it] < 25:
            continue
        if not np.all(np.isfinite(d[int(py) - 3:int(py) + 4, int(px) - 3:int(px) + 4])):
            continue
        blank.append(local_amp(d, px, py, fw / sc))
    nulls, blank = np.array(nulls), np.array(blank)
    blank = blank[np.isfinite(blank)]
    sig_ring = 1.4826 * np.median(np.abs(nulls - np.median(nulls))) if len(nulls) >= 6 else np.nan
    sig_blank = 1.4826 * np.median(np.abs(blank - np.median(blank)))
    sig_use = np.nanmax([sig_ring, sig_blank, obs["amp_err_formal"]])
    # calibration: isolated field stars G 15.5-20, > 25" from the target
    cal = []
    for j in range(len(G)):
        if j == it or not (16.0 <= G[j] <= 21.0):
            continue
        dd = np.hypot(xy_e[:, 0] - xy_e[j, 0], xy_e[:, 1] - xy_e[j, 1]) * sc
        dd[j] = 99
        if np.min(dd[G < G[j] + 3]) < 2.5 * fw or dd[it] < 30:
            continue
        a = local_amp(d, xy_e[j, 0], xy_e[j, 1], fw / sc)
        if np.isfinite(a) and a > 5 * sig_blank:
            cal.append((G[j], a))
    cal = np.array(cal)
    if len(cal) >= 5:
        A = np.c_[np.ones(len(cal)), np.log10(cal[:, 1])]
        cc, *_ = np.linalg.lstsq(A, cal[:, 0], rcond=None)
        for _ in range(3):  # clip 3 sigma
            r = cal[:, 0] - A @ cc
            k = np.abs(r) < 3 * max(1.4826 * np.median(np.abs(r)), 0.05)
            cc, *_ = np.linalg.lstsq(A[k], cal[k, 0], rcond=None)
        cal_rms = float(1.4826 * np.median(np.abs(cal[:, 0] - A @ cc)))
        g_lim = float(cc[0] + cc[1] * np.log10(5 * sig_use))
        # completeness-style check: fraction of Gaia stars within 0.5 mag of g_lim detected >5 sig_blank
        near_lim = [a for (gg, a) in cal if abs(gg - g_lim) < 0.5]
        g_obs = float(cc[0] + cc[1] * np.log10(obs["amp"])) if obs["amp"] > 0 else None
        faintest_cal = float(cal[:, 0].max())
    else:
        cc, cal_rms, g_lim, g_obs, faintest_cal = None, None, None, None, None
    # injection sanity at the 5 sigma amplitude
    inj_amp = 5 * sig_use
    s = fw / sc / 2.3548
    yy, xx = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    d_inj = d + inj_amp * np.exp(-((xx - c_now[0]) ** 2 + (yy - c_now[1]) ** 2) / (2 * s * s))
    oi = scene_fit(d_inj, w, sc, ep, c_now, fw, shift, extra=c_now, mask_r=mask_r)
    snr = obs["amp"] / sig_use
    trust = (sep_move > 2 * fw)
    if not trust:
        status = "no information (star within 2x PSF of the 2026 position)"
    elif g_lim is None:
        status = "no information (calibration failed)"
    elif snr >= 5:
        status = (f"source at 2026 position detected (G~{g_obs:.1f})")
    else:
        status = "no source at 2026 position above 5 sigma"
    row = {"image": key, "survey": h.get("SURVEY", key), "plate": h.get("PLATEID"),
           "date": h.get("DATE-OBS"), "epoch": ep, "scale_as": round(sc, 2),
           "fwhm_as": round(float(fw), 2), "shift_as": [round(dx, 2), round(dy, 2)],
           "star_sep_from_2026_as": round(sep_move, 2), "mask_r_as": round(mask_r, 2),
           "nearest_bright": {"source": int(src[jn]), "G": float(G[jn]), "sep_as": round(float(rr), 2)},
           "amp": obs["amp"], "sig_formal": obs["amp_err_formal"], "sig_ring": float(sig_ring),
           "n_ring": int(len(nulls)), "sig_blank": float(sig_blank), "sig_used": float(sig_use),
           "snr_2026": round(float(snr), 2), "g_obs_if_positive": g_obs,
           "cal_n": int(len(cal)), "cal_rms_mag": cal_rms, "faintest_cal_G": faintest_cal,
           "g_limit_5sig": g_lim, "inj_amp": inj_amp,
           "inj_recovered_snr": round(float(oi["amp"] / sig_use), 2),
           "trusted_geometry": bool(trust), "status": status}
    rows.append(row)
    print(json.dumps({k: row[k] for k in ["image", "epoch", "star_sep_from_2026_as", "fwhm_as",
                                         "snr_2026", "g_limit_5sig", "cal_n", "n_ring",
                                         "inj_recovered_snr", "status"]}))
    # figure
    cut, mod, m, (x0, y0) = obs["cut"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    lo, hi = np.nanpercentile(cut, [5, 99.7])
    for ax, im, tt in zip(axs, [cut, mod, cut - mod], ["data", "model (no extra src)", "residual"]):
        if tt == "residual":
            v = 5 * np.nanstd((cut - mod)[m])
            ax.imshow(im, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
        else:
            ax.imshow(im, origin="lower", cmap="gray_r", vmin=lo, vmax=hi)
        ax.plot(c_now[0] - x0, c_now[1] - y0, "o", mfc="none", mec="r", ms=14, mew=1.5)
        ax.plot(xy_e[it, 0] - x0, xy_e[it, 1] - y0, "b+", ms=14)
        ax.set_title(tt, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"TIC 2102868  {key}  epoch {ep}  star {sep_move:.1f}\" from 2026.9 position "
                 f"(red)  blue + = star at plate epoch\nSNR at 2026 pos {snr:.1f}; 5-sigma limit "
                 f"G~{g_lim if g_lim is None else round(g_lim, 1)}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"tic2102868_{key}.png", dpi=80)
    plt.close(fig)

summary = {"target": 2102868, "pos_2026_9": P_NOW, "T": TMAG, "depth": DEPTH,
           "T_required_max": T_REQ, "G_required_max": G_REQ, "images": rows}
(out / "tic2102868_background.json").write_text(json.dumps(summary, indent=1, default=float))
print(json.dumps({"T_req": T_REQ, "G_req": G_REQ, "pos": P_NOW}, default=float))
