"""Neighbour eclipse test on ZTF DR light curves (IRSA, anonymous).
For each Gaia neighbour inside 63" with required eclipse depth = depth * 10^(0.4 dG) <= 1, fetch
ZTF light curves around it (1.5" cone), convert to BTJD, and test every ephemeris in the 3-sigma
(T0, P) error ellipse: in-transit points |t - tc_n| <= D/2, depth = weighted mean deficit.
Blend excluded: for EVERY allowed ephemeris >= 5 in-transit points and req - depth > 5 sigma.
Blend supported: some allowed ephemeris with depth/sigma >= 5 and |depth - req| < 2 sigma.
Usage: ztf_neighbours.py EPHEM_JSON DOSSIER_JSON DLDIR OUTJSON TIC [TIC ...]"""
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astropy.utils import iers

iers.conf.auto_download = False  # bundled IERS tables; no network download (sub-second effect)

ephf, dosf, dl, outj = sys.argv[1:5]
tics = [int(x) for x in sys.argv[5:] if not x.startswith("--")]
MAX_SINGLE = next((int(x.split("=")[1]) for x in sys.argv[5:] if x.startswith("--max-single=")), 8)
SKIP_CONE = "--skip-cone" in sys.argv
dl = Path(dl)
dl.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "astro-research-client/1.0"}
EPH = json.load(open(ephf))
DOS = json.load(open(dosf))
PALOMAR = EarthLocation.of_site("Palomar") if False else EarthLocation.from_geodetic(
    -116.8597 * u.deg, 33.3563 * u.deg, 1712 * u.m)
out = {"rules": __doc__, "stars": {}}


def fetch(ra, de, rad_deg, fn):
    if (dl / fn).exists() and (dl / fn).stat().st_size > 0:
        return pd.read_csv(dl / fn), "cached:" + fn
    url = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"
    params = {"POS": f"CIRCLE {ra:.6f} {de:.6f} {rad_deg:.6f}", "BANDNAME": "g,r,i",
              "FORMAT": "csv"}
    r = requests.get(url, params=params, headers=UA, timeout=600)
    r.raise_for_status()
    (dl / fn).write_bytes(r.content)
    return pd.read_csv(io.BytesIO(r.content)), r.url


def to_btjd(mjd, ra, de):
    t = Time(np.asarray(mjd, float), format="mjd", scale="utc", location=PALOMAR)
    c = SkyCoord(ra * u.deg, de * u.deg)
    bjd = t.tdb + t.light_travel_time(c, "barycentric")
    return bjd.jd - 2457000.0


def test(t, f, s, eph, D, req, parity=None, nsig=3.0):
    T0, P = eph["T0"], eph["P"]
    cov = np.asarray(eph["cov"])
    if parity is not None:  # 2P: keep epochs with n % 2 == parity, period 2P
        pass
    sT0, sP = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
    n_all = np.round((t - T0) / P).astype(int)
    nmax = max(1, int(np.max(np.abs(n_all))))
    stepT = D / 8
    stepP = D / 8 / nmax
    gT = np.arange(-nsig * sT0, nsig * sT0 + 1e-12, stepT)
    gP = np.arange(-nsig * sP, nsig * sP + 1e-12, stepP)
    ic = np.linalg.inv(cov)
    w = 1 / s**2
    res = []
    for a in gT:
        for b in gP:
            v = np.array([a, b])
            if v @ ic @ v > nsig**2:
                continue
            tt, pp = T0 + a, P + b
            n = np.round((t - tt) / pp).astype(int)
            tc = tt + n * pp
            m = np.abs(t - tc) <= D / 2
            if parity is not None:
                m &= (n % 2) == parity
            k = int(m.sum())
            if k == 0:
                res.append((a, b, 0, np.nan, np.nan))
                continue
            dep = float(np.sum(w[m] * (1 - f[m])) / np.sum(w[m]))
            sig = float(1 / np.sqrt(np.sum(w[m])))
            res.append((a, b, k, dep, sig))
    r = np.array(res, float)
    if len(r) == 0:
        return {"n_ephem": 0}
    k, dep, sig = r[:, 2], r[:, 3], r[:, 4]
    ok = k >= 5
    excl_each = ok & ((req - dep) > 5 * sig)
    supp_each = (k >= 1) & (dep / sig >= 5) & (np.abs(dep - req) < 2 * sig)
    excluded = bool(np.all(excl_each))
    supported = bool(np.any(supp_each))
    i_worst = int(np.nanargmax(np.where(np.isfinite(dep), (dep - req) / np.where(sig > 0, sig, np.nan), -np.inf))) if np.any(np.isfinite(dep)) else None
    return {"n_ephem": int(len(r)), "frac_ephem_ge5pts": float(np.mean(ok)),
            "min_in_transit_pts": int(k.min()), "median_in_transit_pts": float(np.median(k)),
            "max_snr_any_ephem": float(np.nanmax(dep / sig)) if np.any(np.isfinite(dep)) else None,
            "worst_case": (None if i_worst is None else
                           {"dT0_h": r[i_worst, 0] * 24, "dP_h": r[i_worst, 1] * 24,
                            "n_pts": int(k[i_worst]), "depth": dep[i_worst], "sigma": sig[i_worst],
                            "req_minus_depth_over_sigma": (req - dep[i_worst]) / sig[i_worst]}),
            "nominal": (lambda j: {"n_pts": int(k[j]), "depth": dep[j], "sigma": sig[j]})(
                int(np.argmin(np.hypot(r[:, 0], r[:, 1])))),
            "excluded_all_ephem": excluded, "supported_any_ephem": supported}


for tic in tics:
    key = [k for k in EPH if k.startswith(f"survey-{tic}-")][0]
    eph = EPH[key]["ephem"]
    D, depth = eph["D"], eph["depth"]
    nb = DOS[key]["neighbours_63"]
    star = {"key": key, "neighbours": []}
    ra0, de0 = DOS[key]["ra"], DOS[key]["dec"]
    nb = sorted(nb, key=lambda x: x["dG"])
    try:
        if SKIP_CONE:
            raise RuntimeError("cone skipped (earlier attempt: 504 Gateway Timeout after 600 s)")
        cone, curl = fetch(ra0, de0, 66.0 / 3600, f"ztf_{tic}_cone66.csv")
        star["cone_url"] = curl
        star["cone_rows"] = int(len(cone))
        star["cone_objects"] = int(cone.oid.nunique()) if len(cone) else 0
    except Exception as ex:  # noqa: BLE001
        cone = None
        star["cone_error"] = f"not checked: ZTF cone query failed {ex!r}"[:300]
    time.sleep(2)
    for x in nb:
        req = depth * 10 ** (0.4 * x["dG"])
        row = {"source": x["source"], "G": x["G"], "dG": x["dG"], "sep_as": x["sep"],
               "req_depth": req}
        if req > 1:
            row["outcome"] = "cannot produce the dip (required depth > 100%)"
            star["neighbours"].append(row)
            continue
        if x["sep"] < 2.0:
            row["note"] = "unresolved from the target in ZTF (sep < 2\"); light curve is the blend"
        if cone is None:
            # fallback: 1.5" cone per neighbour, brightest first, at most MAX_SINGLE of them
            nsingle = sum(1 for r in star["neighbours"] if r.get("single_query"))
            if nsingle >= MAX_SINGLE:
                row["outcome"] = "not checked (time): cone query failed, per-neighbour cap reached"
                star["neighbours"].append(row)
                continue
            row["single_query"] = True
            try:
                df, url = fetch(x["ra"], x["dec"], 1.5 / 3600, f"ztf_{tic}_{x['source']}.csv")
                row["url"] = url
            except Exception as ex:  # noqa: BLE001
                row["outcome"] = f"not checked: ZTF query failed {ex!r}"[:300]
                star["neighbours"].append(row)
                time.sleep(2)
                continue
            time.sleep(2)
        else:
            df = cone
        if len(df) == 0:
            row["outcome"] = "no information: no ZTF light curve within 1.5\""
            star["neighbours"].append(row)
            continue
        # objects: keep the one(s) whose mean position is nearest the Gaia neighbour
        g = df.groupby("oid").agg(ra=("ra", "median"), dec=("dec", "median"), n=("mag", "size"),
                                  mag=("mag", "median"), fc=("filtercode", "first"))
        g["sep"] = np.hypot((g.ra - x["ra"]) * np.cos(np.radians(x["dec"])), g.dec - x["dec"]) * 3600
        row["ztf_objects"] = g.reset_index().to_dict("records")
        keep = g.index[g.sep < 1.5]
        d = df[df.oid.isin(keep) & (df.catflags == 0)].copy()
        if len(d) < 20:
            row["outcome"] = f"no information: only {len(d)} good ZTF points"
            star["neighbours"].append(row)
            continue
        d["btjd"] = to_btjd(d.mjd, x["ra"], x["dec"])
        f_all, s_all, t_all, b_all = [], [], [], []
        for oid, gg in d.groupby("oid"):
            fl = 10 ** (-0.4 * (gg.mag.values - np.median(gg.mag.values)))
            sg = gg.magerr.values * np.log(10) / 2.5
            z = (fl - 1) / sg
            kfac = max(1.0, 1.4826 * np.median(np.abs(z - np.median(z))))
            f_all.append(fl)
            s_all.append(sg * kfac)
            t_all.append(gg.btjd.values)
            b_all += [gg.filtercode.iloc[0]] * len(gg)
        t = np.concatenate(t_all)
        f = np.concatenate(f_all)
        s = np.concatenate(s_all)
        b = np.array(b_all)
        row["n_points"] = int(len(t))
        row["btjd_span"] = [float(t.min()), float(t.max())]
        row["robust_scatter_ppt"] = float(1.4826e3 * np.median(np.abs(f - np.median(f))))
        row["bands"] = {k: int((b == k).sum()) for k in np.unique(b)}
        row["test_P"] = test(t, f, s, eph, D, req)
        row["test_2P_even"] = test(t, f, s, eph, D, req, parity=0)
        row["test_2P_odd"] = test(t, f, s, eph, D, req, parity=1)
        tp = row["test_P"]
        if tp.get("supported_any_ephem"):
            row["outcome"] = "blend supported on this neighbour"
        elif tp.get("excluded_all_ephem"):
            row["outcome"] = "blend on this neighbour excluded (P fold)"
        else:
            row["outcome"] = "no information"
        if "note" in row and row["outcome"].startswith("blend on"):
            row["outcome"] += " [blend light curve]"
        star["neighbours"].append(row)
        print(tic, x["source"], x["G"], round(req, 4), row["n_points"],
              tp.get("min_in_transit_pts"), tp.get("frac_ephem_ge5pts"), row["outcome"], flush=True)
    out["stars"][str(tic)] = star
    prev = json.loads(Path(outj).read_text()) if Path(outj).exists() else {"stars": {}}
    prev["stars"].update(out["stars"])
    out["stars"] = prev["stars"]
    Path(outj).write_text(json.dumps(out, indent=1, default=float))
Path(outj).write_text(json.dumps(out, indent=1, default=float))
