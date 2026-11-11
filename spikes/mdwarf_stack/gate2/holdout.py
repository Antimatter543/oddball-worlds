"""Gate 2 fresh hold-out of known transiting planets (built before any of them is searched).

build: union of
  - exoplanet.eu (cached copy of 2026-11-11): planet_status Confirmed, detection Primary Transit,
    host Teff < 3900 K, P > 10 d, R < 2.5 R_earth (radius in R_J x 11.209);
  - ExoFOP TOI table (cached 2026-11-11): TFOPWG disposition CP or KP with the same cuts;
  (the NASA Exoplanet Archive TAP returned a maintenance page on 2026-11-11),
  minus LHS 1140, TOI-700 and Gliese 12 (already examined). Planets with P > 60 d are outside the
  10-60 d search range: listed, not scored. Ephemeris: the TOI table (TESS) when the planet is a TOI,
  otherwise exoplanet.eu. TIC ids come from the TOI table or a 10" MAST TIC cone (nearest star).
predict: per planet, sectors in hand, transits in the data and the predicted MES, from the depth
  and the per-sector noise of the PREPROCESSED light curve (no search result is used):
      MES_pred = eff * depth * sqrt(sum over in-data in-transit bins of 1/sigma_sector^2)
  eff = median measured/predicted MES of DEV injections (fixed in the frozen method).

Usage: python -I gate2/holdout.py build | predict
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BIN_D, CAT_DIR, OUT

RJ_RE = 11.209
EXCLUDE_TIC = {92226327, 150428135, 52005579}
EXCLUDE_NAMES = ("LHS 1140", "TOI-700", "Gliese 12", "GJ 12")
G2 = OUT / "gate2"


def _sexa_to_deg(ra, dec):
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    c = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    return c.ra.deg, c.dec.deg


def build() -> None:
    e = pd.read_csv(CAT_DIR / "exoplanet_eu.csv", low_memory=False)
    e = e.assign(re=e["radius"] * RJ_RE)
    m = ((e["planet_status"] == "Confirmed") & e["detection_type"].astype(str).str.contains("Transit")
         & (e["star_teff"] < 3900) & (e["orbital_period"] > 10) & (e["re"] < 2.5))
    e = e[m & ~e["star_name"].astype(str).str.startswith(EXCLUDE_NAMES)].copy()
    t = pd.read_csv(CAT_DIR / "toi.csv", comment="#")
    mt = (t["TFOPWG Disposition"].isin(["CP", "KP"]) & (t["Stellar Eff Temp (K)"] < 3900)
          & (t["Period (days)"] > 10) & (t["Planet Radius (R_Earth)"] < 2.5))
    t = t[mt & ~t["TIC ID"].isin(EXCLUDE_TIC)].copy()
    t["ra_d"], t["dec_d"] = _sexa_to_deg(t["RA"].to_numpy(), t["Dec"].to_numpy())
    rows = []
    used_toi = set()
    for _, r in e.iterrows():
        # same planet in the TOI table: host within 15" and period within 1%
        d = np.hypot((t["ra_d"] - r["ra"]) * math.cos(math.radians(r["dec"])), t["dec_d"] - r["dec"])
        cand = t[(d * 3600 < 15) & (np.abs(t["Period (days)"] / r["orbital_period"] - 1) < 0.01)]
        row = {"planet": r["name"], "host": r["star_name"], "ra": r["ra"], "dec": r["dec"],
               "P_eu": r["orbital_period"], "rp_re": r["re"], "teff": r["star_teff"],
               "rstar": r["star_radius"], "mstar": r["star_mass"]}
        if len(cand):
            c = cand.iloc[0]
            used_toi.add(c["TOI"])
            row.update(tic=int(c["TIC ID"]), toi=float(c["TOI"]), P=float(c["Period (days)"]),
                       t0=float(c["Epoch (BJD)"]) - 2457000.0, dur_h=float(c["Duration (hours)"]),
                       depth_ppm=float(c["Depth (ppm)"]), tmag=float(c["TESS Mag"]), eph="TOI")
        else:
            tz = r["tzero_tr"]
            row.update(tic=None, toi=None, P=float(r["orbital_period"]),
                       t0=float(tz) - 2457000.0 if np.isfinite(tz) else np.nan, dur_h=np.nan,
                       depth_ppm=np.nan, tmag=np.nan, eph="exoplanet.eu")
        rows.append(row)
    for _, c in t[~t["TOI"].isin(used_toi)].iterrows():
        rows.append({"planet": f"TOI-{c['TOI']}", "host": f"TIC {int(c['TIC ID'])}",
                     "ra": c["ra_d"], "dec": c["dec_d"], "P_eu": np.nan,
                     "rp_re": float(c["Planet Radius (R_Earth)"]),
                     "teff": float(c["Stellar Eff Temp (K)"]),
                     "rstar": float(c["Stellar Radius (R_Sun)"]),
                     "mstar": float(c["Stellar Mass (M_Sun)"]), "tic": int(c["TIC ID"]),
                     "toi": float(c["TOI"]), "P": float(c["Period (days)"]),
                     "t0": float(c["Epoch (BJD)"]) - 2457000.0,
                     "dur_h": float(c["Duration (hours)"]), "depth_ppm": float(c["Depth (ppm)"]),
                     "tmag": float(c["TESS Mag"]), "eph": "TOI"})
    df = pd.DataFrame(rows)
    # TIC for non-TOI hosts: nearest TIC star within 10"
    from astroquery.mast import Catalogs
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    for host in df.loc[df["tic"].isna(), "host"].unique():
        r = df[df["host"] == host].iloc[0]
        try:
            cat = Catalogs.query_region(SkyCoord(r["ra"], r["dec"], unit="deg"), radius=10 * u.arcsec,
                                        catalog="TIC")
        except Exception as ex:  # noqa: BLE001
            print("TIC lookup failed", host, repr(ex)[:100])
            continue
        if len(cat) == 0:
            continue
        cat.sort("dstArcSec")
        df.loc[df["host"] == host, "tic"] = int(cat["ID"][0])
        df.loc[df["host"] == host, "tmag"] = float(cat["Tmag"][0])
        for col, src in (("rstar", "rad"), ("mstar", "mass")):
            v = float(cat[src][0]) if np.isfinite(cat[src][0]) else np.nan
            df.loc[(df["host"] == host) & ~np.isfinite(df[col]), col] = v
    # the same planet listed twice with an integer period ratio (exoplanet.eu TOI-2285 b at
    # 13.64 d vs TOI-2285.01 at 27.27 d): keep the TOI (TESS) entry
    drop = []
    for i, r in df[df["eph"] == "exoplanet.eu"].iterrows():
        same = df[(df["tic"] == r["tic"]) & (df["eph"] == "TOI")]
        for _, s in same.iterrows():
            q = max(s["P"], r["P"]) / min(s["P"], r["P"])
            if 2 <= round(q) <= 3 and abs(q - round(q)) < 0.01 * round(q):
                drop.append(i)
    df = df.drop(index=drop)
    df["in_range"] = (df["P"] > 10) & (df["P"] <= 60)
    df = df.sort_values(["in_range", "tmag"], ascending=[False, True])
    df.to_csv(CAT_DIR / "holdout_g2.csv", index=False)
    print(df[["planet", "tic", "eph", "P", "rp_re", "teff", "tmag", "in_range"]].to_string())
    print(f"{len(df)} planets, {int(df['in_range'].sum())} in 10-60 d, hosts "
          f"{df.loc[df['in_range'], 'tic'].nunique()}")
    fetch_list = df.loc[df["in_range"] & df["tic"].notna(), "tic"].astype(int).unique()
    (CAT_DIR / "holdout_fetch.txt").write_text(" ".join(str(x) for x in fetch_list))


def transit_bins(idx, P, t0, dur):
    t = (idx + 0.5) * BIN_D
    n = np.round((t - t0) / P)
    return np.abs(t - t0 - n * P) < dur / 2


def predict(eff: float, rot: str) -> pd.DataFrame:
    from fetch import load
    from search2 import preprocess2
    from inject import duration_14, epochs_in_data

    df = pd.read_csv(CAT_DIR / "holdout_g2.csv")
    out = []
    for _, r in df.iterrows():
        row = r.to_dict()
        p = Path(CAT_DIR).parent / "lc" / f"tic{int(r['tic'])}.npz" if np.isfinite(r["tic"]) else None
        if p is None or not p.exists():
            row.update(n_sectors=0, n_tr=0, mes_pred=0.0, note="no TESS light curve")
            out.append(row)
            continue
        lc, meta = load(int(r["tic"]))
        pp, rinfo = preprocess2(lc, rot)
        M = r["mstar"] if np.isfinite(r["mstar"]) else 0.3
        R = r["rstar"] if np.isfinite(r["rstar"]) else 0.3
        if np.isfinite(r["dur_h"]):
            dur = r["dur_h"] / 24
        else:
            dur = duration_14(r["P"], r["rp_re"], 0.3, M, R)
        depth = (r["depth_ppm"] * 1e-6 if np.isfinite(r["depth_ppm"])
                 else (r["rp_re"] * 0.0091577 / R) ** 2)
        if not np.isfinite(r["t0"]):
            row.update(n_sectors=len(meta["sectors"]), n_tr=0, mes_pred=0.0, note="no ephemeris")
            out.append(row)
            continue
        inb = transit_bins(pp["idx"], r["P"], r["t0"], dur)
        snr = depth * math.sqrt(float(np.sum(1.0 / pp["sigma"][inb] ** 2)))
        n_tr = len(epochs_in_data(pp["idx"], r["P"], r["t0"], dur))
        row.update(n_sectors=len(meta["sectors"]), sectors=[s["sector"] for s in meta["sectors"]],
                   n_tr=n_tr, depth_used=depth, dur_used_h=dur * 24, snr_white=snr,
                   mes_pred=eff * snr if n_tr >= 2 else 0.0, prot=rinfo.get("prot"),
                   rot_applied=rinfo.get("applied"), note="")
        out.append(row)
    o = pd.DataFrame(out)
    o.to_csv(G2 / "holdout_pred.csv", index=False)
    return o


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    elif sys.argv[1] == "predict":
        cfg = json.loads((G2 / "frozen.json").read_text())
        o = predict(cfg["pred_eff"], cfg["rot"])
        print(o[["planet", "tic", "P", "rp_re", "tmag", "n_sectors", "n_tr", "snr_white",
                 "mes_pred"]].to_string())
