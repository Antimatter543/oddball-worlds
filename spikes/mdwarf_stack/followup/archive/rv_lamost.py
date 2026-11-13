"""Re-query LAMOST DR11 (V/162) and DR7 (V/156) and GALAH DR4 (II/385) with robust cell handling;
extract per-epoch RVs and test the pre-registered binarity rule.
Usage: rv_lamost.py OUTJSON"""
import json
import math
import sys
import time

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from scipy import stats

outj = sys.argv[1]
STARS = {2102868: (79.33369680453, 32.12492877993, 180614374211036416),
         234295610: (357.98397801044, -64.7931962242, 6485423779719878016),
         146636926: (42.33950035414, -44.26805062609, 4947259307983420928),
         141816993: (205.44368664769, 58.25523811632, 1658968059093995136),
         57830249: (174.89049075802, -30.66688311479, 3478940625208241920)}
SERVERS = ("vizier.idia.ac.za", "vizier.cfa.harvard.edu")
log, res = {}, {}


def cell(v):
    if np.ma.is_masked(v):
        return None
    try:
        a = np.asarray(v)
        if a.ndim == 0:
            x = a.item()
            if isinstance(x, bytes):
                x = x.decode()
            if isinstance(x, float) and (math.isnan(x) or abs(x) >= 9.9e8):
                return None  # sentinel / NaN
            return x
        return str(v)
    except Exception:  # noqa: BLE001
        return str(v)


for tic, (ra, de, gid) in STARS.items():
    res[tic] = {}
    cats = ["II/385"] + (["V/162", "V/156"] if de > -15 else [])
    for cat in cats:
        for server in SERVERS:
            try:
                v = Vizier(columns=["**", "_r"], row_limit=500, vizier_server=server, timeout=120)
                tl = v.query_region(SkyCoord(ra * u.deg, de * u.deg), radius=10 * u.arcsec,
                                    catalog=cat)
                res[tic][cat] = {t.meta.get("name", cat): [{k: cell(r[k]) for k in t.colnames}
                                                          for r in t] for t in tl}
                log[f"{tic}:{cat}"] = f"ok {server} " + str({k: len(x) for k, x in res[tic][cat].items()})
                break
            except Exception as ex:  # noqa: BLE001
                log[f"{tic}:{cat}@{server}"] = f"failed {ex!r}"[:200]
            time.sleep(1)
        time.sleep(1)


def gid_ok(r, gid):
    for k in ("GaiaDR3", "GaiaDR2", "Gaia", "DR3Name", "dr3_source_id", "Source"):
        if k in r and r[k] is not None:
            return str(r[k]).strip().endswith(str(gid))
    return None


summary = {}
for tic, cats in res.items():
    gid = STARS[tic][2]
    s = {}
    for cat, tabs in cats.items():
        for tn, rows in tabs.items():
            # keep rows that are the target (Gaia id when present, else _r < 6")
            keep = []
            for r in rows:
                g = gid_ok(r, gid)
                if g is True or (g is None and (r.get("_r") or 99) < 6.0):
                    keep.append(r)
            if not keep:
                s[tn] = {"rows": len(rows), "target_rows": 0}
                continue
            rvcols = [c for c in ("RVbr1", "RVlasp1", "RV", "HRV", "rv_comp_1", "rv_gauss",
                                  "rv", "RVr1", "RVb1") if c in keep[0]]
            ent = {"rows": len(rows), "target_rows": len(keep), "rv_cols": rvcols}
            if rvcols:
                c = rvcols[0]
                ec = "e_" + c if "e_" + c in keep[0] else ("e_rv_comp_1" if "e_rv_comp_1" in keep[0] else None)
                tcol = next((k for k in ("MJD", "LMJD", "Obs.Date", "ObsDate", "mjd") if k in keep[0]), None)
                pts = [(r.get(tcol), r.get(c), r.get(ec) if ec else None) for r in keep]
                pts = [p for p in pts if p[1] is not None and p[2] not in (None, 0)]
                ent["rv_col"], ent["err_col"], ent["time_col"] = c, ec, tcol
                ent["points"] = pts
                if len(pts) >= 2 and tcol in ("MJD", "LMJD", "mjd"):
                    # per-night weighted mean
                    nights = {}
                    for t, rv, e in pts:
                        nights.setdefault(int(math.floor(float(t))), []).append((float(rv), float(e)))
                    nm = []
                    for n, v in sorted(nights.items()):
                        a = np.array(v)
                        w = 1 / a[:, 1] ** 2
                        nm.append((n, float(np.sum(w * a[:, 0]) / np.sum(w)),
                                   float(1 / np.sqrt(np.sum(w))), len(a)))
                    ent["nights"] = nm
                    if len(nm) >= 2:
                        a = np.array([(x[1], x[2]) for x in nm])
                        w = 1 / a[:, 1] ** 2
                        mu = np.sum(w * a[:, 0]) / np.sum(w)
                        chi2 = float(np.sum(((a[:, 0] - mu) / a[:, 1]) ** 2))
                        dof = len(a) - 1
                        p = float(stats.chi2.sf(chi2, dof))
                        ptp = float(a[:, 0].max() - a[:, 0].min())
                        i, j = int(np.argmax(a[:, 0])), int(np.argmin(a[:, 0]))
                        zpair = float(ptp / math.hypot(a[i, 1], a[j, 1]))
                        ent.update({"n_nights": len(a), "wmean": float(mu), "chi2": chi2,
                                    "dof": dof, "p_const": p,
                                    "z_equiv": float(stats.norm.isf(p / 2)) if p > 0 else 99.0,
                                    "peak_to_peak": ptp, "max_pair_z": zpair,
                                    "binary_rule": bool((p < 0.0027) and ptp > 2.0)})
            s[tn] = ent
    summary[tic] = s

json.dump({"log": log, "summary": {str(k): v for k, v in summary.items()},
           "raw": {str(k): v for k, v in res.items()}}, open(outj, "w"), indent=1, default=str)
print(json.dumps(log, indent=0))
for tic, s in summary.items():
    for tn, e in s.items():
        print(tic, tn, {k: e.get(k) for k in ("rows", "target_rows", "rv_col", "n_nights", "wmean",
                                              "chi2", "dof", "p_const", "peak_to_peak",
                                              "max_pair_z", "binary_rule")})
