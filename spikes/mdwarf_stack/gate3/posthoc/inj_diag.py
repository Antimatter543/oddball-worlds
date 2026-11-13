"""Diagnose far vs 50 pc reference injection recovery: SNR, R*, sectors, detection vs SNR."""
import json
from pathlib import Path

import numpy as np

G = Path("/home/antiraedus/Projects/oddball-worlds/outputs/planets_spike/gate3/kaggle_out")
TAU = 8.054233220378304
jobs, out, noise = {}, {}, {}
for d in sorted(G.glob("*/run")):
    for x in open(d / "jobs.all.jsonl"):
        r = json.loads(x); jobs[r["id"]] = r
    for f in d.glob("out.h*.jsonl"):
        for x in open(f):
            r = json.loads(x); out[r["id"]] = r
    for x in open(d / "noise.jsonl"):
        r = json.loads(x); noise[(r["tic"], r["kind"])] = r


def det(r):
    return bool(r.get("signals") and r["inj_match"] and r["inj_match"][0][0]
                and r["signals"][0]["mes"] >= TAU)


rows = []
for i, r in out.items():
    if not r.get("inj") or "error" in r:
        continue
    j = jobs[i]
    nz = noise.get((r["tic"], j["kind"]), {})
    ps = [v[0] for v in nz.get("per_sector", {}).values() if v[0] == v[0]]
    cs = [v for v in nz.get("crowdsap", {}).values() if v == v]
    rows.append(dict(kind=j["kind"], cell=j["cell"], R=j["radius"], M=j["mass"], ns=r["n_sectors"],
                     snr=r["inj"]["snr_white"], dep=r["inj"]["depth0"], ntr=r["inj"]["n_tr_data"],
                     sig=np.median(ps) if ps else np.nan, crowd=np.median(cs) if cs else np.nan,
                     det=det(r), top=(r["signals"][0]["mes"] if r.get("signals") else 0.0)))
for c in ("m", "x"):
    print(f"== cell {c}")
    for k in ("far", "ref"):
        s = [x for x in rows if x["kind"] == k and x["cell"] == c]
        a = lambda f: np.array([x[f] for x in s], float)
        print(f"{k}: n {len(s)} det {a('det').mean():.3f}; R* med {np.nanmedian(a('R')):.3f}; depth med "
              f"{1e3*np.median(a('dep')):.2f} ppt; snr_white med {np.median(a('snr')):.1f}; sectors med "
              f"{np.median(a('ns')):.0f}; ntr med {np.median(a('ntr')):.0f}; sig2h med {np.nanmedian(a('sig')):.0f};"
              f" crowd med {np.nanmedian(a('crowd')):.2f}")
        for lo, hi in ((0, 5), (5, 8), (8, 12), (12, 20), (20, 1e9)):
            m = (a("snr") >= lo) & (a("snr") < hi)
            if m.sum():
                print(f"   snr_white {lo}-{hi}: det {int(a('det')[m].sum())}/{int(m.sum())}")
print("== R* by kind (all injections, unique stars)")
for k in ("far", "ref"):
    R = {}
    for i, j in jobs.items():
        if j["kind"] == k and "radius" in j:
            R[j["tic"]] = j["radius"]
    v = np.array([x for x in R.values() if x == x], float)
    print(k, len(v), "R* percentiles 16/50/84", np.percentile(v, [16, 50, 84]).round(3))
print("== kept-ish (detected) far injections")
d = [x for x in rows if x["kind"] == "far" and x["det"]]
print("n", len(d), "ntr median", np.median([x["ntr"] for x in d]), "ntr dist", np.unique([x["ntr"] for x in d], return_counts=True))
print("sectors median", np.median([x["ns"] for x in d]), "depth ppt med", 1e3*np.median([x["dep"] for x in d]))
