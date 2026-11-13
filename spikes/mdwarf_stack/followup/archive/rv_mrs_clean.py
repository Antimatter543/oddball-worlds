"""LAMOST MRS per-night RVs with quality cuts (snr >= 10, duplicate rows removed, per-night error =
max(formal error of the mean, within-night scatter / sqrt(n))); binarity rule: constancy chi2
p < 0.0027 and peak-to-peak > 2 km/s. Usage: rv_mrs_clean.py IN_JSON OUT_JSON"""
import json
import math
import sys

import numpy as np
from scipy import stats

d = json.load(open(sys.argv[1]))
out = {}
for tic, cats in d["raw"].items():
    for cat, tabs in cats.items():
        for tn, rows in tabs.items():
            if not rows or "RVbr1" not in rows[0]:
                continue
            seen, pts = set(), []
            for r in rows:
                if r.get("RVbr1") is None or r.get("e_RVbr1") in (None, 0) or (r.get("snr") or 0) < 10:
                    continue
                k = (r["MJD"], r["RVbr1"], r["e_RVbr1"])
                if k in seen:
                    continue
                seen.add(k)
                pts.append((int(r["MJD"]), float(r["RVbr1"]), float(r["e_RVbr1"]), float(r["snr"])))
            nights = {}
            for m, rv, e, s in pts:
                nights.setdefault(m, []).append((rv, e))
            nm = []
            for m, v in sorted(nights.items()):
                a = np.array(v)
                w = 1 / a[:, 1] ** 2
                mu = float(np.sum(w * a[:, 0]) / np.sum(w))
                ef = float(1 / math.sqrt(np.sum(w)))
                es = float(np.std(a[:, 0], ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0
                nm.append({"mjd": m, "rv": mu, "err": max(ef, es), "n": len(a),
                           "within_night_std": float(np.std(a[:, 0], ddof=1)) if len(a) > 1 else None})
            e = {"n_exposures_kept": len(pts), "nights": nm}
            if len(nm) >= 2:
                rv = np.array([x["rv"] for x in nm])
                er = np.array([x["err"] for x in nm])
                w = 1 / er**2
                mu = np.sum(w * rv) / np.sum(w)
                chi2 = float(np.sum(((rv - mu) / er) ** 2))
                p = float(stats.chi2.sf(chi2, len(rv) - 1))
                e.update({"chi2": chi2, "dof": len(rv) - 1, "p_const": p,
                          "peak_to_peak": float(rv.max() - rv.min()),
                          "binary_rule": bool(p < 0.0027 and rv.max() - rv.min() > 2)})
                # robustness: drop the single-exposure night(s)
                k = np.array([x["n"] for x in nm]) >= 2
                if k.sum() >= 2 and k.sum() < len(k):
                    mu2 = np.sum(w[k] * rv[k]) / np.sum(w[k])
                    c2 = float(np.sum(((rv[k] - mu2) / er[k]) ** 2))
                    e["multi_exposure_nights_only"] = {
                        "n": int(k.sum()), "chi2": c2, "p_const": float(stats.chi2.sf(c2, k.sum() - 1)),
                        "peak_to_peak": float(rv[k].max() - rv[k].min())}
            out[f"{tic}:{tn}"] = e
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps(out, indent=1))
