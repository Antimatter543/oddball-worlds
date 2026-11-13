"""TIC 57830249: which period aliases P_n = (t2 - t1) / n does the survey light curve allow?

For each n = 1..180, the epochs of alias n other than the two detected ones that fall in the
survey light curve (frozen preprocessing) are combined: joint depth at the predicted centres
(weights 1/sigma^2, sigma with the per-sector red-noise factor), observed joint SES, and the same
for a copy with the signal's box injected at all those epochs (measures what a real dip would
give after the detrend). Pre-registered rule (docs/spikes/mdwarf_followup.md): an alias is
excluded only if the injected copy is recovered (joint SES >= 4) and the observed joint SES is
more than 3 below it; "no other epoch in the data" leaves it allowed.
Writes FU_OUT/alias_57830249.json. Usage: python -I followup/alias.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fu
from get_lightcurve import get_lightcurve

KEY = "survey-57830249-long#2"


def joint_at(met, times):
    pts = [met.at(t) for t in times]
    pts = [p for p in pts if p is not None and np.isfinite(p["sigma"])]
    if not pts:
        return None
    w = np.array([1 / p["sigma"] ** 2 for p in pts])
    d = np.array([p["depth"] for p in pts])
    dj, sj = float(np.sum(w * d) / np.sum(w)), float(1 / np.sqrt(np.sum(w)))
    return {"n_ep": len(pts), "depth": dj, "sigma": sj, "ses": dj / sj}


def main():
    sig = fu.load_signals()[KEY]["stats"]
    t1, t2 = sig["epochs"]
    D, depth = sig["dur_h"] / 24, sig["depth"]
    lc, _ = get_lightcurve(57830249, max_bytes=200_000_000)
    pp = fu.prep(lc)
    tlo, thi = (pp["idx"].min() + 0.5) * fu.BIN_D, (pp["idx"].max() + 0.5) * fu.BIN_D
    met = fu.Meter(pp, D, exclude=[t1, t2])
    out = []
    for n in range(1, 181):
        P = (t2 - t1) / n
        k = np.arange(int(np.floor((tlo - t1) / P)) - 1, int(np.ceil((thi - t1) / P)) + 2)
        tt = t1 + k * P
        others = [float(x) for x in tt if min(abs(x - t1), abs(x - t2)) > 0.5
                  and met.at(float(x)) is not None]
        row = {"n": n, "P": P, "n_other": len(others)}
        if others:
            obs = joint_at(met, others)
            lci = lc
            for x in others:
                lci = fu.inject_box(lci, x, D, depth)
            ppi = fu.prep(lci)
            inj = joint_at(fu.Meter(ppi, D, exclude=[t1, t2] + others), others)
            row.update(obs=obs, inj=inj, exp_naive=depth / obs["sigma"])
            rec = inj is not None and inj["ses"] >= 4
            row["outcome"] = ("excluded" if rec and obs["ses"] < inj["ses"] - 3 else
                              "allowed (dip seen at other epochs?)" if obs["ses"] >= 3 else
                              "allowed (not excluded)")
        else:
            row["outcome"] = "allowed (no other epoch in data)"
        out.append(row)
    (fu.FU_OUT / "alias_57830249.json").write_text(json.dumps(out, indent=1, default=float))
    allowed = [r for r in out if not r["outcome"].startswith("excluded")]
    print(f"{len(out)} aliases; excluded {len(out) - len(allowed)}; allowed {len(allowed)}")
    for r in allowed:
        extra = ""
        if r.get("obs"):
            extra = (f" other epochs {r['n_other']}: obs SES {r['obs']['ses']:.1f} "
                     f"({1e3 * r['obs']['depth']:.2f}+-{1e3 * r['obs']['sigma']:.2f} ppt), "
                     f"injected SES {r['inj']['ses']:.1f}")
        print(f"n={r['n']:3d} P={r['P']:9.3f} d {r['outcome']}{extra}")


if __name__ == "__main__":
    main()
