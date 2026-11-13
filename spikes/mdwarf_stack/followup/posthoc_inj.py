"""POST-HOC (not pre-registered, not used for any verdict): the frozen preprocessing removes
part of an injected box (a 4-h box loses about half to the 0.5 d biweight + harm), while the
signal's quoted depth is already a post-detrend depth. So the pre-registered injection (raw
depth = signal depth) is weaker than the real dip would be. Here, for every new-photon epoch
whose window has data, the injection depth is scaled so that its post-detrend depth equals the
signal depth (scale = 1 / survival, survival measured on the same light curve at 5 positions
by differencing injected and plain preprocessed fluxes), and the refute rule is re-applied.
Variant "matched" (fairer, also post-hoc): the signal's quoted depth is post-detrend in the
survey light curve, where the detrend also removed part of it. Raw depth = signal depth /
survival at the detected epochs in the survey light curve (s_det); the dip expected in the new
light curve is raw depth x local survival; injections use the raw depth, and exclusion is tested
against the expected local depth.
Writes FU_OUT/posthoc_inj.json. Usage: python -I followup/posthoc_inj.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fu
from get_lightcurve import get_lightcurve
from tess_test import LCD

NEW = ("gap_in_searched_sector", "sector_not_searched")


def survival(lc, pp, pos, D, depth):
    ppi = fu.prep(fu.inject_box(lc, pos, D, depth))
    _, i1, i2 = np.intersect1d(pp["idx"], ppi["idx"], return_indices=True)
    t = (pp["idx"][i1] + 0.5) * fu.BIN_D
    w = np.abs(t - pos) < D / 2
    if w.sum() < 3:
        return np.nan
    return float(np.mean(pp["flux"][i1][w] - ppi["flux"][i2][w]) / depth)


def main():
    rows = [json.loads(x) for x in (fu.FU_OUT / "tess_epochs.jsonl").read_text().splitlines()]
    eph = json.loads((fu.FU_OUT / "ephem.json").read_text())
    out = []
    s_det = {}
    for r in rows:
        if r["status"] not in NEW or not r.get("n_cov") or r["product"] == "survey":
            continue
        e = eph[r["key"]]["ephem"]
        D, depth = e["D"], e["depth"]
        z = np.load(LCD / f"tic{r['tic']}_s{r['sector']}_{r['product']}.npz", allow_pickle=False)
        lc = {"idx": z["idx"], "flux": z["flux"].astype(float),
              "sector": np.full(len(z["idx"]), r["sector"], np.int16)}
        pp = fu.prep(lc)
        half = fu.NSIG_WIN * r["sig_h"] / 24
        pos = np.linspace(r["t_pred"] - half, r["t_pred"] + half, 5)
        avoid = [x["t"] for x in eph[r["key"]]["epochs"]]
        surv = [survival(lc, pp, float(x), D, depth) for x in pos]
        s = float(np.nanmedian(surv))
        rec = []
        for x in pos:
            ppi = fu.prep(fu.inject_box(lc, float(x), D, depth / max(s, 0.1)))
            m = fu.Meter(ppi, D, exclude=avoid + [float(x)]).at(float(x))
            rec.append(None if m is None else round(float(m["ses"]), 2))
        allrec = all(v is not None and v >= fu.SES_REC for v in rec)
        refute = r["win_cover"] >= fu.WIN_COVER_MIN and allrec and r["excluded_everywhere"]
        if r["key"] not in s_det:
            slc, _ = get_lightcurve(r["tic"], max_bytes=200_000_000)
            spp = fu.prep(slc)
            s_det[r["key"]] = float(np.nanmedian([survival(slc, spp, float(x), D, depth)
                                                  for x in avoid]))
        sd = s_det[r["key"]]
        raw = depth / max(sd, 0.1)
        exp_loc = raw * s
        met = fu.Meter(pp, D, exclude=avoid)
        cov = [m for m in met.window(r["t_pred"], half) if m is not None and np.isfinite(m["ses"])]
        excl2 = bool(cov) and all(m["depth"] < exp_loc - 2 * m["sigma"] for m in cov)
        rec2 = []
        for x in pos:
            ppi = fu.prep(fu.inject_box(lc, float(x), D, raw))
            m = fu.Meter(ppi, D, exclude=avoid + [float(x)]).at(float(x))
            rec2.append(None if m is None else round(float(m["ses"]), 2))
        allrec2 = all(v is not None and v >= fu.SES_REC for v in rec2)
        ap = met.at(r["t_pred"])
        matched = {"s_det": sd, "raw_depth": raw, "exp_local_depth": exp_loc,
                   "at_pred": None if ap is None else [float(ap["depth"]), float(ap["sigma"])],
                   "inj_ses": rec2, "all_rec": allrec2, "excluded_everywhere": excl2,
                   "outcome": "refuted" if (r["win_cover"] >= fu.WIN_COVER_MIN and allrec2
                                            and excl2) else "no information"}
        o = {"id": r["id"], "survival": surv, "matched": matched, "scale": 1 / max(s, 0.1), "inj_ses": rec,
             "all_rec": allrec, "excluded_everywhere": r["excluded_everywhere"],
             "posthoc_outcome": "refuted" if refute else "no information",
             "prereg_outcome": r["outcome"]}
        out.append(o)
        print(json.dumps(o, default=float))
    (fu.FU_OUT / "posthoc_inj.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
