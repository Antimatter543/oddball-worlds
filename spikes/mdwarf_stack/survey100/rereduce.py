"""Is each detected transit of a 100 pc survivor present in other reductions of the same TESS
sector? (task step 4; the follow-up's statistic `followup/fu.Meter`, with injections at the RAW
depth as the follow-up's lesson demands: a detrend can eat a long dip).

Per signal (a key of <OUT>/survey100/vet/graded.jsonl):
  1. survey light curve rebuilt (get_lightcurve; product lists seeded from the survey record),
     frozen preprocessing (fu.prep); ephemeris from the detected epochs (fu.fit_ephem).
  2. raw depth: d_raw = survey box depth / s_det, s_det = median survival of a box of depth
     `depth` injected into the survey's RAW flux at +-1 and +-2 d from each detected epoch
     (followup/posthoc_inj.survival).
  3. per detected epoch and product (QLP, TGLC, TARS, eleanor-lite, T16 where MAST lists them;
     TGLC S1-55 from the bucket; our TESScut 1.5-px aperture), one-sector light curve, frozen
     preprocessing, Meter at the predicted time over +-3 sigma_pred (>= +-1 h):
       expected depth = median depth recovered from d_raw x dil injected at +-1 / +-2 d (same
       sector, real dip untouched); dil = mean CROWDSAP of the sector for aperture products that
       are not crowding-corrected (QLP, TESScut, TARS, eleanor), 1 for PDCSAP and TGLC;
       support threshold = max(3, 95th percentile of 200 random-window maxima);
       present  = window max SES >= threshold and its depth within 2 sigma of the expected depth;
       absent   = expected SES >= 3, >= 3 of 4 injections recovered at SES >= 3, >= 80% of the
                  window covered, and every covered position has depth < expected - 2 sigma;
       no information otherwise (reported with the numbers).
     The survey light curve itself is evaluated the same way as a reference row.
Writes <OUT>/survey100/vet/rereduce.jsonl. Downloads stay in memory (cutout zips: a temp dir that
is deleted at once).

A joint test per product (all evaluated epochs, inverse-variance mean depth at the predicted
times vs the same mean of the expected depths) goes to rereduce_joint.json: present if the joint
SES >= 3 and the depth is within 2 sigma of the expected one; absent if the expected joint SES
>= 3 and the depth is below expected - 2 sigma. TESScut cutouts are fetched for the MAX_TESSCUT
(default 4) sectors holding the strongest epochs only.

Usage: python -I survey100/rereduce.py KEY [KEY ...]
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "followup"))
sys.path.insert(0, str(HERE))
import fu
from common100 import VET

os.environ.setdefault("VET_LC_DIR", str(fu.DATA / "lc_cache" / "lc"))
OUTF = VET / "rereduce.jsonl"
CORRECTED = {"survey", "tglc"}
OFFS = (-2.0, -1.0, 1.0, 2.0)


def one_sector(lc, s):
    m = lc["sector"] == s
    return {k: (v[m] if isinstance(v, np.ndarray) and len(v) == len(m) else v) for k, v in lc.items()}


def survival(lc, pp, pos, D, depth):
    ppi = fu.prep(fu.inject_box(lc, pos, D, depth))
    _, i1, i2 = np.intersect1d(pp["idx"], ppi["idx"], return_indices=True)
    t = (pp["idx"][i1] + 0.5) * fu.BIN_D
    w = np.abs(t - pos) < D / 2
    if w.sum() < 3:
        return np.nan
    return float(np.mean(pp["flux"][i1][w] - ppi["flux"][i2][w]) / depth)


def covered(t, x, D):
    return np.sum(np.abs(t - x) < D / 2) >= 3


def evaluate(lc, tc, half, D, d_inj, avoid, seed):
    pp = fu.prep(lc)
    met = fu.Meter(pp, D, exclude=avoid)
    win = met.window(tc, half)
    cov = [m for m in win if m is not None and np.isfinite(m["ses"])]
    res = {"n_pos": len(win), "cover": len(cov) / max(1, len(win))}
    at = met.at(tc)
    res["at_pred"] = None if at is None else {k: at[k] for k in ("depth", "sigma", "ses", "cover")}
    if cov:
        b = max(cov, key=lambda m: m["ses"])
        res["best"] = {"dt_h": 24 * (b["t"] - tc), "depth": b["depth"], "sigma": b["sigma"],
                       "ses": b["ses"]}
    inj = []
    tt = met.t
    for o in OFFS:
        pos = tc + o
        if not covered(tt, pos, D):
            continue
        ppi = fu.prep(fu.inject_box(lc, pos, D, d_inj))
        mi = fu.Meter(ppi, D, exclude=avoid).at(pos)
        m0 = met.at(pos)
        if mi is None or m0 is None:
            continue
        inj.append({"off_d": o, "depth": mi["depth"] - m0["depth"], "ses": mi["ses"],
                    "sigma": mi["sigma"]})
    res["inj"] = inj
    rw = fu.random_windows(pp, D, half, avoid, n=200, seed=seed)
    thr = max(3.0, float(np.percentile(rw, 95))) if len(rw) >= 20 else float("nan")
    res["thr"] = thr
    if not inj or not cov:
        res["outcome"] = "no information (no window data or no injection position)"
        return res
    exp_d = float(np.median([i["depth"] for i in inj]))
    sig = float(np.median([m["sigma"] for m in cov]))
    res.update(exp_depth=exp_d, exp_ses=exp_d / sig, n_rec=sum(i["ses"] >= 3 for i in inj))
    b = res["best"]
    if np.isfinite(thr) and b["ses"] >= thr and abs(b["depth"] - exp_d) < 2 * b["sigma"]:
        res["outcome"] = "present"
    elif (res["exp_ses"] >= 3 and res["n_rec"] >= min(3, len(inj)) and len(inj) >= 3
          and res["cover"] >= 0.8 and all(m["depth"] < exp_d - 2 * m["sigma"] for m in cov)):
        res["outcome"] = "absent"
    else:
        res["outcome"] = "no information"
    return res


def products_for(tic, ra, dec, sector, gaia, hl, tesscut=True):
    import products

    out = []
    for kind in ("qlp", "tglc", "tars", "eleanor", "t16"):
        uri = hl.get((kind, sector))
        try:
            if kind == "qlp":
                out.append(("qlp",) + products.qlp(tic, sector, uri))
            elif kind == "tglc" and sector <= 55 and gaia:
                out.append(("tglc",) + products.tglc_s3(gaia, ra, dec, sector))
            elif uri:
                out.append((kind,) + products.generic_hlsp(uri, sector, kind))
        except FileNotFoundError:
            if uri and kind == "tglc":
                try:
                    out.append(("tglc",) + products.generic_hlsp(uri, sector, "tglc"))
                except Exception as e:  # noqa: BLE001
                    out.append((kind, None, {"error": repr(e)[:200]}))
        except Exception as e:  # noqa: BLE001
            out.append((kind, None, {"error": repr(e)[:200]}))
    if not tesscut:
        return out
    try:
        out.append(("tesscut",) + products.tesscut(ra, dec, sector))
    except Exception as e:  # noqa: BLE001
        out.append(("tesscut", None, {"error": repr(e)[:200]}))
    return out


def joint(rows):
    """Per product: inverse-variance mean of the depth at the predicted times over all evaluated
    epochs, against the same mean of the expected (injected, raw-depth) depths."""
    res = {}
    for prod in sorted({r["product"] for r in rows}):
        rr = [r for r in rows if r["product"] == prod and r.get("at_pred") and r.get("exp_depth")
              is not None and np.isfinite(r["at_pred"]["sigma"])]
        if not rr:
            continue
        w = np.array([1 / r["at_pred"]["sigma"] ** 2 for r in rr])
        d = float(np.sum(w * [r["at_pred"]["depth"] for r in rr]) / w.sum())
        ex = float(np.sum(w * [r["exp_depth"] for r in rr]) / w.sum())
        sg = float(1 / np.sqrt(w.sum()))
        res[prod] = {"n_epochs": len(rr), "depth": d, "sigma": sg, "ses": d / sg, "exp_depth": ex,
                     "exp_ses": ex / sg, "present": d / sg >= 3 and abs(d - ex) < 2 * sg,
                     "absent": ex / sg >= 3 and d < ex - 2 * sg}
    return res


def run(key, rows, info, sapm):
    from get_lightcurve import get_lightcurve
    from tess_test import mast_hlsp

    r = rows[key]
    s, tic = r["stats"], r["tic"]
    st = info[str(tic)]
    g = (st.get("gaia") or {}).get("target") or {}
    ra, dec = g.get("ra", st["ra"]), g.get("dec", st["dec"])
    gaia = int(g["source"]) if g.get("source") else None
    lc, _ = get_lightcurve(tic, 2_000_000_000)
    pp = fu.prep(lc)
    D = s["dur_h"] / 24
    e = fu.fit_ephem(s)
    t = (pp["idx"] + 0.5) * fu.BIN_D
    surv = []
    for ep in s["epochs"]:
        for o in OFFS:
            if covered(t, ep + o, D):
                surv.append(survival(lc, pp, ep + o, D, s["depth"]))
    s_det = float(np.nanmedian(surv)) if surv else float("nan")
    d_raw = s["depth"] / s_det if s_det and s_det > 0.05 else float("nan")
    crowd = {int(m["sector"]): m.get("crowdsap") for m in sapm.get(str(tic), {}).get("sectors", [])}
    try:
        hl = mast_hlsp(tic)
        hl_status = "ok"
    except Exception as ex:  # noqa: BLE001
        hl, hl_status = {}, f"failed {ex!r}"[:150]
    head = {"key": key, "tic": tic, "P": e["P"], "T0": e["T0"], "D_h": 24 * D, "depth": s["depth"],
            "s_det": s_det, "d_raw": d_raw, "hlsp_listing": hl_status,
            "hlsp": sorted(f"{k}:{n}" for k, n in hl)}
    out = []
    avoid = list(s["epochs"])
    max_tc = int(os.environ.get("MAX_TESSCUT", "4"))
    order = np.argsort(-np.asarray(s["epoch_ses"]))
    tc_secs = []
    for j in order:
        if int(s["epoch_sector"][j]) not in tc_secs and len(tc_secs) < max_tc:
            tc_secs.append(int(s["epoch_sector"][j]))
    cache = {}
    for i, (ep, sec) in enumerate(zip(s["epochs"], s["epoch_sector"])):
        n = round((ep - e["T0"]) / e["P"])
        tp, sp = fu.predict(e, n)
        half = max(3 * sp, 1 / 24)
        dil = crowd.get(int(sec)) or 1.0
        prods = [("survey", one_sector(lc, sec), {"product": "survey"})]
        if int(sec) not in cache:
            cache.clear()
            cache[int(sec)] = products_for(tic, ra, dec, int(sec), gaia, hl,
                                           tesscut=int(sec) in tc_secs)
        prods += cache[int(sec)]
        for kind, plc, meta in prods:
            row = dict(head, epoch=i, n=n, sector=int(sec), t_pred=tp, sig_pred_h=24 * sp,
                       ses_survey=s["epoch_ses"][i], product=kind,
                       meta={k: v for k, v in (meta or {}).items() if k in
                             ("product", "source", "column", "n_cad", "error", "aperture_px",
                              "edge_px", "uri")})
            if plc is None or len(plc["idx"]) < 50:
                row["outcome"] = "not checked: " + ((meta or {}).get("error") or "too little data")
                out.append(row)
                continue
            fac = 1.0 if kind in CORRECTED else float(dil)
            try:
                row.update(dil=fac, **evaluate(plc, tp, half, D, d_raw * fac, avoid, seed=1000 + i))
            except Exception as ex:  # noqa: BLE001
                row["outcome"] = f"not checked: {ex!r}"[:200]
                row["trace"] = traceback.format_exc()[-400:]
            out.append(row)
            print(key, i, sec, kind, row.get("outcome"), {k: (round(row[k], 5) if isinstance(row.get(k), float) else None)
                                                          for k in ("exp_depth", "exp_ses")},
                  (row.get("best") or {}).get("ses"), flush=True)
    js = joint(out)
    for prod, v in js.items():
        print(key, "JOINT", prod, {k: (round(x, 5) if isinstance(x, float) else x) for k, x in v.items()},
              flush=True)
    fp = VET / "rereduce_joint.json"
    old = json.loads(fp.read_text()) if fp.exists() else {}
    old[key] = {"s_det": s_det, "d_raw": d_raw, "tesscut_sectors": tc_secs, "joint": js}
    fp.write_text(json.dumps(old, indent=1, default=float))
    return out


def main():
    rows = {json.loads(x)["key"]: json.loads(x) for x in open(VET / "graded.jsonl") if x.strip()}
    info = json.loads((VET / "star_info.json").read_text())
    sapm = json.loads((VET / "sap_meta.json").read_text()) if (VET / "sap_meta.json").exists() else {}
    done = {json.loads(x)["key"] for x in open(OUTF)} if OUTF.exists() else set()
    with open(OUTF, "a") as fh:
        for k in sys.argv[1:]:
            if k in done:
                continue
            fh.writelines(json.dumps(row, default=float) + "\n" for row in run(k, rows, info, sapm))
            fh.flush()


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
