"""Part 1, source 1: the pre-registered per-epoch test on TESS products the survey did not use.

For every predicted epoch of the five signals that is (a) in a gap of the survey light curve,
(b) in a sector the survey never searched, or (c) a detected epoch (re-reduction = artefact
test), and for every product available in that sector (survey SPOC/TESS-SPOC as reference, QLP,
TGLC, eleanor-lite, T16, our TESScut photometry), on that one sector's light curve:
window coverage, expected SES, max SES and its depth, the 2-sigma exclusion at every position,
5 box injections across the window, 400 random windows -> supported / refuted / no information,
exactly as in docs/spikes/mdwarf_followup.md.

Writes FU_OUT/tess_epochs.jsonl (one row per epoch x product) and FU_OUT/tess_lc/ (small NPZ of
each single-sector light curve used, raw normalised 10-min bins, for figures).
Usage: python -I followup/tess_test.py [TIC ...]
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fu
import products
from get_lightcurve import get_lightcurve

OUTF = fu.FU_OUT / "tess_epochs.jsonl"
LCD = fu.FU_OUT / "tess_lc"
LCD.mkdir(exist_ok=True)
HLSP_KIND = {"QLP": "qlp", "TGLC": "tglc", "GSFC-ELEANOR-LITE": "eleanor", "T16": "t16", "TARS": "tars"}


def mast_hlsp(tic: int) -> dict:
    """(kind, sector) -> dataURL from the MAST observation query (all HLSPs)."""
    import download

    download._set_astroquery_ua()
    from astroquery.mast import Observations

    obs = Observations.query_criteria(target_name=[str(tic)], dataproduct_type="timeseries",
                                      obs_collection=["HLSP"])
    out = {}
    for p, s, u in zip(obs["provenance_name"], obs["sequence_number"], obs["dataURL"]):
        k = HLSP_KIND.get(str(p).upper())
        if k and str(u).endswith(".fits"):
            out.setdefault((k, int(s)), str(u))
    return out


def gaia_id(tic: int) -> int:
    """Gaia DR3 source id of the target: the nearest row of the cached catalogue research."""
    d = json.loads((fu.OUT / "survey" / "vet" / "research" / f"{tic}.json").read_text())
    rows = d["gaia"]["data"]["rows"]
    return int(min(rows, key=lambda r: r["_r"])["Source"])


def seg_edge_dist_h(pp, tc):
    t = (pp["idx"] + 0.5) * fu.BIN_D
    br = np.flatnonzero(np.diff(t) > 0.5)
    edges = np.concatenate([[t[0]], t[br], t[br + 1], [t[-1]]])
    br30 = np.flatnonzero(np.diff(t) > 0.5 / 24 + 1e-6)
    e30 = np.concatenate([[t[0]], t[br30], t[br30 + 1], [t[-1]]])
    return float(24 * np.min(np.abs(edges - tc))), float(24 * np.min(np.abs(e30 - tc)))


def evaluate(lc, tp, sig_d, D, depth, avoid, seed):
    half = fu.NSIG_WIN * sig_d
    pp = fu.prep(lc)
    met = fu.Meter(pp, D, exclude=avoid)
    ms = met.window(tp, half)
    cov = [m for m in ms if m is not None and np.isfinite(m["ses"])]
    row = {"n_pos": len(ms), "n_cov": len(cov), "win_cover": len(cov) / max(1, len(ms)),
           "cad_min": 10 * met.cad_bins}
    if not cov:
        row["outcome"] = "no information"
        row["why"] = "window not covered"
        return row, pp
    sig_med = float(np.median([m["sigma"] for m in cov]))
    best = max(cov, key=lambda m: m["ses"])
    excl = all(m["depth"] < depth - 2 * m["sigma"] for m in cov)
    rnd = fu.random_windows(pp, D, half, avoid, n=400, seed=seed)
    p95 = float(np.percentile(rnd, 95)) if len(rnd) >= 50 else float("nan")
    thr = max(fu.SES_SUPPORT_MIN, p95) if np.isfinite(p95) else float("nan")
    edge, edge30 = seg_edge_dist_h(pp, best["t"])
    # injections at 5 positions spread over the window (only covered ones are meaningful)
    pos = np.linspace(tp - half, tp + half, 5) if half > 0 else np.array([tp])
    inj = []
    for x in pos:
        lci = fu.inject_box(lc, float(x), D, depth)
        ppi = fu.prep(lci)
        mi = fu.Meter(ppi, D, exclude=list(avoid) + [float(x)]).at(float(x))
        inj.append(None if mi is None else round(float(mi["ses"]), 2))
    rec = [v is not None and v >= fu.SES_REC for v in inj]
    consistent = abs(best["depth"] - depth) < 2 * best["sigma"]
    support = (np.isfinite(thr) and best["ses"] >= thr and consistent and edge >= 1.0)
    refute = row["win_cover"] >= fu.WIN_COVER_MIN and all(rec) and excl
    row.update({"exp_ses": depth / sig_med, "sigma_ppt": 1e3 * sig_med,
                "best_t": best["t"], "best_ses": best["ses"], "best_depth_ppt": 1e3 * best["depth"],
                "best_sigma_ppt": 1e3 * best["sigma"], "best_edge_h": edge,
                "best_gap30_h": edge30,
                "at_pred": None if met.at(tp) is None else
                {k: float(v) for k, v in met.at(tp).items() if k in ("depth", "sigma", "ses")},
                "excluded_everywhere": excl, "inj_ses": inj, "inj_all_rec": all(rec),
                "rand_p95": p95, "support_thr": thr,
                "rand_frac_ge_best": float(np.mean(rnd >= best["ses"])) if len(rnd) else None,
                "depth_consistent": consistent})
    if support and refute:
        row["outcome"] = "no information"
        row["why"] = "both rules fire"
    elif support:
        row["outcome"] = "supported"
    elif refute:
        row["outcome"] = "refuted"
    else:
        row["outcome"] = "no information"
        why = []
        if row["win_cover"] < fu.WIN_COVER_MIN:
            why.append(f"window {row['win_cover']:.0%} covered")
        if not all(rec):
            why.append(f"{sum(rec)}/5 injections recovered")
        if not excl:
            why.append("dip of signal depth not excluded at every position")
        if best["ses"] >= (thr if np.isfinite(thr) else 3) and not consistent:
            why.append("significant dip of the wrong depth")
        if best["ses"] >= (thr if np.isfinite(thr) else 3) and edge < 1.0:
            why.append("best position within 1 h of a segment edge")
        row["why"] = "; ".join(why)
    return row, pp


def joint(rows):
    """Weighted joint depth over the new-photon epochs of one product (positions at prediction)."""
    pts = [r["at_pred"] for r in rows if r.get("at_pred")]
    if not pts:
        return None
    w = np.array([1 / p["sigma"] ** 2 for p in pts])
    d = np.array([p["depth"] for p in pts])
    dj = float(np.sum(w * d) / np.sum(w))
    sj = float(1 / np.sqrt(np.sum(w)))
    return {"n": len(pts), "depth_ppt": 1e3 * dj, "sigma_ppt": 1e3 * sj, "ses": dj / sj}


def run_star(key, eph, done):
    tic = eph["tic"]
    e = eph["ephem"]
    D, depth = e["D"], e["depth"]
    dz = json.loads((fu.OUT / "survey" / "vet" / "dossier.json").read_text())[key]
    ra, dec = dz["ra"], dz["dec"]
    avoid = [x["t"] for x in eph["epochs"]]
    eps = [x for x in eph["epochs"] if x["status"] != "future"]
    seen = set()
    todo = []
    for x in eps:
        # 234295610 n=58 sits on the S103/S104 boundary: test it in both sectors
        todo.append(x)
    hl = {}
    try:
        hl = mast_hlsp(tic)
        hl_status = "ok"
    except Exception as ex:  # noqa: BLE001
        hl_status = f"failed {ex!r}"[:200]
    lc_all, meta = get_lightcurve(tic, max_bytes=200_000_000)
    cache = {}
    for x in todo:
        s = x["sector"]
        kinds = ["survey"] if s in eph["sectors_used"] else []
        kinds += ["qlp"] if (("qlp", s) in hl or s <= 98) else []
        kinds += [k for k in ("tglc", "eleanor", "t16", "tars") if (k, s) in hl]
        if s <= 55 and ("tglc", s) not in hl:
            kinds += ["tglc_s3"]  # MAST's API does not list TGLC S12-S55; the bucket has them
        kinds += ["tesscut"]
        for kind in kinds:
            rid = f"{key}|{x['n']}|{s}|{kind}"
            if rid in done or rid in seen:
                continue
            seen.add(rid)
            t0 = time.time()
            row = {"id": rid, "key": key, "tic": tic, "n": x["n"], "t_pred": x["t"],
                   "sig_h": x["sig_h"], "sector": s, "status": x["status"],
                   "detected": x["detected"], "product": kind, "hlsp_query": hl_status}
            try:
                if (s, kind) not in cache:
                    if kind == "survey":
                        m = lc_all["sector"] == s
                        lc = {k: lc_all[k][m] for k in ("idx", "flux", "sector")}
                        info = {"product": "survey " + next(
                            (q["product"] for q in meta["sectors"] if int(q["sector"]) == s), "?")}
                    elif kind == "qlp":
                        lc, info = products.qlp(tic, s, hl.get(("qlp", s)))
                    elif kind == "tesscut":
                        lc, info = products.tesscut(ra, dec, s)
                    elif kind == "tglc_s3":
                        lc, info = products.tglc_s3(gaia_id(tic), ra, dec, s)
                    else:
                        lc, info = products.generic_hlsp(hl[(kind, s)], s, kind)
                    cache[(s, kind)] = (lc, info)
                    np.savez_compressed(LCD / f"tic{tic}_s{s}_{kind}.npz", idx=lc["idx"],
                                        flux=lc["flux"].astype(np.float32),
                                        info=np.array(json.dumps(info, default=str)))
                lc, info = cache[(s, kind)]
                row["info"] = info
                r, _ = evaluate(lc, x["t"], x["sig_h"] / 24, D, depth, avoid, seed=x["n"] + 7)
                row.update(r)
            except FileNotFoundError as ex:
                row["outcome"] = "not available"
                row["why"] = str(ex)[:300]
            except Exception as ex:  # noqa: BLE001
                row["outcome"] = "not checked"
                row["why"] = f"{type(ex).__name__}: {ex}"[:300]
                row["trace"] = traceback.format_exc()[-600:]
            row["wall_s"] = round(time.time() - t0, 1)
            with open(OUTF, "a") as fh:
                fh.write(json.dumps(row, default=float) + "\n")
            print(rid, row.get("outcome"), row.get("why", ""), flush=True)


def main():
    eph = json.loads((fu.FU_OUT / "ephem.json").read_text())
    want = {int(a) for a in sys.argv[1:]}
    done = set()
    if OUTF.exists():
        done = {json.loads(x)["id"] for x in OUTF.read_text().splitlines()}
    for key in fu.KEYS:
        if want and eph[key]["tic"] not in want:
            continue
        run_star(key, eph[key], done)


if __name__ == "__main__":
    main()
