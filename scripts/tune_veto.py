"""Choose the ``cmv2`` flag thresholds on **val** injections and write ``cmv2_flag``.

    uv run python -I scripts/tune_veto.py --sector 32 --inj v1 --scorers classical ssl_v2_iforest

``cmv2_flag = sys_snr >= t_sys and res_snr < t_res`` (``oddball.score.veto.veto_flag``; both
channels, or PDCSAP only). For each grid point and each base scorer, the evaluator's veto rule
is applied on the val split only (flagged real and injected light curves to the bottom, re-rank
within split x stratum). Objective, averaged over the base scorers:

    J = mean(recall in top 1% at ~0.3%, ~1%, ~3% depth) - null false-positive rate

One-standard-error rule: among grid points within ``TOL`` = 0.01 of the best J (about one
standard error of J on val: ~600 injections per depth bin, 300 nulls), the one flagging the
fewest real val stars wins. (A 0.002 tolerance would pick t_sys 6 / t_res 10, 14.5% flagged,
J 0.280; the one-SE rule picks t_sys 10 / t_res 10, 4.6% flagged, J 0.274. Both
are val-only.) The test split and the known positives are never read. The sweep is stored in
the veto's ``provenance.json``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oddball.common.paths import sector_dir
from oddball.common.splits import split_array
from oddball.common.strata import stratum_id
from oddball.eval.harness import (
    DEPTH_EDGES,
    DEPTH_LABELS,
    load_truth,
    rank_within,
    score_dir,
    veto_dir,
)
from oddball.score.veto import veto_flag
from oddball.tess.arrays import dir_metadata

T_SYS = (4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0)
T_RES = (4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0)
DEPTHS = ("~0.3%", "~1%", "~3%")
TOL = 0.01


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_val(scorer: str, sector: int, inj: str, meta: pd.DataFrame, truth: pd.DataFrame):
    sd = score_dir(scorer, sector)
    r = pd.read_parquet(sd / "real.parquet", columns=["tic", "score", "sys_flag"])
    r = r.merge(meta, on="tic", how="left", validate="one_to_one")
    r["split"] = split_array(r.tic.to_numpy())
    r = r[r.split == "val"].reset_index(drop=True)
    r["stratum"] = stratum_id(r.tmag, r.camera, r.ccd)
    i = pd.read_parquet(sd / f"inj_{inj}.parquet", columns=["inj_id", "score", "sys_flag"])
    i = truth.merge(i, on="inj_id", how="left", validate="one_to_one")
    i = i[i.split == "val"].reset_index(drop=True)
    for df in (r, i):
        df["score"] = df.score.astype(float).fillna(-np.inf)
    return r, i


def metrics(r: pd.DataFrame, i: pd.DataFrame, fr: np.ndarray, fi: np.ndarray) -> dict:
    rv = r.assign(score=np.where(fr, -np.inf, r.score))
    iv = i.assign(score=np.where(fi, -np.inf, i.score))
    top = (rank_within(rv, iv, ["split", "stratum"]) >= 0.99) & ~fi
    astro = ~i.is_null.to_numpy()
    out = {d: float(top[astro & (i.depth_bin == d).to_numpy()].mean()) for d in DEPTHS}
    out["all"] = float(top[astro].mean())
    out["null_fpr"] = float(top[~astro].mean())
    out["recall_mean"] = float(np.mean([out[d] for d in DEPTHS]))
    out["J"] = out["recall_mean"] - out["null_fpr"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sector", type=int, default=32)
    ap.add_argument("--inj", default="v1")
    ap.add_argument("--name", default="cmv2")
    ap.add_argument("--scorers", nargs="+", default=["classical", "ssl_v2_iforest"])
    a = ap.parse_args()
    sec = a.sector
    vd = veto_dir(a.name, sec)
    meta = dir_metadata(sector_dir(sec))[["tic", "tmag", "camera", "ccd"]]
    truth = load_truth(a.inj, sec)
    truth["depth_bin"] = pd.cut(truth.depth, DEPTH_EDGES, labels=DEPTH_LABELS, right=False)
    vr_all = pd.read_parquet(vd / "real.parquet")
    vi_all = pd.read_parquet(vd / f"inj_{a.inj}.parquet")

    data = {}
    for s in a.scorers:
        r, i = load_val(s, sec, a.inj, meta, truth)
        vr = r[["tic"]].merge(vr_all, on="tic", how="left", validate="one_to_one")
        vi = i[["inj_id"]].merge(vi_all.drop(columns=["tic"]), on="inj_id", how="left",
                                 validate="one_to_one")  # fmt: skip
        assert vr.cmv2_res_snr.notna().all() and vi.cmv2_res_snr.notna().all()
        data[s] = (r, i, vr, vi)

    base = {}
    for s, (r, i, vr, vi) in data.items():
        none = metrics(r, i, np.zeros(len(r), bool), np.zeros(len(i), bool))
        v0 = metrics(r, i, r.sys_flag.to_numpy(bool), i.sys_flag.to_numpy(bool))
        base[s] = {"none": none, "sys_flag": v0}
        log(f"{s}: no veto {fmt(none)}")
        log(f"{s}: sys_flag (v0, flags {r.sys_flag.mean():.3f}) {fmt(v0)}")

    sweep = []
    for pdc_only in (False, True):
        for ts in T_SYS:
            for tr in T_RES:
                row = {"pdc_only": pdc_only, "t_sys": ts, "t_res": tr}
                per = {}
                for s, (r, i, vr, vi) in data.items():
                    per[s] = metrics(r, i, veto_flag(vr, ts, tr, pdc_only),
                                     veto_flag(vi, ts, tr, pdc_only))  # fmt: skip
                vr0 = data[a.scorers[0]][2]
                row["flag_frac_real_val"] = float(veto_flag(vr0, ts, tr, pdc_only).mean())
                row["J"] = float(np.mean([m["J"] for m in per.values()]))
                row["per_scorer"] = per
                sweep.append(row)
    best_j = max(r["J"] for r in sweep)
    cands = [r for r in sweep if r["J"] >= best_j - TOL]
    best = min(cands, key=lambda r: (r["flag_frac_real_val"], -r["J"]))
    log("top grid points by J:")
    for r in sorted(sweep, key=lambda r: -r["J"])[:12]:
        log(f"  pdc_only={r['pdc_only']} t_sys={r['t_sys']} t_res={r['t_res']} J={r['J']:.4f} "
            f"flag={r['flag_frac_real_val']:.3f} | "
            + " | ".join(f"{s}: {fmt(m)}" for s, m in r["per_scorer"].items()))
    log(f"chosen: pdc_only={best['pdc_only']} t_sys={best['t_sys']} t_res={best['t_res']} "
        f"J={best['J']:.4f} flagged {best['flag_frac_real_val']:.3f} of real val stars")

    ts, tr, po = best["t_sys"], best["t_res"], best["pdc_only"]
    vr_all["cmv2_flag"] = veto_flag(vr_all, ts, tr, po)
    vi_all["cmv2_flag"] = veto_flag(vi_all, ts, tr, po)
    vr_all.to_parquet(vd / "real.parquet", index=False)
    vi_all.to_parquet(vd / f"inj_{a.inj}.parquet", index=False)
    pp = vd / "provenance.json"
    prov = json.loads(pp.read_text()) if pp.exists() else {}
    prov["flag"] = {"rule": "sys_snr >= t_sys and res_snr < t_res", "t_sys": ts, "t_res": tr,
                    "pdc_only": po, "objective": "mean over scorers of mean val recall@1% at "
                    "~0.3/1/3% minus val null FPR; within TOL of the best J -> fewest flagged "
                    "real val stars", "tol": TOL,
                    "scorers": a.scorers, "baselines_val": base, "sweep_val": sweep,
                    "flag_frac_real_all": float(vr_all.cmv2_flag.mean()),
                    "tuned_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}  # fmt: skip
    pp.write_text(json.dumps(prov, indent=1, default=float))
    log(f"wrote cmv2_flag ({vr_all.cmv2_flag.mean():.3f} of all real stars flagged) -> {vd}")


def fmt(m: dict) -> str:
    return (f"0.3% {m['~0.3%']:.3f} 1% {m['~1%']:.3f} 3% {m['~3%']:.3f} all {m['all']:.3f} "
            f"nullFPR {m['null_fpr']:.3f} J {m['J']:.3f}")


if __name__ == "__main__":
    main()
