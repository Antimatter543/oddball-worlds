"""Choose the ``cmv3`` labels and flag thresholds on **val**, and write ``cmv3_flag``.

    uv run python -I scripts/tune_veto3.py --sector 32 --scorers classical ssl_v2_iforest

Grid: every configuration of ``veto3.CONFIGS`` (cmv2 alone, all labels, leave-one-out and
only-one ablations) x {both channels, PDCSAP only} x ``t_sys`` x ``t_res``. The star flag is
``<config>_sys_snr >= t_sys and <config>_res_snr < t_res``. For each grid point and base scorer
the evaluator's veto rule is applied on val: flagged real and injected light curves go to the
bottom, then everything is re-ranked within split x stratum.

Objective, averaged over the base scorers:

    J3 = mean(recall in top 1% at ~0.3 / 1 / 3% depth; v1 astrophysical injections)
         - mean over the five null types of the null false-positive rate
           (v1: null_ccd, null_dump; n2: null_asteroid, null_step, null_column)

The per-type mean keeps the 200 step nulls from outweighing the ~55 asteroid nulls. One-SE
rule as for ``cmv2``: among grid points within ``TOL`` = 0.01 of the best J3, the one that flags
the fewest real val stars. n2 carriers (``split = "carrier"``) are never ranked. Test and the
known positives are never read here.

Writes ``cmv3_flag`` and ``cmv3_{res,sys}_snr[_pdc]``, ``cmv3_{res,sys}_{t,ev}`` (copied from the
chosen configuration) into ``outputs/veto/cmv3/sNNNN/{real,inj_v1,inj_n2}.parquet``, and the
sweep into its ``provenance.json``.
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
from oddball.score import veto3 as V3
from oddball.tess.arrays import dir_metadata

T_SYS = (5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0)
T_RES = (4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0)
DEPTHS = ("~0.3%", "~1%", "~3%")
NULL_TYPES = ("null_ccd", "null_dump", "null_asteroid", "null_step", "null_column")
TOL = 0.01
COPY = ("res_snr", "sys_snr", "res_snr_pdc", "sys_snr_pdc", "res_t", "res_ev", "sys_t", "sys_ev")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_split(scorer: str, sector: int, meta: pd.DataFrame, truth: pd.DataFrame,
               split: str) -> tuple[pd.DataFrame, pd.DataFrame]:  # fmt: skip
    """Real stars and injections (v1 + n2, evaluated only) of one split for one scorer."""
    sd = score_dir(scorer, sector)
    r = pd.read_parquet(sd / "real.parquet", columns=["tic", "score"])
    r = r.merge(meta, on="tic", how="left", validate="one_to_one")
    r["split"] = split_array(r.tic.to_numpy())
    r = r[r.split == split].reset_index(drop=True)
    r["stratum"] = stratum_id(r.tmag, r.camera, r.ccd)
    parts = []
    for name in ("v1", "n2"):
        i = pd.read_parquet(sd / f"inj_{name}.parquet", columns=["inj_id", "score"])
        t = truth[truth.set == name]
        parts.append(t.merge(i, on="inj_id", how="left", validate="one_to_one"))
    i = pd.concat(parts, ignore_index=True)
    i = i[i.split == split].reset_index(drop=True)
    for df in (r, i):
        df["score"] = df.score.astype(float).fillna(-np.inf)
    return r, i


def load_all_truth(sector: int) -> pd.DataFrame:
    parts = []
    for name in ("v1", "n2"):
        t = load_truth(name, sector).assign(set=name)
        parts.append(t)
    t = pd.concat(parts, ignore_index=True)
    t["depth_bin"] = pd.cut(t.depth, DEPTH_EDGES, labels=DEPTH_LABELS, right=False)
    t["stratum"] = stratum_id(t.tmag, t.camera, t.ccd)
    return t


def metrics(r: pd.DataFrame, i: pd.DataFrame, fr: np.ndarray, fi: np.ndarray) -> dict:
    rv = r.assign(score=np.where(fr, -np.inf, r.score))
    iv = i.assign(score=np.where(fi, -np.inf, i.score))
    top = (rank_within(rv, iv, ["split", "stratum"]) >= 0.99) & ~fi
    astro = ~i.is_null.to_numpy()
    out = {d: float(top[astro & (i.depth_bin == d).to_numpy()].mean()) for d in DEPTHS}
    out["all"] = float(top[astro].mean())
    out["astro_flagged"] = float(fi[astro].mean())
    kind = i.kind.to_numpy()
    for k in NULL_TYPES:
        m = kind == k
        out[f"fpr_{k}"] = float(top[m].mean()) if m.any() else float("nan")
        out[f"flag_{k}"] = float(fi[m].mean()) if m.any() else float("nan")
    out["null_fpr_pooled_v1"] = float(top[~astro & np.isin(kind, NULL_TYPES[:2])].mean())
    out["null_fpr_mean"] = float(np.nanmean([out[f"fpr_{k}"] for k in NULL_TYPES]))
    out["recall_mean"] = float(np.mean([out[d] for d in DEPTHS]))
    out["J3"] = out["recall_mean"] - out["null_fpr_mean"]
    out["flag_real"] = float(fr.mean())
    return out


def fmt(m: dict) -> str:
    return (f"0.3% {m['~0.3%']:.3f} 1% {m['~1%']:.3f} 3% {m['~3%']:.3f} | "
            + " ".join(f"{k[5:]} {m['fpr_' + k]:.3f}" for k in NULL_TYPES)
            + f" | J3 {m['J3']:.3f} flag {m['flag_real']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sector", type=int, default=32)
    ap.add_argument("--name", default="cmv3")
    ap.add_argument("--scorers", nargs="+", default=["classical", "ssl_v2_iforest"])
    a = ap.parse_args()
    sec = a.sector
    vd = veto_dir(a.name, sec)
    meta = dir_metadata(sector_dir(sec))[["tic", "tmag", "camera", "ccd"]]
    truth = load_all_truth(sec)
    vr_all = pd.read_parquet(vd / "real.parquet")
    vi_all = {n: pd.read_parquet(vd / f"inj_{n}.parquet") for n in ("v1", "n2")}
    vi_cat = pd.concat(vi_all.values(), ignore_index=True)

    data = {}
    for s in a.scorers:
        r, i = load_split(s, sec, meta, truth, "val")
        vr = r[["tic"]].merge(vr_all, on="tic", how="left", validate="one_to_one")
        vi = i[["inj_id"]].merge(vi_cat.drop(columns=["tic"]), on="inj_id", how="left",
                                 validate="one_to_one")  # fmt: skip
        assert vr.all_res_snr.notna().all() and vi.all_res_snr.notna().all()
        data[s] = (r, i, vr, vi)
        log(f"{s}: val {len(r)} real, {int((~i.is_null).sum())} astro, "
            f"{i[i.is_null].kind.value_counts().to_dict()} nulls")

    base = {}
    for s, (r, i, vr, vi) in data.items():
        none = metrics(r, i, np.zeros(len(r), bool), np.zeros(len(i), bool))
        c2 = metrics(r, i, V3.veto_flag(vr, "cmv2", 10.0, 10.0, True),
                     V3.veto_flag(vi, "cmv2", 10.0, 10.0, True))  # fmt: skip
        base[s] = {"none": none, "cmv2_rule": c2}
        log(f"{s}: no veto   {fmt(none)}")
        log(f"{s}: cmv2 rule {fmt(c2)}")

    sweep = []
    for cfg in V3.CONFIGS:
        for po in (False, True):
            for ts in T_SYS:
                for tr in T_RES:
                    per = {s: metrics(r, i, V3.veto_flag(vr, cfg, ts, tr, po),
                                      V3.veto_flag(vi, cfg, ts, tr, po))
                           for s, (r, i, vr, vi) in data.items()}  # fmt: skip
                    row = {"config": cfg, "pdc_only": po, "t_sys": ts, "t_res": tr,
                           "J3": float(np.mean([m["J3"] for m in per.values()])),
                           "flag_frac_real_val": per[a.scorers[0]]["flag_real"],
                           "per_scorer": per}  # fmt: skip
                    sweep.append(row)
        best_cfg = max((x for x in sweep if x["config"] == cfg), key=lambda x: x["J3"])
        log(f"  {cfg:12s} best J3 {best_cfg['J3']:.4f} (pdc_only={best_cfg['pdc_only']} "
            f"t_sys={best_cfg['t_sys']} t_res={best_cfg['t_res']} "
            f"flag={best_cfg['flag_frac_real_val']:.3f})")
    best_j = max(x["J3"] for x in sweep)
    cands = [x for x in sweep if x["J3"] >= best_j - TOL]
    pick = min(cands, key=lambda x: (x["flag_frac_real_val"], -x["J3"]))
    log("top grid points by J3:")
    for x in sorted(sweep, key=lambda x: -x["J3"])[:10]:
        log(f"  {x['config']} pdc_only={x['pdc_only']} t_sys={x['t_sys']} t_res={x['t_res']} "
            f"J3={x['J3']:.4f} flag={x['flag_frac_real_val']:.3f}")
    log(f"chosen: {pick['config']} pdc_only={pick['pdc_only']} t_sys={pick['t_sys']} "
        f"t_res={pick['t_res']} J3={pick['J3']:.4f} flags {pick['flag_frac_real_val']:.3f} of "
        "real val stars")
    for s, m in pick["per_scorer"].items():
        log(f"  {s}: {fmt(m)}")

    cfg, ts, tr, po = pick["config"], pick["t_sys"], pick["t_res"], pick["pdc_only"]
    for df, path in ((vr_all, vd / "real.parquet"),
                     *((vi_all[n], vd / f"inj_{n}.parquet") for n in ("v1", "n2"))):
        df["cmv3_flag"] = V3.veto_flag(df, cfg, ts, tr, po)
        for c in COPY:
            df[f"cmv3_{c}"] = df[f"{cfg}_{c}"]
        df.to_parquet(path, index=False)
    pp = vd / "provenance.json"
    prov = json.loads(pp.read_text()) if pp.exists() else {}
    prov["flag"] = {"rule": "<config>_sys_snr >= t_sys and <config>_res_snr < t_res",
                    "config": cfg, "labels": list(V3.CONFIGS[cfg]), "t_sys": ts, "t_res": tr,
                    "pdc_only": po, "objective": "J3: mean over scorers of mean val recall@1% "
                    "at ~0.3/1/3% (v1) minus the mean over 5 null types of val null FPR; "
                    "within TOL of the best -> fewest flagged real val stars", "tol": TOL,
                    "scorers": a.scorers, "baselines_val": base, "chosen_val": pick,
                    "best_per_config": {c: max((x for x in sweep if x["config"] == c),
                                               key=lambda x: x["J3"]) for c in V3.CONFIGS},
                    "flag_frac_real_all": float(vr_all.cmv3_flag.mean()),
                    "tuned_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}  # fmt: skip
    pp.write_text(json.dumps(prov, indent=1, default=float))
    log(f"wrote cmv3_flag ({vr_all.cmv3_flag.mean():.3f} of all real stars flagged) -> {vd}")


if __name__ == "__main__":
    main()
