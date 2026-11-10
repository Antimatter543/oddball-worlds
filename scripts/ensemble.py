"""Ensemble scorer from existing score files (docs/ML_DESIGN.md §4, §12).

    uv run python -I scripts/ensemble.py --name ens_v1 --components classical ssl_v2_iforest
    uv run python -I scripts/ensemble.py --name ens_v1_ka --components classical ssl_v2_iforest \
        --ka kc_v1
    uv run python -I scripts/ensemble.py --name ens_v1_seed1 \
        --components classical ssl_v2_seed1_iforest ssl_v2_seed1_recon --fixed ens_v1  # other seed

Each component's ``score`` becomes a percentile within Tmag x CCD stratum against the sector's
real stars, ``p_k``, and a surprise ``s_k = -log(1 - p_k)``. Combiners (chosen on **val**):

* ``mean``     rank-average ``mean_k p_k`` (the §4 rule; penalises one-score extremes). Reference.
* ``max``      ``max_k s_k``: max of calibrated percentiles.
* ``wmax``     ``max(s_1, w s_2, ...)``: weighted max, weights on a small grid.
* ``fisher``   ``sum_k s_k`` (Fisher's method on stratum p-values ``1 - p_k``).
* ``stouffer`` ``sum_k Phi^-1(p_k)``.

The statistic is re-percentiled within stratum against the real stars. With ``--ka`` the
known-class explain-away (``oddball.knownclass.explain.downweight``, alpha and tau swept on val)
is applied to it and the result re-percentiled again. Then the ``--veto`` (``cmv2``; ``cmv3`` for
``ens_v2``, ML_DESIGN §13): flagged light
curves get ``score = -1`` (below every unflagged one); ``score_raw`` keeps the pre-veto value.

Selection objective on val (same as ``tune_veto.py``): mean recall in the top 1% at ~0.3 / 1 / 3%
depth minus the null false-positive rate, with the veto applied. Ties (within 0.002) go to the
simpler rule (order above, then smaller weights). Test and known positives are never read here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oddball.common.paths import outputs_root, sector_dir
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
from oddball.knownclass.explain import downweight, explained_share, stratum_pct
from oddball.tess.arrays import dir_metadata

RULES = ("mean", "max", "wmax", "fisher", "stouffer")
WEIGHTS = (0.5, 0.75, 1.25, 1.5, 2.0)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 0.9)
TAUS = (0.05, 0.1, 0.25, 0.5, 1.0)
DEPTHS = ("~0.3%", "~1%", "~3%")
PASSTHROUGH = ("sys_flag", "sys_frac", "cm_frac", "dump_frac", "n_anom")
VETO_SUFFIXES = ("flag", "res_snr", "sys_snr", "res_snr_pdc", "sys_snr_pdc")


def veto_cols(veto: str) -> tuple[str, ...]:
    return tuple(f"{veto}_{s}" for s in VETO_SUFFIXES)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def combine(P: np.ndarray, rule: str, w: tuple[float, ...] | None = None) -> np.ndarray:
    P = np.clip(P, 0.0, 1.0 - 1e-7)
    S = -np.log1p(-P)
    if rule == "mean":
        return P.mean(1)
    if rule == "max":
        return S.max(1)
    if rule == "wmax":
        return (S * np.asarray(w)[None, :]).max(1)
    if rule == "fisher":
        return S.sum(1)
    if rule == "stouffer":
        return norm.ppf(np.clip(P, 1e-7, 1 - 1e-7)).sum(1)
    raise ValueError(rule)


def candidates(k: int):
    yield ("mean", None)
    yield ("max", None)
    for ws in itertools.product(WEIGHTS, repeat=k - 1):
        yield ("wmax", (1.0, *ws))
    yield ("fisher", None)
    yield ("stouffer", None)


def val_metrics(real, inj, sr, si, fr, fi) -> dict:
    vm_r = (real.split == "val").to_numpy()
    vm_i = (inj.split == "val").to_numpy()
    r = real[vm_r].assign(score=np.where(fr[vm_r], -np.inf, sr[vm_r]))
    i = inj[vm_i].assign(score=np.where(fi[vm_i], -np.inf, si[vm_i]))
    pct = rank_within(r, i, ["split", "stratum"])
    top = (pct >= 0.99) & ~fi[vm_i]
    astro = ~i.is_null.to_numpy()
    out = {d: float(top[astro & (i.depth_bin == d).to_numpy()].mean()) for d in DEPTHS}
    out["all"] = float(top[astro].mean())
    out["null_fpr"] = float(top[~astro].mean())
    out["auroc"] = float(np.where(fi[vm_i], 0.0, pct)[astro].mean())
    out["recall_mean"] = float(np.mean([out[d] for d in DEPTHS]))
    out["J"] = out["recall_mean"] - out["null_fpr"]
    return out


def fmt(m: dict) -> str:
    return (f"0.3% {m['~0.3%']:.3f} 1% {m['~1%']:.3f} 3% {m['~3%']:.3f} all {m['all']:.3f} "
            f"null {m['null_fpr']:.3f} J {m['J']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--components", nargs="+", required=True)
    ap.add_argument("--sector", type=int, default=32)
    ap.add_argument("--inj", default="v1")
    ap.add_argument("--veto", default="cmv2")
    ap.add_argument("--ka", default=None, help="known-class model for explain.downweight")
    ap.add_argument("--fixed", default=None,
                    help="reuse the rule/weights (and alpha/tau) chosen by this ensemble")
    ap.add_argument("--rule", default=None, choices=RULES[:2] + RULES[3:],
                    help="force an unweighted rule instead of choosing on val (diagnostic)")
    a = ap.parse_args()
    sec = a.sector
    comps = a.components
    meta = dir_metadata(sector_dir(sec))[["tic", "tmag", "camera", "ccd"]]
    truth = load_truth(a.inj, sec)
    truth["depth_bin"] = pd.cut(truth.depth, DEPTH_EDGES, labels=DEPTH_LABELS, right=False)

    # ---- assemble real + injected tables
    real = meta.copy()
    inj = truth[["inj_id", "host_tic", "split", "kind", "depth", "is_null", "depth_bin", "tmag",
                 "camera", "ccd"]].copy()  # fmt: skip
    for k, c in enumerate(comps):
        sd = score_dir(c, sec)
        r = pd.read_parquet(sd / "real.parquet")
        i = pd.read_parquet(sd / f"inj_{a.inj}.parquet")
        keep_r = ["tic", "score"] + ([x for x in PASSTHROUGH if x in r] if k == 0 else [])
        keep_i = ["inj_id", "score"] + ([x for x in PASSTHROUGH if x in i] if k == 0 else [])
        real = real.merge(r[keep_r].rename(columns={"score": f"raw_{c}"}), on="tic",
                          how="inner", validate="one_to_one")  # fmt: skip
        inj = inj.merge(i[keep_i].rename(columns={"score": f"raw_{c}"}), on="inj_id",
                        how="left", validate="one_to_one")  # fmt: skip
    real["split"] = split_array(real.tic.to_numpy())
    real["stratum"] = stratum_id(real.tmag, real.camera, real.ccd)
    inj["stratum"] = stratum_id(inj.tmag, inj.camera, inj.ccd)
    vd = veto_dir(a.veto, sec)
    vcols = veto_cols(a.veto)
    vr = pd.read_parquet(vd / "real.parquet", columns=["tic", *vcols])
    vi = pd.read_parquet(vd / f"inj_{a.inj}.parquet", columns=["inj_id", *vcols])
    real = real.merge(vr, on="tic", how="left", validate="one_to_one")
    inj = inj.merge(vi, on="inj_id", how="left", validate="one_to_one")
    fr = real[f"{a.veto}_flag"].fillna(False).to_numpy(bool)
    fi = inj[f"{a.veto}_flag"].fillna(False).to_numpy(bool)
    log(f"{len(real)} real, {len(inj)} injected; veto {a.veto} flags {fr.mean():.3f} real, "
        f"{fi.mean():.3f} injected")

    Pr = np.column_stack([stratum_pct(real[f"raw_{c}"].fillna(-np.inf), real.stratum,
                                      real[f"raw_{c}"].fillna(-np.inf), real.stratum)
                          for c in comps])  # fmt: skip
    Pi = np.column_stack([stratum_pct(real[f"raw_{c}"].fillna(-np.inf), real.stratum,
                                      inj[f"raw_{c}"].fillna(-np.inf), inj.stratum)
                          for c in comps])  # fmt: skip

    for k, c in enumerate(comps):
        log(f"component {c}: val {fmt(val_metrics(real, inj, Pr[:, k], Pi[:, k], fr, fi))}")

    # ---- combiner on val
    sweep = []
    fixed = None
    if a.fixed:
        fixed = json.loads((score_dir(a.fixed, sec) / "provenance.json").read_text())
        cands = [(fixed["rule"], tuple(fixed["weights"]) if fixed["weights"] else None)]
    elif a.rule:
        cands = [(a.rule, None)]
    else:
        cands = list(candidates(len(comps)))
    for rule, w in cands:
        m = val_metrics(real, inj, combine(Pr, rule, w), combine(Pi, rule, w), fr, fi)
        sweep.append({"rule": rule, "weights": w, **m})
        log(f"  {rule:8s} {w!s:22s} {fmt(m)}")
    best = max(s["J"] for s in sweep)
    order = {r: i for i, r in enumerate(RULES)}
    pick = min((s for s in sweep if s["J"] >= best - 0.002),
               key=lambda s: (order[s["rule"]], sum(s["weights"] or ())))  # fmt: skip
    rule, w = pick["rule"], pick["weights"]
    log(f"rule: {rule} {w} (val J {pick['J']:.4f})")
    stat_r, stat_i = combine(Pr, rule, w), combine(Pi, rule, w)
    e_r = stratum_pct(stat_r, real.stratum, stat_r, real.stratum)
    e_i = stratum_pct(stat_r, real.stratum, stat_i, inj.stratum)

    # ---- optional explain-away
    alpha = tau = None
    ka_sweep = []
    if a.ka:
        kd = outputs_root() / "knownclass" / a.ka / f"s{sec:04d}"
        kr = real[["tic"]].merge(pd.read_parquet(kd / "real.parquet").drop(
            columns=["sector", "inj_id"]), on="tic", how="left", validate="one_to_one")
        ki = inj[["inj_id"]].merge(pd.read_parquet(kd / f"inj_{a.inj}.parquet").drop(
            columns=["sector", "tic"]), on="inj_id", how="left", validate="one_to_one")
        if fixed and fixed.get("alpha") is not None:
            grid = [(fixed["alpha"], fixed["tau"])]
        else:
            grid = list(itertools.product(ALPHAS, TAUS))
        for al, ta in grid:
            m = val_metrics(real, inj, downweight(e_r, explained_share(kr, ta), al),
                            downweight(e_i, explained_share(ki, ta), al), fr, fi)  # fmt: skip
            ka_sweep.append({"alpha": al, "tau": ta, **m})
            log(f"  ka alpha {al:.2f} tau {ta:.2f} {fmt(m)}")
        best = max(s["J"] for s in ka_sweep)
        kp = min((s for s in ka_sweep if s["J"] >= best - 0.002),
                 key=lambda s: (s["alpha"], -s["tau"]))  # fmt: skip
        alpha, tau = kp["alpha"], kp["tau"]
        log(f"explain-away: alpha {alpha} tau {tau} (val J {kp['J']:.4f})")
        d_r = downweight(e_r, explained_share(kr, tau), alpha)
        d_i = downweight(e_i, explained_share(ki, tau), alpha)
        e_r, e_i = (stratum_pct(d_r, real.stratum, d_r, real.stratum),
                    stratum_pct(d_r, real.stratum, d_i, inj.stratum))  # fmt: skip
        real["kc_explained"] = explained_share(kr, tau)
        inj["kc_explained"] = explained_share(ki, tau)
    final_val = val_metrics(real, inj, e_r, e_i, fr, fi)
    log(f"final (val, veto applied): {fmt(final_val)}")

    # ---- write §9 files
    od = score_dir(a.name, sec)
    od.mkdir(parents=True, exist_ok=True)
    for df, P, e, f, part in ((real, Pr, e_r, fr, "real"), (inj, Pi, e_i, fi, f"inj_{a.inj}")):
        o = pd.DataFrame({"tic": (df.tic if "tic" in df else df.host_tic).to_numpy(),
                          "sector": sec,
                          "inj_id": df.inj_id.to_numpy() if "inj_id" in df else None,
                          "stratum": df.stratum.to_numpy(),
                          "score": np.where(f, -1.0, e), "score_raw": e})  # fmt: skip
        for k, c in enumerate(comps):
            o[f"score_{c}"] = P[:, k]
        for c in (*PASSTHROUGH, *vcols, "kc_explained"):
            if c in df:
                o[c] = df[c].to_numpy()
        o.to_parquet(od / f"{part}.parquet", index=False)
    prov = {"scorer": a.name, "sector": sec, "inj": a.inj, "components": comps,
            "veto": a.veto, "rule": rule, "weights": list(w) if w else None,
            "ka": a.ka, "alpha": alpha, "tau": tau, "fixed_from": a.fixed,
            "forced_rule": a.rule,
            "objective": "val: mean recall@1% at ~0.3/1/3% minus null FPR, veto applied",
            "val_final": final_val, "val_sweep": sweep, "val_ka_sweep": ka_sweep,
            "vetoed_score": -1.0,
            "built_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}  # fmt: skip
    (od / "provenance.json").write_text(json.dumps(prov, indent=1, default=float))
    log(f"wrote {od}")


if __name__ == "__main__":
    main()
