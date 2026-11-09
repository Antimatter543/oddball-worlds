"""Explain away known variable classes from any scorer (docs/ML_DESIGN.md §4.5).

    uv run python -I scripts/explain_away.py --base classical --kc kc_v1 --sector 32 --inj v1
    # -> outputs/scores/classical_ka/s0032/{real,inj_v1}.parquet

Reads the base scorer's files (§9), the known-class predictions
``outputs/knownclass/<kc>/sNNNN/{real,inj_<name>}.parquet`` and writes a new scorer
``<base>_ka`` (or ``--out``) in the same §9 format, so the evaluator treats it like any other.
The combination rule is ``oddball.knownclass.explain`` (class-conditional percentile, mixed
with the base percentile by ``alpha``). ``--alpha auto`` picks alpha on the **val** injections
only: the mean val recall in the top 1% over the ~0.3%, ~1% and ~3% depth bins (non-null
kinds). The test split and the known positives are never looked at.
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

from oddball.common.paths import outputs_root, sector_dir
from oddball.common.splits import split_array
from oddball.common.strata import stratum_id
from oddball.eval.harness import DEPTH_EDGES, DEPTH_LABELS, load_truth, rank_within
from oddball.knownclass.explain import (
    ExplainAway,
    downweight,
    explained_share,
    prob_matrix,
    stratum_pct,
)
from oddball.knownclass.labels import OTHER
from oddball.tess.arrays import dir_metadata

ALPHAS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
TAUS = (0.05, 0.1, 0.25, 0.5, 1.0)
TUNE_DEPTHS = ("~0.3%", "~1%", "~3%")
PASSTHROUGH = ("sys_flag", "sys_frac", "cm_frac", "dump_frac", "n_anom")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def val_metrics(real: pd.DataFrame, inj: pd.DataFrame, score_r, score_i) -> dict:
    """Val-split recall in the top 1% per depth bin + null false-positive rate (harness rule)."""
    ref = real.assign(score=score_r)[real.split == "val"]
    q = inj.assign(score=score_i)[inj.split == "val"]
    pct = rank_within(ref, q, ["split", "stratum"])
    top = pct >= 0.99
    astro = ~q.is_null.to_numpy()
    out = {}
    for lab in TUNE_DEPTHS:
        m = astro & (q.depth_bin == lab).to_numpy()
        out[lab] = float(top[m].mean()) if m.any() else float("nan")
    out["null_fpr"] = float(top[~astro].mean()) if (~astro).any() else float("nan")
    out["auroc_astro"] = float(np.nanmean(pct[astro]))
    out["objective"] = float(np.nanmean([out[k] for k in TUNE_DEPTHS]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--kc", default="kc_v1")
    ap.add_argument("--out")
    ap.add_argument("--sector", type=int, default=32)
    ap.add_argument("--inj", default="v1")
    ap.add_argument("--alpha", default="auto")
    ap.add_argument("--tau", default=0.1, type=float, help="typicality tolerance (fixed alpha)")
    ap.add_argument("--rule", default="downweight", choices=["downweight", "ccdf"],
                    help="downweight: scale the base surprise by the explained share "
                         "(probability x typicality); ccdf: class-conditional percentile")
    ap.add_argument("--identity", nargs="*", default=["other", "quiet"],
                    help="classes that explain nothing away (F_c(s) = s)")
    a = ap.parse_args()
    out_name = a.out or f"{a.base}_ka"
    sec = a.sector
    sd = outputs_root() / "scores" / a.base / f"s{sec:04d}"
    kd = outputs_root() / "knownclass" / a.kc / f"s{sec:04d}"

    meta = dir_metadata(sector_dir(sec))[["tic", "tmag", "camera", "ccd"]]
    real = pd.read_parquet(sd / "real.parquet")
    real = real.drop(columns=[c for c in ("tmag", "camera", "ccd", "stratum") if c in real])
    real = real.merge(meta, on="tic", how="left", validate="one_to_one")
    real["stratum"] = stratum_id(real.tmag, real.camera, real.ccd)
    real["split"] = split_array(real.tic.to_numpy())
    kr = pd.read_parquet(kd / "real.parquet").drop(columns=["sector", "inj_id"])
    real = real.merge(kr, on="tic", how="left", validate="one_to_one")

    truth = load_truth(a.inj, sec)
    inj = pd.read_parquet(sd / f"inj_{a.inj}.parquet")
    inj = inj.drop(columns=[c for c in ("tmag", "camera", "ccd", "stratum") if c in inj])
    cols_t = ["inj_id", "split", "kind", "depth", "is_null", "tmag", "camera", "ccd"]
    inj = inj.merge(truth[cols_t], on="inj_id", how="left", validate="one_to_one")
    inj["stratum"] = stratum_id(inj.tmag, inj.camera, inj.ccd)
    inj["depth_bin"] = pd.cut(inj.depth, DEPTH_EDGES, labels=DEPTH_LABELS, right=False)
    ki = pd.read_parquet(kd / f"inj_{a.inj}.parquet").drop(columns=["sector", "tic"])
    inj = inj.merge(ki, on="inj_id", how="left", validate="one_to_one")
    classes = [c[5:] for c in kr.columns if c.startswith("kc_p_")]
    for df, nm in ((real, "real"), (inj, "inj")):
        miss = int(df[f"kc_p_{classes[0]}"].isna().sum())
        if miss:
            raise ValueError(f"{miss} {nm} rows have no known-class prediction")

    # 1. base score -> percentile within stratum against all real stars of the sector
    base_r = real.score.astype(float).fillna(-np.inf).to_numpy()
    base_i = inj.score.astype(float).fillna(-np.inf).to_numpy()
    s_r = stratum_pct(base_r, real.stratum, base_r, real.stratum)
    s_i = stratum_pct(base_r, real.stratum, base_i, inj.stratum)
    p_r, p_i = prob_matrix(real, classes), prob_matrix(inj, classes)

    # 2. class-conditional ECDFs from the train-split real stars with a catalogue / quiet target
    ref = ((real.split == "train") & real.target.isin([c for c in classes if c != OTHER])
           & ~real.cat_excluded.astype(bool)).to_numpy()  # fmt: skip
    ea = ExplainAway(classes, identity_classes=tuple(a.identity)).fit(s_r[ref], p_r[ref])
    log(f"reference: {int(ref.sum())} train-split labelled real stars; class mass "
        + ", ".join(f"{c} {m:.0f}" for c, m in ea.class_mass.items()))

    if a.rule == "downweight" and not any(c.startswith("kc_t_") for c in real.columns):
        raise ValueError(f"{a.kc} has no typicality columns (kc_t_*); use --rule ccdf")

    def combined(al, which, tau):
        s, p, df = (s_r, p_r, real) if which == "real" else (s_i, p_i, inj)
        if a.rule == "downweight":
            return downweight(s, explained_share(df, tau), al)
        return ea.score(s, p, al)

    # 3. alpha (and tau for downweight) on val injections
    taus = TAUS if a.rule == "downweight" else (1.0,)
    sweep = {}
    for tau in taus:
        for al in ALPHAS:
            m = val_metrics(real, inj, combined(al, "real", tau), combined(al, "inj", tau))
            sweep[(al, tau)] = m
            log(f"{a.rule} alpha {al:.2f} tau {tau:.2f}: val "
                + ", ".join(f"{k} {v:.3f}" for k, v in m.items()))
    if a.alpha == "auto":
        best = max(v["objective"] for v in sweep.values())
        # ties: least intervention (smallest alpha, then largest tau)
        alpha, tau = min((k for k, v in sweep.items() if v["objective"] >= best - 1e-12),
                         key=lambda k: (k[0], -k[1]))  # fmt: skip
    else:
        alpha, tau = float(a.alpha), float(a.tau)
    log(f"alpha = {alpha}, tau = {tau}")

    def ka_raw(which):  # the fully explained-away score (alpha = 1), for inspection
        return combined(1.0, which, tau)

    # 4. final scores, re-percentiled within stratum against the real stars
    c_r, c_i = combined(alpha, "real", tau), combined(alpha, "inj", tau)
    f_r = stratum_pct(c_r, real.stratum, c_r, real.stratum)
    f_i = stratum_pct(c_r, real.stratum, c_i, inj.stratum)
    od = outputs_root() / "scores" / out_name / f"s{sec:04d}"
    od.mkdir(parents=True, exist_ok=True)
    kc_cols = ["kc_pred", "kc_pmax", "kc_entropy", "kc_energy", "kc_explained"]
    for df, s, ka, fin, part in ((real, s_r, ka_raw("real"), f_r, "real"),
                                 (inj, s_i, ka_raw("inj"), f_i, f"inj_{a.inj}")):  # fmt: skip
        o = pd.DataFrame({"tic": df.tic.to_numpy(), "sector": sec,
                          "inj_id": df.inj_id.to_numpy() if "inj_id" in df else None,
                          "stratum": df.stratum.to_numpy(), "score": fin,
                          "score_base": s, "score_ka_raw": ka})  # fmt: skip
        for c in kc_cols:
            if c in df:
                o[c] = df[c].to_numpy()
        if a.rule == "downweight":
            o["kc_explained"] = explained_share(df, tau)
        for c in PASSTHROUGH:
            if c in df:
                o[c] = df[c].to_numpy()
        o.to_parquet(od / f"{part}.parquet", index=False)
    prov = {"scorer": out_name, "base": a.base, "kc": a.kc, "sector": sec, "inj": a.inj,
            "alpha": alpha, "tau": tau, "alpha_mode": a.alpha, "classes": classes,
            "val_sweep": {f"alpha={k[0]},tau={k[1]}": v for k, v in sweep.items()},
            "class_mass": ea.class_mass, "n_reference": int(ref.sum()),
            "rule": a.rule,
            "rule_text": {"downweight": "score = pct_stratum(1 - (1 - s)^(1 - alpha*E)), E = "
                                        "sum over explaining classes of p_c x min(1, t_c/tau)",
                          "ccdf": "score = pct_stratum((1-alpha)*s + alpha*sum_c p_c F_c(s)); "
                                  "F_c = p_c-weighted ECDF of s over train-split labelled real "
                                  "stars; F_c(s) = s for identity classes"}[a.rule],
            "identity_classes": [c for c, i in zip(classes, ea.identity, strict=True) if i],
            "built_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}  # fmt: skip
    (od / "provenance.json").write_text(json.dumps(prov, indent=1, default=float))
    log(f"wrote {od}")


if __name__ == "__main__":
    main()
