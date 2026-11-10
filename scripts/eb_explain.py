"""``ens_v1_eb``: the ``ens_v1`` ensemble with the light-curve EB test as an explaining term.

    uv run python -I scripts/eb_explain.py --name ens_v1_eb --base ens_v1 --sector 32
    uv run python -I scripts/eb_explain.py --name ens_v1_eb_kc --base ens_v1 --kc kc_v2

Inputs: the base ensemble's rule (its ``provenance.json``: components, combiner, weights), the
component score files, the ``ebf_v1`` EB tables (``scripts/eb_features.py``; ``eb_xsec`` if
present), the ``cmv2`` veto, and two injection sets: ``v1`` (targets + instrument nulls) and
``eb1`` (EB nulls that should not rank high, and dusty-tail transits that should).

The base statistic is recomputed exactly as ``scripts/ensemble.py`` does (asserted against the
base's ``score_raw`` on the files it has), then

    eb_like = max(eb_periodic, w_single * eb_single_raw, eb_xsec * c_shape * c_extra)
    E       = eb_like                                (or 1 - (1 - E_kc)(1 - eb_like) with --kc)
    score   = pct_stratum(1 - (1 - s)^(1 - alpha * E)),   cmv2-flagged -> -1

``alpha``, ``w_single`` (and ``tau`` for the kc share) are chosen on the **val** split only:

    J = mean recall in the top 1% at ~0.3/1/3% (v1) - null FPR (v1) - EB-null FPR (eb1)

with the dusty-tail recall (eb1) not allowed to fall more than 0.02 below alpha = 0. Ties within
``TOL`` = 0.01 of the best J (about one standard error on val) go to the least intervention
(smallest alpha, then smallest w_single, then largest tau). Test and known positives are never
read. Also writes ``<base>/sNNNN/inj_eb1.parquet`` (alpha = 0, the base rule unchanged) so the
evaluator can compare base and EB scorers on the eb1 set.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ensemble import combine

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
from oddball.knownclass.ebtests import combine_eb
from oddball.knownclass.explain import downweight, explained_share, stratum_pct
from oddball.tess.arrays import dir_metadata

ALPHAS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
W_SINGLE = (0.0, 0.25, 0.5, 0.75, 1.0)
TAUS = (0.25, 0.5, 1.0)
DEPTHS = ("~0.3%", "~1%", "~3%")
TOL = 0.01
DUSTY_DROP = 0.02
EB_SETS = ("v1", "eb1")
KEEP_EB = ("eb_periodic", "eb_single_raw", "eb_xsec", "eb_c_shape", "eb_c_extra", "eb_P",
           "eb_n_hit", "eb_anchor_depth", "eb_fwhm", "eb_depth_fex", "eb_n_extra", "eb_asym")  # fmt: skip
VETO_COLS = ("cmv2_flag",)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_eb(sector: int, part: str) -> pd.DataFrame:
    d = outputs_root() / "knownclass" / "features" / "ebf_v1" / f"s{sector:04d}"
    f = pd.read_parquet(d / f"{part}.parquet")
    if "eb_xsec" not in f:
        f["eb_xsec"] = 0.0
    return f[["tic", "inj_id", *KEEP_EB]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="ens_v1_eb")
    ap.add_argument("--base", default="ens_v1")
    ap.add_argument("--sector", type=int, default=32)
    ap.add_argument("--kc", default=None, help="known-class model whose explained share joins E")
    ap.add_argument("--fixed", default=None, help="reuse alpha / w_single / tau of this scorer")
    a = ap.parse_args()
    sec = a.sector
    bprov = json.loads((score_dir(a.base, sec) / "provenance.json").read_text())
    comps, rule = bprov["components"], bprov["rule"]
    w = tuple(bprov["weights"]) if bprov["weights"] else None
    if bprov.get("ka"):
        raise SystemExit(f"{a.base} already has an explain-away step; use the plain ensemble")
    meta = dir_metadata(sector_dir(sec))[["tic", "tmag", "camera", "ccd"]]

    # ---- real stars
    real = meta.copy()
    for c in comps:
        r = pd.read_parquet(score_dir(c, sec) / "real.parquet", columns=["tic", "score"])
        real = real.merge(r.rename(columns={"score": f"raw_{c}"}), on="tic", how="inner",
                          validate="one_to_one")  # fmt: skip
    real["split"] = split_array(real.tic.to_numpy())
    real["stratum"] = stratum_id(real.tmag, real.camera, real.ccd)
    vr = pd.read_parquet(veto_dir("cmv2", sec) / "real.parquet", columns=["tic", *VETO_COLS])
    real = real.merge(vr, on="tic", how="left", validate="one_to_one")
    real = real.merge(load_eb(sec, "real").drop(columns=["inj_id"]), on="tic", how="left",
                      validate="one_to_one")  # fmt: skip
    ref = {c: real[f"raw_{c}"].fillna(-np.inf).to_numpy() for c in comps}
    Pr = np.column_stack([stratum_pct(ref[c], real.stratum, ref[c], real.stratum) for c in comps])
    st_r = combine(Pr, rule, w)
    e_r = stratum_pct(st_r, real.stratum, st_r, real.stratum)
    base_r = pd.read_parquet(score_dir(a.base, sec) / "real.parquet", columns=["tic", "score_raw"])
    chk = real[["tic"]].merge(base_r, on="tic", how="left").score_raw.to_numpy()
    if not np.allclose(chk, e_r, atol=1e-9):
        raise SystemExit("recomputed base statistic does not match the base's score_raw (real)")
    log(f"{len(real)} real stars; base {a.base} reproduced ({rule} {w})")

    # ---- injection sets
    inj = {}
    for name in EB_SETS:
        t = load_truth(name, sec)
        t["depth_bin"] = pd.cut(t.depth, DEPTH_EDGES, labels=DEPTH_LABELS, right=False)
        df = t[["inj_id", "host_tic", "split", "kind", "depth", "is_null", "depth_bin", "tmag",
                "camera", "ccd"]].copy()  # fmt: skip
        for c in comps:
            i = pd.read_parquet(score_dir(c, sec) / f"inj_{name}.parquet",
                                columns=["inj_id", "score"])  # fmt: skip
            df = df.merge(i.rename(columns={"score": f"raw_{c}"}), on="inj_id", how="left",
                          validate="one_to_one")  # fmt: skip
        df["stratum"] = stratum_id(df.tmag, df.camera, df.ccd)
        vi = pd.read_parquet(veto_dir("cmv2", sec) / f"inj_{name}.parquet",
                             columns=["inj_id", *VETO_COLS])  # fmt: skip
        df = df.merge(vi, on="inj_id", how="left", validate="one_to_one")
        df = df.merge(load_eb(sec, f"inj_{name}").drop(columns=["tic"]), on="inj_id", how="left",
                      validate="one_to_one")  # fmt: skip
        P = np.column_stack([stratum_pct(ref[c], real.stratum, df[f"raw_{c}"].fillna(-np.inf),
                                         df.stratum) for c in comps])  # fmt: skip
        st = combine(P, rule, w)
        df["e"] = stratum_pct(st_r, real.stratum, st, df.stratum)
        bf = score_dir(a.base, sec) / f"inj_{name}.parquet"
        if bf.exists():
            b = df[["inj_id"]].merge(pd.read_parquet(bf, columns=["inj_id", "score_raw"]),
                                     on="inj_id", how="left")  # fmt: skip
            if not np.allclose(b.score_raw.to_numpy(), df.e.to_numpy(), atol=1e-9):
                raise SystemExit(f"base statistic mismatch on inj_{name}")
        for col in ("eb_periodic", "eb_single_raw", "eb_xsec"):
            n_miss = int(df[col].isna().sum())
            if n_miss:
                raise SystemExit(f"{n_miss} {name} rows have no {col}")
        inj[name] = df
        log(f"inj {name}: {len(df)} light curves; base reproduced")

    kc_r = kc_i = None
    if a.kc:
        kd = outputs_root() / "knownclass" / a.kc / f"s{sec:04d}"
        kc_r = real[["tic"]].merge(
            pd.read_parquet(kd / "real.parquet").drop(columns=["sector", "inj_id"]),
            on="tic",
            how="left",
            validate="one_to_one",
        )
        kc_i = {n: inj[n][["inj_id"]].merge(pd.read_parquet(kd / f"inj_{n}.parquet").drop(
            columns=["sector", "tic"]), on="inj_id", how="left", validate="one_to_one")
            for n in EB_SETS}  # fmt: skip

    fr = real.cmv2_flag.fillna(False).to_numpy(bool)
    fi = {n: inj[n].cmv2_flag.fillna(False).to_numpy(bool) for n in EB_SETS}

    def share(df, kc, ws, tau):
        e = combine_eb(df, w_single=ws)
        if kc is not None:
            e = 1.0 - (1.0 - explained_share(kc, tau)) * (1.0 - e)
        return e

    def scores(al, ws, tau):
        d_r = downweight(e_r, share(real, kc_r, ws, tau), al)
        out = {"real": stratum_pct(d_r, real.stratum, d_r, real.stratum)}
        for n in EB_SETS:
            d_i = downweight(inj[n].e.to_numpy(), share(inj[n], kc_i[n] if kc_i else None, ws,
                                                        tau), al)  # fmt: skip
            out[n] = stratum_pct(d_r, real.stratum, d_i, inj[n].stratum)
        return out

    def top1(sc, n, split):
        vm_r = (real.split == split).to_numpy()
        r = real[vm_r].assign(score=np.where(fr[vm_r], -np.inf, sc["real"][vm_r]))
        q = inj[n]
        vm = (q.split == split).to_numpy()
        qi = q[vm].assign(score=np.where(fi[n][vm], -np.inf, sc[n][vm]))
        return (rank_within(r, qi, ["split", "stratum"]) >= 0.99) & ~fi[n][vm], q[vm]

    def metrics(sc, split="val"):
        top, q = top1(sc, "v1", split)
        astro = ~q.is_null.to_numpy()
        m = {d: float(top[astro & (q.depth_bin == d).to_numpy()].mean()) for d in DEPTHS}
        m["all"] = float(top[astro].mean())
        for k in sorted(q.kind[astro].unique()):
            m[f"kind_{k}"] = float(top[astro & (q.kind == k).to_numpy()].mean())
        m["null_fpr"] = float(top[~astro].mean())
        top2, q2 = top1(sc, "eb1", split)
        isn = q2.is_null.to_numpy()
        m["eb_null_fpr"] = float(top2[isn].mean())
        m["dusty_recall"] = float(top2[~isn].mean())
        m["recall_mean"] = float(np.mean([m[d] for d in DEPTHS]))
        m["J"] = m["recall_mean"] - m["null_fpr"] - m["eb_null_fpr"]
        return m

    if a.fixed:
        fp = json.loads((score_dir(a.fixed, sec) / "provenance.json").read_text())
        grid = [(fp["alpha"], fp["w_single"], fp["tau"])]
    else:
        grid = list(itertools.product(ALPHAS, W_SINGLE, TAUS if a.kc else (None,)))
    sweep = []
    base_m = metrics(scores(0.0, 0.0, TAUS[-1] if a.kc else None))
    log(f"alpha 0: {fmt(base_m)}")
    for al, ws, tau in grid:
        if al == 0.0 and (ws, tau) != (0.0, TAUS[-1] if a.kc else None):
            continue  # alpha 0 is the base whatever the other knobs
        m = metrics(scores(al, ws, tau))
        m.update(
            alpha=al,
            w_single=ws,
            tau=tau,
            ok_dusty=bool(m["dusty_recall"] >= base_m["dusty_recall"] - DUSTY_DROP),
        )
        sweep.append(m)
        log(
            f"  alpha {al:.2f} w_single {ws:.2f} tau {tau}: {fmt(m)}"
            + ("" if m["ok_dusty"] else "  [dusty-tail recall drop]")
        )
    ok = [s for s in sweep if s["ok_dusty"]]
    best = max(s["J"] for s in ok)
    pick = min((s for s in ok if s["J"] >= best - TOL),
               key=lambda s: (s["alpha"], s["w_single"], -(s["tau"] or 0)))  # fmt: skip
    al, ws, tau = pick["alpha"], pick["w_single"], pick["tau"]
    log(f"chosen alpha {al} w_single {ws} tau {tau} (val J {pick['J']:.4f}, best {best:.4f})")

    sc = scores(al, ws, tau)
    sc0 = scores(0.0, 0.0, tau)
    od = score_dir(a.name, sec)
    od.mkdir(parents=True, exist_ok=True)
    frames = {"real": (real, fr, sc["real"], e_r, kc_r)}
    for n in EB_SETS:
        frames[f"inj_{n}"] = (inj[n], fi[n], sc[n], inj[n].e.to_numpy(),
                              kc_i[n] if kc_i else None)  # fmt: skip
    for part, (df, f, s, e, kc) in frames.items():
        o = pd.DataFrame({"tic": (df.tic if "tic" in df else df.host_tic).to_numpy(),
                          "sector": sec,
                          "inj_id": df.inj_id.to_numpy() if "inj_id" in df else None,
                          "stratum": df.stratum.to_numpy(),
                          "score": np.where(f, -1.0, s), "score_raw": s, "score_base": e,
                          "eb_like": combine_eb(df, w_single=ws),
                          "explained": share(df, kc, ws, tau), "cmv2_flag": f})  # fmt: skip
        for c in KEEP_EB:
            o[c] = df[c].to_numpy()
        o.to_parquet(od / f"{part}.parquet", index=False)
    # the base rule on the eb1 set, for the evaluator (alpha = 0: identical to the base)
    bdir = score_dir(a.base, sec)
    if not (bdir / "inj_eb1.parquet").exists():
        df = inj["eb1"]
        pd.DataFrame(
            {
                "tic": df.host_tic.to_numpy(),
                "sector": sec,
                "inj_id": df.inj_id.to_numpy(),
                "stratum": df.stratum.to_numpy(),
                "score": np.where(fi["eb1"], -1.0, sc0["eb1"]),
                "score_raw": sc0["eb1"],
                "cmv2_flag": fi["eb1"],
            }
        ).to_parquet(bdir / "inj_eb1.parquet", index=False)
        log(f"wrote {bdir / 'inj_eb1.parquet'} (base rule, for evaluation)")
    prov = {"scorer": a.name, "base": a.base, "sector": sec, "components": comps, "rule": rule,
            "weights": list(w) if w else None, "kc": a.kc, "alpha": al, "w_single": ws,
            "tau": tau, "veto": "cmv2", "sets": list(EB_SETS),
            "eb_like": "max(eb_periodic, w_single * eb_single_raw, eb_xsec * c_shape * c_extra)",
            "explained": "eb_like" if not a.kc else "1 - (1 - E_kc(tau)) (1 - eb_like)",
            "rule_text": "score = pct_stratum(1 - (1 - s)^(1 - alpha * E)); cmv2 flagged -> -1",
            "objective": "val: mean recall@1% (v1, ~0.3/1/3%) - null FPR (v1) - EB-null FPR "
                         f"(eb1); dusty-tail recall >= alpha-0 value - {DUSTY_DROP}; "
                         f"ties within {TOL}: smallest alpha, then w_single, then largest tau",
            "val_alpha0": base_m, "val_chosen": pick, "val_sweep": sweep,
            "fixed_from": a.fixed,
            "built_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}  # fmt: skip
    (od / "provenance.json").write_text(json.dumps(prov, indent=1, default=float))
    log(f"wrote {od}")


def fmt(m: dict) -> str:
    return (
        f"0.3% {m['~0.3%']:.3f} 1% {m['~1%']:.3f} 3% {m['~3%']:.3f} all {m['all']:.3f} "
        f"null {m['null_fpr']:.3f} ebnull {m['eb_null_fpr']:.3f} dusty "
        f"{m['dusty_recall']:.3f} J {m['J']:.4f}"
    )


if __name__ == "__main__":
    main()
