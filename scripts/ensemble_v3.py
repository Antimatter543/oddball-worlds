"""``ens_v3``: combiner and EB down-weight chosen jointly on val with a dipper-aware objective.

    uv run python -I scripts/ensemble_v3.py --sector 32 --name ens_v3

Components ``classical``, ``ssl_v2_iforest``, ``ssl_v2_recon`` (their score files), the ``ebf_v1``
EB tables, the ``cmv3`` veto, and four injection sets: ``v1`` (generic targets + CCD / dump
nulls), ``n2`` (asteroid, step, column nulls), ``eb1`` (EB nulls, dusty-tail targets) and
``yso1`` (dipper-style targets on catalogued young hosts + matched nulls). Every candidate is
``oddball.score.ens3``'s pipeline; its val objective J and the tie rule are documented there.

Grid: rule in {mean, max, fisher, stouffer} plus weighted max ``max(s_c, w2 s_iforest, w3 s_recon)``
with w2, w3 in ``WEIGHTS`` (the classical-floor family is w2, w3 <= 1), times ``alpha`` in
``ALPHAS`` (``eb_like = max(eb_periodic, eb_xsec * shape * extra)``, ``w_single = 0`` as chosen for
``ens_v1_eb``). Only val rows are scored for selection; test and the known positives are never
read here (real stars of every split are written to the score files, as for every scorer).

Also, for the evaluator, the two reference scorers are applied unchanged to the sets they lack
(never overwriting an existing file): ``ens_v2`` (wmax 1:2:2, alpha 0, cmv3) and ``ens_v1_eb``
(wmax 1:1.5:1.5, alpha 0.75, cmv2). Both are first reproduced exactly from their own files.
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
from oddball.eval.harness import DEPTH_EDGES, DEPTH_LABELS, load_truth, score_dir, veto_dir
from oddball.knownclass.ebtests import combine_eb
from oddball.knownclass.explain import downweight, stratum_pct
from oddball.score import ens3 as E3
from oddball.tess.arrays import dir_metadata

COMPS = ("classical", "ssl_v2_iforest", "ssl_v2_recon")
SETS = ("v1", "n2", "eb1", "yso1")
WEIGHTS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
REFS = {  # name -> (rule, weights, alpha, veto)
    "ens_v2": ("wmax", (1.0, 2.0, 2.0), 0.0, "cmv3"),
    "ens_v1_eb": ("wmax", (1.0, 1.5, 1.5), 0.75, "cmv2"),
}
EBF = ("eb_periodic", "eb_single_raw", "eb_xsec", "eb_c_shape", "eb_c_extra")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_eb(sec: int, part: str) -> pd.DataFrame:
    d = outputs_root() / "knownclass" / "features" / "ebf_v1" / f"s{sec:04d}"
    return pd.read_parquet(d / f"{part}.parquet", columns=["tic", "inj_id", *EBF])


def combiners():
    yield ("mean", None)
    yield ("max", None)
    for w2, w3 in itertools.product(WEIGHTS, repeat=2):
        if (w2, w3) != (1.0, 1.0):  # = max
            yield ("wmax", (1.0, w2, w3))
    yield ("fisher", None)
    yield ("stouffer", None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="ens_v3")
    ap.add_argument("--sector", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--components", nargs="+", default=list(COMPS),
                    help="component scorers (e.g. another ssl_v2 seed's), in COMPS order")
    ap.add_argument("--fixed", default=None,
                    help="reuse this scorer's rule / weights / alpha (no sweep; seed checks)")
    a = ap.parse_args()
    sec = a.sector
    comps = tuple(a.components)
    t_start = time.time()
    meta = dir_metadata(sector_dir(sec))[["tic", "tmag", "camera", "ccd"]]

    # ---- real stars
    real = meta.copy()
    for c in comps:
        r = pd.read_parquet(score_dir(c, sec) / "real.parquet", columns=["tic", "score"])
        real = real.merge(r.rename(columns={"score": f"raw_{c}"}), on="tic", how="inner",
                          validate="one_to_one")  # fmt: skip
    real["split"] = split_array(real.tic.to_numpy())
    real["stratum"] = stratum_id(real.tmag, real.camera, real.ccd)
    for v in ("cmv2", "cmv3"):
        f = pd.read_parquet(veto_dir(v, sec) / "real.parquet", columns=["tic", f"{v}_flag"])
        real = real.merge(f, on="tic", how="left", validate="one_to_one")
    real = real.merge(load_eb(sec, "real").drop(columns=["inj_id"]), on="tic", how="left",
                      validate="one_to_one")  # fmt: skip
    ref = {c: real[f"raw_{c}"].fillna(-np.inf).to_numpy() for c in comps}
    P = {"real": np.column_stack([stratum_pct(ref[c], real.stratum, ref[c], real.stratum)
                                  for c in comps])}  # fmt: skip
    eb = {"real": combine_eb(real, w_single=0.0)}
    flags = {v: {"real": real[f"{v}_flag"].fillna(False).to_numpy(bool)} for v in ("cmv2", "cmv3")}
    frames = {"real": real}

    # ---- injection sets
    for name in SETS:
        t = load_truth(name, sec)
        t["depth_bin"] = pd.cut(t.depth, DEPTH_EDGES, labels=DEPTH_LABELS, right=False)
        cols = ["inj_id", "host_tic", "split", "kind", "depth", "is_null", "depth_bin", "tmag",
                "camera", "ccd", "event_id"]  # fmt: skip
        q = t[cols].copy()
        for c in comps:
            i = pd.read_parquet(score_dir(c, sec) / f"inj_{name}.parquet",
                                columns=["inj_id", "score"])  # fmt: skip
            q = q.merge(i.rename(columns={"score": f"raw_{c}"}), on="inj_id", how="left",
                        validate="one_to_one")  # fmt: skip
        q["stratum"] = stratum_id(q.tmag, q.camera, q.ccd)
        for v in ("cmv2", "cmv3"):
            p = veto_dir(v, sec) / f"inj_{name}.parquet"
            if p.exists():
                f = pd.read_parquet(p, columns=["inj_id", f"{v}_flag"])
                q = q.merge(f, on="inj_id", how="left", validate="one_to_one")
            flags[v][name] = (q[f"{v}_flag"].fillna(False).to_numpy(bool) if f"{v}_flag" in q
                              else None)  # fmt: skip
        if flags["cmv3"][name] is None:
            raise SystemExit(f"no cmv3 veto for {name}: run scripts/veto3_inj_set.py")
        q = q.merge(load_eb(sec, f"inj_{name}").drop(columns=["tic"]), on="inj_id", how="left",
                    validate="one_to_one")  # fmt: skip
        miss = {c: int(q[f"raw_{c}"].isna().sum()) for c in comps}
        miss["eb"] = int(q.eb_periodic.isna().sum())
        if any(miss.values()):
            raise SystemExit(f"{name}: missing inputs {miss}")
        P[name] = np.column_stack([stratum_pct(ref[c], real.stratum, q[f"raw_{c}"], q.stratum)
                                   for c in comps])  # fmt: skip
        eb[name] = combine_eb(q, w_single=0.0)
        frames[name] = q
        log(f"{name}: {len(q)} light curves ({int((q.split == 'val').sum())} val); cmv3 flags "
            f"{flags['cmv3'][name].mean():.3f}")

    parts = ("real", *SETS)

    def base(rule, w):
        st_r = combine(P["real"], rule, w)
        out = {"real": stratum_pct(st_r, real.stratum, st_r, real.stratum)}
        for n in SETS:
            out[n] = stratum_pct(st_r, real.stratum, combine(P[n], rule, w), frames[n].stratum)
        return out

    def final(e, alpha):
        """EB down-weight then re-percentile (identity at alpha = 0, as in eb_explain)."""
        if alpha == 0.0:
            return e
        d = {p: downweight(e[p], eb[p], alpha) for p in parts}
        return {p: stratum_pct(d["real"], real.stratum, d[p], frames[p].stratum) for p in parts}

    def val_hits(sc, veto="cmv3"):
        return {n: E3.top1(real, frames[n], sc["real"], sc[n], flags[veto]["real"],
                           flags[veto][n], "val") for n in SETS}  # fmt: skip

    vrows = {n: frames[n][frames[n].split == "val"].reset_index(drop=True) for n in SETS}

    def metrics(hits):
        t = {}
        for n in SETS:
            t.update(E3.set_terms(n, vrows[n], hits[n]))
        return {**t, **E3.objective(t)}

    # ---- reproduce the reference scorers from their own files (seed-0 components only)
    checks = {}
    for nm, (rule, w, al, _v) in (REFS.items() if comps == COMPS else ()):
        sc = final(base(rule, w), al)
        for p in parts:
            f = score_dir(nm, sec) / ("real.parquet" if p == "real" else f"inj_{p}.parquet")
            if not f.exists():
                continue
            key = "tic" if p == "real" else "inj_id"
            old = frames[p][[key]].merge(pd.read_parquet(f, columns=[key, "score_raw"]), on=key,
                                         how="left")  # fmt: skip
            dmax = float(np.nanmax(np.abs(old.score_raw.to_numpy() - sc[p])))
            checks[f"{nm}/{p}"] = dmax
            if not dmax < 1e-9:
                raise SystemExit(f"{nm}/{p}: reproduction failed (max |diff| {dmax})")
    log(f"reference scorers reproduced: {checks}")

    # ---- the val sweep
    sweep = []
    fixed = None
    if a.fixed:
        fixed = json.loads((score_dir(a.fixed, sec) / "provenance.json").read_text())
        cands = [(fixed["rule"], tuple(fixed["weights"]) if fixed["weights"] else None)]
        fixed["alpha"] = float(fixed.get("alpha") or 0.0)  # ens_v2 has no EB step: alpha 0
        alphas = (0.0, fixed["alpha"]) if fixed["alpha"] else (0.0,)
    else:
        cands, alphas = list(combiners()), ALPHAS
    for rule, w in cands:
        e = base(rule, w)
        dusty0 = None
        for al in alphas:
            m = metrics(val_hits(final(e, al)))
            if al == 0.0:
                dusty0 = m["dusty"]
            m.update(rule=rule, weights=list(w) if w else None, alpha=al,
                     ok_dusty=bool(m["dusty"] >= dusty0 - E3.DUSTY_DROP))  # fmt: skip
            sweep.append(m)
        best = max((s for s in sweep[-len(alphas):]), key=lambda s: s["J"])
        log(f"  {rule:8s} {w!s:18s} best alpha {best['alpha']:.2f}: J {best['J']:.4f} "
            f"R_gen {best['R_gen']:.3f} R_yso {best['R_yso']:.3f} N_inst {best['N_inst']:.3f} "
            f"N_eb {best['N_eb']:.3f} N_yso {best['N_yso']:.3f} dusty {best['dusty']:.3f}")
    if fixed:
        pick = next(s for s in sweep if s["alpha"] == fixed["alpha"])
    else:
        pick = E3.pick(sweep)
    best_j = max(s["J"] for s in sweep if s["ok_dusty"])
    log(f"chosen: {pick['rule']} {pick['weights']} alpha {pick['alpha']} (val J {pick['J']:.4f}, "
        f"best {best_j:.4f})")

    families, old_obj, boot, yso_ci = {}, None, {}, None
    if not fixed:
        def fam(s):
            if s["rule"] != "wmax":
                return s["rule"]
            return "wmax_floor" if max(s["weights"][1:]) <= 1.0 else "wmax_ml_heavy"

        families = {}
        for s in sweep:
            if s["ok_dusty"] and (fam(s) not in families or s["J"] > families[fam(s)]["J"]):
                families[fam(s)] = s
        for k, s in families.items():
            log(f"  family {k:14s}: {s['rule']} {s['weights']} alpha {s['alpha']} J {s['J']:.4f} "
                f"R_gen {s['R_gen']:.3f} R_yso {s['R_yso']:.3f} N_inst {s['N_inst']:.3f}")
        old_obj = E3.pick(sweep, key="J_v2")
        log(f"under the ens_v2 objective (R_gen - pooled v1 null FPR) the same grid picks "
            f"{old_obj['rule']} {old_obj['weights']} alpha {old_obj['alpha']}")

        def cand(rule, w, al):
            return next(s for s in sweep if s["rule"] == rule and s["alpha"] == al
                        and (s["weights"] == (list(w) if w else None)))  # fmt: skip

        # ---- uncertainty: paired cluster bootstrap of J against the references
        def hits_of(rule, w, al):
            return val_hits(final(base(rule, tuple(w) if w else None), al))

        h_pick = hits_of(pick["rule"], pick["weights"], pick["alpha"])
        top = max((s for s in sweep if s["ok_dusty"]), key=lambda s: s["J"])
        comps_ = {"ens_v2 rule": ("wmax", (1.0, 2.0, 2.0), 0.0),
                  "ens_v1_eb rule": ("wmax", (1.0, 1.5, 1.5), 0.75),
                  "max, alpha 0.75": ("max", None, 0.75),
                  "best J": (top["rule"], top["weights"], top["alpha"])}  # fmt: skip
        boot = {}
        for k, (rule, w, al) in comps_.items():
            d0, lo, hi = E3.paired_delta_j(vrows, h_pick, hits_of(rule, w, al), n_boot=a.n_boot)
            c = cand(rule, w, al)
            boot[k] = {"rule": rule, "weights": list(w) if w else None, "alpha": al, "J": c["J"],
                       "R_gen": c["R_gen"], "R_yso": c["R_yso"], "N_inst": c["N_inst"],
                       "N_eb": c["N_eb"], "N_yso": c["N_yso"], "dJ_pick_minus": d0,
                       "dJ_90": [lo, hi]}  # fmt: skip
            log(f"  J(pick) - J({k}) = {d0:+.4f} [90%: {lo:+.4f}, {hi:+.4f}]")
        yso_ci = E3.cluster_bootstrap(h_pick["yso1"][~vrows["yso1"].is_null.to_numpy()],
                                      vrows["yso1"].host_tic[~vrows["yso1"].is_null].to_numpy())
        log(f"val yso1 recall (pick, all depths): {yso_ci[0]:.3f} [90% host bootstrap "
            f"{yso_ci[1]:.3f}, {yso_ci[2]:.3f}]")


    # ---- write ens_v3 and the reference scorers' missing set files
    def write(nm, rule, w, al, veto, overwrite):
        e = base(rule, tuple(w) if w else None)
        sc = final(e, al)
        od = score_dir(nm, sec)
        od.mkdir(parents=True, exist_ok=True)
        wrote = []
        for p in parts:
            path = od / ("real.parquet" if p == "real" else f"inj_{p}.parquet")
            if path.exists() and not overwrite:
                continue
            fl = flags[veto][p]
            if fl is None:
                log(f"  {nm}/{p}: no {veto} flags, not written")
                continue
            df = frames[p]
            o = pd.DataFrame({"tic": (df.tic if p == "real" else df.host_tic).to_numpy(),
                              "sector": sec,
                              "inj_id": None if p == "real" else df.inj_id.to_numpy(),
                              "stratum": df.stratum.to_numpy(),
                              "score": np.where(fl, -1.0, sc[p]), "score_raw": sc[p],
                              "score_base": e[p], "eb_like": eb[p], f"{veto}_flag": fl})  # fmt: skip
            for k, c in enumerate(comps):
                o[f"score_{c}"] = P[p][:, k]
            o.to_parquet(path, index=False)
            wrote.append(path.name)
        return wrote

    written = {a.name: write(a.name, pick["rule"], pick["weights"], pick["alpha"], "cmv3", True)}
    for nm, (rule, w, al, veto) in (REFS.items() if not fixed else ()):
        written[nm] = write(nm, rule, w, al, veto, False)
    log(f"written: {written}")
    od = score_dir(a.name, sec)
    pd.DataFrame(sweep).to_csv(od / "val_sweep.csv", index=False)
    prov = {"scorer": a.name, "sector": sec, "components": list(comps), "sets": list(SETS),
            "veto": "cmv3", "rule": pick["rule"], "weights": pick["weights"],
            "alpha": pick["alpha"], "w_single": 0.0,
            "rule_text": "score = pct_stratum(1 - (1 - e)^(1 - alpha * eb_like)), "
                         "e = pct_stratum(combine(p; rule, w)); cmv3 flagged -> -1",
            "objective": "val: J = R_gen(v1 ~0.3/1/3%) + R_yso(yso1 ~0.3/1/3/10%) - N_inst (mean "
                         "of 5 instrument null types) - N_eb (eb1); dusty-tail recall >= alpha-0 "
                         f"value - {E3.DUSTY_DROP}; ties within {E3.TOL}: simplest rule, "
                         "weights closest to equal, smallest alpha",
            "grid": {"rules": ["mean", "max", "wmax", "fisher", "stouffer"],
                     "weights": list(WEIGHTS), "alphas": list(ALPHAS)},
            "val_chosen": pick, "val_best_J": best_j, "families": families,
            "pick_under_ens_v2_objective": ({k: old_obj[k] for k in ("rule", "weights", "alpha",
                                                                    "J", "J_v2")}
                                            if old_obj else None), "fixed_from": a.fixed,
            "bootstrap": boot, "val_yso_recall_host_ci90": yso_ci,
            "reproduction_maxdiff": checks, "written": written,
            "seconds": round(time.time() - t_start),
            "built_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}  # fmt: skip
    (od / "provenance.json").write_text(json.dumps(prov, indent=1, default=float))
    log(f"wrote {od} in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
