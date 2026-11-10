"""Ensemble selection with a dipper-aware val objective (``ens_v3``; docs/ML_DESIGN.md §15).

A candidate scorer is a combiner of the component stratum percentiles (``rule``, ``weights``)
followed by the EB down-weight (``alpha``) and the ``cmv3`` veto::

    s     = combine(p_1..p_k; rule, w)                   (scripts/ensemble.py)
    e     = pct_stratum(s)                               (against the sector's real stars)
    score = pct_stratum(1 - (1 - e)^(1 - alpha * eb_like)),   cmv3-flagged -> -1

Val objective (pre-registered before any number was seen; test and known positives unread)::

    J = R_gen + R_yso - N_inst - N_eb
    R_gen  mean recall@1% over the ~0.3 / 1 / 3 % depth bins, v1 targets
    R_yso  mean recall@1% over the ~0.3 / 1 / 3 / 10 % depth bins, yso1 targets (young hosts)
    N_inst mean over the five instrument null types (v1 null_ccd, null_dump; n2 null_asteroid,
           null_step, null_column) of the false-positive rate in the top 1%
    N_eb   eb1 null_eb false-positive rate

subject to the eb1 dusty-tail recall not falling more than ``DUSTY_DROP`` below the same
combiner's value at ``alpha = 0``. Among candidates within ``TOL`` of the best J the simplest
wins (:func:`simplicity`). The yso1 matched-null rate (``N_yso``) is reported, not penalised:
real dippers *are* variable young stars, so pushing young-star variability down would push the
targets down with it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from oddball.eval.harness import rank_within

RULE_ORDER = ("mean", "max", "wmax", "fisher", "stouffer")
GEN_DEPTHS = ("~0.3%", "~1%", "~3%")
YSO_DEPTHS = ("~0.3%", "~1%", "~3%", "~10%")
INST_NULLS = ("null_ccd", "null_dump", "null_asteroid", "null_step", "null_column")
TOL = 0.01
DUSTY_DROP = 0.02


def top1(real: pd.DataFrame, q: pd.DataFrame, real_score: np.ndarray, q_score: np.ndarray,
         real_flag: np.ndarray, q_flag: np.ndarray, split: str) -> np.ndarray:  # fmt: skip
    """In the top 1% after the symmetric veto: ``q`` rows of ``split`` ranked among the real
    stars of the same split and stratum (the evaluator's rule). Returns a bool array over the
    ``q`` rows of that split (in order)."""
    vr = (real.split == split).to_numpy()
    vq = (q.split == split).to_numpy()
    r = pd.DataFrame({"split": split, "stratum": real.stratum.to_numpy()[vr],
                      "score": np.where(real_flag[vr], -np.inf, real_score[vr])})  # fmt: skip
    i = pd.DataFrame({"split": split, "stratum": q.stratum.to_numpy()[vq],
                      "score": np.where(q_flag[vq], -np.inf, q_score[vq])})  # fmt: skip
    return (rank_within(r, i, ["split", "stratum"]) >= 0.99) & ~q_flag[vq]


def _mean(x: np.ndarray) -> float:
    return float(x.mean()) if len(x) else float("nan")


def set_terms(name: str, q: pd.DataFrame, hit: np.ndarray) -> dict:
    """Per-set metric terms from the val hits of one injection set (rows of one split)."""
    astro = ~q.is_null.to_numpy()
    kind = q.kind.to_numpy()
    dbin = q.depth_bin.astype(object).to_numpy()
    out = {}
    if name == "v1":
        for d in GEN_DEPTHS:
            out[f"gen_{d}"] = _mean(hit[astro & (dbin == d)])
        out["gen_all"] = _mean(hit[astro])
        for k in ("null_ccd", "null_dump"):
            out[f"fpr_{k}"] = _mean(hit[kind == k])
        out["fpr_v1_pooled"] = _mean(hit[~astro])
    elif name == "n2":
        for k in ("null_asteroid", "null_step", "null_column"):
            out[f"fpr_{k}"] = _mean(hit[kind == k])
    elif name == "eb1":
        out["fpr_null_eb"] = _mean(hit[kind == "null_eb"])
        out["dusty"] = _mean(hit[kind == "dusty_tail"])
    elif name == "yso1":
        for d in YSO_DEPTHS:
            out[f"yso_{d}"] = _mean(hit[astro & (dbin == d)])
        out["yso_all"] = _mean(hit[astro])
        out["fpr_null_yso"] = _mean(hit[~astro])
    else:
        raise ValueError(name)
    return out


def objective(t: dict) -> dict:
    """The J terms from the merged per-set terms."""
    r_gen = float(np.mean([t[f"gen_{d}"] for d in GEN_DEPTHS]))
    r_yso = float(np.mean([t[f"yso_{d}"] for d in YSO_DEPTHS]))
    n_inst = float(np.mean([t[f"fpr_{k}"] for k in INST_NULLS]))
    n_eb = t["fpr_null_eb"]
    return {"R_gen": r_gen, "R_yso": r_yso, "N_inst": n_inst, "N_eb": n_eb,
            "N_yso": t["fpr_null_yso"], "J": r_gen + r_yso - n_inst - n_eb,
            # the ens_v2 objective, for reference: v1 recall minus pooled v1 null FPR
            "J_v2": r_gen - t["fpr_v1_pooled"]}  # fmt: skip


def simplicity(c: dict) -> tuple:
    """Sort key: rule order, then closeness of the weights to equal, then smaller alpha."""
    w = c.get("weights") or ()
    return (RULE_ORDER.index(c["rule"]), float(np.sum(np.abs(np.log2(w)))) if w else 0.0,
            float(c["alpha"]))  # fmt: skip


def pick(sweep: list[dict], tol: float = TOL, key: str = "J") -> dict:
    """Simplest admissible candidate within ``tol`` of the best admissible ``key``."""
    ok = [c for c in sweep if c.get("ok_dusty", True)]
    if not ok:
        raise ValueError("no admissible candidate")
    best = max(c[key] for c in ok)
    return min((c for c in ok if c[key] >= best - tol), key=simplicity)


def cluster_bootstrap(hit: np.ndarray, cluster: np.ndarray, n_boot: int = 2000,
                      seed: int = 0) -> tuple[float, float, float]:  # fmt: skip
    """Mean of ``hit`` and a 90% interval, resampling clusters (e.g. hosts) with replacement."""
    hit = np.asarray(hit, float)
    u, inv = np.unique(np.asarray(cluster), return_inverse=True)
    s = np.bincount(inv, weights=hit, minlength=len(u))
    n = np.bincount(inv, minlength=len(u)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(n_boot, len(u)))
    m = s[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
    return float(hit.mean()), float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95))


def clusters(q: pd.DataFrame, name: str) -> np.ndarray:
    """Resampling unit per injection: the host for yso1 (its replicas share the host's own
    variability), the shared event for instrument nulls (every host of one event carries the
    identical signal), else the injection itself."""
    if name == "yso1":
        return q.host_tic.astype(str).to_numpy()
    ev = q.event_id.to_numpy() if "event_id" in q else np.full(len(q), -1)
    return np.where(ev >= 0, [f"ev{e}" for e in ev], q.inj_id.astype(str).to_numpy())


def paired_delta_j(sets: dict, hits_a: dict, hits_b: dict, n_boot: int = 500,
                   seed: int = 0) -> tuple[float, float, float]:  # fmt: skip
    """J(a) - J(b) and a 90% interval, resampling the clusters of every set (the same draw for
    both candidates). ``sets``: name -> val rows of that set; ``hits_*``: name -> bool array.

    Clusters are resampled within strata (yso1: all hosts; other sets: per kind, with all
    astrophysical kinds of v1 as one stratum), so no null type can vanish from a draw. Draws
    in which a term is still undefined (an empty depth bin) are dropped.
    """
    rng = np.random.default_rng(seed)
    strata = {}
    for n, q in sets.items():
        cl = clusters(q, n)
        if n == "yso1":
            st = np.zeros(len(q), int)
        else:
            st = np.unique(np.where(q.is_null.to_numpy(), q.kind.astype(str).to_numpy(), "astro"),
                           return_inverse=True)[1]  # fmt: skip
        groups = []
        for k in np.unique(st):
            rows = np.flatnonzero(st == k)
            u, inv = np.unique(cl[rows], return_inverse=True)
            groups.append([rows[inv == i] for i in range(len(u))])
        strata[n] = groups

    def j(idx, hits):
        t = {}
        for n, q in sets.items():
            t.update(set_terms(n, q.iloc[idx[n]], hits[n][idx[n]]))
        return objective(t)["J"]

    full = {n: np.arange(len(q)) for n, q in sets.items()}
    d0 = j(full, hits_a) - j(full, hits_b)
    ds = []
    for _ in range(n_boot):
        idx = {n: np.concatenate([g[k] for g in strata[n]
                                  for k in rng.integers(0, len(g), len(g))])
               for n in sets}  # fmt: skip
        d = j(idx, hits_a) - j(idx, hits_b)
        if np.isfinite(d):
            ds.append(d)
    if not ds:
        return float(d0), float("nan"), float("nan")
    return float(d0), float(np.quantile(ds, 0.05)), float(np.quantile(ds, 0.95))
