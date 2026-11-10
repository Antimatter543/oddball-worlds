"""Candidate list across sectors (the first multi-sector search, PLAN 2026-11-10).

Per sector the frozen S32 rules give two rankings of the real stars (higher score = weirder):

* ``cls``: the classical baseline, with the same EB down-weight and ``cmv3`` veto as ``ens_v3``:
  ``pct_stratum(1 - (1 - c)^(1 - alpha * eb_like))``, ``c`` the classical stratum percentile,
  ``alpha`` = ens_v3's 0.9 (not tuned for classical), cmv3-flagged stars -> -1.
* ``ens``: ``ens_v3`` as written by ``ensemble_v3.py`` (EB down-weight and cmv3 already in).

A star is a candidate in a sector when it is in the top ``n`` of either ranking. Candidates
are merged across sectors by TIC, keeping every sector's ranks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from oddball.knownclass.explain import downweight, stratum_pct

TOP_N = 300
EB_LIKE_FLAG = 0.5


def rank_desc(score: np.ndarray, tie: np.ndarray, tic: np.ndarray) -> np.ndarray:
    """1-based rank by ``score`` (desc), ties broken by ``tie`` (desc), then TIC (asc).
    Deterministic, so the same inputs always give the same list."""
    order = np.lexsort((np.asarray(tic), -np.nan_to_num(np.asarray(tie, float), nan=-np.inf),
                        -np.nan_to_num(np.asarray(score, float), nan=-np.inf)))  # fmt: skip
    r = np.empty(len(order), dtype=np.int64)
    r[order] = np.arange(1, len(order) + 1)
    return r


def classical_eb(c_pct: np.ndarray, eb_like: np.ndarray, stratum: np.ndarray,
                 flag: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """(score with veto, score before veto) for the classical ranking (module docstring)."""
    d = downweight(c_pct, eb_like, alpha) if alpha else np.asarray(c_pct, float)
    raw = stratum_pct(d, stratum, d, stratum)
    return np.where(np.asarray(flag, bool), -1.0, raw), raw


def add_ranks(t: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """``t``: one sector's real stars with ``tic, cls_score, cls_raw, ens_score, ens_raw``.
    Returns a copy with ``cls_rank``, ``ens_rank`` (1 = weirdest) and ``in_cls``, ``in_ens``
    (rank <= ``n``)."""
    t = t.copy()
    tic = t.tic.to_numpy()
    t["cls_rank"] = rank_desc(t.cls_score.to_numpy(), t.cls_raw.to_numpy(), tic)
    t["ens_rank"] = rank_desc(t.ens_score.to_numpy(), t.ens_raw.to_numpy(), tic)
    t["in_cls"] = t.cls_rank <= n
    t["in_ens"] = t.ens_rank <= n
    return t


def sector_candidates(t: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """The candidate rows of one sector: the union of both top-``n`` lists (:func:`add_ranks`)."""
    t = t if "cls_rank" in t else add_ranks(t, n)
    return t[t.in_cls | t.in_ens].reset_index(drop=True)


def merge_sectors(long: pd.DataFrame, observed: dict[int, set[int]]) -> pd.DataFrame:
    """One row per TIC from the per-sector candidate rows ``long`` (``tic, sector, cls_rank,
    ens_rank, ...``). ``observed``: sector -> TICs ingested there (to count the sectors a star
    was looked at in, candidate or not)."""
    long = long.copy()
    long["best_rank"] = long[["cls_rank", "ens_rank"]].min(axis=1)
    g = long.sort_values(["tic", "best_rank", "sector"]).groupby("tic", sort=True)
    out = g.agg(n_sectors_cand=("sector", "nunique"),
                sectors_cand=("sector", lambda s: " ".join(str(x) for x in sorted(s))),
                best_rank=("best_rank", "min"),
                best_sector=("sector", "first")).reset_index()  # fmt: skip
    secs = sorted(observed)
    obs = {s: observed[s] for s in secs}
    out["n_sectors_obs"] = [sum(int(t) in obs[s] for s in secs) for t in out.tic]
    out["sectors_obs"] = [" ".join(str(s) for s in secs if int(t) in obs[s]) for t in out.tic]
    out["multi"] = out.n_sectors_cand >= 2
    wide = long.pivot(index="tic", columns="sector", values=["cls_rank", "ens_rank"])
    wide.columns = [f"{v}_s{s:04d}" for v, s in wide.columns]
    return out.merge(wide.reset_index(), on="tic", how="left")


def ccd_excess(cand: pd.DataFrame, stars: pd.DataFrame) -> pd.DataFrame:
    """Candidates per camera/CCD against the sector's share of stars there: ``ratio`` = observed
    / expected, with a Poisson upper-tail p-value for an excess (one sector)."""
    from scipy.stats import poisson

    n_all = len(stars)
    base = stars.groupby(["camera", "ccd"]).size().rename("n_stars")
    obs = cand.groupby(["camera", "ccd"]).size().rename("n_cand")
    t = pd.concat([base, obs], axis=1).fillna(0).astype(int).reset_index()
    t["expected"] = t.n_stars / n_all * len(cand)
    t["ratio"] = np.where(t.expected > 0, t.n_cand / t.expected.clip(lower=1e-9), np.nan)
    t["p_excess"] = poisson.sf(t.n_cand - 1, t.expected)
    return t


def time_clusters(cand: pd.DataFrame, bin_days: float = 0.5, min_n: int = 3) -> pd.DataFrame:
    """Candidates whose strongest residual event (``cmv3_res_t``) falls in the same
    ``bin_days`` bin on the same camera/CCD. Stratified percentiles spread the list evenly over
    CCDs by construction, so a CCD-wide artefact shows up as a cluster in time, not as a CCD
    excess. Returns clusters with at least ``min_n`` stars, largest first."""
    c = cand[np.isfinite(cand.cmv3_res_t)].copy()
    c["tbin"] = np.floor(c.cmv3_res_t / bin_days).astype(int)
    g = c.groupby(["camera", "ccd", "tbin"]).agg(n=("tic", "size"),
                                                 t_mid=("cmv3_res_t", "median"),
                                                 events=("cmv3_res_ev", lambda s: ", ".join(
                                                     sorted(set(map(str, s)))[:3])))  # fmt: skip
    return g[g.n >= min_n].sort_values("n", ascending=False).reset_index()
