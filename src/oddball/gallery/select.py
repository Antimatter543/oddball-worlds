"""Which stars go into a labelling gallery (docs/ML_DESIGN.md §7, "choosing what to label next").

Exploitation plus about 30% exploration, so the labels are not only the head of one ranking:

* ``top``      the best-ranked stars by the scorer's ``score``.
* ``explore-top5``  uniform random draws from the rest of the top 5%.
* ``explore-mid``   "uncertain" mid-ranks: random draws from ranks in the (5%, 25%] band,
  weighted by how much the scorer's own sub-scores disagree (one sub-score extreme while the
  combination is lukewarm). Without sub-scores the draw is uniform. Once a re-ranker exists,
  this is where its uncertainty goes.

Known positives (split ``known``) are never shown: they are evaluation-only (§0 "no peeking").
Stars already labelled are skipped by default, so a new gallery shows new stars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from oddball.common.splits import split_array

NON_SUBSCORE = {"score", "score_rankavg"}


def subscore_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("score_") and c not in NON_SUBSCORE
            and pd.api.types.is_numeric_dtype(df[c])]  # fmt: skip


def rank_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Global rank (1 = weirdest) by ``score``; ties broken by ``score_rankavg`` if present.

    Rows with a NaN score are dropped. Adds ``rank`` and ``rank_pct`` (rank / n * 100).
    """
    d = scores[np.isfinite(scores["score"].to_numpy(np.float64))].copy()
    keys = ["score"] + (["score_rankavg"] if "score_rankavg" in d else [])
    d = d.sort_values(keys, ascending=False, kind="stable").reset_index(drop=True)
    d["rank"] = np.arange(1, len(d) + 1)
    d["rank_pct"] = 100.0 * d["rank"] / len(d)
    return d


def disagreement(d: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Spread (max - min) of the sub-scores' global percentiles; 0 with fewer than 2."""
    if len(cols) < 2:
        return np.zeros(len(d))
    p = np.column_stack([d[c].rank(pct=True).to_numpy() for c in cols])
    return np.nan_to_num(np.nanmax(p, 1) - np.nanmin(p, 1))


def select(scores: pd.DataFrame, n: int, explore_frac: float = 0.3, seed: int = 0,
           exclude: set[tuple[int, int]] | frozenset = frozenset(), top_band: float = 5.0,
           mid_band: tuple[float, float] = (5.0, 25.0)) -> pd.DataFrame:  # fmt: skip
    """Return the chosen rows (with ``rank``, ``rank_pct``, ``split``, ``reason``, ``pick``).

    ``exclude``: ``(tic, sector)`` pairs to skip (already labelled).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    d = rank_scores(scores)
    d["split"] = split_array(d.tic.to_numpy())
    keep = (d.split != "known").to_numpy()
    if exclude:
        keys = set(zip(d.tic.astype(int), d.sector.astype(int), strict=True))
        drop = keys & set(exclude)
        if drop:
            keep &= ~np.array([(int(t), int(s)) in drop for t, s in zip(d.tic, d.sector,
                                                                        strict=True)])  # fmt: skip
    pool = d[keep]
    rng = np.random.default_rng(seed)
    n_explore = round(explore_frac * n)
    n_top = n - n_explore

    top = pool.head(n_top).copy()
    top["pick"] = "top"
    top["reason"] = [f"top: rank {r} of {len(d)} by score" for r in top["rank"]]
    taken = set(top.index)

    n_t5 = (n_explore + 1) // 2
    band5 = pool[(pool.rank_pct <= top_band) & ~pool.index.isin(taken)]
    t5 = band5.loc[rng.choice(band5.index, size=min(n_t5, len(band5)), replace=False)].copy()
    t5["pick"] = "explore-top5"
    t5["reason"] = [f"explore: random draw from the top {top_band:g}% (rank {r}, top "
                    f"{p:.2f}%)" for r, p in zip(t5["rank"], t5.rank_pct, strict=True)]  # fmt: skip
    taken |= set(t5.index)

    n_mid = n_explore - len(t5)
    lo, hi = mid_band
    mid_pool = pool[(pool.rank_pct > lo) & (pool.rank_pct <= hi) & ~pool.index.isin(taken)]
    cols = subscore_columns(d)
    dis_all = pd.Series(disagreement(d, cols), index=d.index)
    dis = dis_all.loc[mid_pool.index].to_numpy()
    w = dis**2 + 1e-3 if len(cols) >= 2 else np.ones(len(mid_pool))
    k = min(n_mid, len(mid_pool))
    idx = rng.choice(mid_pool.index, size=k, replace=False, p=w / w.sum()) if k else []
    mid = mid_pool.loc[idx].copy()
    mid["pick"] = "explore-mid"
    how = (f"weighted by sub-score disagreement ({dis_all.loc[i]:.2f})" if len(cols) >= 2
           else "uniform" for i in mid.index)  # fmt: skip
    mid["reason"] = [f"explore: uncertain mid-rank, ranks {lo:g}-{hi:g}% (rank {r}, top "
                     f"{p:.1f}%), {h}" for r, p, h in zip(mid["rank"], mid.rank_pct, how,
                                                          strict=True)]  # fmt: skip
    out = pd.concat([top, t5, mid])
    out["disagreement"] = dis_all.loc[out.index].to_numpy()
    return out.reset_index(drop=True)
