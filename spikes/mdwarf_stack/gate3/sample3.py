"""Gate 3 sample (pre-registered, docs/spikes/mdwarf_100pc.md section 2).

Pool = targets_100pc.csv minus TOI hosts (any disposition, local toi.csv) and exoplanet.eu hosts
within 30". 300 stars, strata = predicted-sector bucket (6-12, 13-25, 26+) x Tmag (< 13, 13-14,
14-15), proportional (largest remainder), seeded order per stratum (20261116), draw order
interleaved by rank / quota. Stars without data are kept (no replacement).
Also assigns shards (draw position mod 5) to the sample and to the 150 v2-validation 50 pc
reference stars. Writes <OUT>/gate3/sample.csv, ref.csv. Usage: python -I gate3/sample3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import CAT_DIR, OUT

G3 = OUT / "gate3"
SEED = 20261116
N = 300
NSHARD = 5
SB = (("6-12", 6, 12), ("13-25", 13, 25), ("26+", 26, 10_000))
TB = (("<13", -5, 13), ("13-14", 13, 14), ("14-15", 14, 15))


def _sep(ra1, de1, ra2, de2):
    r = np.radians
    c = np.sin(r(de1)) * np.sin(r(de2)) + np.cos(r(de1)) * np.cos(r(de2)) * np.cos(r(ra1 - ra2))
    return np.degrees(np.arccos(np.clip(c, -1, 1))) * 3600


def strata(t: pd.DataFrame) -> pd.Series:
    sb = pd.Series("", index=t.index)
    tb = pd.Series("", index=t.index)
    for name, lo, hi in SB:
        sb[(t.n_sectors >= lo) & (t.n_sectors <= hi)] = name
    for name, lo, hi in TB:
        tb[(t.Tmag >= lo) & (t.Tmag < hi)] = name
    return sb + "|" + tb


def pool(t: pd.DataFrame) -> pd.DataFrame:
    toi = pd.read_csv(CAT_DIR / "toi.csv", low_memory=False)
    excl = {int(x) for x in toi["TIC ID"].dropna()}
    eu = pd.read_csv(CAT_DIR / "exoplanet_eu.csv", low_memory=False)
    era, ede = eu["ra"].to_numpy(float), eu["dec"].to_numpy(float)
    ok = np.isfinite(era) & np.isfinite(ede)
    era, ede = era[ok], ede[ok]
    n0 = len(t)
    t = t[~t["tic"].isin(excl)]
    n1 = len(t)
    has = np.array([bool(np.any(_sep(r.ra, r.dec, era, ede) < 30)) for r in t.itertuples()])
    t = t[~has]
    print(f"pool: {n0} targets; -{n0 - n1} TOI hosts; -{int(has.sum())} exoplanet.eu hosts "
          f"-> {len(t)}")
    return t.copy()


def main():
    t = pd.read_csv(G3 / "targets_100pc.csv")
    t["stratum"] = strata(t)
    p = pool(t)
    sizes = p["stratum"].value_counts()
    raw = {s: N * sizes[s] / len(p) for s in sizes.index}
    quota = {s: int(np.floor(v)) for s, v in raw.items()}
    for s in sorted(raw, key=lambda s: raw[s] - quota[s], reverse=True)[: N - sum(quota.values())]:
        quota[s] += 1
    rng = np.random.default_rng(SEED)
    out = []
    for s in sorted(sizes.index):
        g = p[p.stratum == s].sort_values("tic").reset_index(drop=True)
        g = g.iloc[rng.permutation(len(g))].reset_index(drop=True)
        g["rank"] = np.arange(len(g))
        g["quota"] = quota[s]
        out.append(g.head(quota[s]))
    d = pd.concat(out, ignore_index=True)
    d = d.assign(_k=(d["rank"] + 0.5) / d["quota"]).sort_values(["_k", "stratum"]) \
        .drop(columns="_k").reset_index(drop=True)
    d["order"] = np.arange(len(d))
    d["shard"] = d["order"] % NSHARD
    d.to_csv(G3 / "sample.csv", index=False)
    full = t["stratum"].value_counts()
    tab = pd.DataFrame({"full_list": full, "pool": sizes, "quota": pd.Series(quota)}).fillna(0)
    print(tab.astype(int).sort_index().to_string())
    from scipy.stats import ks_2samp

    print("KS Tmag p =", ks_2samp(d.Tmag, t.Tmag).pvalue, "; KS n_sectors p =",
          ks_2samp(d.n_sectors, t.n_sectors).pvalue)
    print("medians sample / full: Tmag", d.Tmag.median(), t.Tmag.median(), "; n_sectors",
          d.n_sectors.median(), t.n_sectors.median())
    ref = pd.read_csv(OUT / "vetting" / "v2" / "val_set.csv").reset_index(drop=True)
    ref["order"] = np.arange(len(ref))
    ref["shard"] = ref["order"] % NSHARD
    ref.to_csv(G3 / "ref.csv", index=False)
    print("sample", len(d), "ref", len(ref))


if __name__ == "__main__":
    main()
