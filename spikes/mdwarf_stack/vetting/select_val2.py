"""Vetting v2: the FRESH validation set (pre-registered in docs/spikes/mdwarf_vetting.md, "v2").

  stars   python -I vetting/select_val2.py stars   -> <OUT>/vetting/v2/val_stars.csv
          Survey targets (gate2/survey_targets.csv) with >= 6 predicted sectors, minus every gate-2
          DEV / TEST control, every hold-out host, the gate-1 hosts, and any star with a TOI (any
          disposition, local toi.csv) or an exoplanet.eu planet within 30" (as for the gate-2
          controls: rolled copies of a star with a real transit make planet-shaped false alarms).
          Stratified by predicted-sector bucket (6-12, 13-25, 26+), proportional to the pool,
          N_STARS in total (largest remainder), each stratum drawn in a seeded random order.
          The CSV lists the whole seeded order per stratum (rank); stars past the quota are the
          replacements, taken in rank order when a drawn star has no TESS product.
  jobs    python -I vetting/select_val2.py jobs    -> <OUT>/vetting/v2/jobs/val2.jsonl
          After the light curves are downloaded (vetting/fetch_val2.py). Per star, in draw order:
          4 long-range null realisations (inv0, invroll11, roll21, roll22: the v1 TEST-null set),
          2 headline injections (1.0-1.5 R_e, 15-30 d) and 1 long injection (1.0-2.0 R_e,
          30-55 d), log-uniform P, b U(0, 0.9), t0 uniform in the first period of the data.
          Frozen gate-2 search (gate2/run2.py), rot from gate2/frozen.json, top signal only.

BLINDING: these are survey stars. No job is ever a real, unaltered search (mode "real" without an
injection); `jobs` refuses to write one and `check_jobs` re-checks the file before every run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "gate2"))
from common import CAT_DIR, LC_DIR, OUT

V2 = OUT / "vetting" / "v2"
JOBS = V2 / "jobs"
SEED_STARS = 20261212
SEED_INJ = 20261213
N_STARS = 150
BUCKETS = (("6-12", 6, 12), ("13-25", 13, 25), ("26+", 26, 10_000))
NULLS = (("inv", 0), ("invroll", 11), ("roll", 21), ("roll", 22))
N_HEADLINE, N_LONG = 2, 1


def _sep(ra1, de1, ra2, de2):
    r = np.radians
    c = np.sin(r(de1)) * np.sin(r(de2)) + np.cos(r(de1)) * np.cos(r(de2)) * np.cos(r(ra1 - ra2))
    return np.degrees(np.arccos(np.clip(c, -1, 1))) * 3600


def pool() -> pd.DataFrame:
    from make_jobs import HOSTS

    t = pd.read_csv(OUT / "gate2" / "survey_targets.csv")
    t = t[t["n_sectors"] >= 6].copy()
    excl = set(pd.read_csv(CAT_DIR / "controls_g2.csv")["tic"].astype(int))
    h = pd.read_csv(CAT_DIR / "holdout_g2.csv")
    excl |= {int(x) for x in h["tic"].dropna()}
    excl |= {int(x) for x in HOSTS}
    toi = pd.read_csv(CAT_DIR / "toi.csv", low_memory=False)
    excl_toi = {int(x) for x in toi["TIC ID"].dropna()}
    eu = pd.read_csv(CAT_DIR / "exoplanet_eu.csv", low_memory=False)
    era, ede = eu["ra"].to_numpy(float), eu["dec"].to_numpy(float)
    ok = np.isfinite(era) & np.isfinite(ede)
    era, ede = era[ok], ede[ok]
    n0 = len(t)
    t = t[~t["tic"].isin(excl)]
    n1 = len(t)
    t = t[~t["tic"].isin(excl_toi)]
    n2 = len(t)
    has_eu = np.array([bool(np.any(_sep(r.ra, r.dec, era, ede) < 30)) for r in t.itertuples()])
    t = t[~has_eu]
    print(f"pool: {n0} survey targets with >= 6 predicted sectors; -{n0 - n1} gate-2/hold-out/"
          f"gate-1 stars; -{n1 - n2} TOI hosts; -{int(has_eu.sum())} exoplanet.eu hosts -> {len(t)}")
    return t


def stars() -> pd.DataFrame:
    t = pool()
    t["bucket"] = ""
    for name, lo, hi in BUCKETS:
        t.loc[(t.n_sectors >= lo) & (t.n_sectors <= hi), "bucket"] = name
    sizes = t["bucket"].value_counts()
    raw = {b: N_STARS * sizes.get(b, 0) / len(t) for b, _, _ in BUCKETS}
    quota = {b: int(np.floor(v)) for b, v in raw.items()}
    for b in sorted(raw, key=lambda b: raw[b] - quota[b], reverse=True)[: N_STARS - sum(quota.values())]:
        quota[b] += 1
    rng = np.random.default_rng(SEED_STARS)
    out = []
    for b, _, _ in BUCKETS:
        s = t[t.bucket == b].sort_values("tic").reset_index(drop=True)
        order = rng.permutation(len(s))
        s = s.iloc[order].reset_index(drop=True)
        s["rank"] = np.arange(len(s))
        s["quota"] = quota[b]
        out.append(s)
    out = pd.concat(out, ignore_index=True)
    V2.mkdir(parents=True, exist_ok=True)
    cols = ["tic", "ra", "dec", "Tmag", "Teff", "rad", "mass", "n_sectors", "bucket", "rank",
            "quota"]
    out[cols].to_csv(V2 / "val_stars.csv", index=False)
    print("quota", quota, "pool sizes", dict(sizes))
    return out


def drawn(have_data=None) -> pd.DataFrame:
    """The set: per stratum, the first `quota` stars in rank order that have data."""
    s = pd.read_csv(V2 / "val_stars.csv")
    out = []
    for b, g in s.groupby("bucket", sort=False):
        g = g.sort_values("rank")
        if have_data is not None:
            g = g[[have_data(int(x)) for x in g.tic]]
        out.append(g.head(int(g["quota"].iloc[0])))
    d = pd.concat(out)
    # draw order across strata: interleave by rank / quota so any prefix stays proportional
    d = d.assign(_k=d["rank"] / d["quota"]).sort_values(["_k", "bucket"]).drop(columns="_k")
    return d.reset_index(drop=True)


def check_jobs(jobs):
    bad = [j["id"] for j in jobs if j.get("mode", "real") == "real" and not j.get("inj")]
    if bad:
        raise SystemExit(f"BLINDING: {len(bad)} real unaltered search jobs, refusing: {bad[:3]}")


def jobs():
    from make_jobs2 import _inj

    rot = json.loads((OUT / "gate2" / "frozen.json").read_text())["rot"]
    status = _fetch_status()
    d = drawn(lambda t: status.get(t) == "ok")
    rng = np.random.default_rng(SEED_INJ)
    out = []
    for i, row in d.iterrows():
        t = int(row["tic"])
        z = np.load(LC_DIR / f"tic{t}.npz", allow_pickle=False)
        idx0 = int(z["idx"].min())
        m = float(row["mass"]) if np.isfinite(row["mass"]) else 0.3
        r = float(row["rad"]) if np.isfinite(row["rad"]) else 0.3
        for mode, seed in NULLS:
            out.append({"id": f"v2null-{mode}{seed}-{t}", "tic": t, "mode": mode, "seed": seed,
                        "rot": rot, "range": "long", "max_sig": 1, "mass": m, "radius": r,
                        "order": i})
        rowd = {"tic": t, "mass": m, "rad": r}
        specs = [((1.0, 1.5), (15.0, 30.0), "head")] * N_HEADLINE + \
                [((1.0, 2.0), (30.0, 55.0), "long")] * N_LONG
        for k, (rr, pr, tag) in enumerate(specs):
            b = _inj(rng, rowd, rr, pr, idx0, f"v2{tag}")
            out.append(dict(b, id=f"v2inj{tag}{k}-{t}", rot=rot, order=i))
    check_jobs(out)
    JOBS.mkdir(parents=True, exist_ok=True)
    with open(JOBS / "val2.jsonl", "w") as fh:
        fh.writelines(json.dumps(j) + "\n" for j in out)
    d.to_csv(V2 / "val_set.csv", index=False)
    print(len(d), "stars", len(out), "jobs", dict(d.bucket.value_counts()))


def _fetch_status() -> dict:
    p = V2 / "fetch_status.jsonl"
    st = {}
    if p.exists():
        for x in open(p):
            if x.strip():
                r = json.loads(x)
                st[int(r["tic"])] = r["status"]
    return st


if __name__ == "__main__":
    {"stars": stars, "jobs": jobs}[sys.argv[1]]()
