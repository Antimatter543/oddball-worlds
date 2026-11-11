"""Gate 2 job files (fixed seeds). Stages:

  dev     DEV only: paired injections (rot none vs harm, long search, top signal), real DEV
          controls under both arms (long, 3 signals), then the null realisations under ROT
          (long: 7 per star, short: 5 per star; top signal).
  test    TEST real controls (long + short), TEST injections (long, top signal, rot ROT) and the
          paired rot=none arm for every TEST injection (criterion d).
  hosts   the 3 gate-1 hosts and the hold-out hosts (long + short, 6 signals).

Usage: python -I gate2/make_jobs2.py dev|test|hosts [--rot harm]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import BIN_D, CAT_DIR, LC_DIR, OUT
from make_jobs import FALLBACK_M, FALLBACK_R, HOSTS

G2 = OUT / "gate2"
JOBS = G2 / "jobs"
LONG_NULLS = (("inv", 0), ("invroll", 11), ("invroll", 12), ("roll", 21), ("roll", 22),
              ("roll", 23), ("roll", 24))
SHORT_NULLS = (("inv", 0), ("invroll", 11), ("roll", 21), ("roll", 22), ("roll", 23))
# TEST injection cells: (R range, P range, count). Headline cell first, 6+ sector stars only.
TEST_CELLS = [((1.0, 1.5), (15.0, 30.0), 200, 6), ((1.0, 1.5), (30.0, 50.0), 80, 6),
              ((1.5, 2.0), (15.0, 30.0), 60, 6), ((1.5, 2.0), (30.0, 50.0), 60, 6),
              ((1.0, 1.5), (15.0, 30.0), 36, 1)]  # last: headline cell on 2-5 sector stars


def _mr(row):
    m = row["mass"] if np.isfinite(row["mass"]) else FALLBACK_M
    r = row["rad"] if np.isfinite(row["rad"]) else FALLBACK_R
    return float(m), float(r)


def _ctl(split):
    c = pd.read_csv(CAT_DIR / "controls_g2.csv")
    c = c[c["split"] == split]
    have = [(LC_DIR / f"tic{t}.npz").exists() for t in c["tic"]]
    if not all(have):
        print(f"WARNING {split}: {len(c) - sum(have)} controls have no light curve yet")
    return c[have]


def _nsec(t):
    z = np.load(LC_DIR / f"tic{t}.npz", allow_pickle=False)
    return len(json.loads(str(z["meta"]))["sectors"]), int(z["idx"].min())


def dump(name, jobs):
    JOBS.mkdir(parents=True, exist_ok=True)
    with open(JOBS / name, "w") as fh:
        fh.writelines(json.dumps(j) + "\n" for j in jobs)
    print(name, len(jobs))


def _inj(rng, row, rr, pr, idx0, tag):
    m, r = _mr(row)
    P = float(np.exp(rng.uniform(np.log(pr[0]), np.log(pr[1]))))
    spec = {"rp": float(rng.uniform(*rr)), "P": P,
            "t0": float((idx0 + 0.5) * BIN_D + rng.uniform(0, P)), "b": float(rng.uniform(0, 0.9))}
    return {"tic": int(row["tic"]), "mode": "real", "seed": 0, "range": "long", "max_sig": 1,
            "mass": m, "radius": r, "inj": spec, "tag": tag}


def dev(rot):
    c = _ctl("DEV")
    rng = np.random.default_rng(20261123)
    jobs = []
    for k in range(2):  # 2 paired injections per DEV star, 1.0-2.0 R_e, 15-50 d
        for _, row in c.iterrows():
            _, i0 = _nsec(int(row["tic"]))
            base = _inj(rng, row, (1.0, 2.0), (15.0, 50.0), i0, "devpair")
            for arm in ("none", "harm"):
                jobs.append(dict(base, id=f"devinj{k}-{int(row['tic'])}-{arm}", rot=arm))
    for _, row in c.iterrows():
        m, r = _mr(row)
        t = int(row["tic"])
        for arm in ("none", "harm"):
            jobs.append({"id": f"devreal-{t}-{arm}", "tic": t, "mode": "real", "seed": 0,
                         "rot": arm, "range": "long", "max_sig": 3, "mass": m, "radius": r})
    dump("dev_paired.jsonl", jobs)
    nl, ns = [], []
    for _, row in c.iterrows():
        m, r = _mr(row)
        t = int(row["tic"])
        for mode, seed in LONG_NULLS:
            nl.append({"id": f"nullL-{mode}{seed}-{t}", "tic": t, "mode": mode, "seed": seed,
                       "rot": rot, "range": "long", "max_sig": 1, "mass": m, "radius": r})
        for mode, seed in SHORT_NULLS:
            ns.append({"id": f"nullS-{mode}{seed}-{t}", "tic": t, "mode": mode, "seed": seed,
                       "rot": rot, "range": "short", "max_sig": 1, "mass": m, "radius": r})
        ns.append({"id": f"devrealS-{t}", "tic": t, "mode": "real", "seed": 0, "rot": rot,
                   "range": "short", "max_sig": 3, "mass": m, "radius": r})
    dump(f"dev_nulls_long_{rot}.jsonl", nl)
    dump(f"dev_nulls_short_{rot}.jsonl", ns)


def test(rot):
    c = _ctl("TEST")
    jobs = []
    for _, row in c.iterrows():
        m, r = _mr(row)
        t = int(row["tic"])
        for rg in ("long", "short"):
            jobs.append({"id": f"testreal-{rg}-{t}", "tic": t, "mode": "real", "seed": 0,
                         "rot": rot, "range": rg, "max_sig": 3, "mass": m, "radius": r})
    dump("test_real.jsonl", jobs)
    rng = np.random.default_rng(20261124)
    info = {int(t): _nsec(int(t)) for t in c["tic"]}
    inj, paired = [], []
    for ci, (rr, pr, n, min_sec) in enumerate(TEST_CELLS):
        pool = [row for _, row in c.iterrows()
                if (info[int(row["tic"])][0] >= 6) == (min_sec >= 6)]
        order = rng.permutation(len(pool))
        for i in range(n):
            row = pool[order[i % len(pool)]]
            base = _inj(rng, row, rr, pr, info[int(row["tic"])][1], f"cell{ci}")
            base["inj"]["cell"] = ci
            jid = f"testinj{ci}-{i:03d}-{int(row['tic'])}"
            inj.append(dict(base, id=jid, rot=rot))
            paired.append(dict(base, id=jid + "-none", rot="none"))
    # interleave cells so that any prefix is balanced
    inj.sort(key=lambda j: (int(j["id"].split("-")[1]) / {0: 200, 1: 80, 2: 60, 3: 60, 4: 36}[
        j["inj"]["cell"]]))
    dump("test_inj.jsonl", inj)
    dump("test_inj_none.jsonl", paired)


def hosts(rot):
    h = pd.read_csv(CAT_DIR / "holdout_g2.csv")
    h = h[h["in_range"] & h["tic"].notna()]
    jobs = []
    seen = set()
    stars = [(t, m, r, "g1host") for t, (m, r) in HOSTS.items()]
    for _, row in h.iterrows():
        t = int(row["tic"])
        if t in seen:
            continue
        seen.add(t)
        m = row["mstar"] if np.isfinite(row["mstar"]) else FALLBACK_M
        r = row["rstar"] if np.isfinite(row["rstar"]) else FALLBACK_R
        stars.append((t, float(m), float(r), "holdout"))
    for t, m, r, kind in stars:
        if not (LC_DIR / f"tic{t}.npz").exists():
            continue
        for rg in ("long", "short"):
            jobs.append({"id": f"{kind}-{rg}-{t}", "tic": t, "mode": "real", "seed": 0, "rot": rot,
                         "range": rg, "max_sig": 6, "mass": m, "radius": r})
    dump("hosts.jsonl", jobs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--rot", default="harm")
    a = ap.parse_args()
    {"dev": dev, "test": test, "hosts": hosts}[a.stage](a.rot)
