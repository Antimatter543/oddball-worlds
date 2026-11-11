"""Write the job files for the spike (fixed seeds).

  controls_real.jsonl  identical search on the 30 real control light curves (max 6 signals)
  nulls.jsonl          4 null realisations per control (inv, invroll, roll x2; max 3 signals)
  inj.jsonl            injections into RAW control flux (max 3 signals); --n-per-cell
  hosts.jsonl          the three known hosts (max 6 signals)

Usage: uv run python -I spikes/mdwarf_stack/make_jobs.py [--n-per-cell 200]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BIN_D, CAT_DIR, OUT
from fetch import load

FALLBACK_M, FALLBACK_R = 0.3, 0.3
# TIC v8 stellar parameters as listed by ExoFOP (same source as the controls' TIC values).
HOSTS = {92226327: (0.184589, 0.215535), 150428135: (0.413261, 0.419457),
         52005579: (0.242168, 0.269427)}
CELLS = [((1.0, 1.5), (15.0, 30.0)), ((1.0, 1.5), (30.0, 50.0)),
         ((1.5, 2.0), (15.0, 30.0)), ((1.5, 2.0), (30.0, 50.0))]


def _mr(row):
    m = row["mass"] if np.isfinite(row["mass"]) else FALLBACK_M
    r = row["rad"] if np.isfinite(row["rad"]) else FALLBACK_R
    return float(m), float(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cell", type=int, default=200)
    a = ap.parse_args()
    ctl = pd.read_csv(CAT_DIR / "controls.csv")
    ctl = ctl[[(Path(CAT_DIR).parent / "lc" / f"tic{t}.npz").exists() for t in ctl["tic"]]]
    jobdir = OUT / "jobs"
    jobdir.mkdir(exist_ok=True)

    def dump(name, jobs):
        with open(jobdir / name, "w") as fh:
            fh.writelines(json.dumps(j) + "\n" for j in jobs)
        print(name, len(jobs))

    real, nulls = [], []
    for _, row in ctl.iterrows():
        m, r = _mr(row)
        t = int(row["tic"])
        real.append({"id": f"real-{t}", "tic": t, "mode": "real", "seed": 0, "max_sig": 6,
                     "mass": m, "radius": r})
        for mode, seed in (("inv", 0), ("invroll", 11), ("roll", 21), ("roll", 22)):
            nulls.append({"id": f"{mode}{seed}-{t}", "tic": t, "mode": mode, "seed": seed,
                          "max_sig": 3, "mass": m, "radius": r})
    dump("controls_real.jsonl", real)
    dump("nulls.jsonl", nulls)

    rng = np.random.default_rng(20261112)
    inj = []
    rows = list(ctl.iterrows())
    n = 0
    for i in range(a.n_per_cell):
        for c, ((r_lo, r_hi), (p_lo, p_hi)) in enumerate(CELLS):
            _, row = rows[n % len(rows)]
            n += 1
            m, r = _mr(row)
            t = int(row["tic"])
            lc, _ = load(t)
            P = float(np.exp(rng.uniform(np.log(p_lo), np.log(p_hi))))
            tmin = (lc["idx"].min() + 0.5) * BIN_D
            spec = {"rp": float(rng.uniform(r_lo, r_hi)), "P": P,
                    "t0": float(tmin + rng.uniform(0, P)), "b": float(rng.uniform(0, 0.9)),
                    "cell": c}
            inj.append({"id": f"inj{i:03d}-c{c}-{t}", "tic": t, "mode": "real", "seed": 0,
                        "max_sig": 3, "mass": m, "radius": r, "inj": spec})
    # interleave so that any prefix is balanced across cells and stars
    dump("inj.jsonl", inj)

    hosts = [{"id": f"host-{t}", "tic": t, "mode": "real", "seed": 0, "max_sig": 6,
              "mass": m, "radius": r} for t, (m, r) in HOSTS.items()]
    dump("hosts.jsonl", hosts)


if __name__ == "__main__":
    main()
