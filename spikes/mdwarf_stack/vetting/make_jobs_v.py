"""Search jobs that the vetting calibration needs beyond gate 2 (frozen gate-2 search, unchanged).

  dev_inj_v    200 extra DEV injections (1.0-2.0 R_e, 15-55 d log-uniform, b 0-0.9) on DEV stars
               with 6+ sectors, top signal, rot = frozen arm. Gate 2 left only 56 matched DEV
               injections, too few to set a per-test threshold.
  test_nulls_v TEST null realisations, long range, 4 per TEST star (inv0, invroll11, roll21,
               roll22: the first seeds of the gate-2 DEV long-null list). Gate 2 ran nulls on DEV
               only; the vetting's false-alarm rejection is measured on these.

Both are run by the frozen gate-2 runner (gate2/run2.py), ONE niced worker. Outputs go to
outputs/planets_spike/vetting/. Usage: python -I vetting/make_jobs_v.py
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
from common import CAT_DIR, LC_DIR, OUT  # noqa: E402
from make_jobs2 import _inj, _mr, _nsec  # noqa: E402

V = OUT / "vetting"
JOBS = V / "jobs"
TEST_NULLS = (("inv", 0), ("invroll", 11), ("roll", 21), ("roll", 22))


def main() -> None:
    JOBS.mkdir(parents=True, exist_ok=True)
    rot = json.loads((OUT / "gate2" / "frozen.json").read_text())["rot"]
    c = pd.read_csv(CAT_DIR / "controls_g2.csv")
    c = c[[(LC_DIR / f"tic{t}.npz").exists() for t in c["tic"]]]
    dev = c[c["split"] == "DEV"]
    info = {int(t): _nsec(int(t)) for t in dev["tic"]}
    pool = [row for _, row in dev.iterrows() if info[int(row["tic"])][0] >= 6]
    rng = np.random.default_rng(20261201)
    order = rng.permutation(len(pool))
    jobs = []
    for i in range(200):
        row = pool[order[i % len(pool)]]
        b = _inj(rng, row, (1.0, 2.0), (15.0, 55.0), info[int(row["tic"])][1], "devinjv")
        jobs.append(dict(b, id=f"devinjv-{i:03d}-{int(row['tic'])}", rot=rot))
    with open(JOBS / "dev_inj_v.jsonl", "w") as fh:
        fh.writelines(json.dumps(j) + "\n" for j in jobs)
    print("dev_inj_v", len(jobs), "on", len(pool), "DEV stars with 6+ sectors")
    test = c[c["split"] == "TEST"]
    nl = []
    for _, row in test.iterrows():
        m, r = _mr(row)
        t = int(row["tic"])
        for mode, seed in TEST_NULLS:
            nl.append({"id": f"tnullL-{mode}{seed}-{t}", "tic": t, "mode": mode, "seed": seed,
                       "rot": rot, "range": "long", "max_sig": 1, "mass": m, "radius": r})
    with open(JOBS / "test_nulls_v.jsonl", "w") as fh:
        fh.writelines(json.dumps(j) + "\n" for j in nl)
    print("test_nulls_v", len(nl))


if __name__ == "__main__":
    main()
