"""Gate 3 vetting measurement: frozen v2 statistics (vetting/run_vet2.run, unchanged) on far-star
search rows.

Tasks (same definitions as run_vet2's val and survey stages):
  g3_true      injected copies whose top signal matches the injection with MES >= tau_long
  g3_fa_null   null realisations whose top signal has MES >= 7
  g3_real      real long-search signals >= tau_long (every slot)
Grading happens later (gate3/analyze3.py, vet2.grade2 with frozen_vet2.json).
Usage: python -I gate3/vet3.py ROWS.jsonl JOBS.jsonl OUT.jsonl I/N
"""

from __future__ import annotations

import os

for _v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "gate2"))
sys.path.insert(0, str(HERE.parent / "vetting"))
sys.path.insert(0, str(HERE.parents[2] / "src"))

NULL_MES = 7.0


def tasks(rows_file, jobs_file, tau):
    jobs = {}
    for x in open(jobs_file):
        if x.strip():
            j = json.loads(x)
            jobs[j["id"]] = j
    out = []
    for x in open(rows_file):
        if not x.strip():
            continue
        r = json.loads(x)
        if "error" in r or not r.get("signals") or r.get("range") != "long":
            continue
        job = jobs[r["id"]]
        if r.get("inj"):
            if r["inj_match"][0][0] and r["signals"][0]["mes"] >= tau:
                out.append(("g3_true", r, job, [0]))
        elif r["mode"] != "real":
            if r["signals"][0]["mes"] >= NULL_MES:
                out.append(("g3_fa_null", r, job, [0]))
        else:
            ks = [i for i, s in enumerate(r["signals"]) if s["mes"] >= tau]
            if ks:
                out.append(("g3_real", r, job, ks))
    return out


def main():
    from run_vet import tau_long
    from run_vet2 import run

    rows_file, jobs_file, out, sh = sys.argv[1:5]
    shard = tuple(int(x) for x in sh.split("/"))
    t = tasks(rows_file, jobs_file, tau_long())
    run("gate3", t, Path(out), extra_only=False, shard=shard)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
