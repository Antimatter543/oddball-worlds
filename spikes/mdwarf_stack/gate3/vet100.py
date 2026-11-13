"""Vetting v2 (frozen, vetting/run_vet2.py survey stage unchanged) for the 50-100 pc extension,
with the catalogue info read from --info instead of the shared <OUT>/vetting/star_info.json.

Usage (VET_LC_DIR as for run_vet2):
  python -I gate3/vet100.py --info STAR_INFO.json --rows ROWS.jsonl --jobs JOBS.jsonl \
      --out OUT.jsonl [--shard 0/2]
"""

from __future__ import annotations

import os

for _v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "gate2"))
sys.path.insert(0, str(HERE.parent / "vetting"))
sys.path.insert(0, str(HERE.parents[2] / "src"))


def main():
    import catalog
    from run_vet import tasks_survey
    from run_vet2 import run

    ap = argparse.ArgumentParser()
    ap.add_argument("--info", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    catalog.INFO = Path(a.info)
    run("survey100", tasks_survey(a.rows, a.jobs), Path(a.out),
        shard=tuple(int(x) for x in a.shard.split("/")))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
