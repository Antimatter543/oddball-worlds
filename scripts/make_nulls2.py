"""Build the second null set ``n2`` (asteroid tracks, dump steps, single columns).

    uv run python -I scripts/make_nulls2.py --sector 32

Needs the ``cmv2`` tess-point cache (``outputs/veto/cmv2/sNNNN/_cache/cells.parquet``; built by
``scripts/build_veto.py``). See ``oddball.inject.nulls2``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oddball.common.paths import outputs_root
from oddball.inject.nulls2 import build_null2_set


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sector", type=int, default=32)
    ap.add_argument("--name", default="n2")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--n-tracks", type=int, default=40)
    ap.add_argument("--n-columns", type=int, default=40)
    ap.add_argument("--n-steps", type=int, default=10)
    ap.add_argument("--hosts-per-step", type=int, default=20)
    a = ap.parse_args()
    cells = pd.read_parquet(outputs_root() / "veto" / "cmv2" / f"s{a.sector:04d}" / "_cache"
                            / "cells.parquet")  # fmt: skip
    for split in a.splits:
        t0 = time.time()
        print(f"== {a.name} s{a.sector:04d} {split}")
        prov = build_null2_set(a.sector, split, cells, name=a.name, seed=a.seed,
                               n_tracks=a.n_tracks, n_columns=a.n_columns, n_steps=a.n_steps,
                               hosts_per_step=a.hosts_per_step)  # fmt: skip
        print(f"   {prov['n_evaluated']} evaluated + {prov['n_carrier']} carriers "
              f"{prov['evaluated_by_kind']} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
