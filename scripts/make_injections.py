"""Build the injection sets ``<data>/injections/<name>/sNNNN/{val,test}/``.

    ODDBALL_DATA=... uv run python -I scripts/make_injections.py --sector 32 --name v1
    # the EB check set (ML_DESIGN §14): EB nulls + dusty-tail transits, hosts disjoint from v1
    uv run python -I scripts/make_injections.py --sector 32 --name eb1 --seed 20261110 \
        --kinds null_eb null_eb dusty_tail --n-astro 2400 --n-events 0 --exclude-sets v1

Hosts come only from the matching split, stratified by Tmag bin x camera/CCD; known positives
are never hosts; the seed is fixed. See ``oddball.inject.build`` for the rules.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from oddball.common.paths import data_root
from oddball.inject.build import build_injection_set
from oddball.inject.sampling import EB_KINDS, KINDS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sector", type=int, default=32)
    ap.add_argument("--name", default="v1")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--n-astro", type=int, default=3000)
    ap.add_argument("--n-events", type=int, default=10)
    ap.add_argument("--hosts-per-event", type=int, default=30)
    ap.add_argument("--n-dump-events", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None, help="override output dir (one split)")
    ap.add_argument("--kinds", nargs="+", default=list(KINDS), choices=[*KINDS, *EB_KINDS],
                    help="astrophysical-host kinds, cycled (repeat a kind to weight it)")
    ap.add_argument("--exclude-sets", nargs="*", default=[],
                    help="injection sets whose hosts are not reused")
    a = ap.parse_args()
    excl = set()
    for name in a.exclude_sets:
        for p in (data_root() / "injections" / name / f"s{a.sector:04d}").glob(
                "*/injections.parquet"):
            excl.update(pd.read_parquet(p, columns=["host_tic"]).host_tic.astype(int).tolist())
    for split in a.splits:
        t0 = time.time()
        print(f"== {a.name} s{a.sector:04d} {split}")
        prov = build_injection_set(
            a.sector, split, name=a.name, seed=a.seed, n_astro=a.n_astro, n_events=a.n_events,
            hosts_per_event=a.hosts_per_event, n_dump_events=a.n_dump_events, out_dir=a.out,
            kinds=tuple(a.kinds), exclude_tics=excl or None,
        )  # fmt: skip
        print(f"   {prov['n_astro']} astrophysical + {prov['n_null']} null "
              f"({prov['n_null_events']} events) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
