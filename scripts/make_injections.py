"""Build the injection sets ``<data>/injections/<name>/sNNNN/{val,test}/``.

    ODDBALL_DATA=... uv run python -I scripts/make_injections.py --sector 32 --name v1

Hosts come only from the matching split, stratified by Tmag bin x camera/CCD; known positives
are never hosts; the seed is fixed. See ``oddball.inject.build`` for the rules.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oddball.inject.build import build_injection_set


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
    a = ap.parse_args()
    for split in a.splits:
        t0 = time.time()
        print(f"== {a.name} s{a.sector:04d} {split}")
        prov = build_injection_set(
            a.sector, split, name=a.name, seed=a.seed, n_astro=a.n_astro, n_events=a.n_events,
            hosts_per_event=a.hosts_per_event, n_dump_events=a.n_dump_events, out_dir=a.out,
        )  # fmt: skip
        print(f"   {prov['n_astro']} astrophysical + {prov['n_null']} null "
              f"({prov['n_null_events']} events) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
