"""Build the fixed-grid ML cache for a shard directory (real sector or injection set).

    ODDBALL_DATA=... uv run python -I scripts/ml_prepare.py --sector 32
    ODDBALL_DATA=... uv run python -I scripts/ml_prepare.py --shards data/injections/v1/s0032/val
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from oddball.common.paths import sector_dir
from oddball.ml.data import build_cache, cache_dir_for


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sector", type=int)
    g.add_argument("--shards", type=Path)
    a = ap.parse_args()
    src = sector_dir(a.sector) if a.sector is not None else a.shards
    t0 = time.time()
    out = build_cache(src, cache_dir_for(src))
    print(f"cache ready: {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
