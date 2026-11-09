"""Fetch young-star catalogues from VizieR into data/rubin/catalogs/.

    uv run python -I scripts/rubin_fetch_catalogs.py [spicy gaia_yso known_yso] [--force]
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from oddball.rubin import catalog


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=list(catalog.CATALOGS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for name in args.names:
        t = time.time()
        p = catalog.fetch(name, force=args.force)
        df = pd.read_parquet(p)
        print(f"{name}: {len(df)} rows -> {p} ({p.stat().st_size/1e6:.1f} MB, "
              f"{time.time()-t:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
