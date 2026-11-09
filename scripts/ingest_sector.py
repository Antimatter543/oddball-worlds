"""Ingest one TESS-SPOC FFI sector into binned Parquet shards.

Run with ``-I`` (it parses downloaded files)::

    uv run python -I scripts/ingest_sector.py 39 --limit 2000 --include-known
    uv run python -I scripts/ingest_sector.py 39                # full sector, resumable

Data goes to ``$ODDBALL_DATA/tess-spoc/sNNNN`` (default ``<repo>/data``).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # -I drops cwd/script paths

from oddball.common.paths import REPO_ROOT
from oddball.tess.ingest import ingest_sector
from oddball.tess.lightcurve import DEFAULT_BIN_MINUTES, DEFAULT_QUALITY_BITMASK

KNOWN = REPO_ROOT / "docs" / "known_positives.csv"


def known_tics(path: Path = KNOWN) -> set[int]:
    if not path.exists():
        return set()
    with path.open() as fh:
        return {int(r["tic"]) for r in csv.DictReader(fh) if r.get("tic")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("sector", type=int)
    ap.add_argument("--limit", type=int, default=None, help="only the first N LCs of the script")
    ap.add_argument("--include-known", action="store_true",
                    help=f"also ingest any TIC in {KNOWN.relative_to(REPO_ROOT)} (with --limit)")
    ap.add_argument("--tic", type=int, action="append", default=[], help="extra TIC(s) to include")
    ap.add_argument("--workers", type=int, default=32, help="concurrent downloads (default 32)")
    ap.add_argument("--shard-size", type=int, default=10_000)
    ap.add_argument("--bin-minutes", type=float, default=DEFAULT_BIN_MINUTES)
    ap.add_argument("--bitmask", type=int, default=DEFAULT_QUALITY_BITMASK)
    a = ap.parse_args()

    include = set(a.tic) | (known_tics() if a.include_known else set())
    ingest_sector(
        a.sector,
        limit=a.limit,
        include_tics=include,
        workers=a.workers,
        shard_size=a.shard_size,
        bin_minutes=a.bin_minutes,
        bitmask=a.bitmask,
        log=lambda s: print(s, flush=True),
    )


if __name__ == "__main__":
    main()
