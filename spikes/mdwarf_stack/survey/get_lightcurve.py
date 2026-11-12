"""On-demand light curve for vetting: the survey's exact NPZ for one star, from a capped cache.

The survey deletes each star's NPZ after searching it (disk budget). Vetting calls
`get_lightcurve(tic)`, which rebuilds the identical NPZ (same product rule, same bytes: the
public AWS bucket with MAST fallback, same conversion as gate 2, see download.py) into
DATA/lc_cache/, and evicts the least recently used files once the cache exceeds `max_bytes`.

    from get_lightcurve import get_lightcurve
    lc, meta = get_lightcurve(141769356)          # dict of arrays, provenance dict

CLI: python -I survey/get_lightcurve.py TIC [TIC ...] [--max-mb 300]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from common import DATA  # noqa: E402

CACHE = DATA / "lc_cache"
MAX_BYTES = 300_000_000  # about 600 stars at the survey's ~0.5 MB median
_lock = threading.Lock()
_dl = None


def _downloader():
    global _dl
    if _dl is None:
        from download import Downloader

        log = CACHE / "errors.jsonl"

        def errlog(r):
            with open(log, "a") as fh:
                fh.write(json.dumps(r, default=str) + "\n")

        _dl = Downloader(CACHE, errlog)
    return _dl


def _evict(max_bytes: int, keep: Path):
    files = sorted((p for p in (CACHE / "lc").glob("tic*.npz")), key=lambda p: p.stat().st_atime)
    total = sum(p.stat().st_size for p in files)
    for p in files:
        if total <= max_bytes:
            break
        if p == keep:
            continue
        total -= p.stat().st_size
        p.unlink(missing_ok=True)


def get_lightcurve(tic: int, max_bytes: int = MAX_BYTES, refresh: bool = False):
    """(lc, meta) for one star, as the survey searched it. Raises if MAST lists no product."""
    from searchjob import load_npz

    tic = int(tic)
    with _lock:
        dl = _downloader()
        p = dl.npz_path(tic)
        if refresh or not p.exists():
            prods, _ = dl.products(tic)
            if not prods:
                raise LookupError(f"TIC {tic}: no SPOC 2-min or TESS-SPOC light curve at MAST")
            recs, metas, fails = dl.fetch_star(tic, prods)
            if not recs:
                raise OSError(f"TIC {tic}: every sector failed: {fails}")
            p = dl.write_npz(tic, recs, metas, fails)
        os.utime(p)  # LRU by access time, set explicitly (noatime mounts)
        _evict(max_bytes, keep=p)
        return load_npz(p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tics", nargs="+", type=int)
    ap.add_argument("--max-mb", type=float, default=MAX_BYTES / 1e6)
    a = ap.parse_args()
    for t in a.tics:
        lc, meta = get_lightcurve(t, int(a.max_mb * 1e6))
        srcs = [s.get("source") for s in meta["sectors"]]
        print(t, len(meta["sectors"]), "sectors", len(lc["idx"]), "bins", "failures",
              len(meta["failures"]), "from", {s: srcs.count(s) for s in set(srcs)})
