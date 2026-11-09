"""Integrity check for an ingested sector directory (see ingest.py for the layout).

Used after pulling a sector down from Kaggle, but works on any sector dir.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

PROV_REQUIRED = (
    "sector", "product", "hlsp_version", "doi", "script_url", "download_dates_utc",
    "bin_minutes", "quality_bitmask", "n_in_script", "n_converted", "n_shards", "bytes_shards",
    "runs",
)  # fmt: skip


def _read_tics(path: Path, deep: bool) -> tuple[list[int], int]:
    """TICs in a shard. ``deep`` decodes every column of every row group (catches corrupt
    pages, not just a readable footer), one row group at a time to keep memory flat."""
    pf = pq.ParquetFile(path)
    tics: list[int] = []
    for i in range(pf.num_row_groups):
        cols = None if deep else ["tic"]
        tics += pf.read_row_group(i, columns=cols).column("tic").to_pylist()
    if len(tics) != pf.metadata.num_rows:
        raise ValueError(f"row groups hold {len(tics)} rows, footer says {pf.metadata.num_rows}")
    return tics, pf.metadata.num_rows


def verify_sector_dir(sdir: Path, sector: int | None = None, deep: bool = True) -> dict:
    """Return ``{"ok", "problems", "n_rows", "n_shards", "bytes_shards", "n_in_script", ...}``.

    Checks: provenance present with the required keys (and the right sector); every
    ``lc_*.parquet`` and ``meta_*.parquet`` opens and decodes with pyarrow; lc and meta shards
    pair up one to one with the same TICs; no TIC appears twice; ``metadata.parquet`` and
    provenance ``n_converted`` agree with the shard row count; ``n_shards``/``bytes_shards``
    agree with the files; no FITS or half-written ``.tmp`` files are present.
    """
    sdir = Path(sdir)
    problems: list[str] = []
    out: dict = {"dir": str(sdir), "problems": problems}

    prov: dict = {}
    pp = sdir / "provenance.json"
    if not pp.exists():
        problems.append("provenance.json missing")
    else:
        try:
            prov = json.loads(pp.read_text())
        except (OSError, ValueError) as e:
            problems.append(f"provenance.json unreadable: {e}")
        missing = [k for k in PROV_REQUIRED if k not in prov]
        if prov and missing:
            problems.append(f"provenance.json missing keys: {missing}")
        if sector is not None and prov.get("sector") not in (None, sector):
            problems.append(f"provenance sector {prov.get('sector')} != {sector}")

    shard_dir = sdir / "shards"
    lc = {int(f.stem.split("_")[1]): f for f in sorted(shard_dir.glob("lc_*.parquet"))}
    meta = {int(f.stem.split("_")[1]): f for f in sorted(shard_dir.glob("meta_*.parquet"))}
    if not lc:
        problems.append("no lc_*.parquet shards")
    if set(lc) != set(meta):
        problems.append(f"lc/meta shard indices differ: lc-only {sorted(set(lc) - set(meta))}, "
                        f"meta-only {sorted(set(meta) - set(lc))}")  # fmt: skip

    tic_count: Counter = Counter()
    n_rows = 0
    for idx, f in lc.items():
        try:
            tics, n = _read_tics(f, deep)
        except Exception as e:  # noqa: BLE001 - any read failure is a verification failure
            problems.append(f"{f.name} unreadable: {type(e).__name__}: {e}")
            continue
        n_rows += n
        tic_count.update(tics)
        if idx in meta:
            try:
                mtics, _ = _read_tics(meta[idx], deep=True)
            except Exception as e:  # noqa: BLE001
                problems.append(f"{meta[idx].name} unreadable: {type(e).__name__}: {e}")
                continue
            if sorted(mtics) != sorted(tics):
                problems.append(f"shard {idx}: lc and meta TICs differ")
    dups = [t for t, c in tic_count.items() if c > 1]
    if dups:
        problems.append(f"{len(dups)} TICs appear in more than one row, e.g. {dups[:5]}")

    mp = sdir / "metadata.parquet"
    if not mp.exists():
        problems.append("metadata.parquet missing")
    else:
        try:
            m_rows = pq.ParquetFile(mp).metadata.num_rows
            m_tics = pq.read_table(mp, columns=["tic"]).column("tic").to_pylist()
        except Exception as e:  # noqa: BLE001
            problems.append(f"metadata.parquet unreadable: {type(e).__name__}: {e}")
        else:
            if m_rows != n_rows:
                problems.append(f"metadata.parquet has {m_rows} rows, shards have {n_rows}")
            elif set(m_tics) != set(tic_count):
                problems.append("metadata.parquet TICs differ from shard TICs")

    shard_bytes = sum(f.stat().st_size for f in lc.values())
    if prov:
        if prov.get("n_converted") != n_rows:
            problems.append(f"provenance n_converted {prov.get('n_converted')} != {n_rows} rows")
        if prov.get("n_shards") != len(lc):
            problems.append(f"provenance n_shards {prov.get('n_shards')} != {len(lc)} files")
        if prov.get("bytes_shards") != shard_bytes:
            problems.append(f"provenance bytes_shards {prov.get('bytes_shards')} != {shard_bytes}")

    stray = [p.name for p in sdir.rglob("*") if p.suffix.lower() in (".fits", ".tmp", ".part")
             or p.name.endswith(".fits.gz")]  # fmt: skip
    if stray:
        problems.append(f"stray files: {stray[:5]}")

    out.update(
        ok=not problems,
        n_rows=n_rows,
        n_shards=len(lc),
        bytes_shards=shard_bytes,
        n_in_script=prov.get("n_in_script"),
        n_failures=_count_failures(sdir / "failures.csv"),
    )
    return out


def _count_failures(path: Path) -> dict:
    if not path.exists():
        return {}
    import csv

    with path.open() as fh:
        return dict(Counter(r["kind"] for r in csv.DictReader(fh)))
