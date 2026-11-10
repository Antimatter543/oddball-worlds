"""Batched, NaN-padded 2D views of any shard directory (real sector or injection set).

Every light curve lives on the global 30-min bin grid (see lightcurve.py), so a batch of light
curves maps onto one ``(n, L)`` array whose column ``j`` is global bin ``g0 + j`` for every
row. Gaps and the padding outside each light curve are NaN (``qflags`` 0, ``npts`` 0).

A *shard directory* is anything with ``shards/lc_*.parquet`` (+ ``meta_*.parquet`` or a
consolidated ``metadata.parquet``): ``<data>/tess-spoc/sNNNN`` or
``<data>/injections/<name>/sNNNN/<split>`` (docs/ML_DESIGN.md §9).

Memory: one batch of ``batch_rows`` rows costs ~``batch_rows * L * 4 B`` per channel
(2000 x 1250 x 4 B = 10 MB), so a whole sector is never in memory.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from oddball.common.paths import sector_dir

LIST_COLUMNS = ("flux", "flux_err", "npts", "qflags", "sap_flux", "bkg")


def dir_metadata(d: Path) -> pd.DataFrame:
    d = Path(d)
    if (d / "metadata.parquet").exists():
        return pq.read_table(d / "metadata.parquet").to_pandas()
    files = sorted((d / "shards").glob("meta_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no shard metadata under {d}")
    return pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)


def lc_files(d: Path) -> list[Path]:
    return sorted((Path(d) / "shards").glob("lc_*.parquet"))


@dataclass(frozen=True)
class Grid:
    """Global bin window: column j <-> global bin g0 + j; bin width in days."""

    g0: int
    length: int
    bin_days: float

    @property
    def time(self) -> np.ndarray:
        return (self.g0 + np.arange(self.length) + 0.5) * self.bin_days


def sector_grid(sector: int, pad: int = 2) -> Grid:
    """Grid covering every real light curve of a sector (injections are subsets of it)."""
    m = dir_metadata(sector_dir(sector))
    g0 = int(m.bin0.min()) - pad
    g1 = int((m.bin0 + m.n_bins).max()) + pad
    return Grid(g0, g1 - g0, float(m.bin_minutes.iloc[0]) / 1440.0)


def _pad_list(col: pa.Array, bin0: np.ndarray, grid: Grid, dtype, fill) -> np.ndarray:
    col = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    n = len(col)
    lengths = col.value_lengths().to_numpy(zero_copy_only=False).astype(np.int64)
    values = col.flatten().to_numpy(zero_copy_only=False)
    out = np.full((n, grid.length), fill, dtype=dtype)
    rows = np.repeat(np.arange(n), lengths)
    starts = np.repeat(np.cumsum(lengths) - lengths, lengths)
    cols = (bin0 - grid.g0)[rows] + (np.arange(lengths.sum()) - starts)
    ok = (cols >= 0) & (cols < grid.length)
    if not ok.all():
        raise ValueError("light curve extends outside the grid; rebuild the grid")
    out[rows, cols] = values.astype(dtype, copy=False)
    return out


def iter_batches(
    d: Path,
    grid: Grid,
    batch_rows: int = 2000,
    columns: tuple[str, ...] = LIST_COLUMNS,
    files: list[Path] | None = None,
) -> Iterator[dict]:
    """Yield ``{tic, inj_id (or None), host_tic, bin0, <col>: (n, L) array}`` per batch.

    Float columns are float32 with NaN padding; ``qflags`` int32 / ``npts`` uint16 with 0.
    ``files`` restricts the pass to some of the directory's ``lc_*.parquet`` files (parallel
    workers).
    """
    for f in files if files is not None else lc_files(d):
        pf = pq.ParquetFile(f)
        names = set(pf.schema_arrow.names)
        want = ["tic", "bin0", *columns] + [c for c in ("inj_id", "host_tic") if c in names]
        for b in pf.iter_batches(batch_size=batch_rows, columns=want):
            bin0 = b.column("bin0").to_numpy().astype(np.int64)
            out = {
                "tic": b.column("tic").to_numpy().astype(np.int64),
                "bin0": bin0,
                "inj_id": b.column("inj_id").to_pylist() if "inj_id" in names else None,
                "shard_file": f.name,
            }
            for c in columns:
                if c == "qflags":
                    out[c] = _pad_list(b.column(c), bin0, grid, np.int32, 0)
                elif c == "npts":
                    out[c] = _pad_list(b.column(c), bin0, grid, np.uint16, 0)
                else:
                    out[c] = _pad_list(b.column(c), bin0, grid, np.float32, np.nan)
            yield out
