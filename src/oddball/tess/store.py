"""Readers for the ingested Parquet shards (see ingest.py for the layout)."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from oddball.common.paths import sector_dir


def metadata(sector: int) -> pd.DataFrame:
    sdir = sector_dir(sector)
    path = sdir / "metadata.parquet"
    if path.exists():
        return pq.read_table(path).to_pandas()
    files = sorted((sdir / "shards").glob("meta_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no ingested data for sector {sector} under {sdir}")
    return pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)


def _lc_files(sector: int) -> list[Path]:
    return sorted((sector_dir(sector) / "shards").glob("lc_*.parquet"))


def _row_to_dict(row: dict, bin_minutes: float) -> dict:
    n = len(row["flux"])
    out = {k: np.asarray(row[k]) for k in ("flux", "flux_err", "npts", "qflags", "sap_flux", "bkg")}
    out["time"] = (row["bin0"] + np.arange(n) + 0.5) * (bin_minutes / 1440.0)
    out.update(tic=int(row["tic"]), sector=int(row["sector"]), bin0=int(row["bin0"]))
    return out


def load_lightcurves(sector: int, tics: Iterable[int], bin_minutes: float | None = None) -> dict:
    """Return ``{tic: {time, flux, flux_err, npts, qflags, sap_flux, bkg, ...}}``.

    Uses the metadata shard index so only the shards holding the requested TICs are read.
    """
    tics = [int(t) for t in tics]
    meta = metadata(sector)
    sel = meta[meta.tic.isin(tics)]
    if bin_minutes is None:
        bin_minutes = float(meta.bin_minutes.iloc[0])
    out: dict[int, dict] = {}
    sdir = sector_dir(sector) / "shards"
    for shard, grp in sel.groupby("shard"):
        tbl = ds.dataset(sdir / f"lc_{int(shard):05d}.parquet").to_table(
            filter=ds.field("tic").isin(grp.tic.tolist())
        )
        for row in tbl.to_pylist():
            out[int(row["tic"])] = _row_to_dict(row, bin_minutes)
    return out


def iter_lightcurves(sector: int, batch_rows: int = 1000) -> Iterator[dict]:
    """Stream every light curve of a sector, one row-group batch at a time (bounded RAM)."""
    bin_minutes = float(metadata(sector).bin_minutes.iloc[0])
    for f in _lc_files(sector):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=batch_rows):
            for row in batch.to_pylist():
                yield _row_to_dict(row, bin_minutes)
