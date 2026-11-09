"""Data module: shard directory -> fixed-grid, robust-standardised fp16 memmap cache -> batches.

One example is one star in one sector on a fixed 30-min grid of ``GRID_LEN`` bins (ML_DESIGN §1).
Gaps stay NaN in the cache and become a validity mask at load time; nothing is interpolated.

Cache layout (``<cache_dir>/``)::

    x.npy        float16 [N, 3, GRID_LEN]: pdc, sap (shape views), err (all NaN where unobserved)
    index.parquet one row per example: row, tic, sector, inj_id, host_tic, camera, ccd, tmag,
                 split, log_scatter, log_p2p, n_obs
    done.json    source files + origin bin, written last (cache is valid only if present)

Shape view: ``y = A * asinh(((f - median) / s) / A)`` with ``s`` = 1.4826 * MAD of the star's
flux. Linear for |dev| < A robust sigmas, logarithmic beyond, so a 30% dip on a quiet star does not
reach fp16/NLL-blowing magnitudes but stays ordered. Errors are propagated through the same
transform (``err_y = err / s / sqrt(1 + (dev/A)^2)``). Amplitude lives in the scalar context.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from oddball.common.paths import data_root
from oddball.common.splits import split_array

GRID_LEN = 1280  # 80 patches of 16 bins = 26.7 d at 30 min; a TESS sector is ~1240 bins
PATCH = 16
ASINH_A = 10.0
ERR_FLOOR = 1e-3  # in robust-sigma units
CTX_NAMES = ("log_scatter", "log_p2p", "tmag")
TMAG_FILL = 12.0  # only used if a host has no Tmag


# --------------------------------------------------------------------------------------------
# Shard discovery and reading


def find_lc_files(shard_dir: Path) -> list[Path]:
    """Light-curve shard files under ``shard_dir`` (ingest ``shards/lc_*.parquet`` layout, or any
    Parquet file carrying a ``flux`` list column, e.g. injection sets)."""
    shard_dir = Path(shard_dir)
    files = sorted(shard_dir.glob("shards/lc_*.parquet")) or sorted(shard_dir.glob("lc_*.parquet"))
    if files:
        return files
    out = []
    for f in sorted(shard_dir.rglob("*.parquet")):
        try:
            names = pq.read_schema(f).names
        except Exception:  # noqa: BLE001 - not a readable parquet: skip
            continue
        if "flux" in names and "bin0" in names:
            out.append(f)
    return out


def _list_to_numpy(col) -> tuple[np.ndarray, np.ndarray]:
    """pyarrow ListArray (chunked or not) -> (offsets int64 [n+1], float values)."""
    arr = col.combine_chunks() if hasattr(col, "combine_chunks") else col
    offsets = arr.offsets.to_numpy().astype(np.int64)
    values = arr.values.to_numpy(zero_copy_only=False).astype(np.float64)
    base = offsets[0]
    return offsets - base, values[base : offsets[-1]]


def rows_to_grid(batch, origin: int, length: int = GRID_LEN) -> dict[str, np.ndarray]:
    """Place a pyarrow RecordBatch of light curves on the fixed grid (NaN = no data)."""
    n = batch.num_rows
    bin0 = batch.column("bin0").to_numpy().astype(np.int64)
    out = {}
    for key in ("flux", "flux_err", "sap_flux"):
        offs, vals = _list_to_numpy(batch.column(key))
        g = np.full((n, length), np.nan, dtype=np.float64)
        for i in range(n):
            seg = vals[offs[i] : offs[i + 1]]
            start = int(bin0[i] - origin)
            lo, hi = max(0, start), min(length, start + len(seg))
            if hi > lo:
                g[i, lo:hi] = seg[lo - start : hi - start]
        out[key] = g
    return out


def _robust_center_scale(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(f, axis=1)
            mad = np.nanmedian(np.abs(f - med[:, None]), axis=1)
    s = 1.4826 * mad
    s = np.where(np.isfinite(s) & (s > 1e-7), s, 1e-7)
    return med, s


def compress(dev: np.ndarray) -> np.ndarray:
    return ASINH_A * np.arcsinh(dev / ASINH_A)


def standardise(flux: np.ndarray, err: np.ndarray, sap: np.ndarray) -> dict[str, np.ndarray]:
    """Robust-standardise grids [B, L] (NaN = gap). Returns shape views, errors and context."""
    obs = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    flux = np.where(obs, flux, np.nan)
    med, s = _robust_center_scale(flux)
    dev = (flux - med[:, None]) / s[:, None]
    pdc = compress(dev)
    with np.errstate(all="ignore"):
        e = err / s[:, None] / np.sqrt(1.0 + (dev / ASINH_A) ** 2)
    e = np.where(obs, np.maximum(e, ERR_FLOOR), np.nan)

    sap_obs = obs & np.isfinite(sap)
    sapf = np.where(sap_obs, sap, np.nan)
    smed, ss = _robust_center_scale(sapf)
    sdev = (sapf - smed[:, None]) / ss[:, None]
    sapv = compress(sdev)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p_hi = np.nanpercentile(flux, 99.5, axis=1)
        p_lo = np.nanpercentile(flux, 0.5, axis=1)
    p2p = np.where(np.isfinite(p_hi - p_lo), np.maximum(p_hi - p_lo, 1e-7), 1e-7)
    return {
        "pdc": pdc,
        "sap": sapv,
        "err": e,
        "log_scatter": np.log10(s),
        "log_p2p": np.log10(p2p),
        "n_obs": obs.sum(1),
    }


# --------------------------------------------------------------------------------------------
# Cache


def cache_root() -> Path:
    return data_root() / "ml_cache"


def cache_dir_for(shard_dir: Path) -> Path:
    """Deterministic cache location for a shard directory (path relative to the data root when
    possible, e.g. ``ml_cache/tess-spoc__s0032`` or ``ml_cache/injections__v1__s0032__val``)."""
    p = Path(shard_dir).expanduser().resolve()
    try:
        rel = p.relative_to(data_root().resolve())
        tag = "__".join(rel.parts)
    except ValueError:
        import hashlib

        tag = "ext_" + p.name + "_" + hashlib.blake2b(str(p).encode(), digest_size=4).hexdigest()
    return cache_root() / tag


def _real_metadata_lookup(sectors) -> pd.DataFrame | None:
    """camera/ccd/tmag per TIC from the ingested real metadata (used for injected hosts)."""
    from oddball.tess.store import metadata

    frames = []
    for s in sectors:
        try:
            m = metadata(int(s))[["tic", "sector", "camera", "ccd", "tmag"]]
            frames.append(m)
        except FileNotFoundError:
            continue
    return pd.concat(frames, ignore_index=True) if frames else None


def sector_origin(sector: int) -> int | None:
    """Grid origin (first bin index) for a sector, from the real ingest metadata."""
    from oddball.tess.store import metadata

    try:
        return int(metadata(int(sector)).bin0.min())
    except FileNotFoundError:
        return None


def build_cache(
    shard_dir: Path, cache_dir: Path, origin: int | None = None, batch_rows: int = 1000, log=print
) -> Path:
    """Convert every light curve under ``shard_dir`` to the fixed-grid cache. Idempotent."""
    shard_dir, cache_dir = Path(shard_dir), Path(cache_dir)
    files = find_lc_files(shard_dir)
    if not files:
        raise FileNotFoundError(f"no light-curve shards under {shard_dir}")
    sig = [[str(f.resolve()), f.stat().st_size, int(f.stat().st_mtime)] for f in files]
    done = cache_dir / "done.json"
    if done.exists() and json.loads(done.read_text()).get("files") == sig:
        return cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # pass 1: row counts, sectors, origin (bin0 only: cheap)
    n_total, sectors, min_bin0 = 0, set(), None
    for f in files:
        t = pq.read_table(f, columns=["sector", "bin0"])
        n_total += t.num_rows
        sectors.update(int(x) for x in np.unique(t.column("sector").to_numpy()))
        b = int(t.column("bin0").to_numpy().min())
        min_bin0 = b if min_bin0 is None else min(min_bin0, b)
    if origin is None:
        origins = {sector_origin(s) for s in sectors}
        origin = origins.pop() if len(origins) == 1 and None not in origins else min_bin0
    log(f"cache {cache_dir}: {n_total} rows from {len(files)} files, sectors {sorted(sectors)}, "
        f"origin bin {origin}")

    x = np.lib.format.open_memmap(
        cache_dir / "x.npy.tmp", mode="w+", dtype=np.float16, shape=(n_total, 3, GRID_LEN)
    )
    meta_lookup = _real_metadata_lookup(sectors)
    recs = []
    row = 0
    for f in files:
        pf = pq.ParquetFile(f)
        names = pf.schema_arrow.names
        cols = [c for c in ("tic", "sector", "bin0", "flux", "flux_err", "sap_flux",
                            "inj_id", "host_tic") if c in names]  # fmt: skip
        for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
            g = rows_to_grid(batch, origin)
            st = standardise(g["flux"], g["flux_err"], g["sap_flux"])
            b = batch.num_rows
            x[row : row + b, 0] = st["pdc"].astype(np.float16)
            x[row : row + b, 1] = st["sap"].astype(np.float16)
            x[row : row + b, 2] = np.minimum(st["err"], 6e4).astype(np.float16)
            tic = batch.column("tic").to_numpy().astype(np.int64)
            host = (batch.column("host_tic").to_numpy().astype(np.int64)
                    if "host_tic" in cols else tic)  # fmt: skip
            inj = (batch.column("inj_id").to_pylist() if "inj_id" in cols else [None] * b)
            recs.append(pd.DataFrame({
                "row": np.arange(row, row + b), "tic": tic,
                "sector": batch.column("sector").to_numpy().astype(np.int64),
                "inj_id": pd.Series(inj, dtype=object), "host_tic": host,
                "log_scatter": st["log_scatter"], "log_p2p": st["log_p2p"], "n_obs": st["n_obs"],
            }))  # fmt: skip
            row += b
        log(f"  {f.name}: {row}/{n_total}")
    x.flush()
    del x
    idx = pd.concat(recs, ignore_index=True)
    if meta_lookup is not None:
        meta_lookup = meta_lookup.drop_duplicates(["tic", "sector"])
        idx = idx.merge(meta_lookup.rename(columns={"tic": "host_tic"}),
                        on=["host_tic", "sector"], how="left")  # fmt: skip
    for c, fill in (("camera", 0), ("ccd", 0), ("tmag", TMAG_FILL)):
        if c not in idx:
            idx[c] = fill
        idx[c] = idx[c].fillna(fill)
    idx["camera"] = idx["camera"].astype(int)
    idx["ccd"] = idx["ccd"].astype(int)
    idx["split"] = split_array(idx["host_tic"].values)
    idx = idx.sort_values("row").reset_index(drop=True)
    idx.to_parquet(cache_dir / "index.parquet")
    (cache_dir / "x.npy.tmp").rename(cache_dir / "x.npy")
    done.write_text(json.dumps({"files": sig, "origin": origin, "n": n_total,
                                "sectors": sorted(sectors)}))  # fmt: skip
    return cache_dir


class Cache:
    """Read-only view of a built cache (memmap: the OS page cache holds it, not the process)."""

    def __init__(self, cache_dir: Path):
        self.dir = Path(cache_dir)
        self.x = np.load(self.dir / "x.npy", mmap_mode="r")
        self.index = pd.read_parquet(self.dir / "index.parquet")
        self.info = json.loads((self.dir / "done.json").read_text())
        assert len(self.index) == self.x.shape[0]

    def __len__(self) -> int:
        return self.x.shape[0]

    def rows(self, split: str | tuple[str, ...]) -> np.ndarray:
        splits = (split,) if isinstance(split, str) else tuple(split)
        return self.index.index[self.index.split.isin(splits)].to_numpy()

    def ctx(self, rows: np.ndarray) -> np.ndarray:
        return self.index.loc[rows, list(CTX_NAMES)].to_numpy(np.float32)

    def stratum_key(self, rows: np.ndarray) -> np.ndarray:
        """camera*10+ccd, used to pick the per-CCD common-gap mask."""
        sub = self.index.loc[rows]
        return (sub.camera.to_numpy() * 10 + sub.ccd.to_numpy()).astype(np.int64)


def training_rows(cache: Cache, split: str = "train") -> np.ndarray:
    """Rows usable for training: the named split only, real stars only, with enough data.

    ``known`` and ``test`` are never returned for training (asserted)."""
    assert split in ("train", "val"), "only train/val are ever used for fitting or early stopping"
    idx = cache.index
    sel = (idx.split == split) & idx.inj_id.isna() & (idx.n_obs >= 200)
    rows = idx.index[sel].to_numpy()
    assert not idx.loc[rows, "split"].isin(["known", "test"]).any()
    return rows


def common_gap_masks(cache: Cache, rows: np.ndarray, min_frac: float = 0.9) -> dict[int, np.ndarray]:
    """Per-sector 'instrument-blind' observation mask: a bin is usable only if, on *every*
    camera/CCD, at least ``min_frac`` of stars have data there. This removes the per-CCD gap
    fingerprint (scattered-light / quality-flag windows) that otherwise identifies the CCD.
    Returns {0: mask [L] bool} (single global mask; key kept for future per-sector use)."""
    keys = cache.stratum_key(rows)
    fr = []
    rng = np.random.default_rng(0)
    for k in np.unique(keys):
        r = rows[keys == k]
        r = np.sort(rng.choice(r, size=min(len(r), 1500), replace=False))
        fr.append(np.isfinite(cache.x[r, 0].astype(np.float32)).mean(0))
    fr = np.stack(fr)
    return {0: (fr >= min_frac).all(0)}


# --------------------------------------------------------------------------------------------
# Batching (memory-light: index -> memmap gather, one batch at a time)


def iter_batches(
    cache: Cache, rows: np.ndarray, batch_size: int, shuffle: bool, seed: int = 0
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield ``(rows, x float32 [B,3,L], ctx [B,3])``. Rows inside a batch are sorted so the memmap
    read is near-sequential."""
    order = np.random.default_rng(seed).permutation(rows) if shuffle else np.asarray(rows)
    for i in range(0, len(order), batch_size):
        r = np.sort(order[i : i + batch_size])
        yield r, np.asarray(cache.x[r], dtype=np.float32), cache.ctx(r)


# --------------------------------------------------------------------------------------------
# Torch side: augmentation and model inputs (GPU)


def augment(x, gen, max_shift: int = 48, gap_p: float = 0.5, noise_max: float = 0.5,
            amp_jitter: float = 0.1):  # fmt: skip
    """Training-only augmentations (ML_DESIGN §1) on a torch tensor [B,3,L] with NaN gaps.

    * random shift (crop at the edges; vacated bins become gaps, never wrapped),
    * extra gap-like masked spans,
    * small amplitude jitter,
    * flux noise matched to the per-bin error (error inflated consistently).

    Never time-reversed, never injects dips."""
    import torch

    B, C, L = x.shape
    dev = x.device
    shift = torch.randint(-max_shift, max_shift + 1, (B, 1), device=dev, generator=gen)
    src = torch.arange(L, device=dev)[None, :] - shift
    valid = (src >= 0) & (src < L)
    src = src.clamp(0, L - 1)
    x = torch.gather(x, 2, src[:, None, :].expand(B, C, L))
    x = torch.where(valid[:, None, :], x, torch.full_like(x, float("nan")))

    # 0-2 extra gap spans of 8-64 bins
    for _ in range(2):
        on = torch.rand(B, 1, device=dev, generator=gen) < gap_p
        ln = torch.randint(8, 65, (B, 1), device=dev, generator=gen)
        st = (torch.rand(B, 1, device=dev, generator=gen) * L).long()
        pos = torch.arange(L, device=dev)[None, :]
        span = on & (pos >= st) & (pos < st + ln)
        x = torch.where(span[:, None, :], torch.full_like(x, float("nan")), x)

    amp = 1.0 + amp_jitter * (2 * torch.rand(B, 1, device=dev, generator=gen) - 1)
    a = noise_max * torch.rand(B, 1, device=dev, generator=gen)
    err = x[:, 2]
    noise = torch.randn(B, L, device=dev, generator=gen) * a * err
    pdc = x[:, 0] * amp + noise
    sap = x[:, 1] * amp + torch.randn(B, L, device=dev, generator=gen) * a * err
    err = err * torch.sqrt(1 + a**2)
    return torch.stack([pdc, sap, err], 1)


def make_inputs(x, obs_mask=None):
    """[B,3,L] (NaN gaps) -> model inputs.

    Returns ``inp [B,P,PATCH*4]`` (pdc, sap, log err, obs per bin; unobserved bins zeroed),
    ``target [B,2,L]`` (pdc, sap; 0 where unobserved), ``err [B,L]``, ``obs [B,L]`` (pdc observed),
    ``sap_obs [B,L]``. ``obs_mask`` [L] or [B,L] bool further restricts observed bins (the common
    gap mask)."""
    import torch

    pdc, sap, err = x[:, 0], x[:, 1], x[:, 2]
    obs = torch.isfinite(pdc) & torch.isfinite(err) & (err > 0)
    if obs_mask is not None:
        obs = obs & obs_mask
    sap_obs = obs & torch.isfinite(sap)
    z = torch.zeros_like(pdc)
    pdc = torch.where(obs, pdc, z)
    sap = torch.where(sap_obs, sap, z)
    err = torch.where(obs, err, torch.ones_like(err))
    lerr = torch.where(obs, torch.log(err).clamp(-7, 7), z)
    B, L = pdc.shape
    P = L // PATCH
    feats = torch.stack([pdc, sap, lerr, obs.float()], -1)  # [B,L,4]
    inp = feats.reshape(B, P, PATCH * 4)
    return inp, torch.stack([pdc, sap], 1), err, obs, sap_obs


def contiguous_block_mask(B: int, P: int, ratio: float, rng: np.random.Generator,
                          min_len: int = 2, max_len: int = 8) -> np.ndarray:  # fmt: skip
    """[B,P] bool: ~``ratio`` of patches hidden in contiguous blocks of min_len..max_len."""
    target = round(ratio * P)
    m = np.zeros((B, P), dtype=bool)
    for b in range(B):
        while m[b].sum() < target:
            ln = rng.integers(min_len, max_len + 1)
            st = rng.integers(0, P - ln + 1)
            m[b, st : st + ln] = True
    return m


def scoring_masks(P: int, n_masks: int, rng: np.random.Generator, block: int = 4) -> np.ndarray:
    """[n_masks, P] bool: blocks of ``block`` patches; every block is hidden in exactly
    ``n_masks*0.4`` masks (2 of 5), so each patch gets a surprise estimate, each mask hides 40%."""
    nb = int(np.ceil(P / block))
    per = max(1, round(0.4 * n_masks))
    perm = rng.permutation(nb)
    m = np.zeros((n_masks, P), dtype=bool)
    for j in range(nb):
        for q in range(per):
            g = (perm[j] + q) % n_masks
            m[g, j * block : (j + 1) * block] = True
    return m


# --------------------------------------------------------------------------------------------
# Several caches (one per sector) behind one global row index


class MultiCache:
    """Several caches (e.g. one per training sector) addressed by one global row index.

    Global row ``g`` lives in cache ``cid = searchsorted(offsets, g, 'right') - 1`` at local row
    ``g - offsets[cid]``. Each sector keeps its own grid origin (bins are sector-relative)."""

    def __init__(self, cache_dirs):
        self.caches = [Cache(Path(d)) for d in cache_dirs]
        self.offsets = np.cumsum([0] + [len(c) for c in self.caches]).astype(np.int64)
        self.sectors = [int(c.index.sector.iloc[0]) for c in self.caches]

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def split_global(self, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cid = np.searchsorted(self.offsets, g, side="right") - 1
        return cid, g - self.offsets[cid]

    def training_rows(self, split: str) -> np.ndarray:
        return np.concatenate([training_rows(c, split) + o
                               for c, o in zip(self.caches, self.offsets, strict=False)])

    def index_of(self, g: np.ndarray) -> pd.DataFrame:
        cid, loc = self.split_global(np.asarray(g))
        parts = [self.caches[c].index.loc[loc[cid == c]] for c in np.unique(cid)]
        return pd.concat(parts)

    def iter_batches(self, grows: np.ndarray, batch_size: int, shuffle: bool, seed: int = 0):
        """Yield ``(global rows, cache id per row, x float32 [B,3,L], ctx [B,3])``; inside a batch
        rows are grouped by cache and sorted, so each memmap read is near-sequential."""
        grows = np.asarray(grows)
        order = np.random.default_rng(seed).permutation(grows) if shuffle else grows
        for i in range(0, len(order), batch_size):
            g = np.sort(order[i : i + batch_size])
            cid, loc = self.split_global(g)
            xs, cs = [], []
            for c in np.unique(cid):
                r = loc[cid == c]
                xs.append(np.asarray(self.caches[c].x[r], dtype=np.float32))
                cs.append(self.caches[c].ctx(r))
            yield g, cid, np.concatenate(xs), np.concatenate(cs)


def sector_gap_masks(mc: MultiCache, split_rows: np.ndarray, min_frac: float = 0.9) -> np.ndarray:
    """[n_caches, L] bool: the common-gap mask (``common_gap_masks``) of each cache, computed on
    the given (train) rows of that cache."""
    cid, loc = mc.split_global(split_rows)
    return np.stack([common_gap_masks(c, loc[cid == k], min_frac)[0]
                     for k, c in enumerate(mc.caches)])  # fmt: skip


def signal_dominated(x: np.ndarray, max_err: float = 0.3, min_obs: int = 400) -> np.ndarray:
    """[B] bool: stars whose median per-bin error is below ``max_err`` robust sigmas, i.e. whose
    scatter is mostly stellar variability (predictable from context), not white noise."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(x[:, 2], axis=1)
    return (med < max_err) & (np.isfinite(x[:, 0]).sum(1) >= min_obs)


def signal_flags(cache: Cache, chunk: int = 4096) -> np.ndarray:
    """``signal_dominated`` for every row of a cache (computed once, stored as sigdom.npy)."""
    f = cache.dir / "sigdom.npy"
    if f.exists():
        return np.load(f)
    out = np.zeros(len(cache), dtype=bool)
    for i in range(0, len(cache), chunk):
        out[i : i + chunk] = signal_dominated(np.asarray(cache.x[i : i + chunk], np.float32))
    np.save(f, out)
    return out


def mixed_order(rows: np.ndarray, sig_rows: np.ndarray, frac: float, batch_size: int,
                rng: np.random.Generator) -> np.ndarray:  # fmt: skip
    """An epoch's row order in which every batch has ``frac`` of its rows drawn (with
    replacement) from ``sig_rows`` and the rest from a permutation of ``rows``."""
    if frac <= 0 or len(sig_rows) == 0:
        return rng.permutation(rows)
    n_sig = int(round(frac * batch_size))
    base = rng.permutation(rows)
    n_b = int(np.ceil(len(base) / (batch_size - n_sig)))
    out = []
    for b in range(n_b):
        part = base[b * (batch_size - n_sig) : (b + 1) * (batch_size - n_sig)]
        out += [part, rng.choice(sig_rows, n_sig)]
    return np.concatenate(out)
