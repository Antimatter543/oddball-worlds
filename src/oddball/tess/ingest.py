"""Stream one TESS-SPOC FFI sector into compact Parquet shards.

Layout under ``<data>/tess-spoc/sNNNN/``::

    dl-lc.sh                 MAST bulk curl script (the URL list; cached)
    shards/lc_NNNNN.parquet  binned light curves, ~shard_size rows each
    shards/meta_NNNNN.parquet  per-LC metadata for the same rows
    metadata.parquet         consolidated metadata (rebuilt by finalize)
    failures.csv             one line per failed attempt (tic, kind, message, utc)
    provenance.json          product, version, URLs, bin config, run history

FITS bytes are held in memory only and dropped as soon as they are converted, so no FITS file
ever touches the disk. Resumable: TICs already in a written shard, or recorded as permanent
failures, are skipped. Shards are written to a temp name and renamed, so a crash never leaves
a half-written shard; Ctrl-C flushes the in-memory buffer as a (short) final shard.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import re
import resource
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from oddball.common.http import ANON_USER_AGENT
from oddball.common.paths import sector_dir
from oddball.tess.lightcurve import (
    DEFAULT_BIN_MINUTES,
    DEFAULT_QUALITY_BITMASK,
    LightCurveError,
    convert_fits_bytes,
)

PRODUCT = "TESS-SPOC"
HLSP_VERSION = "v1"
HLSP_DOI = "10.17909/t9-wpz1-8s54"
SCRIPT_URL = (
    "https://archive.stsci.edu/hlsps/tess-spoc/download_scripts/"
    "hlsp_tess-spoc_tess_phot_s{sector:04d}_tess_{version}_dl-lc.sh"
)
_URL_RE = re.compile(r"'(https://[^']+_lc\.fits)'")
_TIC_RE = re.compile(r"hlsp_tess-spoc_tess_phot_(\d{16})-s(\d{4})_")

LC_SCHEMA = pa.schema(
    [
        ("tic", pa.int64()),
        ("sector", pa.int16()),
        ("bin0", pa.int64()),
        ("flux", pa.list_(pa.float32())),
        ("flux_err", pa.list_(pa.float32())),
        ("npts", pa.list_(pa.uint16())),
        ("qflags", pa.list_(pa.int32())),
        ("sap_flux", pa.list_(pa.float32())),
        ("bkg", pa.list_(pa.float32())),
    ]
)
# Columns of the metadata table (order kept stable for readers).
META_COLUMNS = [
    "tic", "sector", "camera", "ccd", "tmag", "ra", "dec", "teff", "radius",
    "crowdsap", "flfrcsap", "cadence_s", "n_cadences", "n_good", "n_bins", "n_bins_filled",
    "rms_bin", "p2p_native", "median_flux", "bin0", "bin_minutes", "t_start", "t_stop",
    "data_rel", "procver", "pdcmethod", "shard",
]  # fmt: skip

# Hard ceiling on bins per light curve: a sector is <~30 d, so anything past 40 d of span
# means a corrupt time column, and would otherwise make one giant row.
MAX_SPAN_DAYS = 40.0


@dataclass(frozen=True)
class Target:
    tic: int
    url: str


# --------------------------------------------------------------------------------------------
# URL list


def fetch_script(sector: int, dest: Path, version: str = HLSP_VERSION) -> tuple[Path, str]:
    url = SCRIPT_URL.format(sector=sector, version=version)
    if not dest.exists():
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        tmp = dest.with_suffix(".tmp")
        tmp.write_bytes(r.content)
        tmp.rename(dest)
    return dest, url


def parse_script(text: str, sector: int) -> list[Target]:
    out, seen = [], set()
    for url in _URL_RE.findall(text):
        m = _TIC_RE.search(url)
        if not m or int(m.group(2)) != sector:
            continue
        tic = int(m.group(1))
        if tic not in seen:
            seen.add(tic)
            out.append(Target(tic, url))
    return out


# --------------------------------------------------------------------------------------------
# Resume state


def _shard_files(sdir: Path, kind: str) -> list[Path]:
    return sorted((sdir / "shards").glob(f"{kind}_*.parquet"))


def done_tics(sdir: Path) -> set[int]:
    """TICs already converted (in a shard) or permanently failed."""
    done: set[int] = set()
    for f in _shard_files(sdir, "lc"):
        done.update(pq.read_table(f, columns=["tic"]).column("tic").to_pylist())
    fail = sdir / "failures.csv"
    if fail.exists():
        with fail.open() as fh:
            for row in csv.DictReader(fh):
                if row["kind"] == "permanent":
                    done.add(int(row["tic"]))
    return done


def _next_shard_index(sdir: Path) -> int:
    idx = [int(f.stem.split("_")[1]) for f in _shard_files(sdir, "lc")]
    return max(idx) + 1 if idx else 0


# --------------------------------------------------------------------------------------------
# Writing


def _list_array(arrs: list[np.ndarray], dtype) -> pa.ListArray:
    lengths = np.fromiter((len(a) for a in arrs), dtype=np.int32, count=len(arrs))
    offsets = np.zeros(len(arrs) + 1, dtype=np.int32)
    np.cumsum(lengths, out=offsets[1:])
    values = np.concatenate(arrs).astype(dtype, copy=False) if arrs else np.array([], dtype)
    return pa.ListArray.from_arrays(pa.array(offsets), pa.array(values))


def _atomic_write(table: pa.Table, path: Path, **kw) -> None:
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd", **kw)
    os.replace(tmp, path)


class ShardWriter:
    def __init__(self, sdir: Path, sector: int, shard_size: int):
        self.dir = sdir / "shards"
        self.dir.mkdir(parents=True, exist_ok=True)
        for stale in self.dir.glob("*.tmp"):  # leftovers from a crash mid-write
            stale.unlink()
        self.sector = sector
        self.shard_size = shard_size
        self.index = _next_shard_index(sdir)
        self.metas: list[dict] = []
        self.bins: list = []
        self.written = 0

    def add(self, meta: dict, binned) -> None:
        meta["shard"] = self.index
        meta["bin0"] = binned.bin0
        # Downcast to storage dtypes now: the buffer holds shard_size LCs, and float64 doubles
        # it (~1 GB vs ~0.5 GB at 10k rows).
        binned.flux = binned.flux.astype(np.float32)
        binned.flux_err = binned.flux_err.astype(np.float32)
        binned.npts = binned.npts.astype(np.uint16)
        binned.extras = {
            k: v.astype(np.int32 if k == "qflags" else np.float32) for k, v in binned.extras.items()
        }
        self.metas.append(meta)
        self.bins.append(binned)
        if len(self.bins) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.bins:
            return
        b = self.bins
        lc = pa.Table.from_arrays(
            [
                pa.array([m["tic"] for m in self.metas], pa.int64()),
                pa.array([self.sector] * len(b), pa.int16()),
                pa.array([x.bin0 for x in b], pa.int64()),
                _list_array([x.flux for x in b], np.float32),
                _list_array([x.flux_err for x in b], np.float32),
                _list_array([x.npts for x in b], np.uint16),
                _list_array([x.extras["qflags"] for x in b], np.int32),
                _list_array([x.extras["sap_flux"] for x in b], np.float32),
                _list_array([x.extras["bkg"] for x in b], np.float32),
            ],
            schema=LC_SCHEMA,
        )
        meta = pa.Table.from_pylist([{k: m.get(k) for k in META_COLUMNS} for m in self.metas])
        # Small row groups so a reader pulling a handful of TICs decodes ~1k rows, not 10k.
        _atomic_write(lc, self.dir / f"lc_{self.index:05d}.parquet", row_group_size=1000)
        _atomic_write(meta, self.dir / f"meta_{self.index:05d}.parquet")
        self.written += len(b)
        self.index += 1
        self.metas, self.bins = [], []


class FailureLog:
    def __init__(self, path: Path):
        new = not path.exists()
        self.fh = path.open("a", newline="")
        self.w = csv.writer(self.fh)
        if new:
            self.w.writerow(["tic", "kind", "message", "utc"])
        self.count = {"permanent": 0, "transient": 0}

    def log(self, tic: int, kind: str, msg: str) -> None:
        self.count[kind] += 1
        self.w.writerow([tic, kind, msg[:300], _utcnow()])
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


# --------------------------------------------------------------------------------------------
# Download + convert (runs in worker threads)

_tls = threading.local()


def _session() -> requests.Session:
    s = getattr(_tls, "session", None)
    if s is None:
        s = requests.Session()
        s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=2, pool_maxsize=2))
        s.headers["User-Agent"] = ANON_USER_AGENT
        _tls.session = s
    return s


def _download(url: str, retries: int = 4, timeout: float = 60.0) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = _session().get(url, timeout=timeout)
            if r.status_code == 404:
                raise LightCurveError("HTTP 404")
            r.raise_for_status()
            return r.content
        except LightCurveError:
            raise
        except requests.RequestException as e:  # network / 5xx / 429: back off and retry
            last = e
            time.sleep(min(30.0, 2.0 * 2**attempt))
    raise ConnectionError(f"download failed after {retries} tries: {last}")


def process_target(t: Target, sector: int, bin_minutes: float, bitmask: int):
    data = _download(t.url)
    try:
        meta, binned = convert_fits_bytes(data, bin_minutes=bin_minutes, bitmask=bitmask)
    finally:
        del data  # the only copy of the FITS; never written to disk
    if int(meta["tic"]) != t.tic:
        raise LightCurveError(f"header TICID {meta['tic']} != filename TIC {t.tic}")
    if meta["sector"] is not None and int(meta["sector"]) != sector:
        raise LightCurveError(f"header SECTOR {meta['sector']} != {sector}")
    if binned.nbins * bin_minutes / 1440.0 > MAX_SPAN_DAYS:
        raise LightCurveError(f"time span {binned.nbins * bin_minutes / 1440:.1f} d too long")
    return meta, binned


# --------------------------------------------------------------------------------------------
# Driver


def _utcnow() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rss_mb() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def select_targets(
    targets: list[Target], done: set[int], limit: int | None, include_tics: set[int]
) -> list[Target]:
    """``limit`` takes the first N of the script order (deterministic), then adds any
    ``include_tics`` (e.g. known positives) found in the sector; already-done TICs drop out."""
    chosen = targets[:limit] if limit else list(targets)
    have = {t.tic for t in chosen}
    chosen += [t for t in targets if t.tic in include_tics and t.tic not in have]
    return [t for t in chosen if t.tic not in done]


def ingest_sector(
    sector: int,
    limit: int | None = None,
    include_tics: set[int] | None = None,
    workers: int = 32,
    shard_size: int = 10_000,
    bin_minutes: float = DEFAULT_BIN_MINUTES,
    bitmask: int = DEFAULT_QUALITY_BITMASK,
    log=print,
) -> dict:
    sdir = sector_dir(sector)
    sdir.mkdir(parents=True, exist_ok=True)
    _check_bin_config(sdir, bin_minutes, bitmask)

    script, script_url = fetch_script(sector, sdir / "dl-lc.sh")
    targets = parse_script(script.read_text(), sector)
    done = done_tics(sdir)
    todo = select_targets(targets, done, limit, include_tics or set())
    log(f"s{sector:04d}: {len(targets)} LCs in script, {len(done)} done, {len(todo)} to do; "
        f"workers={workers} shard={shard_size} bin={bin_minutes} min -> {sdir}")

    writer = ShardWriter(sdir, sector, shard_size)
    fails = FailureLog(sdir / "failures.csv")
    run = {"started_utc": _utcnow(), "requested": len(todo), "limit": limit,
           "workers": workers, "bin_minutes": bin_minutes, "quality_bitmask": bitmask}  # fmt: skip
    t0 = time.time()
    ok = 0
    interrupted = False
    window = 2 * workers  # bounded in-flight futures: memory is O(window), not O(sector)
    it = iter(todo)
    pending: dict = {}
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            def refill():
                while len(pending) < window:
                    t = next(it, None)
                    if t is None:
                        return
                    pending[ex.submit(process_target, t, sector, bin_minutes, bitmask)] = t

            refill()
            try:
                while pending:
                    finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in finished:
                        t = pending.pop(fut)
                        try:
                            meta, binned = fut.result()
                            writer.add(meta, binned)
                            ok += 1
                        except LightCurveError as e:
                            fails.log(t.tic, "permanent", str(e))
                        except Exception as e:  # noqa: BLE001 - logged, retried next run
                            fails.log(t.tic, "transient", f"{type(e).__name__}: {e}")
                        n = ok + sum(fails.count.values())
                        if n % 1000 == 0:
                            el = time.time() - t0
                            rate = n / el
                            log(f"  {n}/{len(todo)}  {rate:.1f}/s  eta {(len(todo)-n)/rate/60:.1f} min"
                                f"  fail p={fails.count['permanent']} t={fails.count['transient']}"
                                f"  rss {_rss_mb():.0f} MB")
                    refill()
            except KeyboardInterrupt:
                interrupted = True
                log("interrupted: cancelling pending downloads, flushing buffer ...")
                for f in pending:
                    f.cancel()
                ex.shutdown(wait=True, cancel_futures=True)
    finally:
        writer.flush()
        fails.close()
        run.update({
            "finished_utc": _utcnow(), "seconds": round(time.time() - t0, 1), "converted": ok,
            "failed_permanent": fails.count["permanent"],
            "failed_transient": fails.count["transient"], "interrupted": interrupted,
        })  # fmt: skip
        summary = finalize(sector, script_url, len(targets), run)
    log(json.dumps({k: summary[k] for k in ("n_converted", "n_in_script", "bytes_shards")}))
    if interrupted:
        sys.exit(130)
    return summary


def _check_bin_config(sdir: Path, bin_minutes: float, bitmask: int) -> None:
    """Refuse to mix bin widths / masks within one sector directory."""
    prov = sdir / "provenance.json"
    if prov.exists():
        p = json.loads(prov.read_text())
        if p.get("n_converted", 0) and (
            p["bin_minutes"] != bin_minutes or p["quality_bitmask"] != bitmask
        ):
            raise SystemExit(
                f"{sdir} was built with bin={p['bin_minutes']} min, bitmask={p['quality_bitmask']};"
                " use a different ODDBALL_DATA or delete it first"
            )


def finalize(sector: int, script_url: str, n_in_script: int, run: dict | None = None) -> dict:
    """Consolidate shard metadata into metadata.parquet and (re)write provenance.json."""
    sdir = sector_dir(sector)
    metas = _shard_files(sdir, "meta")
    if metas:
        tbl = pa.concat_tables([pq.read_table(f) for f in metas], promote_options="default")
        _atomic_write(tbl, sdir / "metadata.parquet")
        n_conv = tbl.num_rows
        data_rel = sorted({str(x) for x in tbl.column("data_rel").to_pylist()})
        procver = sorted({str(x) for x in tbl.column("procver").to_pylist()})
        cadence = sorted({round(float(x), 1) for x in tbl.column("cadence_s").to_pylist()})
    else:
        n_conv, data_rel, procver, cadence = 0, [], [], []

    prov_path = sdir / "provenance.json"
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {"runs": []}
    if run:
        prov["runs"].append(run)
    lc_files = _shard_files(sdir, "lc")
    prov.update({
        "sector": sector,
        "product": PRODUCT,
        "product_full": "TESS-SPOC FFI light curves, MAST HLSP hlsp_tess-spoc",
        "hlsp_version": HLSP_VERSION,
        "doi": HLSP_DOI,
        "hlsp_page": "https://archive.stsci.edu/hlsp/tess-spoc",
        "script_url": script_url,
        "file_url_template": "https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:HLSP/"
        "tess-spoc/s{sector:04d}/target/{tic16[0:4]}/{tic16[4:8]}/{tic16[8:12]}/"
        "{tic16[12:16]}/hlsp_tess-spoc_tess_phot_{tic16}-s{sector:04d}_tess_v1_lc.fits",
        "download_dates_utc": sorted({r["started_utc"][:10] for r in prov["runs"]}),
        "spoc_data_release": data_rel,
        "spoc_procver": procver,
        "native_cadence_s": cadence,
        "flux_column": "PDCSAP_FLUX (normalised by median of good cadences); no detrending",
        "bin_minutes": run["bin_minutes"] if run else prov.get("bin_minutes"),
        "bin_grid": "global: bin k covers [k*w, (k+1)*w) BTJD, w=bin_minutes/1440; "
        "time_centre = (bin0 + i + 0.5) * w",
        "quality_bitmask": run["quality_bitmask"] if run else prov.get(
            "quality_bitmask", DEFAULT_QUALITY_BITMASK),
        "n_in_script": n_in_script,
        "n_converted": n_conv,
        "n_shards": len(lc_files),
        "bytes_shards": sum(f.stat().st_size for f in lc_files),
        "bytes_metadata": sum(f.stat().st_size for f in _shard_files(sdir, "meta")),
        "finalized_utc": _utcnow(),
    })  # fmt: skip
    tmp = prov_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(prov, indent=2))
    os.replace(tmp, prov_path)
    return prov
