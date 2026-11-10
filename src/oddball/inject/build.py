"""Build one injection set: ``<data>/injections/<name>/sNNNN/<split>/`` (docs/ML_DESIGN.md §9).

Layout (same shard schema as ingest, so any scorer reads it like a real sector)::

    shards/lc_NNNNN.parquet    LC_SCHEMA + inj_id (str) + host_tic (int); tic == host_tic
    shards/meta_NNNNN.parquet  the host's metadata row + inj_id, host_tic, kind, is_null
    metadata.parquet           consolidated meta
    injections.parquet         one row per inj_id (see INJ_COLUMNS)
    provenance.json            seed, counts, source-sector provenance, git SHA

Host selection (failure catalogue #3, optimistic injections; #4 leakage):

* hosts come **only** from ``split_of(tic) == split``; ``known`` stars are never hosts
  (``split_of`` already returns ``known`` for them, and we assert it);
* astrophysical hosts are a **proportional stratified random sample** over Tmag bin x
  camera/CCD (``oddball.common.strata``), with no quietness or quality cut, so variable,
  faint and crowded stars are injected at their real rate;
* null-event hosts are disjoint from astrophysical hosts; each host carries one injection.

Null (instrument-like) events: ``n_events`` events on distinct CCDs, ``hosts_per_event`` hosts
each, every host of an event getting the *identical* model (same shape, time, depth).
The last ``n_dump_events`` are aligned with a momentum dump read from QUALITY bit 32.

Processing is shard by shard: only the hosts of one source shard are in memory at a time.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from oddball.common.paths import REPO_ROOT, data_root, sector_dir
from oddball.common.splits import known_positive_tics, split_array
from oddball.common.strata import ccd_index, stratum_id
from oddball.inject.apply import apply_model, sap_dilution
from oddball.inject.sampling import (
    DEPTH_RANGE,
    KINDS,
    PER_HOST_NULL_KINDS,
    draw_null_event,
    draw_params,
)
from oddball.tess.arrays import dir_metadata
from oddball.tess.ingest import LC_SCHEMA

SPLIT_CODE = {"train": 0, "val": 1, "test": 2}
INJ_SCHEMA = pa.schema(list(LC_SCHEMA) + [("inj_id", pa.string()), ("host_tic", pa.int64())])
INJ_COLUMNS = [
    "inj_id", "host_tic", "sector", "split", "kind", "depth", "duration_d", "t0_btjd",
    "params_json", "is_null",
    # extras (not required by §9, used by the evaluator and for audits)
    "event_id", "tmag", "camera", "ccd", "stratum", "crowdsap", "sap_depth", "binned_depth",
    "peak_bin_valid",
]  # fmt: skip


def injection_dir(name: str, sector: int, split: str) -> Path:
    return data_root() / "injections" / name / f"s{sector:04d}" / split


def stratified_sample(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Proportional allocation over ``df.stratum`` (largest remainder), then random draws."""
    counts = df.stratum.value_counts().sort_index()
    quota = counts / counts.sum() * n
    alloc = np.floor(quota).astype(int)
    rem = int(n - alloc.sum())
    if rem > 0:
        order = np.argsort(-(quota - alloc).to_numpy(), kind="stable")
        alloc.iloc[order[:rem]] += 1
    parts = []
    for s, k in alloc.items():
        grp = df[df.stratum == s]
        k = min(int(k), len(grp))
        if k:
            parts.append(grp.iloc[rng.choice(len(grp), size=k, replace=False)])
    return pd.concat(parts).sort_values("tic").reset_index(drop=True)


def select_hosts(
    meta: pd.DataFrame,
    split: str,
    n_astro: int,
    n_events: int,
    hosts_per_event: int,
    rng: np.random.Generator,
    exclude_tics=None,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Return ``(astro_hosts, [null_hosts_event_0, ...])`` from ``split`` only.

    ``exclude_tics``: hosts already used by another injection set (kept disjoint).
    """
    m = meta.copy()
    m["split"] = split_array(m.tic.to_numpy())
    cand = m[m.split == split].copy()
    if exclude_tics is not None:
        cand = cand[~cand.tic.isin({int(x) for x in exclude_tics})]
    known = known_positive_tics()
    assert not cand.tic.isin(known).any(), "known positives leaked into the host pool"
    cand["stratum"] = stratum_id(cand.tmag, cand.camera, cand.ccd)
    cand["ccd_idx"] = ccd_index(cand.camera, cand.ccd)

    ccds = sorted(cand.ccd_idx.unique())
    ev_ccds = rng.choice(ccds, size=min(n_events, len(ccds)), replace=False)
    null_groups, used = [], set()
    for c in ev_ccds:
        pool = cand[(cand.ccd_idx == c) & ~cand.tic.isin(used)]
        g = pool.iloc[rng.choice(len(pool), size=min(hosts_per_event, len(pool)), replace=False)]
        used.update(g.tic.tolist())
        null_groups.append(g.sort_values("tic").reset_index(drop=True))
    astro = stratified_sample(cand[~cand.tic.isin(used)], n_astro, rng)
    return astro, null_groups


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10, check=False).stdout.strip()  # fmt: skip
    except Exception:  # noqa: BLE001
        return "unknown"


def _read_hosts(sector: int, shard: int, tics: list[int]) -> dict[int, dict]:
    path = sector_dir(sector) / "shards" / f"lc_{int(shard):05d}.parquet"
    tbl = ds.dataset(path).to_table(filter=ds.field("tic").isin(tics))
    return {int(r["tic"]): r for r in tbl.to_pylist()}


def _times(bin0: int, n: int, bin_days: float) -> np.ndarray:
    return (bin0 + np.arange(n) + 0.5) * bin_days


def _event_coverage(sector, hosts: pd.DataFrame, bin_days: float):
    """Per-bin fraction of an event's hosts with data, on a common grid, plus dump times."""
    rows = {}
    for shard, grp in hosts.groupby("shard"):
        rows.update(_read_hosts(sector, int(shard), grp.tic.tolist()))
    g0 = min(int(r["bin0"]) for r in rows.values())
    g1 = max(int(r["bin0"]) + len(r["flux"]) for r in rows.values())
    cover = np.zeros(g1 - g0)
    dump = np.zeros(g1 - g0, dtype=bool)
    for r in rows.values():
        f = np.asarray(r["flux"], np.float64)
        q = np.asarray(r["qflags"], np.int64)
        j = int(r["bin0"]) - g0
        cover[j : j + len(f)] += np.isfinite(f)
        dump[j : j + len(f)] |= (q & 32) > 0
    t = _times(g0, g1 - g0, bin_days)
    # one time per contiguous run of dump-flagged bins
    idx = np.flatnonzero(dump)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if idx.size else []
    dump_times = np.array([t[r].mean() for r in runs])
    return t, cover / len(rows), dump_times


def build_injection_set(
    sector: int,
    split: str,
    name: str = "v1",
    seed: int = 20261109,
    n_astro: int = 3000,
    n_events: int = 10,
    hosts_per_event: int = 30,
    n_dump_events: int = 3,
    out_dir: Path | None = None,
    log=print,
    kinds: tuple[str, ...] = KINDS,
    exclude_tics=None,
) -> dict:
    if split not in ("val", "test"):
        raise ValueError("injection sets are built for val or test only")
    out = Path(out_dir) if out_dir else injection_dir(name, sector, split)
    if (out / "injections.parquet").exists():
        raise SystemExit(f"{out} already exists; delete it to rebuild")
    (out / "shards").mkdir(parents=True, exist_ok=True)

    meta = dir_metadata(sector_dir(sector))
    bin_days = float(meta.bin_minutes.iloc[0]) / 1440.0
    ss = np.random.SeedSequence([seed, sector, SPLIT_CODE[split]])
    rng_sel, rng_ev, rng_kind = (np.random.default_rng(s) for s in ss.spawn(3))
    # per-injection streams are keyed by the injection's index, not processing order
    inj_key = [seed, sector, SPLIT_CODE[split], 7]

    astro, null_groups = select_hosts(meta, split, n_astro, n_events, hosts_per_event, rng_sel,
                                      exclude_tics)  # fmt: skip

    # ---- plan: one row per injection
    kinds = np.resize(np.array(kinds), len(astro))
    rng_kind.shuffle(kinds)
    plan = astro.assign(kind=kinds, event_id=-1, params=None)
    n_ev = len(null_groups)
    # event depths: evenly spread in log over the depth range, shuffled (10 events cover it)
    edges = np.linspace(np.log(DEPTH_RANGE[0]), np.log(DEPTH_RANGE[1]), n_ev + 1)
    ev_depths = np.exp(rng_ev.uniform(edges[:-1], edges[1:]))
    rng_ev.shuffle(ev_depths)
    ev_rows = []
    for e, grp in enumerate(null_groups):
        kind = "null_dump" if e >= n_ev - n_dump_events else "null_ccd"
        t, cover, dumps = _event_coverage(sector, grp, bin_days)
        p = draw_null_event(kind, rng_ev, t, cover, float(ev_depths[e]), dump_times=dumps)
        p["event_id"] = e
        p["event_ccd"] = f"cam{int(grp.camera.iloc[0])}ccd{int(grp.ccd.iloc[0])}"
        log(f"  null event {e}: {kind} {p['event_ccd']} n={len(grp)} depth={p['depth']:.4f} "
            f"dur={p['duration_d']:.3f} d t0={p['t0_btjd']:.3f}")
        ev_rows.append(grp.assign(kind=kind, event_id=e, params=[p] * len(grp)))
    plan = pd.concat([plan, *ev_rows], ignore_index=True)
    plan["inj_id"] = [f"{name}-s{sector:04d}-{split}-{i:05d}" for i in range(len(plan))]
    plan["inj_index"] = np.arange(len(plan))
    assert plan.tic.is_unique

    # ---- shard-by-shard injection
    inj_records = []
    n_written = 0
    meta_cols = [c for c in meta.columns if c != "shard"]
    for out_idx, (shard, grp) in enumerate(plan.groupby("shard", sort=True)):
        hosts = _read_hosts(sector, int(shard), grp.tic.tolist())
        lc_rows, meta_rows = [], []
        for r in grp.itertuples(index=False):
            h = hosts[int(r.tic)]
            flux = np.asarray(h["flux"], np.float64)
            sap = np.asarray(h["sap_flux"], np.float64)
            t = _times(int(h["bin0"]), len(flux), bin_days)
            valid = np.isfinite(flux)
            if r.params is None:
                rng = np.random.default_rng([*inj_key, int(r.inj_index)])
                rad = float(r.radius) if r.radius is not None else np.nan
                p = draw_params(r.kind, rng, t, valid, host={"radius": rad})
                p["is_null"] = r.kind in PER_HOST_NULL_KINDS
            else:
                p = dict(r.params)
            f_inj, s_inj, m = apply_model(flux, sap, t, p, bin_days, crowdsap=r.crowdsap)
            dev = np.abs(m - 1.0)
            k = int(np.argmin(np.abs(t - p["t0_btjd"])))
            lc_rows.append({
                "tic": int(r.tic), "sector": int(sector), "bin0": int(h["bin0"]),
                "flux": f_inj, "flux_err": np.asarray(h["flux_err"], np.float32),
                "npts": np.asarray(h["npts"], np.uint16),
                "qflags": np.asarray(h["qflags"], np.int32), "sap_flux": s_inj,
                "bkg": np.asarray(h["bkg"], np.float32),
                "inj_id": r.inj_id, "host_tic": int(r.tic),
            })  # fmt: skip
            mrow = {c: getattr(r, c) for c in meta_cols}
            mrow.update(inj_id=r.inj_id, host_tic=int(r.tic), kind=r.kind,
                        is_null=bool(p["is_null"]), shard=out_idx)  # fmt: skip
            meta_rows.append(mrow)
            inj_records.append({
                "inj_id": r.inj_id, "host_tic": int(r.tic), "sector": int(sector),
                "split": split, "kind": r.kind, "depth": float(p["depth"]),
                "duration_d": float(p["duration_d"]), "t0_btjd": float(p["t0_btjd"]),
                "params_json": json.dumps(p, sort_keys=True), "is_null": bool(p["is_null"]),
                "event_id": int(r.event_id), "tmag": float(r.tmag), "camera": int(r.camera),
                "ccd": int(r.ccd), "stratum": int(r.stratum),
                "crowdsap": float(r.crowdsap) if r.crowdsap is not None else np.nan,
                "sap_depth": float(p["depth"]) * sap_dilution(r.crowdsap),
                "binned_depth": float(np.max(dev[valid])) if valid.any() else 0.0,
                "peak_bin_valid": bool(valid[k]),
            })  # fmt: skip
        lc_tbl = pa.Table.from_pylist(lc_rows, schema=INJ_SCHEMA)
        _atomic(lc_tbl, out / "shards" / f"lc_{out_idx:05d}.parquet", row_group_size=1000)
        _atomic(pa.Table.from_pylist(meta_rows), out / "shards" / f"meta_{out_idx:05d}.parquet")
        n_written += len(lc_rows)
        log(f"  shard {int(shard):3d} -> lc_{out_idx:05d}: {len(lc_rows)} injections")

    inj = pd.DataFrame(inj_records, columns=INJ_COLUMNS)
    _atomic(pa.Table.from_pandas(inj, preserve_index=False), out / "injections.parquet")
    metas = [pq.read_table(f) for f in sorted((out / "shards").glob("meta_*.parquet"))]
    _atomic(pa.concat_tables(metas, promote_options="default"), out / "metadata.parquet")

    src_prov = sector_dir(sector) / "provenance.json"
    src = json.loads(src_prov.read_text()) if src_prov.exists() else {}
    prov = {
        "name": name, "sector": sector, "split": split, "seed": seed,
        "created_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _git_sha(),
        "n_injections": len(inj), "n_astro": int((~inj.is_null).sum()),
        "n_null": int(inj.is_null.sum()), "n_null_events": n_ev,
        "kinds": inj.kind.value_counts().to_dict(),
        "host_rule": "split_of(tic)==split (known excluded); proportional stratified by "
                     "Tmag bin x camera/CCD; no quality/quietness cut; one injection per host",
        "model": "multiplicative; PDCSAP *= m; SAP *= 1 - CROWDSAP*(1-m); errors unchanged; "
                 "model averaged over each 30-min bin (5 sub-samples)",
        "source": {k: src.get(k) for k in ("product", "hlsp_version", "doi",
                                           "download_dates_utc", "spoc_data_release",
                                           "bin_minutes", "quality_bitmask", "n_converted")},
    }  # fmt: skip
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    log(f"wrote {n_written} injections to {out}")
    return prov


def _atomic(table: pa.Table, path: Path, **kw) -> None:
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd", **kw)
    os.replace(tmp, path)
