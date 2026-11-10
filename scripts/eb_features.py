"""Light-curve eclipsing-binary tests (``oddball.knownclass.ebtests``) for a sector.

    uv run python -I scripts/eb_features.py --sector 32 --inj v1 eb1 --workers 6
    uv run python -I scripts/eb_features.py --sector 33 --workers 6          # real stars only
    uv run python -I scripts/eb_features.py --xsec 32 --others 33 38 43 44 55

Writes ``outputs/knownclass/features/ebf_v1/sNNNN/{real,inj_<name>}.parquet`` (keys ``tic,
sector, inj_id``; columns ``eb_*``) and ``provenance.json``. ``--xsec`` adds ``eb_xsec``
(cross-sector support, E7) to a sector's files from the other sectors' real files: for injected
light curves the host's other-sector light curves are used (they hold the host's real behaviour,
not the injection). Shards are processed in parallel by ``--workers`` processes (~0.2 GB each).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oddball.common.paths import REPO_ROOT, data_root, outputs_root, sector_dir
from oddball.knownclass import ebtests as E
from oddball.tess.arrays import dir_metadata, lc_files, sector_grid

NAME = "ebf_v1"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
    except Exception:  # noqa: BLE001
        return "unknown"


def out_dir(sector: int) -> Path:
    return outputs_root() / "knownclass" / "features" / NAME / f"s{sector:04d}"


def _work(args) -> pd.DataFrame:
    d, f, sector, batch_rows = args
    grid = sector_grid(sector)
    meta = dir_metadata(d)
    return E.eb_dir(d, grid, meta, sector, batch_rows, files=[f])


def run_dir(d: Path, sector: int, workers: int, batch_rows: int) -> pd.DataFrame:
    files = lc_files(d)
    jobs = [(d, f, sector, batch_rows) for f in files]
    if workers <= 1:
        parts = [_work(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            parts = []
            for k, p in enumerate(ex.map(_work, jobs)):
                parts.append(p)
                log(f"    {k + 1}/{len(jobs)} files")
    return pd.concat(parts, ignore_index=True)


def add_xsec(sector: int, others: list[int]) -> dict:
    od = out_dir(sector)
    oth = []
    for s in others:
        p = out_dir(s) / "real.parquet"
        if p.exists():
            oth.append(pd.read_parquet(p))
        else:
            log(f"  no {p}: sector {s} skipped")
    o = pd.concat(oth, ignore_index=True) if oth else pd.DataFrame()
    info = {}
    for f in sorted(od.glob("*.parquet")):
        df = pd.read_parquet(f)
        if f.stem == "real":
            df["host_tic"] = df.tic
        else:
            name = f.stem[len("inj_") :]
            truth = pd.concat([pd.read_parquet(p, columns=["inj_id", "host_tic"]) for p in
                               (data_root() / "injections" / name / f"s{sector:04d}").glob(
                                   "*/injections.parquet")], ignore_index=True)  # fmt: skip
            df = df.drop(columns=[c for c in ("host_tic",) if c in df]).merge(
                truth, on="inj_id", how="left", validate="one_to_one"
            )
        df["eb_xsec"] = E.cross_sector(df, o)
        df = df.drop(columns=["host_tic"])
        df.to_parquet(f, index=False)
        info[f.stem] = {"n": len(df), "frac_xsec_gt_0.5": float((df.eb_xsec > 0.5).mean())}
        log(f"  {f.name}: eb_xsec > 0.5 for {(df.eb_xsec > 0.5).mean():.4f}")
    return {
        "xsec_others": [s for s in others if (out_dir(s) / "real.parquet").exists()],
        "xsec_info": info,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sector", type=int)
    ap.add_argument("--inj", nargs="*", default=[])
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch-rows", type=int, default=500)
    ap.add_argument("--xsec", type=int, help="add cross-sector support to this sector's files")
    ap.add_argument("--others", type=int, nargs="*", default=[33, 38, 43, 44, 55])
    a = ap.parse_args()

    if a.xsec is not None:
        info = add_xsec(a.xsec, [s for s in a.others if s != a.xsec])
        pp = out_dir(a.xsec) / "provenance.json"
        prov = json.loads(pp.read_text()) if pp.exists() else {}
        prov.update(info)
        pp.write_text(json.dumps(prov, indent=1))
        return

    od = out_dir(a.sector)
    od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dirs = [] if a.skip_real else [("real", sector_dir(a.sector))]
    for name in a.inj:
        for split in ("val", "test"):
            d = data_root() / "injections" / name / f"s{a.sector:04d}" / split
            if (d / "injections.parquet").exists():
                dirs.append((f"inj_{name}_{split}", d))
    parts = {}
    timing = {}
    for tag, d in dirs:
        t1 = time.time()
        log(f"{tag}: {d}")
        parts[tag] = run_dir(d, a.sector, a.workers, a.batch_rows)
        timing[tag] = round(time.time() - t1)
        log(
            f"  {len(parts[tag])} light curves in {timing[tag]} s; eb_periodic > 0.5: "
            f"{(parts[tag].eb_periodic > 0.5).mean():.4f}"
        )
    if "real" in parts:
        parts["real"].to_parquet(od / "real.parquet", index=False)
    for name in a.inj:
        ps = [parts[k] for k in parts if k.startswith(f"inj_{name}_")]
        if ps:
            pd.concat(ps, ignore_index=True).to_parquet(od / f"inj_{name}.parquet", index=False)
    pp = od / "provenance.json"
    prov = json.loads(pp.read_text()) if pp.exists() else {}
    prov.update({"feature_table": NAME, "sector": a.sector, "git_sha": git_sha(),
                 "built_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                 "source_provenance": str(sector_dir(a.sector) / "provenance.json"),
                 "constants": {k: getattr(E, k) for k in (
                     "EB_BLOCK", "EB_WIN", "EV_SNR", "ANCHOR_SNR", "N_ANCHOR", "P_MIN_D",
                     "HIT_FRAC_DEPTH", "HIT_SIG", "MIN_COVER", "EXTRA_DEPTH", "EXTRA_SNR",
                     "EDGE_D", "SEC_SNR", "SEC_HITFRAC", "FEX_TOL", "WIDTH_TOL", "ASYM_TOL",
                     "EXTRA_FACTOR")}})  # fmt: skip
    prov.setdefault("parts", {}).update({k: len(v) for k, v in parts.items()})
    prov.setdefault("seconds", {}).update(timing)
    pp.write_text(json.dumps(prov, indent=1))
    log(f"done in {time.time() - t0:.0f} s -> {od}")


if __name__ == "__main__":
    main()
