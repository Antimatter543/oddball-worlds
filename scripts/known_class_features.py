"""Build the ``kcf_v1`` feature table (oddball.knownclass.features) for a sector + injection sets.

    uv run python -I scripts/known_class_features.py --sector 32 --inj v1
    uv run python -I scripts/known_class_features.py --sector 33 --inj

Writes ``outputs/knownclass/features/kcf_v1/sNNNN/{real,inj_<name>}.parquet`` (one row per light
curve, keys ``tic, sector, inj_id``) and ``provenance.json``. ~11 min and ~0.6 GB peak per
10-min sector on this machine (1000-row batches).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oddball.common.paths import REPO_ROOT, data_root, outputs_root, sector_dir
from oddball.knownclass.features import featurize_dir
from oddball.tess.arrays import dir_metadata, sector_grid

NAME = "kcf_v1"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sector", type=int, required=True)
    ap.add_argument("--inj", nargs="*", default=[])
    ap.add_argument("--batch-rows", type=int, default=1000)
    ap.add_argument("--skip-real", action="store_true")
    a = ap.parse_args()

    out = outputs_root() / "knownclass" / "features" / NAME / f"s{a.sector:04d}"
    out.mkdir(parents=True, exist_ok=True)
    grid = sector_grid(a.sector)
    t0 = time.time()
    dirs = [] if a.skip_real else [("real", sector_dir(a.sector))]
    for name in a.inj:
        for split in ("val", "test"):
            d = data_root() / "injections" / name / f"s{a.sector:04d}" / split
            if (d / "injections.parquet").exists():
                dirs.append((f"inj_{name}_{split}", d))
    frames = {}
    for tag, d in dirs:
        meta = dir_metadata(d)
        log(f"{tag}: {d} ({len(meta)} light curves)")
        frames[tag] = featurize_dir(d, grid, meta, a.sector, a.batch_rows,
                                    log=log if tag == "real" else None)  # fmt: skip
    if "real" in frames:
        frames["real"].to_parquet(out / "real.parquet", index=False)
    for name in a.inj:
        parts = [frames[k] for k in frames if k.startswith(f"inj_{name}_")]
        if parts:
            pd.concat(parts, ignore_index=True).to_parquet(out / f"inj_{name}.parquet", index=False)
    prov_p = out / "provenance.json"
    prov = json.loads(prov_p.read_text()) if prov_p.exists() else {}
    prov.update({"feature_table": NAME, "sector": a.sector, "git_sha": git_sha(),
                 "built_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                 "grid": [grid.g0, grid.length, grid.bin_days],
                 "source_provenance": str(sector_dir(a.sector) / "provenance.json")})
    prov.setdefault("parts", {}).update({k: len(v) for k, v in frames.items()})
    prov_p.write_text(json.dumps(prov, indent=1))
    log(f"done in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
