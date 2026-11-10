"""Build the young-star host injection set ``yso1`` (docs/ML_DESIGN.md §15).

    uv run python -I scripts/make_yso_set.py --sector 32 --name yso1 \
        --exclude-sets v1 eb1 n2 --replicas 20

Hosts: catalogued young stars (VSX young/irregular types within 21", Gaia DR3 YSO within 3"; the
types ``kc_v1`` hard-excludes) of the val and test splits, not near a known positive, not an
earlier injection host (``oddball.inject.yso``). Each host carries ``--replicas`` dipper-style
injections and one unchanged copy (the matched null). Catalogue inputs are the cached cone
results under ``<data>/knownclass/raw/{vsx,gaia_vclass}`` (their ``provenance.json`` records the
service, mirror and query dates); nothing is downloaded here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from oddball.common.paths import data_root, sector_dir
from oddball.inject.yso import build_host_set, young_host_pool
from oddball.knownclass import catalogs as K
from oddball.tess.arrays import dir_metadata


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sector", type=int, default=32)
    ap.add_argument("--name", default="yso1")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--seed", type=int, default=20261111)
    ap.add_argument("--replicas", type=int, default=20)
    ap.add_argument("--exclude-sets", nargs="*", default=["v1", "eb1", "n2"])
    a = ap.parse_args()
    excl = set()
    for name in a.exclude_sets:
        for p in (data_root() / "injections" / name / f"s{a.sector:04d}").glob(
                "*/injections.parquet"):
            excl.update(pd.read_parquet(p, columns=["host_tic"]).host_tic.astype(int).tolist())
    meta = dir_metadata(sector_dir(a.sector))
    pool, counts = young_host_pool(meta, K.load_raw("vsx"), K.load_raw("gaia_vclass"),
                                   splits=tuple(a.splits), exclude_tics=excl)  # fmt: skip
    print(json.dumps(counts, indent=1, default=int))
    print(pool.groupby(["split", "yso_source"]).size().to_string())
    cat_prov = {s: json.loads((K.raw_dir(s) / "provenance.json").read_text())
                for s in ("vsx", "gaia_vclass")}  # fmt: skip
    for split in a.splits:
        t0 = time.time()
        hosts = pool[pool.split == split]
        print(f"== {a.name} s{a.sector:04d} {split}: {len(hosts)} hosts")
        prov = build_host_set(a.sector, split, hosts, name=a.name, seed=a.seed,
                              replicas=a.replicas)  # fmt: skip
        prov["pool_counts"] = counts
        prov["exclude_sets"] = a.exclude_sets
        prov["catalogues"] = {s: {k: v.get(k) for k in ("service", "catalog", "radius_arcsec",
                                                         "first_query_utc", "last_query_utc")}
                              for s, v in cat_prov.items()}  # fmt: skip
        out = data_root() / "injections" / a.name / f"s{a.sector:04d}" / split
        (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
        print(f"   {prov['n_astro']} injections + {prov['n_null']} nulls in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
