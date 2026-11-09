"""Build catalogue labels for the known-class (explain-away) score (docs/ML_DESIGN.md §4.5).

    uv run python -I scripts/build_known_class_labels.py --fetch vsx prsa gaia
    uv run python -I scripts/build_known_class_labels.py --build

Targets are every TIC in the sector metadata tables (``--sectors``, default 32 33 38 43 44 55).
``--fetch`` queries the anonymous services (resumable; cached under
``<data>/knownclass/raw/<source>/`` with ``provenance.json``). ``--build`` maps catalogue types
to known classes (``oddball.knownclass.labels``), applies the hard exclusions (excluded types,
``docs/known_positives.csv``, every injection host under ``<data>/injections``) and writes
``<data>/knownclass/labels.parquet`` + ``labels_provenance.json``. Labels are per TIC; the
quiet class is assigned per light curve at training time.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oddball.common.paths import data_root, sector_dir
from oddball.common.splits import known_positive_tics, split_array
from oddball.knownclass import catalogs as K
from oddball.knownclass import labels as L

SECTORS = (32, 33, 38, 43, 44, 55)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def targets(sectors) -> pd.DataFrame:
    parts = []
    for s in sectors:
        m = pd.read_parquet(sector_dir(s) / "metadata.parquet", columns=["tic", "ra", "dec", "tmag"])
        parts.append(m)
    t = pd.concat(parts, ignore_index=True).drop_duplicates("tic").sort_values("tic")
    return t.reset_index(drop=True)


def injection_hosts() -> dict[int, str]:
    out = {}
    for p in sorted((data_root() / "injections").glob("*/s*/*/injections.parquet")):
        for t in pd.read_parquet(p, columns=["host_tic"]).host_tic.unique():
            out[int(t)] = "injection_host"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sectors", type=int, nargs="+", default=list(SECTORS))
    ap.add_argument("--fetch", nargs="*", default=[], choices=["vsx", "prsa", "gaia", "gaia_vclass"])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--vsx-chunk", type=int, default=5000)
    ap.add_argument("--gaia-chunk", type=int, default=20000)
    a = ap.parse_args()

    tg = targets(a.sectors)
    log(f"{len(tg)} unique TICs in sectors {a.sectors}")
    for src in a.fetch:
        log(f"fetch {src}")
        if src == "vsx":
            K.fetch_vsx(tg, chunk=a.vsx_chunk, log=log)
        elif src == "prsa":
            K.fetch_prsa(log=log)
        elif src == "gaia":
            K.fetch_gaia(tg, chunk=a.gaia_chunk, log=log)
        elif src == "gaia_vclass":
            K.fetch_gaia_vclass(tg, chunk=a.vsx_chunk, log=log)
    if not a.build:
        return

    def maybe(src):
        try:
            return K.load_raw(src)
        except FileNotFoundError as e:
            log(f"WARNING {e}")
            return None

    vsx, prsa, gaia = maybe("vsx"), maybe("prsa"), maybe("gaia")
    if gaia is None:  # fall back to VizieR's copy of the DR3 classifier table
        vc = maybe("gaia_vclass")
        gaia = K.gaia_from_vclass(vc) if vc is not None and len(vc) else None
    excl = injection_hosts()
    excl.update({t: "known_positive" for t in known_positive_tics()})
    lab = L.build_labels(tg.tic.to_numpy(), tg.tmag.to_numpy(), vsx, prsa, gaia, excl)
    lab["split"] = split_array(lab.tic.to_numpy())
    out = data_root() / "knownclass" / "labels.parquet"
    lab.to_parquet(out, index=False)

    counts = (lab[lab.label != ""].groupby(["label", "label_source"]).size()
              .unstack(fill_value=0))  # fmt: skip
    prov = {
        "built_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "sectors": a.sectors,
        "n_tics": len(lab),
        "sources": {s: json.loads((K.raw_dir(s) / "provenance.json").read_text())
                    for s in ("vsx", "prsa", "gaia", "gaia_vclass")
                    if (K.raw_dir(s) / "provenance.json").exists()},
        "label_counts": {c: {k: int(v) for k, v in r.items()} for c, r in counts.iterrows()},
        "excluded": lab[lab.excluded].exclude_reason.value_counts().to_dict(),
        "flagged_any": int(lab.flagged_any.sum()),
        "exclusion_rules": {"vsx_exact": sorted(L.VSX_EXCLUDE_EXACT),
                            "vsx_prefix": list(L.VSX_EXCLUDE_PREFIX),
                            "gaia_classes": sorted(L.GAIA_EXCLUDE)},
        "radii_arcsec": {"vsx_label": L.VSX_LABEL_RADIUS, "vsx_exclude": K.VSX_RADIUS,
                         "gaia_label": L.GAIA_LABEL_RADIUS, "gaia_exclude": K.GAIA_RADIUS},
    }  # fmt: skip
    (data_root() / "knownclass" / "labels_provenance.json").write_text(json.dumps(prov, indent=1))
    log(f"labels -> {out}")
    print(counts.assign(total=counts.sum(1)).to_string())
    print("excluded:", prov["excluded"])
    print("labelled by split:", lab[lab.label != ""].split.value_counts().to_dict())
    print("sources disagree on", int((lab.n_sources_disagree > 0).sum()), "labelled TICs")


if __name__ == "__main__":
    main()
