"""Apply an existing per-event veto (``cmv2``) to one more injection set, without touching the
real-star files.

    uv run python -I scripts/veto_inj_set.py --sector 32 --inj eb1
    uv run python -I scripts/veto_inj_set.py --sector 32 --inj yso1 --real-masks-only

``build_veto.py`` re-runs its pass over the real sector and rewrites ``real.parquet`` (dropping
``cmv2_flag``, which ``tune_veto.py`` adds). This script instead reuses the cached real maps
(``_cache/maps_real.npz``) and pixel cells, builds the set's own masks exactly as
``build_veto.py`` does (``oddball.score.veto.set_masks``), classifies its events, and applies the
flag thresholds already chosen on val (``provenance.json["flag"]``). It writes only
``outputs/veto/<name>/sNNNN/inj_<set>.parquet`` and records the set in ``provenance.json``.

``--real-masks-only`` skips the set's own binomial CCD map (as ``build_veto3.py`` does for
``n2``). Use it for sets with replicas (``yso1``, ML_DESIGN §15): copies of one host share its
real events at the same bins and would otherwise look like a CCD-wide event.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oddball.common.paths import data_root, outputs_root
from oddball.score import veto as V
from oddball.tess.arrays import dir_metadata, sector_grid


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sector", type=int, default=32)
    ap.add_argument("--inj", required=True)
    ap.add_argument("--name", default="cmv2")
    ap.add_argument("--batch-rows", type=int, default=2000)
    ap.add_argument("--real-masks-only", action="store_true",
                    help="judge the set with the real sector's masks only")
    a = ap.parse_args()
    sec = a.sector
    vd = outputs_root() / "veto" / a.name / f"s{sec:04d}"
    prov = json.loads((vd / "provenance.json").read_text())
    flag = prov["flag"]
    z = np.load(vd / "_cache" / "maps_real.npz")
    maps = V.Maps(z["num"], z["den"], z["num_cell"], z["den_cell"], z["scat"], z["dump"],
                  z["n_ccd"], int(z["n"]))  # fmt: skip
    masks = V.real_masks(maps, prov.get("k_mad", V.K_MAD))
    cells = pd.read_parquet(vd / "_cache" / "cells.parquet")
    grid = sector_grid(sec)
    parts, extra = [], {}
    for split in ("val", "test"):
        d = data_root() / "injections" / a.inj / f"s{sec:04d}" / split
        if not (d / "injections.parquet").exists():
            continue
        mi = dir_metadata(d)
        if a.real_masks_only:
            sm = masks
            extra[f"{a.inj}/{split}"] = 0
        else:
            own = V.scan_maps(d, grid, mi, cells, a.batch_rows)
            sm = V.set_masks(masks, own)
            extra[f"{a.inj}/{split}"] = sm.info["set_extra_cells"]
        f, _ = V.classify_dir(d, grid, mi, cells, sm, a.batch_rows)
        f.insert(2, "sector", sec)
        parts.append(f)
        log(f"{a.inj}/{split}: {len(f)} light curves, {extra[f'{a.inj}/{split}']} extra cells")
    out = pd.concat(parts, ignore_index=True)
    out["cmv2_flag"] = V.veto_flag(out, flag["t_sys"], flag["t_res"], flag["pdc_only"])
    out.to_parquet(vd / f"inj_{a.inj}.parquet", index=False)
    prov.setdefault("extra_sets", {})[a.inj] = {
        "set_extra_cells": extra,
        "real_masks_only": bool(a.real_masks_only),
        "flag_frac": float(out.cmv2_flag.mean()),
        "built_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "script": "scripts/veto_inj_set.py (real maps reused; flag thresholds from 'flag')",
    }
    (vd / "provenance.json").write_text(json.dumps(prov, indent=1, default=float))
    log(f"wrote {vd / f'inj_{a.inj}.parquet'}: flagged {out.cmv2_flag.mean():.3f}")


if __name__ == "__main__":
    main()
