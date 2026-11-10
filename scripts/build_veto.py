"""Build the per-event systematics veto ``cmv2`` for a sector and its injection sets.

    uv run python -I scripts/build_veto.py --sector 32 --inj v1
    uv run python -I scripts/tune_veto.py --sector 32 --inj v1      # picks t_sys / t_res on val

Writes ``outputs/veto/cmv2/sNNNN/{real,inj_<name>}.parquet``: one row per light curve (keys
``tic, inj_id``) with the per-event aggregates of ``oddball.score.veto`` (``cmv2_res_snr``,
``cmv2_sys_snr``, their PDCSAP-only versions, the strongest residual / systematic event's time
and label, ``cmv2_sys_share``, event counts). ``tune_veto.py`` adds ``cmv2_flag``. Also writes
``masks.npz`` (the common-mode, scattered-light and dump masks), ``events_known.parquet`` (every
event of the known-positive light curves, for inspection only) and ``provenance.json``.

Cost on S32: two streaming passes over the real sector plus tess-point positions (cached).
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

from oddball.common.paths import data_root, outputs_root, sector_dir
from oddball.common.splits import split_array
from oddball.score import veto as V
from oddball.tess.arrays import dir_metadata, sector_grid


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_maps(m: V.Maps, p: Path) -> None:
    np.savez_compressed(p, num=m.num, den=m.den, num_cell=m.num_cell, den_cell=m.den_cell,
                        scat=m.scat, dump=m.dump, n_ccd=m.n_ccd, n=m.n)  # fmt: skip


def load_maps(p: Path) -> V.Maps:
    z = np.load(p)
    return V.Maps(z["num"], z["den"], z["num_cell"], z["den_cell"], z["scat"], z["dump"],
                  z["n_ccd"], int(z["n"]))  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sector", type=int, default=32)
    ap.add_argument("--inj", nargs="*", default=["v1"])
    ap.add_argument("--name", default="cmv2")
    ap.add_argument("--batch-rows", type=int, default=2000)
    ap.add_argument("--reuse", action="store_true", help="reuse cached cells / real maps")
    ap.add_argument("--k-mad", type=float, default=V.K_MAD)
    a = ap.parse_args()
    sec = a.sector
    t0 = time.time()
    out = outputs_root() / "veto" / a.name / f"s{sec:04d}"
    cache = out / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    grid = sector_grid(sec)
    sdir = sector_dir(sec)
    meta = dir_metadata(sdir)

    cp = cache / "cells.parquet"
    if a.reuse and cp.exists():
        cells = pd.read_parquet(cp)
    else:
        log("tess-point pixel positions ...")
        cells = V.pixel_cells(meta, sec)
        cells.to_parquet(cp, index=False)
    log(f"cells: {len(cells)} stars, {int((~cells.own_ccd).sum())} not on their own CCD "
        f"per tess-point; stars per pixel cell index median "
        f"{int(cells.groupby('cell').size().median())} (summed over the 16 CCDs)")

    mp = cache / "maps_real.npz"
    if a.reuse and mp.exists():
        maps = load_maps(mp)
        log("reused real maps")
    else:
        log(f"pass 1 (maps) over {sdir}")
        maps = V.scan_maps(sdir, grid, meta, cells, a.batch_rows, log=log)
        save_maps(maps, mp)
    masks = V.real_masks(maps, a.k_mad)
    log(f"masks: {json.dumps({k: v for k, v in masks.info.items() if k != 'k_mad'})}")
    np.savez_compressed(out / "masks.npz", cm_ccd=masks.cm_ccd, cm_cell=masks.cm_cell,
                        scat=masks.scat, dump_bins=masks.dump_bins, g0=grid.g0,
                        bin_days=grid.bin_days, series=np.array(V.SERIES))  # fmt: skip

    log("pass 2 (events) over the real sector")
    known = set(meta.tic[split_array(meta.tic.to_numpy()) == "known"].tolist())
    feats, ev_known = V.classify_dir(sdir, grid, meta, cells, masks, a.batch_rows, log=log,
                                     keep_events=known)  # fmt: skip
    feats.insert(2, "sector", sec)
    feats.to_parquet(out / "real.parquet", index=False)
    if ev_known is not None:
        ev_known.to_parquet(out / "events_known.parquet", index=False)
    log(f"real: {len(feats)} light curves; median events {feats.cmv2_n_ev.median():.0f}, "
        f"median sys share {feats.cmv2_sys_share.median():.2f}")

    set_info = {}
    for name in a.inj:
        parts = []
        for split in ("val", "test"):
            d = data_root() / "injections" / name / f"s{sec:04d}" / split
            if not (d / "injections.parquet").exists():
                log(f"skip {d}: not built")
                continue
            mi = dir_metadata(d)
            own = V.scan_maps(d, grid, mi, cells, a.batch_rows)
            sm = V.set_masks(masks, own)
            set_info[f"{name}/{split}"] = sm.info["set_extra_cells"]
            log(f"  {name}/{split}: {sm.info['set_extra_cells']} extra common-mode cells from "
                "the set's own map")
            f, _ = V.classify_dir(d, grid, mi, cells, sm, a.batch_rows)
            f.insert(2, "sector", sec)
            parts.append(f)
        if parts:
            pd.concat(parts, ignore_index=True).to_parquet(out / f"inj_{name}.parquet",
                                                           index=False)  # fmt: skip
            log(f"wrote inj_{name}")

    prov = {"name": a.name, "sector": sec, "inj": a.inj, "widths_bins": list(V.WIDTHS),
            "anom_snr": V.ANOM_SNR, "ncell": V.NCELL, "k_mad": a.k_mad, "floor": V.FLOOR,
            "min_count": V.MIN_COUNT, "sys_share": V.SYS_SHARE, "scat_bit": V.SCAT_BIT,
            "scat_frac": V.SCAT_FRAC, "dump_bins": masks.dump_bins.tolist(),
            "dump_btjd": [float(grid.time[i]) for i in masks.dump_bins],
            "set_pmax": V.DIR_CM_PMAX, "set_extra_cells": set_info, "masks": masks.info,
            "source_provenance": str(sdir / "provenance.json"),
            "seconds": round(time.time() - t0),
            "built_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}  # fmt: skip
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, default=float))
    log(f"done in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
