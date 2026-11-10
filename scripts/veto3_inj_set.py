"""Apply the tuned ``cmv3`` veto to more injection sets, without touching the real-star files.

    uv run python -I scripts/veto3_inj_set.py --sector 32 --inj eb1
    uv run python -I scripts/veto3_inj_set.py --sector 32 --inj yso1 --real-masks-only yso1

``build_veto3.py`` re-runs pass C over the whole real sector (~30 min) and rewrites
``real.parquet`` without ``cmv3_flag`` (``tune_veto3.py`` adds it). This script instead reuses
the cached real scan (``outputs/veto/cmv3/sNNNN/_cache/{lcs,events,points,stats}_real``) and the
``cmv2`` masks, runs the three passes on the set exactly as ``build_veto3.py`` does (scan,
cross-star labels against the real sector minus the hosts' real copies, classify), and applies
the flag already chosen on val (``provenance.json["flag"]``: configuration, t_sys, t_res,
PDCSAP-only). It writes only ``outputs/veto/cmv3/sNNNN/inj_<set>.parquet`` and records the set
under ``provenance.json["extra_sets"]``.

Sets with a ``replica`` column (``yso1``, ML_DESIGN §15) carry several copies of each host. The
cross-star labels (``track``, ``column``) are then computed **per replica**, so that a host's
copies never count as each other's neighbours; the matched nulls (``replica = -1``) form their
own group. Such sets should also use ``--real-masks-only``.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_veto3 import load_maps, load_scan

from oddball.common.paths import data_root, outputs_root
from oddball.score import veto as V
from oddball.score import veto3 as V3
from oddball.tess.arrays import dir_metadata, sector_grid

COPY = ("res_snr", "sys_snr", "res_snr_pdc", "sys_snr_pdc", "res_t", "res_ev", "sys_t", "sys_ev")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cross_by_group(scan, groups: pd.Series, cells, n_ccd, L, ref):
    """``V3.cross_labels`` run separately on each group of light curves (by ``inj_id``)."""
    lcs, ev, pts, _ = scan
    g = lcs.inj_id.map(groups)
    if g.isna().any():
        raise SystemExit(f"{int(g.isna().sum())} light curves have no replica")
    labs, n_tr, n_pairs = [], 0, 0
    for idx in lcs.groupby(g.to_numpy()).groups.values():
        sl = lcs.loc[idx]
        keep = set(sl.lc.tolist())
        lab, _, tr = V3.cross_labels(sl, ev[ev.lc.isin(keep)], pts[pts.lc.isin(keep)], cells,
                                     n_ccd, L, ref=ref)  # fmt: skip
        labs.append(lab)
        n_pairs += len(tr)
        n_tr += int(lab.track.sum()) if len(lab) else 0
    lab = pd.concat(labs, ignore_index=True) if labs else pd.DataFrame(
        columns=["lc", "ch", "sign", "w", "start", "track", "column"])
    return lab, n_pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sector", type=int, default=32)
    ap.add_argument("--inj", nargs="+", required=True)
    ap.add_argument("--real-masks-only", nargs="*", default=[])
    ap.add_argument("--name", default="cmv3")
    ap.add_argument("--batch-rows", type=int, default=2000)
    ap.add_argument("--check", action="store_true",
                    help="compare with the existing inj_<set>.parquet instead of writing")
    a = ap.parse_args()
    sec = a.sector
    vd = outputs_root() / "veto" / a.name / f"s{sec:04d}"
    prov = json.loads((vd / "provenance.json").read_text())
    flag = prov["flag"]
    cfg, ts, tr, po = flag["config"], flag["t_sys"], flag["t_res"], flag["pdc_only"]
    c2 = outputs_root() / "veto" / "cmv2" / f"s{sec:04d}" / "_cache"
    cells = pd.read_parquet(c2 / "cells.parquet")
    masks = V.real_masks(load_maps(c2 / "maps_real.npz"))
    grid = sector_grid(sec)
    lcs_r, ev_r, pts_r, st_r = load_scan(vd / "_cache", "real")
    log(f"real scan reused: {len(lcs_r)} light curves, {len(ev_r)} events; flag {cfg} "
        f"t_sys {ts} t_res {tr} pdc_only {po}")
    for name in a.inj:
        parts, info = [], {}
        for split in ("val", "test"):
            d = data_root() / "injections" / name / f"s{sec:04d}" / split
            if not (d / "injections.parquet").exists():
                log(f"skip {d}: not built")
                continue
            mi = dir_metadata(d)
            truth = pd.read_parquet(d / "injections.parquet")
            if name in a.real_masks_only:
                sm, extra = masks, 0
            else:
                own = V.scan_maps(d, grid, mi, cells, a.batch_rows)
                sm = V.set_masks(masks, own)
                extra = sm.info["set_extra_cells"]
            scan_i = V3.scan_dir(d, grid, mi, cells, sm, a.batch_rows)
            ref = (lcs_r, ev_r, pts_r)
            if "replica" in truth:
                groups = truth.set_index("inj_id").replica
                lab_i, n_pairs = cross_by_group(scan_i, groups, cells, st_r.n_ccd, grid.length,
                                                ref)  # fmt: skip
                grouping = "per replica"
            else:
                lab_i, _, tr_i = V3.cross_labels(scan_i[0], scan_i[1], scan_i[2], cells,
                                                 st_r.n_ccd, grid.length, ref=ref)  # fmt: skip
                n_pairs, grouping = len(tr_i), "whole set"
            f, _ = V3.classify_dir(d, grid, mi, cells, sm, st_r, lab_i, batch_rows=a.batch_rows)
            f.insert(2, "sector", sec)
            parts.append(f)
            info[f"{name}/{split}"] = {
                "set_extra_cm_cells": extra, "real_masks_only": name in a.real_masks_only,
                "cross_labels": grouping,
                "track_events": int(lab_i.track.sum()) if len(lab_i) else 0,
                "column_events": int(lab_i.column.sum()) if len(lab_i) else 0,
                "track_pairs": n_pairs}  # fmt: skip
            log(f"  {name}/{split}: {len(f)} light curves; {info[f'{name}/{split}']}")
        if not parts:
            continue
        out = pd.concat(parts, ignore_index=True)
        out["cmv3_flag"] = V3.veto_flag(out, cfg, ts, tr, po)
        for c in COPY:
            out[f"cmv3_{c}"] = out[f"{cfg}_{c}"]
        if a.check:  # reproduction test against the published file
            old = pd.read_parquet(vd / f"inj_{name}.parquet")
            j = old.merge(out, on="inj_id", suffixes=("_old", ""))
            nf = int((j.cmv3_flag_old != j.cmv3_flag).sum())
            dmax = {c: float(np.nanmax(np.abs(j[f"cmv3_{c}_old"] - j[f"cmv3_{c}"])))
                    for c in ("res_snr", "sys_snr", "res_snr_pdc", "sys_snr_pdc")}  # fmt: skip
            log(f"check {name}: {len(j)} of {len(old)} rows matched; flag mismatches {nf}; "
                f"max |diff| {dmax}")
            continue
        out.to_parquet(vd / f"inj_{name}.parquet", index=False)
        prov.setdefault("extra_sets", {})[name] = {
            "sets": info, "flag_frac": float(out.cmv3_flag.mean()),
            "built_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "script": "scripts/veto3_inj_set.py (cached real scan; flag from 'flag')"}  # fmt: skip
        log(f"wrote inj_{name}: flagged {out.cmv3_flag.mean():.3f} "
            f"({int(out.cmv3_flag.sum())} of {len(out)})")
        (vd / "provenance.json").write_text(json.dumps(prov, indent=1, default=float))


if __name__ == "__main__":
    main()
