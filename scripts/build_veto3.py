"""Build the per-event veto ``cmv3`` for a sector and its injection sets (ML_DESIGN §13).

    uv run python -I scripts/build_veto3.py --sector 32 --inj v1 n2 --real-masks-only n2
    uv run python -I scripts/tune_veto3.py --sector 32          # chooses labels and thresholds

Reuses ``cmv2``'s tess-point positions and real common-mode maps
(``outputs/veto/cmv2/sNNNN/_cache``), so the cmv2 masks are identical. Three passes per light
curve set (``oddball.score.veto3``): scan (events, track points, per-CCD statistics), cross-star
labels (asteroid tracks, same-column events) and classify (all labels, stage 2 for every
configuration in ``veto3.CONFIGS``).

Writes ``outputs/veto/cmv3/sNNNN/``: ``real.parquet`` and ``inj_<name>.parquet`` (one row per
light curve, ``<config>_{res,sys}_snr[_pdc]`` and strongest-event columns per configuration, plus
per-label event counts for ``all``), ``tracks_real.parquet`` / ``windows_real.parquet`` (the
asteroid tracks found and the light-curve windows they cover), ``events_watch.parquet`` (every
event of the known positives and of ``--watch`` TICs, for inspection only), ``provenance.json``
(with the chance-track null: the same track search on time-shuffled points).

``--real-masks-only``: injection sets judged with the real sector's masks only, without the
set's own binomial CCD map. The ``n2`` nulls are local (a track, a column, per-star steps); the
set's own map would see e.g. a column's ~15 hosts as a CCD-wide event, which the real sky would
not show.
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
from oddball.score import veto3 as V3
from oddball.tess.arrays import dir_metadata, sector_grid


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_maps(p: Path) -> V.Maps:
    z = np.load(p)
    return V.Maps(z["num"], z["den"], z["num_cell"], z["den_cell"], z["scat"], z["dump"],
                  z["n_ccd"], int(z["n"]))  # fmt: skip


def save_scan(cache: Path, tag: str, scan) -> None:
    lcs, ev, pts, st = scan
    lcs.to_parquet(cache / f"lcs_{tag}.parquet", index=False)
    ev.to_parquet(cache / f"events_{tag}.parquet", index=False)
    pts.to_parquet(cache / f"points_{tag}.parquet", index=False)
    np.savez_compressed(cache / f"stats_{tag}.npz", slope_hist=st.slope_hist, qbit=st.qbit,
                        n_ccd=st.n_ccd)  # fmt: skip


def load_scan(cache: Path, tag: str):
    z = np.load(cache / f"stats_{tag}.npz")
    return (pd.read_parquet(cache / f"lcs_{tag}.parquet"),
            pd.read_parquet(cache / f"events_{tag}.parquet"),
            pd.read_parquet(cache / f"points_{tag}.parquet"),
            V3.ScanStats(z["slope_hist"], z["qbit"], z["n_ccd"]))  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sector", type=int, default=32)
    ap.add_argument("--inj", nargs="*", default=["v1", "n2"])
    ap.add_argument("--real-masks-only", nargs="*", default=["n2"])
    ap.add_argument("--name", default="cmv3")
    ap.add_argument("--batch-rows", type=int, default=2000)
    ap.add_argument("--reuse", action="store_true", help="reuse the cached real scan")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="classify only these veto3.CONFIGS (default: all 18, as tuned on S32). "
                         "Sectors with a fixed rule need only 'cmv2' (reproduction check) and "
                         "the chosen one ('all'); ~9x faster")
    ap.add_argument("--watch", type=Path, default=None,
                    help="CSV with a tic column: keep their full event lists (inspection)")
    a = ap.parse_args()
    if a.configs:
        bad = set(a.configs) - set(V3.CONFIGS)
        if bad:
            raise SystemExit(f"unknown configs {sorted(bad)}")
    configs = {k: V3.CONFIGS[k] for k in (a.configs or V3.CONFIGS)}
    sec = a.sector
    t0 = time.time()
    out = outputs_root() / "veto" / a.name / f"s{sec:04d}"
    cache = out / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    c2 = outputs_root() / "veto" / "cmv2" / f"s{sec:04d}" / "_cache"
    grid = sector_grid(sec)
    sdir = sector_dir(sec)
    meta = dir_metadata(sdir)
    cells = pd.read_parquet(c2 / "cells.parquet")
    masks = V.real_masks(load_maps(c2 / "maps_real.npz"))
    log(f"cmv2 masks reused: dump bins {masks.dump_bins.tolist()}")

    # ---- real sector
    if a.reuse and (cache / "stats_real.npz").exists():
        scan_r = load_scan(cache, "real")
        log("reused real scan")
    else:
        log(f"pass A (scan) over {sdir}")
        scan_r = V3.scan_dir(sdir, grid, meta, cells, masks, a.batch_rows, log=log)
        save_scan(cache, "real", scan_r)
    lcs_r, ev_r, pts_r, st_r = scan_r
    sq = st_r.slope_q()
    log(f"real scan: {len(lcs_r)} light curves, {len(ev_r)} events, {len(pts_r)} track points "
        f"({int((pts_r.kind == 1).sum())} background bumps); slope q90 per CCD "
        f"{np.round(np.log10(sq), 2).tolist()} (log10); QUALITY-window bins per CCD "
        f"{st_r.qwin().sum(1).tolist()}")
    log("pass B (cross-star labels) on the real sector")
    lab_r, win_r, tr_r = V3.cross_labels(lcs_r, ev_r, pts_r, cells, st_r.n_ccd, grid.length,
                                         log=log)  # fmt: skip
    tr_r.to_parquet(out / "tracks_real.parquet", index=False)
    win_r.merge(lcs_r[["lc", "tic"]], on="lc").to_parquet(out / "windows_real.parquet",
                                                          index=False)  # fmt: skip
    log(f"real cross labels: {int(lab_r.track.sum())} track events on "
        f"{lab_r[lab_r.track].lc.nunique()} light curves; {int(lab_r.column.sum())} column "
        f"events on {lab_r[lab_r.column].lc.nunique()} light curves")
    # chance-track null: the same search with point times shuffled within each CCD
    stars = V3._with_pos(lcs_r[["lc"]], lcs_r, cells)
    pts_pos = V3._with_pos(pts_r, lcs_r, cells).rename(columns={"tic": "star"})
    null_info = {}
    for seed in (1, 2):
        w0, t0_ = V3.track_labels(pts_pos, stars, st_r.n_ccd, shuffle_seed=seed)
        null_info[f"seed{seed}"] = {"track_pairs": len(t0_),
                                    "lcs_in_windows": int(w0.lc.nunique()) if len(w0) else 0}
    log(f"chance-track null (shuffled times): {null_info}; real: {len(tr_r)} track pairs, "
        f"{win_r.lc.nunique() if len(win_r) else 0} light curves in windows")

    known = set(meta.tic[split_array(meta.tic.to_numpy()) == "known"].tolist())
    if a.watch:
        known |= set(pd.read_csv(a.watch).tic.astype(int).tolist())
    log("pass C (classify) over the real sector")
    feats, ev_watch = V3.classify_dir(sdir, grid, meta, cells, masks, st_r, lab_r,
                                      configs=configs, batch_rows=a.batch_rows, log=log,
                                      keep_events=known)  # fmt: skip
    feats.insert(2, "sector", sec)
    feats.to_parquet(out / "real.parquet", index=False)
    if ev_watch is not None:
        ev_watch.to_parquet(out / "events_watch.parquet", index=False)
    # sanity: the cmv2 configuration must reproduce the cmv2 build
    p2 = outputs_root() / "veto" / "cmv2" / f"s{sec:04d}" / "real.parquet"
    chk = {}
    if p2.exists() and "cmv2" in configs:
        r2 = pd.read_parquet(p2, columns=["tic", "cmv2_res_snr", "cmv2_sys_snr",
                                          "cmv2_res_snr_pdc", "cmv2_sys_snr_pdc"])  # fmt: skip
        j = feats[["tic", "cmv2_res_snr", "cmv2_sys_snr", "cmv2_res_snr_pdc",
                   "cmv2_sys_snr_pdc"]].merge(r2, on="tic", suffixes=("", "_ref"))  # fmt: skip
        for k in ("res_snr", "sys_snr", "res_snr_pdc", "sys_snr_pdc"):
            chk[k] = float(np.abs(j[f"cmv2_{k}"] - j[f"cmv2_{k}_ref"]).max())
        log(f"cmv2 reproduction, max |diff|: {chk}")

    # ---- injection sets
    set_info = {}
    for name in a.inj:
        parts = []
        for split in ("val", "test"):
            d = data_root() / "injections" / name / f"s{sec:04d}" / split
            if not (d / "injections.parquet").exists():
                log(f"skip {d}: not built")
                continue
            mi = dir_metadata(d)
            if name in a.real_masks_only:
                sm = masks
                extra = 0
            else:
                own = V.scan_maps(d, grid, mi, cells, a.batch_rows)
                sm = V.set_masks(masks, own)
                extra = sm.info["set_extra_cells"]
            scan_i = V3.scan_dir(d, grid, mi, cells, sm, a.batch_rows)
            lab_i, _, tr_i = V3.cross_labels(scan_i[0], scan_i[1], scan_i[2], cells,
                                                 st_r.n_ccd, grid.length,
                                                 ref=(lcs_r, ev_r, pts_r))  # fmt: skip
            f, _ = V3.classify_dir(d, grid, mi, cells, sm, st_r, lab_i, configs=configs,
                                   batch_rows=a.batch_rows)  # fmt: skip
            f.insert(2, "sector", sec)
            parts.append(f)
            set_info[f"{name}/{split}"] = {
                "set_extra_cm_cells": extra, "real_masks_only": name in a.real_masks_only,
                "track_events": int(lab_i.track.sum()), "column_events": int(lab_i.column.sum()),
                "track_pairs": len(tr_i)}  # fmt: skip
            log(f"  {name}/{split}: {set_info[f'{name}/{split}']}")
        if parts:
            pd.concat(parts, ignore_index=True).to_parquet(out / f"inj_{name}.parquet",
                                                           index=False)  # fmt: skip
            log(f"wrote inj_{name}")

    prov = {"name": a.name, "sector": sec, "inj": a.inj, "real_masks_only": a.real_masks_only,
            "labels": list(V3.LABELS), "configs": {k: list(v) for k, v in configs.items()},
            "params": {k: getattr(V3, k) for k in (
                "SAP_MIN_SNR", "BKG_FRAC", "RAW_SNR", "PDC_RATIO", "PDC_SIG", "GAP_BINS",
                "GAP_DUR", "SLOPE_Q", "QBITS", "QB_FRAC", "POINT_SNR", "POINT_MAXLEN", "TMAX",
                "VMIN", "VMAX", "TOL", "SUPPORT_WIN", "CROWD_FRAC", "TRACK_R", "TRACK_MARGIN",
                "COL_HALF", "COL_CTRL", "COL_MIN", "COL_P")},
            "slope_q90_per_ccd": st_r.slope_q().tolist(),
            "qwin_bins_per_ccd": st_r.qwin().sum(1).tolist(),
            "real_counts": {"events": len(ev_r), "points": len(pts_r),
                            "track_pairs": len(tr_r),
                            "track_lcs": int(lab_r[lab_r.track].lc.nunique()),
                            "column_lcs": int(lab_r[lab_r.column].lc.nunique())},
            "chance_track_null": null_info, "cmv2_reproduction_maxdiff": chk,
            "sets": set_info, "cmv2_cache": str(c2),
            "source_provenance": str(sdir / "provenance.json"),
            "seconds": round(time.time() - t0),
            "built_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}  # fmt: skip
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, default=float))
    log(f"done in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
