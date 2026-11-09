"""Score a sector (and its injection sets) with the classical baseline.

    ODDBALL_DATA=... uv run python -I scripts/score_classical.py --sector 32 --inj v1

Writes, per docs/ML_DESIGN.md §9, ``outputs/scores/<scorer>/sNNNN/{real,inj_<name>}.parquet``
for three variants:

* ``classical``      rank-average of the sub-score percentiles (the pre-registered baseline).
* ``classical_tail`` the same sub-scores combined so the most extreme ones dominate
                     (mean of ``-log(1 - p)``); an ablation of the combination rule.
* ``classical_cm``   the systematics filter v0: common-mode bins (per channel x CCD) and bins
                     next to each star's momentum-dump flags are **removed**, every feature is
                     recomputed, then rank-averaged as in ``classical``. Comparing it with
                     ``classical`` is the "with / without common-mode" test.

Every score file also carries the unmasked pass's ``cm_frac``, ``dump_frac``, ``sys_frac``,
``sys_flag`` and ``n_anom`` (the per-light-curve systematics feature and flag).

Calibration (sub-score and combined percentiles) is always against the **real** light curves
of the sector, within Tmag bin x camera/CCD strata (each variant against its own real pass).
The common-mode map for an injection set is the real map OR-ed with the set's own map,
thresholded against the set's own typical level, so an instrument event shared by the set's
hosts on one CCD is visible to it, while the set's scattered astrophysical signals are not.

Cost: two passes over the real sector (~8 min each for S32 on this machine), seconds for
injection sets. ``--reuse-real`` reuses the cached real passes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oddball.common.paths import data_root, outputs_root, sector_dir
from oddball.score import classical as C
from oddball.tess.arrays import dir_metadata, sector_grid

VARIANTS = {  # variant -> (feature pass, combine rule)
    "classical": ("raw", "mean"),
    "classical_tail": ("raw", "tail"),
    "classical_cm": ("masked", "mean"),
}
DIR_CM_MIN_FRAC = 0.05  # floor on an injection set's own common-mode threshold (and >= 5 LCs)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_pass(res: C.DirResult, path: Path) -> None:
    arrs = {f"{ch}_{k}": v for ch, sp in res.anom.items() for k, v in sp.items()}
    np.savez_compressed(path.with_suffix(".npz"), ccd=res.ccd, map_num=res.map_num,
                        map_den=res.map_den, **arrs)  # fmt: skip
    res.feats.to_parquet(path.with_suffix(".parquet"), index=False)


def load_pass(path: Path) -> C.DirResult:
    z = np.load(path.with_suffix(".npz"))
    anom = {ch: {k: z[f"{ch}_{k}"] for k in ("rows", "cols", "w", "dump")} for ch in C.CHANNELS}
    return C.DirResult(pd.read_parquet(path.with_suffix(".parquet")), anom, z["ccd"],
                       z["map_num"], z["map_den"])  # fmt: skip


def anomaly_fraction(res: C.DirResult) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(res.map_den > 0, res.map_num / res.map_den, 0.0)


class Calibrator:
    """Sub-score and combined-score ECDFs per stratum, fitted on real light curves only."""

    def __init__(self, hows: tuple[str, ...]):
        self.hows = hows

    def fit(self, sub: pd.DataFrame, strata: np.ndarray) -> Calibrator:
        self.cols = list(sub.columns)
        self.sub = C.StratumECDF().fit(sub.to_numpy(), strata)
        pct = self.sub.transform(sub.to_numpy(), strata)
        self.comb = {h: C.StratumECDF().fit(C.combine(pct, h)[:, None], strata)
                     for h in self.hows}  # fmt: skip
        return self

    def transform(self, sub: pd.DataFrame, strata: np.ndarray):
        pct = self.sub.transform(sub[self.cols].to_numpy(), strata)
        return pct, {h: self.comb[h].transform(C.combine(pct, h)[:, None], strata)[:, 0]
                     for h in self.hows}  # fmt: skip


def score_frame(base: pd.DataFrame, pct, cols, score, sysf) -> pd.DataFrame:
    out = base.copy()
    out["score"] = score
    for j, c in enumerate(cols):
        out[f"score_{c}"] = pct[:, j]
    for c in ("cm_frac", "dump_frac", "sys_frac", "sys_flag", "n_anom"):
        out[c] = sysf[c].to_numpy()
    return out


def run_pass(d, grid, meta, cache: Path, name: str, reuse: bool, batch_rows: int, mask=None,
             verbose=False) -> C.DirResult:  # fmt: skip
    if reuse and (cache / f"{name}.npz").exists():
        log(f"reused cached pass {name}")
        return load_pass(cache / name)
    t0 = time.time()
    log(f"pass {name}: {d} ({len(meta)} light curves)")
    res = C.featurize_dir(d, grid, meta, batch_rows, log=log if verbose else None, mask=mask)
    save_pass(res, cache / name)
    log(f"pass {name} done in {time.time() - t0:.0f} s")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sector", type=int, default=32)
    ap.add_argument("--inj", nargs="*", default=["v1"], help="injection set names")
    ap.add_argument("--reuse-real", action="store_true", help="reuse the cached real passes")
    ap.add_argument("--batch-rows", type=int, default=2000)
    a = ap.parse_args()

    sec = a.sector
    t_start = time.time()
    grid = sector_grid(sec)
    cache = outputs_root() / "scores" / "classical" / f"s{sec:04d}" / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    meta_r = dir_metadata(sector_dir(sec))
    sdir = sector_dir(sec)

    # ---- real: raw pass, common-mode map, masked pass
    res_r = {"raw": run_pass(sdir, grid, meta_r, cache, "real", a.reuse_real, a.batch_rows,
                             verbose=True)}  # fmt: skip
    frac_r = anomaly_fraction(res_r["raw"])
    thresh = C.cm_threshold(frac_r, res_r["raw"].map_den)
    cm_r = C.common_mode_bins(res_r["raw"].map_num, res_r["raw"].map_den, thresh)
    np.savez_compressed(cache / "commonmode_real.npz", frac=frac_r, den=res_r["raw"].map_den,
                        thresh=thresh, cm=cm_r, g0=grid.g0, bin_days=grid.bin_days)  # fmt: skip
    log(f"common-mode map: threshold median pdc {np.median(thresh[0]):.4f} sap "
        f"{np.median(thresh[1]):.4f}; common-mode bins per CCD: pdc {cm_r[0].sum(1).mean():.1f},"
        f" sap {cm_r[1].sum(1).mean():.1f} of {grid.length}")
    res_r["masked"] = run_pass(sdir, grid, meta_r, cache, "real_masked", a.reuse_real,
                               a.batch_rows, mask=cm_r, verbose=True)  # fmt: skip

    tics = res_r["raw"].feats.tic.to_numpy()
    assert np.array_equal(tics, res_r["masked"].feats.tic.to_numpy())
    strata_r = C.strata_for(meta_r.set_index("tic").loc[tics])
    sys_r = C.systematics_features(res_r["raw"], cm_r)
    cals = {}
    for p in ("raw", "masked"):
        hows = tuple(sorted({h for (pp, h) in VARIANTS.values() if pp == p}))
        cals[p] = Calibrator(hows).fit(C.subscores(res_r[p].feats), strata_r)
    base_r = pd.DataFrame({"tic": tics, "sector": sec, "inj_id": None, "stratum": strata_r})
    for v, (p, h) in VARIANTS.items():
        pct, scores = cals[p].transform(C.subscores(res_r[p].feats), strata_r)
        d = outputs_root() / "scores" / v / f"s{sec:04d}"
        d.mkdir(parents=True, exist_ok=True)
        score_frame(base_r, pct, cals[p].cols, scores[h], sys_r).to_parquet(d / "real.parquet",
                                                                            index=False)
    log(f"real scores written; sys_flag rate {sys_r.sys_flag.mean():.2%}")

    # ---- injection sets
    for name in a.inj:
        frames = {v: [] for v in VARIANTS}
        for split in ("val", "test"):
            d = data_root() / "injections" / name / f"s{sec:04d}" / split
            if not (d / "injections.parquet").exists():
                log(f"skip {d}: not built")
                continue
            meta_i = dir_metadata(d)
            res_i = {"raw": run_pass(d, grid, meta_i, cache, f"inj_{name}_{split}", False,
                                     a.batch_rows)}  # fmt: skip
            # The set's own map, thresholded against its *own* typical level: every light curve
            # in an injection set carries a signal, so a fixed fraction would flag the
            # astrophysical injections themselves. Only time-aligned excesses survive.
            th_i = C.cm_threshold(anomaly_fraction(res_i["raw"]), res_i["raw"].map_den,
                                  floor=DIR_CM_MIN_FRAC)  # fmt: skip
            cm_i = cm_r | C.common_mode_bins(res_i["raw"].map_num, res_i["raw"].map_den, th_i,
                                             min_count=5)  # fmt: skip
            n_extra = int((cm_i & ~cm_r).sum())
            log(f"  {name}/{split}: {n_extra} extra common-mode (channel, CCD, bin) cells from "
                "the set's own map")
            res_i["masked"] = run_pass(d, grid, meta_i, cache, f"inj_{name}_{split}_masked",
                                       False, a.batch_rows, mask=cm_i)  # fmt: skip
            ids = res_i["raw"].feats.inj_id.to_numpy()
            assert np.array_equal(ids, res_i["masked"].feats.inj_id.to_numpy())
            sys_i = C.systematics_features(res_i["raw"], cm_i)
            strata_i = C.strata_for(meta_i.set_index("inj_id").loc[ids])
            base_i = pd.DataFrame({"tic": res_i["raw"].feats.tic.to_numpy(), "sector": sec,
                                   "inj_id": ids, "stratum": strata_i})  # fmt: skip
            for v, (p, h) in VARIANTS.items():
                pct, scores = cals[p].transform(C.subscores(res_i[p].feats), strata_i)
                frames[v].append(score_frame(base_i, pct, cals[p].cols, scores[h], sys_i))
        for v in VARIANTS:
            if frames[v]:
                out = outputs_root() / "scores" / v / f"s{sec:04d}" / f"inj_{name}.parquet"
                pd.concat(frames[v], ignore_index=True).to_parquet(out, index=False)
        log(f"injection scores written for {name}")

    prov = {"sector": sec, "variants": {k: list(v) for k, v in VARIANTS.items()},
            "inj": a.inj, "subscores": cals["raw"].cols,
            "grid": [grid.g0, grid.length, grid.bin_days],
            "dir_cm_min_frac": DIR_CM_MIN_FRAC, "seconds": round(time.time() - t_start),
            "source_provenance": str(sdir / "provenance.json")}  # fmt: skip
    for v in VARIANTS:
        p = outputs_root() / "scores" / v / f"s{sec:04d}" / "provenance.json"
        p.write_text(json.dumps({**prov, "variant": v}, indent=2))
    log("done")


if __name__ == "__main__":
    main()
