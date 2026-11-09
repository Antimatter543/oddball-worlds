"""Score a sector (and its injection sets) with the classical baseline.

    ODDBALL_DATA=... uv run python -I scripts/score_classical.py --sector 32 --inj v1

Writes, per docs/ML_DESIGN.md §9, ``outputs/scores/<scorer>/sNNNN/{real,inj_<name>}.parquet``
for three variants that share one feature pass:

* ``classical``      rank-average of the sub-score percentiles (the pre-registered baseline);
* ``classical_cm``   the same, multiplied by ``1 - sys_frac`` (common-mode / momentum-dump
                     down-weighting, v0) before the stratum percentile;
* ``classical_tail`` rank-based combination dominated by the most extreme sub-scores
                     (mean of ``-log(1 - p)``), reported as an ablation.

Calibration (sub-score and combined percentiles) is always against the **real** light curves
of the sector, within Tmag bin x camera/CCD strata. The common-mode map for an injection set is
the real map OR-ed with a map of the injection set itself, so an instrument event shared by
the set's hosts on one CCD is visible to it (see ``oddball.score.classical``).
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

VARIANTS = ("classical", "classical_cm", "classical_tail")
DIR_CM_MIN_FRAC = 0.05  # an injection-set bin is common-mode if >=5% (and >=5) of its CCD dips


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_pass(res: C.DirResult, path: Path) -> None:
    np.savez_compressed(path.with_suffix(".npz"), anom_pdc=res.anom["pdc"],
                        anom_sap=res.anom["sap"], dumpadj=res.dumpadj, ccd=res.ccd,
                        map_num=res.map_num, map_den=res.map_den)  # fmt: skip
    res.feats.to_parquet(path.with_suffix(".parquet"), index=False)


def load_pass(path: Path) -> C.DirResult:
    z = np.load(path.with_suffix(".npz"))
    return C.DirResult(pd.read_parquet(path.with_suffix(".parquet")),
                       {"pdc": z["anom_pdc"], "sap": z["anom_sap"]}, z["dumpadj"], z["ccd"],
                       z["map_num"], z["map_den"])  # fmt: skip


class Calibrator:
    """All ECDFs, fitted on the real light curves only."""

    def fit(self, sub: pd.DataFrame, strata: np.ndarray, sysf: pd.DataFrame):
        self.cols = list(sub.columns)
        self.sub = C.StratumECDF().fit(sub.to_numpy(), strata)
        pct = self.sub.transform(sub.to_numpy(), strata)
        raws = self._raws(pct, sysf)
        self.comb = {v: C.StratumECDF().fit(r[:, None], strata) for v, r in raws.items()}
        return self

    @staticmethod
    def _raws(pct, sysf):
        mean = C.combine(pct, "mean")
        return {
            "classical": mean,
            "classical_cm": mean * (1.0 - sysf.sys_frac.to_numpy()),
            "classical_tail": C.combine(pct, "tail"),
        }

    def transform(self, sub: pd.DataFrame, strata: np.ndarray, sysf: pd.DataFrame):
        pct = self.sub.transform(sub[self.cols].to_numpy(), strata)
        raws = self._raws(pct, sysf)
        scores = {v: self.comb[v].transform(r[:, None], strata)[:, 0] for v, r in raws.items()}
        return pct, scores


def score_frame(base: pd.DataFrame, pct, cols, score, sysf) -> pd.DataFrame:
    out = base.copy()
    out["score"] = score
    for j, c in enumerate(cols):
        out[f"score_{c}"] = pct[:, j]
    for c in ("cm_frac", "dump_frac", "sys_frac", "sys_flag", "n_anom"):
        out[c] = sysf[c].to_numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sector", type=int, default=32)
    ap.add_argument("--inj", nargs="*", default=["v1"], help="injection set names")
    ap.add_argument("--reuse-real", action="store_true", help="reuse the cached real pass")
    ap.add_argument("--batch-rows", type=int, default=2000)
    a = ap.parse_args()

    sec = a.sector
    grid = sector_grid(sec)
    cache = outputs_root() / "scores" / "classical" / f"s{sec:04d}" / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    meta_r = dir_metadata(sector_dir(sec))

    t0 = time.time()
    if a.reuse_real and (cache / "real.npz").exists():
        res_r = load_pass(cache / "real")
        log(f"reused real pass ({len(res_r.feats)} light curves)")
    else:
        log(f"real pass over {sector_dir(sec)} ({len(meta_r)} light curves)")
        res_r = C.featurize_dir(sector_dir(sec), grid, meta_r, a.batch_rows, log=log)
        save_pass(res_r, cache / "real")
        log(f"real pass done in {time.time() - t0:.0f} s")

    with np.errstate(invalid="ignore", divide="ignore"):
        frac_r = np.where(res_r.map_den > 0, res_r.map_num / res_r.map_den, 0.0)
    thresh = C.cm_threshold(frac_r, res_r.map_den)
    cm_r = C.common_mode_bins(res_r.map_num, res_r.map_den, thresh)
    np.savez_compressed(cache / "commonmode_real.npz", frac=frac_r, den=res_r.map_den,
                        thresh=thresh, cm=cm_r, g0=grid.g0, bin_days=grid.bin_days)  # fmt: skip
    log(f"common-mode map: threshold pdc median {np.median(thresh[0]):.4f}, sap median "
        f"{np.median(thresh[1]):.4f}; common-mode bins per CCD: pdc "
        f"{cm_r[0].sum(1).mean():.1f}, sap {cm_r[1].sum(1).mean():.1f} of {grid.length}")

    mr = meta_r.set_index("tic").loc[res_r.feats.tic]
    strata_r = C.strata_for(mr)
    sys_r = C.systematics_features(res_r, cm_r, grid.length)
    sub_r = C.subscores(res_r.feats)
    cal = Calibrator().fit(sub_r, strata_r, sys_r)
    pct_r, scores_r = cal.transform(sub_r, strata_r, sys_r)
    base_r = pd.DataFrame({"tic": res_r.feats.tic.to_numpy(), "sector": sec, "inj_id": None,
                           "stratum": strata_r})  # fmt: skip
    for v in VARIANTS:
        d = outputs_root() / "scores" / v / f"s{sec:04d}"
        d.mkdir(parents=True, exist_ok=True)
        score_frame(base_r, pct_r, cal.cols, scores_r[v], sys_r).to_parquet(d / "real.parquet",
                                                                             index=False)
    log(f"real scores written; sys_flag rate {sys_r.sys_flag.mean():.3%}")

    for name in a.inj:
        frames = {v: [] for v in VARIANTS}
        for split in ("val", "test"):
            d = data_root() / "injections" / name / f"s{sec:04d}" / split
            if not (d / "injections.parquet").exists():
                log(f"skip {d}: not built")
                continue
            meta_i = dir_metadata(d)
            log(f"injection pass {name}/{split} ({len(meta_i)} light curves)")
            res_i = C.featurize_dir(d, grid, meta_i, a.batch_rows)
            save_pass(res_i, cache / f"inj_{name}_{split}")
            cm_i = cm_r | C.common_mode_bins(res_i.map_num, res_i.map_den, thresh,
                                             min_count=5, min_frac=DIR_CM_MIN_FRAC)  # fmt: skip
            sys_i = C.systematics_features(res_i, cm_i, grid.length)
            mi = meta_i.set_index("inj_id").loc[res_i.feats.inj_id]
            strata_i = C.strata_for(mi)
            pct_i, scores_i = cal.transform(C.subscores(res_i.feats), strata_i, sys_i)
            base_i = pd.DataFrame({"tic": res_i.feats.tic.to_numpy(), "sector": sec,
                                   "inj_id": res_i.feats.inj_id.to_numpy(),
                                   "stratum": strata_i})  # fmt: skip
            for v in VARIANTS:
                frames[v].append(score_frame(base_i, pct_i, cal.cols, scores_i[v], sys_i))
        for v in VARIANTS:
            if frames[v]:
                out = outputs_root() / "scores" / v / f"s{sec:04d}" / f"inj_{name}.parquet"
                pd.concat(frames[v], ignore_index=True).to_parquet(out, index=False)
        log(f"injection scores written for {name}")

    prov = {"sector": sec, "variants": list(VARIANTS), "inj": a.inj,
            "subscores": cal.cols, "grid": [grid.g0, grid.length, grid.bin_days],
            "dir_cm_min_frac": DIR_CM_MIN_FRAC, "seconds": round(time.time() - t0)}  # fmt: skip
    for v in VARIANTS:
        p = outputs_root() / "scores" / v / f"s{sec:04d}" / "provenance.json"
        p.write_text(json.dumps(prov, indent=2))
    log("done")


if __name__ == "__main__":
    main()
