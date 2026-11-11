"""Score an extra injection set with a trained encoder **without** re-scoring the real stars.

    uv run python -I scripts/ml_score_inj.py --run runs/ssl_v2_s0 --shards data/injections/n2/s0032 \
        --variant iforest=ssl_v2_iforest --variant recon=ssl_v2_recon

``scripts/ml_score.py`` re-encodes the real sector whenever its per-run cache
(``<run>/score/real_sub_*.parquet``, ``emb_real_*.npy``) is missing, which rewrites every real
score file. This entry point reuses what the published score files and the run already hold:

* the fitted reference state ``<run>/score/ref_state_sNNNN.pkl`` (S32 before 2026-11-10: ``ref_state.pkl``) (embedding statistics, IsolationForest)
  and the cached common-gap mask ``<run>/score/obs_mask_sNNNN.npy``;
* the real reference sub-scores from ``outputs/scores/<ref-scorer>/sNNNN/real.parquet`` (every
  ``score_*`` column), with Tmag / camera / CCD from the sector metadata, for calibration.

Only variants that need no kNN reference embedding are supported (``iforest``, ``recon``).
``--check v1`` re-scores an existing set into a scratch directory and compares it with the
published files (the reproduction test).
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_score import (
    VARIANT_COLS,
    attach_sys,
    load_model,
    ref_state_path,
    score_cache,
    write_scores,
)

from oddball.common.paths import REPO_ROOT, outputs_root, sector_dir
from oddball.ml.data import Cache, build_cache, cache_dir_for
from oddball.ml.scoring import combine, embedding_matrix, obs_mask_for
from oddball.tess.arrays import dir_metadata


def log(s):
    print(s, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--sector", type=int, default=32)
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--ref-scorer", default="ssl_v2_iforest")
    ap.add_argument("--n-masks", type=int, default=5)
    ap.add_argument("--mask-block", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask-seeding", choices=("tic", "stream"), default="tic")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="write score files under this directory instead of outputs/scores")
    a = ap.parse_args()
    run = a.run if a.run.is_absolute() else REPO_ROOT / a.run
    model, ck = load_model(run / "best.pt")
    work = run / "score"
    sdir = f"s{a.sector:04d}"
    scorers = {}
    for v in a.variant:
        kind, name = v.split("=")
        if kind not in ("iforest", "recon"):
            raise SystemExit("only iforest / recon variants are supported here")
        scorers[name] = VARIANT_COLS[kind]
    real_cache = Cache(build_cache(sector_dir(a.sector), cache_dir_for(sector_dir(a.sector)),
                                   log=log))  # fmt: skip
    obs_mask = obs_mask_for(ck, a.sector, real_cache, work, log)
    with open(ref_state_path(work, a.sector), "rb") as fh:
        st = pickle.load(fh)
    meta = dir_metadata(sector_dir(a.sector))[["tic", "camera", "ccd", "tmag"]]
    real_sub = pd.read_parquet(outputs_root() / "scores" / a.ref_scorer / sdir / "real.parquet")
    real_sub = real_sub.merge(meta, on="tic", how="left", validate="one_to_one")

    shards = a.shards.resolve()
    parts = [shards / s for s in ("val", "test") if (shards / s).is_dir()] or [shards]
    name = a.name or (shards.parent.name if shards.name.startswith("s0") else
                      shards.parent.parent.name)  # fmt: skip
    frames = []
    for p in parts:
        log(f"scoring injection shards {p}")
        cache = Cache(build_cache(p, cache_dir_for(p), origin=real_cache.info["origin"], log=log))
        tag = "inj_" + "_".join(cache.dir.name.split("__")[-3:]) + f"_b{a.mask_block}"
        if a.out_root:
            tag += "_check"
        sub, pooled, ctx = score_cache(model, ck, cache, work, tag, a.n_masks, a.seed, obs_mask,
                                       a.mask_block, a.mask_seeding)  # fmt: skip
        emb = embedding_matrix(pooled, ctx, st["stats"])
        sub["score_iforest"] = -st["iforest"].score_samples(emb)
        frames.append(sub)
    allsub = attach_sys(pd.concat(frames, ignore_index=True), a.sector, f"inj_{name}")
    root = a.out_root or (outputs_root() / "scores")
    for sc, cols in scorers.items():
        s = allsub.copy()
        s["score_rankavg"], s["score"] = combine(s, real_sub, cols)
        write_scores(s, root / sc / sdir / f"inj_{name}.parquet", real=False)
    if a.out_root:  # reproduction check against the published files
        for sc in scorers:
            new = pd.read_parquet(root / sc / sdir / f"inj_{name}.parquet")
            old = pd.read_parquet(outputs_root() / "scores" / sc / sdir / f"inj_{name}.parquet")
            j = old[["inj_id", "score"]].merge(new[["inj_id", "score"]], on="inj_id",
                                               suffixes=("_old", "_new"))  # fmt: skip
            d = np.abs(j.score_old - j.score_new)
            log(f"check {sc}: {len(j)} rows, max |diff| {d.max():.2e}, "
                f"corr {np.corrcoef(j.score_old, j.score_new)[0, 1]:.6f}")


if __name__ == "__main__":
    main()
