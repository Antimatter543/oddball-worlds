"""Score real and/or injected light curves with a trained encoder; write ML_DESIGN §9 score files.

    # all real S32 stars (every split, incl. known) -> outputs/scores/ssl_v1/s0032/real.parquet
    ODDBALL_DATA=... uv run python -I scripts/ml_score.py --run runs/ssl_v1_s0 --scorer ssl_v1
    # an injection set (a dir with val/ and test/ subdirs, or one split dir)
    ODDBALL_DATA=... uv run python -I scripts/ml_score.py --run runs/ssl_v1_s0 --scorer ssl_v1 \
        --shards data/injections/v1/s0032

The real stars are always scored first (or reloaded) because they are the kNN / IForest reference
(train + val) and the percentile reference for calibration. Injected rows are compared against the
real reference with their host TIC excluded from the neighbours.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from oddball.common.paths import REPO_ROOT, data_root, outputs_root, sector_dir
from oddball.ml.data import Cache, build_cache, cache_dir_for
from oddball.ml.scoring import (
    combine,
    embedding_matrix,
    encode_and_surprise,
    fit_embedding_stats,
    knn_score,
)
from oddball.ml.train import load_model

TOP_TIME_RESOLVED = 500


def log(s):
    print(s, flush=True)


def score_cache(model, ck, cache: Cache, workdir: Path, tag: str, n_masks: int, seed: int):
    rows = np.arange(len(cache))
    sur = np.lib.format.open_memmap(workdir / f"surprise_{tag}.npy", mode="w+",
                                    dtype=np.float16, shape=(len(rows), cache.x.shape[2]))  # fmt: skip
    t0 = time.time()
    pooled, rpk, rmn, rsap = encode_and_surprise(
        model, cache, rows, obs_mask=ck.get("obs_mask"), drop_sap=ck["train_cfg"]["drop_sap"],
        n_masks=n_masks, seed=seed, surprise_out=sur, amp=False, log=log,
    )  # fmt: skip
    sur.flush()
    log(f"  encoded {len(rows)} rows in {time.time() - t0:.0f} s")
    idx = cache.index.reset_index(drop=True)
    sub = idx[["tic", "sector", "inj_id", "host_tic", "camera", "ccd", "tmag", "split"]].copy()
    sub["score_recon"] = rpk
    sub["score_recon_mean"] = rmn
    sub["score_recon_sap"] = rsap
    return sub, pooled, cache.ctx(rows)


def write_scores(sub: pd.DataFrame, path: Path, real: bool) -> None:
    out = sub[["tic", "sector", "inj_id", "score"] + sorted(c for c in sub if c.startswith("score_"))].copy()
    if real:
        out["inj_id"] = pd.Series([None] * len(out), dtype=object)
    out["tic"] = out["tic"].astype("int64")
    out["sector"] = out["sector"].astype("int64")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    log(f"  wrote {path} ({len(out)} rows)")


def save_time_resolved(sub, sur_path: Path, out_path: Path, top: int) -> None:
    sur = np.load(sur_path, mmap_mode="r")
    pick = np.argsort(-sub["score"].to_numpy(), kind="stable")[:top]
    pick = np.sort(pick)
    df = sub.iloc[pick][["tic", "sector", "inj_id", "score"]].copy()
    df["surprise"] = [np.asarray(sur[i], dtype=np.float32) for i in pick]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log(f"  wrote time-resolved surprise for top {len(df)} -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--scorer", default="ssl_v1")
    ap.add_argument("--sector", type=int, default=32)
    ap.add_argument("--shards", type=Path, action="append", default=[])
    ap.add_argument("--name", default=None, help="injection set name (default: from the path)")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--n-masks", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rescore-real", action="store_true")
    ap.add_argument("--out-root", type=Path, default=outputs_root())
    a = ap.parse_args()

    run = a.run if a.run.is_absolute() else REPO_ROOT / a.run
    model, ck = load_model(run / "best.pt")
    work = run / "score"
    work.mkdir(exist_ok=True)
    sdir = f"s{a.sector:04d}"
    score_dir = a.out_root / "scores" / a.scorer / sdir
    ml_dir = a.out_root / "ml" / a.scorer

    # ---------------- real stars (reference) ----------------
    ref_file = work / f"real_sub_{sdir}.parquet"
    if a.rescore_real or not ref_file.exists():
        log(f"scoring real {sdir} with {run.name} (epoch {ck['epoch']}, val {ck['val_nll']:.4f})")
        cache = Cache(build_cache(sector_dir(a.sector), cache_dir_for(sector_dir(a.sector)),
                                  log=log))  # fmt: skip
        sub, pooled, ctx = score_cache(model, ck, cache, work, f"real_{sdir}", a.n_masks, a.seed)
        is_ref = sub.split.isin(["train", "val"]).to_numpy()
        stats = fit_embedding_stats(pooled[is_ref], ctx[is_ref])
        emb = embedding_matrix(pooled, ctx, stats)
        t0 = time.time()
        sub["score_knn"] = knn_score(emb, sub.tic.to_numpy(), emb[is_ref],
                                     sub.tic.to_numpy()[is_ref], k=a.k)  # fmt: skip
        log(f"  kNN (k={a.k}) in {time.time() - t0:.0f} s")
        iso = IsolationForest(n_estimators=300, max_samples=4096, random_state=a.seed, n_jobs=4)
        iso.fit(emb[is_ref])
        sub["score_iforest"] = -iso.score_samples(emb)
        sub["score_rankavg"], sub["score"] = combine(sub, sub)
        np.save(work / f"emb_real_{sdir}.npy", emb.astype(np.float16))
        np.save(work / f"pooled_real_{sdir}.npy", pooled.astype(np.float32))
        with open(work / "ref_state.pkl", "wb") as fh:
            pickle.dump({"stats": stats, "iforest": iso, "is_ref": is_ref}, fh)
        sub.to_parquet(ref_file, index=False)
        write_scores(sub, score_dir / "real.parquet", real=True)
        save_time_resolved(sub, work / f"surprise_real_{sdir}.npy",
                           ml_dir / f"timeresolved_{sdir}_real.parquet", TOP_TIME_RESOLVED)  # fmt: skip
        (ml_dir / "score_meta.json").write_text(json.dumps({
            "run": str(run), "epoch": ck["epoch"], "val_nll": ck["val_nll"], "k": a.k,
            "n_masks": a.n_masks, "combine": ["score_knn", "score_recon", "score_iforest"],
            "reference": "real train+val", "data_root": str(data_root()),
        }, indent=1))  # fmt: skip
    real_sub = pd.read_parquet(ref_file)

    # ---------------- injection sets ----------------
    if a.shards:
        with open(work / "ref_state.pkl", "rb") as fh:
            st = pickle.load(fh)
        emb_real = np.load(work / f"emb_real_{sdir}.npy").astype(np.float32)
        ref_mask = st["is_ref"]
        ref_emb, ref_tic = emb_real[ref_mask], real_sub.tic.to_numpy()[ref_mask]
    for shards in a.shards:
        shards = shards.resolve()
        parts = [shards / s for s in ("val", "test") if (shards / s).is_dir()] or [shards]
        name = a.name or (shards.parent.name if shards.name.startswith("s0") else
                          shards.parent.parent.name)  # fmt: skip
        frames = []
        for p in parts:
            log(f"scoring injection shards {p}")
            cache = Cache(build_cache(p, cache_dir_for(p), origin=ck.get("origin"), log=log))
            tag = "inj_" + "_".join(cache.dir.name.split("__")[-3:])
            sub, pooled, ctx = score_cache(model, ck, cache, work, tag, a.n_masks, a.seed)
            emb = embedding_matrix(pooled, ctx, st["stats"])
            sub["score_knn"] = knn_score(emb, sub.host_tic.to_numpy(), ref_emb, ref_tic, k=a.k)
            sub["score_iforest"] = -st["iforest"].score_samples(emb)
            sub["score_rankavg"], sub["score"] = combine(sub, real_sub)
            sub["_sur"] = str(work / f"surprise_{tag}.npy")
            frames.append(sub)
        allsub = pd.concat(frames, ignore_index=True)
        write_scores(allsub, score_dir / f"inj_{name}.parquet", real=False)
        for p, f in zip(parts, frames, strict=True):
            save_time_resolved(f, Path(f["_sur"].iloc[0]),
                               ml_dir / f"timeresolved_{sdir}_inj_{name}_{p.name}.parquet",
                               TOP_TIME_RESOLVED)  # fmt: skip


if __name__ == "__main__":
    main()
