"""Score real and/or injected light curves with a trained encoder; write ML_DESIGN §9 score files.

    # all real S32 stars (every split, incl. known) -> outputs/scores/ssl_v2/s0032/real.parquet
    uv run python -I scripts/ml_score.py --run runs/ssl_v2_s0 --scorer ssl_v2 \
        --variant recon=ssl_v2_recon --shards data/injections/v1/s0032

The real stars are always scored first (or reloaded) because they are the kNN / IForest reference
(train + val) and the percentile reference for calibration. Injected rows are compared against the
real reference with their host TIC excluded from the neighbours.

* **Held-out sector.** A model trained on other sectors stores per-sector common-gap masks; for a
  sector it never saw (S32 for ssl_v2) the mask is computed from that sector's real train + val
  stars (which bins have data; no labels) and reused for its injection sets.
* **Variants.** ``--variant recon=<scorer>`` also writes a scorer whose ``score`` is the
  stratified percentile of ``score_recon`` alone (same sub-scores, same files otherwise).
* **Systematics veto columns.** ``cm_frac, dump_frac, sys_frac, sys_flag, n_anom`` are copied
  from the classical scorer's files for the same light curves (by TIC for real stars, by
  ``inj_id`` for injections): the common-mode / momentum-dump flag of
  ``oddball.score.classical.systematics_features`` (binomial common-mode test for injection
  sets). The evaluator applies the veto symmetrically when the column is present.
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
    COMBINE,
    combine,
    embedding_matrix,
    encode_and_surprise,
    fit_embedding_stats,
    knn_score,
    obs_mask_for,
)
from oddball.ml.train import load_model

TOP_TIME_RESOLVED = 500
SYS_COLS = ("cm_frac", "dump_frac", "sys_frac", "sys_flag", "n_anom")
VARIANT_COLS = {"combined": COMBINE, "recon": ("score_recon",), "knn": ("score_knn",),
                "iforest": ("score_iforest",)}  # fmt: skip


def log(s):
    print(s, flush=True)


def score_cache(model, ck, cache: Cache, workdir: Path, tag: str, n_masks: int, seed: int,
                obs_mask, block: int = 4):  # fmt: skip
    rows = np.arange(len(cache))
    enc_file = workdir / f"enc_{tag}.npz"
    if enc_file.exists() and (workdir / f"surprise_{tag}.npy").exists():
        d = np.load(enc_file)
        if len(d["pooled"]) == len(rows):  # resume: encoding is the slow part
            log(f"  reusing encodings {enc_file}")
            pooled, rpk, rmn, rsap = d["pooled"], d["rpk"], d["rmn"], d["rsap"]
            rows = None
    if rows is not None:
        sur = np.lib.format.open_memmap(workdir / f"surprise_{tag}.npy", mode="w+",
                                        dtype=np.float16, shape=(len(rows), cache.x.shape[2]))  # fmt: skip
        t0 = time.time()
        pooled, rpk, rmn, rsap = encode_and_surprise(
            model, cache, rows, obs_mask=obs_mask, drop_sap=ck["train_cfg"]["drop_sap"],
            n_masks=n_masks, seed=seed, surprise_out=sur, amp=False, log=log, block=block,
        )  # fmt: skip
        sur.flush()
        del sur
        np.savez(enc_file, pooled=pooled, rpk=rpk, rmn=rmn, rsap=rsap)
        log(f"  encoded {len(pooled)} rows in {time.time() - t0:.0f} s")
    rows = np.arange(len(cache))
    idx = cache.index.reset_index(drop=True)
    sub = idx[["tic", "sector", "inj_id", "host_tic", "camera", "ccd", "tmag", "split"]].copy()
    sub["score_recon"] = rpk
    sub["score_recon_mean"] = rmn
    if not ck["train_cfg"]["drop_sap"]:
        sub["score_recon_sap"] = rsap
    return sub, pooled, cache.ctx(rows)


def attach_sys(sub: pd.DataFrame, sector: int, which: str, source: str = "classical") -> pd.DataFrame:
    """Copy the classical scorer's per-light-curve systematics columns onto ``sub``."""
    f = outputs_root() / "scores" / source / f"s{sector:04d}" / f"{which}.parquet"
    if not f.exists():
        log(f"  no {f}: score file written without systematics columns (no veto possible)")
        return sub
    cols = [c for c in SYS_COLS if c in pd.read_parquet(f, columns=None).columns]
    key = "tic" if which == "real" else "inj_id"
    src = pd.read_parquet(f, columns=[key, *cols]).drop_duplicates(key)
    out = sub.drop(columns=[c for c in cols if c in sub]).merge(src, on=key, how="left",
                                                               validate="many_to_one")  # fmt: skip
    miss = int(out[cols[0]].isna().sum())
    if miss:
        log(f"  warning: {miss} rows have no classical systematics flag (treated as unflagged)")
    out["sys_flag"] = out["sys_flag"].fillna(False).astype(bool)
    return out


def write_scores(sub: pd.DataFrame, path: Path, real: bool) -> None:
    keep = ["tic", "sector", "inj_id", "score"]
    keep += sorted(c for c in sub if c.startswith("score_")) + [c for c in SYS_COLS if c in sub]
    out = sub[keep].copy()
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


def emit(sub: pd.DataFrame, real_sub: pd.DataFrame, scorers: dict, sdir: str, fname: str,
         real: bool) -> None:  # fmt: skip
    """Write one score file per scorer variant (score = its combination of sub-scores)."""
    for name, cols in scorers.items():
        s = sub.copy()
        s["score_rankavg"], s["score"] = combine(s, real_sub, cols)
        write_scores(s, outputs_root() / "scores" / name / sdir / fname, real)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--scorer", default="ssl_v1")
    ap.add_argument("--variant", action="append", default=[],
                    help="extra scorer as <kind>=<name>, kind in recon/knn/iforest/combined")
    ap.add_argument("--sector", type=int, default=32)
    ap.add_argument("--shards", type=Path, action="append", default=[])
    ap.add_argument("--name", default=None, help="injection set name (default: from the path)")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--n-masks", type=int, default=5)
    ap.add_argument("--mask-block", type=int, default=8,
                    help="scoring-mask block in patches (ssl_v1: 4; 8 chosen on val injections)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rescore-real", action="store_true")
    a = ap.parse_args()

    run = a.run if a.run.is_absolute() else REPO_ROOT / a.run
    model, ck = load_model(run / "best.pt")
    work = run / "score"
    work.mkdir(exist_ok=True)
    sdir = f"s{a.sector:04d}"
    scorers = {a.scorer: COMBINE}
    for v in a.variant:
        kind, name = v.split("=")
        scorers[name] = VARIANT_COLS[kind]
    ml_dir = outputs_root() / "ml" / a.scorer
    held_out = a.sector not in [int(s) for s in ck.get("sectors", [a.sector])]
    log(f"run {run.name} (epoch {ck['epoch']}, val {ck['val_nll']:.4f}); trained on "
        f"{ck.get('sectors')}; s{a.sector:04d} held out: {held_out}")

    # ---------------- real stars (reference) ----------------
    ref_file = work / f"real_sub_{sdir}_b{a.mask_block}.parquet"
    real_cache = Cache(build_cache(sector_dir(a.sector), cache_dir_for(sector_dir(a.sector)),
                                   log=log))  # fmt: skip
    obs_mask = obs_mask_for(ck, a.sector, real_cache, work, log)
    if a.rescore_real or not ref_file.exists():
        sub, pooled, ctx = score_cache(model, ck, real_cache, work, f"real_{sdir}_b{a.mask_block}",
                                       a.n_masks, a.seed, obs_mask, a.mask_block)  # fmt: skip
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
        sub = attach_sys(sub, a.sector, "real")
        np.save(work / f"emb_real_{sdir}.npy", emb.astype(np.float16))
        np.save(work / f"pooled_real_{sdir}.npy", pooled.astype(np.float32))
        with open(work / "ref_state.pkl", "wb") as fh:
            pickle.dump({"stats": stats, "iforest": iso, "is_ref": is_ref}, fh)
        sub.to_parquet(ref_file, index=False)
        save_time_resolved(sub, work / f"surprise_real_{sdir}_b{a.mask_block}.npy",
                           ml_dir / f"timeresolved_{sdir}_real.parquet", TOP_TIME_RESOLVED)  # fmt: skip
    real_sub = pd.read_parquet(ref_file)
    emit(real_sub, real_sub, scorers, sdir, "real.parquet", real=True)
    ml_dir.mkdir(parents=True, exist_ok=True)
    (ml_dir / "score_meta.json").write_text(json.dumps({
        "run": str(run), "epoch": ck["epoch"], "val_nll": ck["val_nll"], "k": a.k,
        "n_masks": a.n_masks, "mask_block": a.mask_block,
        "scorers": {k: list(v) for k, v in scorers.items()},
        "trained_on_sectors": ck.get("sectors"), "held_out": held_out,
        "reference": "real train+val", "data_root": str(data_root()),
        "sys_columns": "copied from outputs/scores/classical (sys_flag veto)",
    }, indent=1))  # fmt: skip

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
            cache = Cache(build_cache(p, cache_dir_for(p), origin=real_cache.info["origin"],
                                      log=log))  # fmt: skip
            tag = "inj_" + "_".join(cache.dir.name.split("__")[-3:]) + f"_b{a.mask_block}"
            sub, pooled, ctx = score_cache(model, ck, cache, work, tag, a.n_masks, a.seed,
                                           obs_mask, a.mask_block)  # fmt: skip
            emb = embedding_matrix(pooled, ctx, st["stats"])
            sub["score_knn"] = knn_score(emb, sub.host_tic.to_numpy(), ref_emb, ref_tic, k=a.k)
            sub["score_iforest"] = -st["iforest"].score_samples(emb)
            sub["score_rankavg"], sub["score"] = combine(sub, real_sub)
            sub["_sur"] = str(work / f"surprise_{tag}.npy")
            frames.append(sub)
        allsub = attach_sys(pd.concat(frames, ignore_index=True), a.sector, f"inj_{name}")
        sub_file = work / f"inj_{name}_sub_{sdir}_b{a.mask_block}.parquet"
        allsub.drop(columns=["_sur"]).to_parquet(sub_file, index=False)
        emit(allsub, real_sub, scorers, sdir, f"inj_{name}.parquet", real=False)
        for p, f in zip(parts, frames, strict=True):
            save_time_resolved(f, Path(f["_sur"].iloc[0]),
                               ml_dir / f"timeresolved_{sdir}_inj_{name}_{p.name}.parquet",
                               TOP_TIME_RESOLVED)  # fmt: skip


if __name__ == "__main__":
    main()
