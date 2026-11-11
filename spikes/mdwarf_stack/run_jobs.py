"""Run search jobs (real / null / injection) in parallel worker processes, 1 numba thread each.

Jobs file: JSON lines {id, tic, mode, seed, max_sig, mass, radius, inj?}. mode is one of
  real | inv | invroll | roll   (inv: flip the detrended residual; roll: per-sector circular
  shift of the detrended values with `seed`; inj: injection params, applied to RAW flux).
Results are appended to the output JSONL; finished ids are skipped on restart.

Usage: uv run python -I spikes/mdwarf_stack/run_jobs.py JOBS.jsonl OUT.jsonl [--procs 6]
"""

from __future__ import annotations

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))


def run_job(job: dict) -> dict:
    from fetch import load
    from inject import binned_model, duration_14, epochs_in_data, match
    from search import preprocess, search_star

    c0, w0 = time.process_time(), time.time()
    lc, meta = load(job["tic"])
    out = {k: job[k] for k in ("id", "tic", "mode", "seed")}
    out["n_sectors"] = len(meta["sectors"])
    out["sectors"] = [s["sector"] for s in meta["sectors"]]
    inj = job.get("inj")
    if inj:
        lc = dict(lc)
        lc["flux"] = lc["flux"] * binned_model(
            lc["idx"], P=inj["P"], t0=inj["t0"], rp_re=inj["rp"], b=inj["b"],
            mass=job["mass"], radius=job["radius"])
    pp = preprocess(lc)
    mode = job["mode"]
    if mode in ("inv", "invroll"):
        pp["flux"] = 2.0 - pp["flux"]
    if mode in ("roll", "invroll"):
        rng = np.random.default_rng(job["seed"])
        for s in np.unique(pp["sector"]):
            m = np.flatnonzero(pp["sector"] == s)
            sh = int(rng.integers(1, len(m)))
            pp["flux"][m] = np.roll(pp["flux"][m], sh)
            pp["sigma"][m] = np.roll(pp["sigma"][m], sh)
    c1 = time.process_time()
    sigs, stats = search_star(pp, job["mass"], job["radius"], job["max_sig"])
    out["cpu_pre"] = c1 - c0
    out["cpu_s"] = time.process_time() - c0
    out["wall_s"] = time.time() - w0
    out.update(stats)
    out["signals"] = [s.as_dict() for s in sigs]
    if inj:
        dur = duration_14(inj["P"], inj["rp"], inj["b"], job["mass"], job["radius"])
        out["inj"] = dict(inj, dur14=dur)
        out["inj"]["n_tr_data"] = int(len(epochs_in_data(pp["idx"], inj["P"], inj["t0"], dur)))
        out["inj_match"] = [list(match(s, inj["P"], inj["t0"], dur, pp["idx"]))
                            for s in out["signals"]]
    return out


def _safe(job):
    try:
        return run_job(job)
    except Exception as e:  # noqa: BLE001
        return {"id": job["id"], "tic": job["tic"], "error": repr(e)[:300]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    jobs = [json.loads(x) for x in open(a.jobs)]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(x)["id"] for x in open(a.out) if x.strip()}
    todo = [j for j in jobs if j["id"] not in done]
    print(f"{len(todo)} jobs to run ({len(done)} done)", flush=True)
    t0 = time.time()
    with get_context("spawn").Pool(a.procs, maxtasksperchild=50) as pool, open(a.out, "a") as fh:
        for i, r in enumerate(pool.imap_unordered(_safe, todo, chunksize=1)):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if (i + 1) % 10 == 0 or "error" in r:
                print(f"{i + 1}/{len(todo)} {time.time() - t0:.0f}s last={r.get('id')} "
                      f"cpu={r.get('cpu_s', float('nan')):.1f} {r.get('error', '')}", flush=True)


if __name__ == "__main__":
    main()
