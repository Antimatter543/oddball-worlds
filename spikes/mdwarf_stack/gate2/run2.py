"""Gate 2 job runner (CPU engine, numba 1 thread per worker, resumable per job).

Job: {id, tic, mode, seed, rot, range, max_sig, mass, radius, inj?}
  mode  real | inv | invroll | roll   (applied after preprocessing, as in gate 1)
  rot   none | harm                    (search2.preprocess2)
  range long (10-60 d) | short (1-10 d)
  inj   injection params, applied to the RAW binned flux before rotation removal and detrending
Jobs that share (tic, rot) and have no injection share one preprocessing pass.
Results are appended to OUT.jsonl; finished ids are skipped on restart.

Usage: python -I gate2/run2.py JOBS.jsonl OUT.jsonl [--procs 3]
"""

from __future__ import annotations

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

RANGES = {"long": (10.0, 60.0), "short": (1.0, 10.0)}


def _apply_mode(pp, mode, seed):
    pp = {k: v.copy() for k, v in pp.items()}
    if mode in ("inv", "invroll"):
        pp["flux"] = 2.0 - pp["flux"]
    if mode in ("roll", "invroll"):
        rng = np.random.default_rng(seed)
        for s in np.unique(pp["sector"]):
            m = np.flatnonzero(pp["sector"] == s)
            sh = int(rng.integers(1, len(m)))
            pp["flux"][m] = np.roll(pp["flux"][m], sh)
            pp["sigma"][m] = np.roll(pp["sigma"][m], sh)
    return pp


def _rot_short(info):
    keep = ("applied", "n_sec_applied", "n_sec", "prot", "amp")
    out = {k: info.get(k) for k in keep}
    secs = info.get("sectors") or {}
    out["max_power"] = max([v["power"] for v in secs.values()], default=0.0)
    return out


def run_task(jobs: list[dict]) -> list[dict]:
    from fetch import load
    from inject import binned_model, duration_14, epochs_in_data, match
    from search import search_star
    from search2 import preprocess2

    out = []
    try:
        lc, meta = load(jobs[0]["tic"])
    except Exception as e:  # noqa: BLE001
        return [{"id": j["id"], "tic": j["tic"], "error": repr(e)[:300]} for j in jobs]
    cache = {}
    for job in jobs:
        try:
            c0, w0 = time.process_time(), time.time()
            inj = job.get("inj")
            key = (job["rot"], None if inj else "real")
            if inj:
                lcj = dict(lc)
                lcj["flux"] = lc["flux"] * binned_model(
                    lc["idx"], P=inj["P"], t0=inj["t0"], rp_re=inj["rp"], b=inj["b"],
                    mass=job["mass"], radius=job["radius"])
                pp, rinfo = preprocess2(lcj, job["rot"])
            elif key in cache:
                pp, rinfo = cache[key]
            else:
                pp, rinfo = preprocess2(lc, job["rot"])
                cache[key] = (pp, rinfo)
            c1 = time.process_time()
            q = _apply_mode(pp, job["mode"], job["seed"])
            sigs, stats = search_star(q, job["mass"], job["radius"], job["max_sig"],
                                      prange=RANGES[job["range"]])
            r = {k: job[k] for k in ("id", "tic", "mode", "seed", "rot", "range", "max_sig")}
            r.update(n_sectors=len(meta["sectors"]), sectors=[s["sector"] for s in meta["sectors"]],
                     cpu_pre=c1 - c0, cpu_s=time.process_time() - c0, wall_s=time.time() - w0,
                     engine="cpu", rot_info=_rot_short(rinfo), **stats)
            r["signals"] = [s.as_dict() for s in sigs]
            if inj:
                dur = duration_14(inj["P"], inj["rp"], inj["b"], job["mass"], job["radius"])
                r["inj"] = dict(inj, dur14=dur)
                r["inj"]["n_tr_data"] = len(epochs_in_data(pp["idx"], inj["P"], inj["t0"], dur))
                r["inj_match"] = [list(match(s, inj["P"], inj["t0"], dur, pp["idx"]))
                                  for s in r["signals"]]
                # white-noise expected SNR of the injection in the preprocessed data
                from holdout import transit_bins

                depth = 1 - float(binned_model(np.array([0]), P=inj["P"], t0=0.5 * 10 / 1440,
                                               rp_re=inj["rp"], b=inj["b"], mass=job["mass"],
                                               radius=job["radius"])[0])
                inb = transit_bins(pp["idx"], inj["P"], inj["t0"], dur)
                r["inj"]["snr_white"] = float(depth * np.sqrt(np.sum(1 / pp["sigma"][inb] ** 2)))
                r["inj"]["depth0"] = depth
            out.append(r)
        except Exception as e:  # noqa: BLE001
            out.append({"id": job["id"], "tic": job["tic"], "error": repr(e)[:300]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=3)
    a = ap.parse_args()
    a.procs = min(a.procs, 3)  # machine rule: at most 3 CPU workers
    jobs = [json.loads(x) for x in open(a.jobs) if x.strip()]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(x)["id"] for x in open(a.out) if x.strip()}
    todo = [j for j in jobs if j["id"] not in done]
    # group: same (tic, rot) without injection share preprocessing; keep the file order
    tasks, idx = [], {}
    for j in todo:
        if j.get("inj"):
            tasks.append([j])
            continue
        k = (j["tic"], j["rot"])
        if k not in idx:
            idx[k] = len(tasks)
            tasks.append([])
        tasks[idx[k]].append(j)
    print(f"{len(todo)} jobs in {len(tasks)} tasks ({len(done)} done)", flush=True)
    t0 = time.time()
    n = 0
    with get_context("spawn").Pool(a.procs, maxtasksperchild=40) as pool, open(a.out, "a") as fh:
        for rs in pool.imap_unordered(run_task, tasks, chunksize=1):
            for r in rs:
                fh.write(json.dumps(r) + "\n")
                n += 1
            fh.flush()
            last = rs[-1]
            if n % 20 < len(rs) or "error" in last:
                print(f"{n}/{len(todo)} {time.time() - t0:.0f}s last={last.get('id')} "
                      f"cpu={last.get('cpu_s', float('nan')):.1f} {last.get('error', '')}",
                      flush=True)


if __name__ == "__main__":
    main()
