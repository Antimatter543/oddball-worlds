"""Gate 3 shard driver (runs inside a Kaggle CPU session, or locally for a smoke test).

Stages, each resumable and each bounded by the --deadline (unix time):
  fetch   PDCSAP + SAP NPZ per star (vetting/fetch_val2.fetch_one: the survey's product rule and
          S3 route, one read per sector), status -> fetch_status.jsonl
  noise   per sector robust scatter of 2-h bins of the frozen-preprocessed light curve
  jobs    per far star with data: real long + short (6 signals), 4 nulls, 5 injections;
          per 50 pc reference star: 2 injections (1.5-2.0, 2.0-3.0 R_e); seeds per (seed, TIC)
  search  frozen gate2/run2.py on two job halves in parallel (2 + 2 workers = 4 vCPU)
  vet     frozen vetting v2 measurement (run_vet2.run) on far-star rows: matched injections
          >= tau_long, nulls with top MES >= 7, real long signals >= tau_long (3 processes)
All outputs go to <OUT>/gate3/run/ (OUT from ODDBALL_MAIN). Pre-registration:
docs/spikes/mdwarf_100pc.md. Usage:
  python -I gate3/kdriver3.py --sample S.csv --ref R.csv --info star_info.json --deadline T
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPIKE = HERE.parent
sys.path.insert(0, str(SPIKE))
sys.path.insert(0, str(SPIKE / "gate2"))
sys.path.insert(0, str(SPIKE / "survey"))
sys.path.insert(0, str(SPIKE / "vetting"))
sys.path.insert(0, str(SPIKE.parents[1] / "src"))
from common import BIN_D, DATA, LC_DIR, OUT

RUN = OUT / "gate3" / "run"
SEED_FAR, SEED_REF = 20261117, 20261118
NULLS = (("inv", 0), ("invroll", 11), ("roll", 21), ("roll", 22))
FAR_INJ = [((1.0, 1.5), (15.0, 30.0), "h"), ((1.0, 1.5), (15.0, 30.0), "h"),
           ((1.0, 1.5), (30.0, 50.0), "l"), ((1.5, 2.0), (15.0, 50.0), "m"),
           ((2.0, 3.0), (15.0, 50.0), "x")]
REF_INJ = [((1.5, 2.0), (15.0, 50.0), "m"), ((2.0, 3.0), (15.0, 50.0), "x")]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def rows(p):
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(x) for x in open(p) if x.strip()]


def stage_fetch(stars: list[int], deadline: float):
    import download as D
    import fetch_val2 as F

    F.CREATED = RUN / "created_files.txt"
    st_path = RUN / "fetch_status.jsonl"
    done = {r["tic"] for r in rows(st_path)}

    def errlog(r):
        with open(RUN / "fetch_errors.jsonl", "a") as fh:
            fh.write(json.dumps(r, default=str) + "\n")

    dl = D.Downloader(DATA / "g3dl", errlog)
    pool = ThreadPoolExecutor(4, thread_name_prefix="g3dl")
    t0 = time.time()
    for i, tic in enumerate(stars):
        if tic in done:
            continue
        if time.time() > deadline:
            log("fetch: deadline")
            break
        c0 = time.time()
        try:
            st, why = F.fetch_one(dl, tic, pool)
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            st, why = "error", repr(e)[:200]
        with open(st_path, "a") as fh:
            fh.write(json.dumps({"tic": tic, "status": st, "why": why,
                                 "wall_s": time.time() - c0}) + "\n")
        if i % 10 == 0:
            log(f"fetch {i + 1}/{len(stars)} {time.time() - t0:.0f}s")
    pool.shutdown()


def have_data(tic: int) -> bool:
    return (LC_DIR / f"tic{tic}.npz").exists()


def sigma_2h(flux, idx, cad_bins):
    """followup/noise.sigma_2h: MAD scatter of 2-h bin means, bins >= 50% filled."""
    t = (idx + 0.5) * BIN_D
    k = np.floor(t * 12).astype(np.int64)
    _, inv = np.unique(k, return_inverse=True)
    n = np.bincount(inv)
    b = np.bincount(inv, weights=flux) / n
    b = b[n >= 0.5 * 12 / cad_bins]
    if len(b) < 20:
        return float("nan"), len(b)
    return float(1.4826 * np.median(np.abs(b - np.median(b)))), len(b)


def stage_noise(stars: list[tuple[int, str]], rot: str, deadline: float):
    from fetch import load
    from search2 import preprocess2

    out = RUN / "noise.jsonl"
    done = {r["tic"] for r in rows(out)}
    for tic, kind in stars:
        if tic in done or not have_data(tic):
            continue
        if time.time() > deadline:
            log("noise: deadline")
            break
        try:
            lc, meta = load(tic)
            pp, _ = preprocess2(lc, rot)
            per = {}
            for s in np.unique(pp["sector"]):
                m = pp["sector"] == s
                ms = lc["sector"] == s
                d = np.diff(lc["idx"][ms])
                cad = max(1, int(np.median(d[d > 0]))) if (d > 0).any() else 1
                v, nb = sigma_2h(pp["flux"][m].astype(float), pp["idx"][m], cad)
                per[int(s)] = [1e6 * v, nb, 10 * cad]
            prods = {int(x["sector"]): x.get("product") for x in meta["sectors"]}
            rec = {"tic": tic, "kind": kind, "per_sector": per, "products": prods,
                   "crowdsap": {int(x["sector"]): x.get("crowdsap") for x in meta["sectors"]}}
        except Exception as e:  # noqa: BLE001
            rec = {"tic": tic, "kind": kind, "error": repr(e)[:200]}
        with open(out, "a") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")


def stage_jobs(sample: pd.DataFrame, ref: pd.DataFrame, rot: str) -> list[dict]:
    from make_jobs2 import _inj

    def mr(row):
        m = float(row["mass"]) if np.isfinite(row["mass"]) else 0.3
        r = float(row["rad"]) if np.isfinite(row["rad"]) else 0.3
        return m, r

    jobs = []
    for kind, df, seed, specs in (("far", sample, SEED_FAR, FAR_INJ),
                                  ("ref", ref, SEED_REF, REF_INJ)):
        for _, row in df.iterrows():
            t = int(row["tic"])
            if not have_data(t):
                continue
            z = np.load(LC_DIR / f"tic{t}.npz", allow_pickle=False)
            idx0 = int(z["idx"].min())
            m, r = mr(row)
            base = {"tic": t, "rot": rot, "mass": m, "radius": r, "order": int(row["order"]),
                    "kind": kind}
            if kind == "far":
                for rg in ("long", "short"):
                    jobs.append(dict(base, id=f"g3real-{rg}-{t}", mode="real", seed=0, range=rg,
                                     max_sig=6))
                for mode, sd in NULLS:
                    jobs.append(dict(base, id=f"g3null-{mode}{sd}-{t}", mode=mode, seed=sd,
                                     range="long", max_sig=1))
            rng = np.random.default_rng([seed, t])
            for k, (rr, pr, tag) in enumerate(specs):
                b = _inj(rng, {"tic": t, "mass": m, "rad": r}, rr, pr, idx0, f"g3{kind}{tag}")
                jobs.append(dict(b, id=f"g3inj{kind}-{tag}{k}-{t}", rot=rot, order=base["order"],
                                 kind=kind, cell=tag))
    jobs.sort(key=lambda j: (j["kind"] != "far", j["order"]))
    return jobs


def stage_search(jobs: list[dict], deadline: float, halves: int = 2, procs: int = 2):
    files = []
    stars = sorted({(j["order"], j["kind"], j["tic"]) for j in jobs})
    part = {(k, t): i % halves for i, (_, k, t) in enumerate(stars)}
    for h in range(halves):
        f = RUN / f"jobs.h{h}.jsonl"
        with open(f, "w") as fh:
            fh.writelines(json.dumps(j) + "\n" for j in jobs if part[(j["kind"], j["tic"])] == h)
        files.append(f)
    env = dict(os.environ, NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    ps = []
    for h, f in enumerate(files):
        lf = open(RUN / f"search.h{h}.log", "a")
        ps.append((subprocess.Popen([sys.executable, "-I", str(SPIKE / "gate2" / "run2.py"),
                                     str(f), str(RUN / f"out.h{h}.jsonl"), "--procs", str(procs)],
                                    stdout=lf, stderr=subprocess.STDOUT, env=env,
                                    cwd=str(SPIKE)), lf))
    while any(p.poll() is None for p, _ in ps):
        if time.time() > deadline:
            log("search: deadline, stopping run2")
            for p, _ in ps:
                if p.poll() is None:
                    p.terminate()
            time.sleep(20)
            for p, _ in ps:
                if p.poll() is None:
                    p.kill()
            break
        time.sleep(30)
    for p, lf in ps:
        p.wait()
        lf.close()
    n = sum(len(rows(RUN / f"out.h{h}.jsonl")) for h in range(halves))
    log(f"search: {n} of {len(jobs)} job rows")


def stage_vet(jobs: list[dict], deadline: float, halves: int = 2, nproc: int = 3):
    allrows = [r for h in range(halves) for r in rows(RUN / f"out.h{h}.jsonl")]
    jobd = {j["id"]: j for j in jobs}
    keep = [r for r in allrows if jobd.get(r["id"], {}).get("kind") == "far" and "error" not in r]
    with open(RUN / "vet_input.jsonl", "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in keep)
    env = dict(os.environ, NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    ps = []
    for i in range(nproc):
        lf = open(RUN / f"vet.s{i}.log", "a")
        ps.append((subprocess.Popen([sys.executable, "-I", str(HERE / "vet3.py"),
                                     str(RUN / "vet_input.jsonl"), str(RUN / "jobs.all.jsonl"),
                                     str(RUN / f"vet3.s{i}of{nproc}.jsonl"), f"{i}/{nproc}"],
                                    stdout=lf, stderr=subprocess.STDOUT, env=env,
                                    cwd=str(SPIKE)), lf))
    while any(p.poll() is None for p, _ in ps):
        if time.time() > deadline:
            log("vet: deadline")
            for p, _ in ps:
                if p.poll() is None:
                    p.terminate()
            break
        time.sleep(20)
    for p, lf in ps:
        p.wait()
        lf.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--info", required=True)
    ap.add_argument("--deadline", type=float, required=True, help="unix time to stop the search")
    ap.add_argument("--vet-minutes", type=float, default=25.0)
    ap.add_argument("--fetch-minutes", type=float, default=45.0)
    ap.add_argument("--halves", type=int, default=2, help="parallel run2 processes")
    ap.add_argument("--procs", type=int, default=2, help="workers per run2 process")
    ap.add_argument("--vet-procs", type=int, default=3)
    a = ap.parse_args()
    t_start = time.time()
    RUN.mkdir(parents=True, exist_ok=True)
    fz_repo = SPIKE / "gate2" / "frozen.json"
    (OUT / "gate2").mkdir(parents=True, exist_ok=True)
    if not (OUT / "gate2" / "frozen.json").exists():
        shutil.copy2(fz_repo, OUT / "gate2" / "frozen.json")
    (OUT / "vetting").mkdir(parents=True, exist_ok=True)
    if not (OUT / "vetting" / "star_info.json").exists():
        shutil.copy2(a.info, OUT / "vetting" / "star_info.json")
    rot = json.loads(fz_repo.read_text())["rot"]
    sample = pd.read_csv(a.sample)
    ref = pd.read_csv(a.ref)
    stars = [int(x) for x in sample["tic"]] + [int(x) for x in ref["tic"]]
    log(f"start: {len(sample)} far + {len(ref)} ref stars; rot {rot}")
    stage_fetch(stars, min(a.deadline, t_start + 60 * a.fetch_minutes))
    stage_noise([(int(t), "far") for t in sample["tic"]] + [(int(t), "ref") for t in ref["tic"]],
                rot, a.deadline)
    jobs = stage_jobs(sample, ref, rot)
    with open(RUN / "jobs.all.jsonl", "w") as fh:
        fh.writelines(json.dumps(j) + "\n" for j in jobs)
    log(f"jobs: {len(jobs)}")
    stage_search(jobs, a.deadline, a.halves, a.procs)
    stage_vet(jobs, a.deadline + 60 * a.vet_minutes, a.halves, a.vet_procs)
    log(f"done in {(time.time() - t_start) / 3600:.2f} h")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
