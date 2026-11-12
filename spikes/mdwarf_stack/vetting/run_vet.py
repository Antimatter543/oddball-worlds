"""Measure every vetting statistic for one calibration stage (or a survey result), ONE process.

Stages (docs/spikes/mdwarf_vetting.md):
  dev      DEV true signals (gate-2 dev_paired harm injections + vetting dev_inj_v; matched,
           MES >= 7.0) and DEV false alarms (DEV long nulls and DEV real-control signals, MES >= 6)
  test     TEST injections (harm, matched, MES >= 6), TEST real-control signals, TEST nulls
  holdout  every long-search signal >= tau_long on the hold-out and gate-1 hosts
  survey   python -I vetting/run_vet.py survey --rows OUT.jsonl --jobs JOBS.jsonl
           (every long signal >= tau_long; needs SAP + star info fetched for those stars first)

Rows (raw statistics only; grading is done later with frozen thresholds) are appended to
<OUT>/vetting/vet_<stage>.jsonl, resumable by row key.
"""

from __future__ import annotations

import os

for _v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "gate2"))
sys.path.insert(0, str(HERE))
from common import OUT  # noqa: E402

G2 = OUT / "gate2"
V = OUT / "vetting"
DEV_TRUE_MES = 7.0
FA_MES = 6.0


def _rows(p):
    p = Path(p)
    return [json.loads(x) for x in open(p) if x.strip()] if p.exists() else []


def _jobs(*files):
    out = {}
    for f in files:
        for j in _rows(f):
            out[j["id"]] = j
    return out


def tau_long():
    return json.loads((G2 / "frozen.json").read_text())["tau_long"]


def tasks_dev():
    jobs = _jobs(G2 / "jobs" / "dev_paired.jsonl", G2 / "jobs" / "dev_nulls_long_harm.jsonl",
                 V / "jobs" / "dev_inj_v.jsonl")
    out = []
    for f, src in ((G2 / "dev_paired.out.jsonl", "g2"), (V / "dev_inj_v.out.jsonl", "v")):
        for r in _rows(f):
            if "error" in r or r.get("rot") != "harm":
                continue
            if r.get("inj"):
                if not (r["signals"] and r["inj_match"][0][0]
                        and r["signals"][0]["mes"] >= DEV_TRUE_MES):
                    continue
                out.append(("dev_true", r, jobs[r["id"]], [0]))
            elif r["id"].startswith("devreal"):
                ks = [i for i, s in enumerate(r["signals"]) if s["mes"] >= FA_MES]
                if ks:
                    out.append(("dev_fa_real", r, jobs[r["id"]], ks))
    for r in _rows(G2 / "dev_nulls_long_harm.out.jsonl"):
        if r.get("signals") and r["signals"][0]["mes"] >= FA_MES:
            out.append(("dev_fa_null", r, jobs[r["id"]], [0]))
    return out


def tasks_test():
    jobs = _jobs(G2 / "jobs" / "test_inj.jsonl", G2 / "jobs" / "test_real.jsonl",
                 V / "jobs" / "test_nulls_v.jsonl")
    out = []
    for r in _rows(G2 / "test_inj.out.jsonl"):
        if "error" in r or not r.get("inj"):
            continue
        if r["signals"] and r["inj_match"][0][0] and r["signals"][0]["mes"] >= FA_MES:
            out.append(("test_true", r, jobs[r["id"]], [0]))
    for r in _rows(G2 / "test_real.out.jsonl"):
        ks = [i for i, s in enumerate(r.get("signals", [])) if s["mes"] >= FA_MES]
        if ks:
            out.append(("test_fa_real", r, jobs[r["id"]], ks))
    for r in _rows(V / "test_nulls_v.out.jsonl"):
        if r.get("signals") and r["signals"][0]["mes"] >= FA_MES:
            out.append(("test_fa_null", r, jobs[r["id"]], [0]))
    return out


def tasks_holdout():
    jobs = _jobs(G2 / "jobs" / "hosts.jsonl")
    tau = tau_long()
    out = []
    for r in _rows(G2 / "hosts.out.jsonl"):
        ks = [i for i, s in enumerate(r.get("signals", [])) if s["mes"] >= tau]
        if ks:
            out.append(("holdout", r, jobs[r["id"]], ks))
    return out


def tasks_survey(rows_file, jobs_file):
    jobs = _jobs(jobs_file)
    tau = tau_long()
    out = []
    for r in _rows(rows_file):
        if r.get("range", "long") != "long" or "error" in r:
            continue
        ks = [i for i, s in enumerate(r.get("signals", [])) if s["mes"] >= tau]
        if ks:
            out.append(("survey", r, jobs[r["id"]], ks))
    return out


def _load_star(tic):
    from catalog import load_info
    from fetch import load
    from fetch_sap import load_sap

    lc, meta = load(tic)
    sap, smeta = load_sap(tic)
    crowd = {int(s["sector"]): s.get("crowdsap") for s in (smeta or {}).get("sectors", [])}
    return lc, meta, sap, crowd, load_info().get(str(tic))


def run(stage, tasks, out_path):
    from vet import attach_star, base_series, make_ctx, measure

    done = {r["key"] for r in _rows(out_path)}
    # group by (tic, injection id): one preprocessing pass per group
    groups = {}
    for pop, r, job, ks in tasks:
        g = (r["tic"], r["id"] if r.get("inj") else None)
        groups.setdefault(g, []).append((pop, r, job, ks))
    todo = sum(len(ks) for _, _, _, ks in tasks)
    print(f"{stage}: {todo} signals in {len(groups)} groups ({len(done)} done)", flush=True)
    n, t_start = 0, time.time()
    star_cache = {}
    with open(out_path, "a") as fh:
        for (tic, inj_id), items in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
            keys = [f"{r['id']}#{k}" for _, r, _, ks in items for k in ks]
            if all(k in done for k in keys):
                continue
            c0 = time.process_time()
            try:
                if tic not in star_cache:
                    star_cache = {tic: _load_star(tic)}  # keep one star in memory
                lc, meta, sap, crowd, info = star_cache[tic]
                job = items[0][2]
                base = base_series(lc, sap, crowd, inj=items[0][1].get("inj") if inj_id else None,
                                   mass=job["mass"], radius=job["radius"])
            except Exception as e:  # noqa: BLE001
                for k in keys:
                    fh.write(json.dumps({"key": k, "stage": stage, "tic": tic,
                                         "error": f"prepare: {e!r}"[:300]}) + "\n")
                fh.flush()
                continue
            for pop, r, job, ks in items:
                ctx = make_ctx(base, tic, job["mass"], job["radius"], r["mode"], r["seed"],
                               crowd, info)
                for k in ks:
                    key = f"{r['id']}#{k}"
                    if key in done:
                        continue
                    sig = r["signals"][k]
                    row = {"key": key, "stage": stage, "pop": pop, "id": r["id"], "tic": tic,
                           "sig_index": k, "n_sectors": r.get("n_sectors"),
                           "mode": r["mode"], "seed": r["seed"]}
                    if r.get("inj"):
                        row["inj"] = r["inj"]
                        row["inj_match"] = r["inj_match"][k]
                    try:
                        st = measure(ctx, sig)
                        if ctx.mode == "real":
                            attach_star(st, info)
                        row["stats"] = st
                    except Exception as e:  # noqa: BLE001
                        row["error"] = f"measure: {e!r}"[:300]
                        row["trace"] = traceback.format_exc()[-800:]
                    row["cpu_s"] = time.process_time() - c0
                    fh.write(json.dumps(row, default=float) + "\n")
                    n += 1
            fh.flush()
            if n % 25 < len(keys):
                print(f"  {n}/{todo} {time.time() - t_start:.0f}s", flush=True)
    print(f"{stage}: done {n} in {time.time() - t_start:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--rows")
    ap.add_argument("--jobs")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.stage == "test" or a.stage == "holdout":
        if not (V / "frozen_vet.json").exists():
            sys.exit("refusing: TEST / hold-out vetting needs the frozen thresholds first")
    tasks = {"dev": tasks_dev, "test": tasks_test, "holdout": tasks_holdout}.get(a.stage)
    tasks = tasks() if tasks else tasks_survey(a.rows, a.jobs)
    if a.limit:
        tasks = tasks[: a.limit]
    out = Path(a.out) if a.out else V / f"vet_{a.stage}.jsonl"
    run(a.stage, tasks, out)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
