"""Vetting v2 measurement (raw statistics only; grading is done later with frozen thresholds).

Stages
  dev     the v1 DEV tasks (run_vet.tasks_dev); only the v2-only statistics are computed and
          stored by key (v1 statistics stay in vetting/vet_dev.jsonl) -> v2/vet2_dev.jsonl
  val     fresh validation set (v2/val2.out.jsonl): null top signals with MES >= 6 and matched
          injection top signals with MES >= tau_long; full v1 measure + v2 statistics
          -> v2/vet2_val.jsonl (both versions are graded from the same row)
  survey  --rows SURVEY_ROWS.jsonl [--jobs JOBS.jsonl]: every long signal >= tau_long, full v1 +
          v2 statistics -> v2/vet2_survey.jsonl (same tasks as run_vet.py survey)

--shard I/N splits by star (TIC mod N) so two niced processes can share a stage; resumable by key.
BLINDING (val): only null realisations and injected copies; a real unaltered row aborts the run,
and an injected copy's top signal is used only when it matches the injection.
"""

from __future__ import annotations

import os

for _v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "gate2"))
sys.path.insert(0, str(HERE))
from common import OUT
from run_vet import _jobs, _load_star, _rows, tasks_dev, tasks_survey, tau_long

V2 = OUT / "vetting" / "v2"
FA_MES = 6.0


def tasks_val():
    jobs = _jobs(V2 / "jobs" / "val2.jsonl")
    tau = tau_long()
    out = []
    for r in _rows(V2 / "val2.out.jsonl"):
        if "error" in r:
            continue
        if r.get("mode", "real") == "real" and not r.get("inj"):
            raise SystemExit(f"BLINDING: real unaltered row {r['id']} in the validation output")
        if not r.get("signals"):
            continue
        if r.get("inj"):
            if r["inj_match"][0][0] and r["signals"][0]["mes"] >= tau:
                out.append(("val_true", r, jobs[r["id"]], [0]))
        elif r["signals"][0]["mes"] >= FA_MES:
            out.append(("val_fa_null", r, jobs[r["id"]], [0]))
    return out


def run(stage, tasks, out_path, extra_only=False, shard=(0, 1)):
    from vet import attach_star, base_series, make_ctx
    from vet2 import measure2, measure_extra

    i_sh, n_sh = shard
    tasks = [t for t in tasks if int(t[1]["tic"]) % n_sh == i_sh]
    done = {r["key"] for r in _rows(out_path)}
    groups = {}
    for pop, r, job, ks in tasks:
        g = (r["tic"], r["id"] if r.get("inj") else None)
        groups.setdefault(g, []).append((pop, r, job, ks))
    todo = sum(len(ks) for _, _, _, ks in tasks)
    print(f"{stage} shard {i_sh}/{n_sh}: {todo} signals in {len(groups)} groups "
          f"({len(done)} done)", flush=True)
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
                    star_cache = {tic: _load_star(tic)}
                lc, meta, sap, crowd, info = star_cache[tic]
                job = items[0][2]
                base = base_series(lc, sap, crowd, inj=items[0][1].get("inj") if inj_id else None,
                                   mass=job["mass"], radius=job["radius"])
            except Exception as e:  # noqa: BLE001
                fh.writelines(json.dumps({"key": k, "stage": stage, "tic": tic,
                                         "error": f"prepare: {e!r}"[:300]}) + "\n" for k in keys)
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
                           "mode": r["mode"], "seed": r["seed"], "mass": job["mass"],
                           "radius": job["radius"]}
                    if r.get("inj"):
                        row["inj"] = r["inj"]
                        row["inj_match"] = r["inj_match"][k]
                    try:
                        if extra_only:
                            row["v2"] = measure_extra(ctx, sig)
                        else:
                            st = measure2(ctx, sig)
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
    ap.add_argument("stage", choices=("dev", "val", "survey"))
    ap.add_argument("--rows")
    ap.add_argument("--jobs")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out")
    a = ap.parse_args()
    shard = tuple(int(x) for x in a.shard.split("/"))
    V2.mkdir(parents=True, exist_ok=True)
    if a.stage in ("val", "survey") and not (V2 / "frozen_vet2.json").exists():
        sys.exit("refusing: validation / survey vetting needs frozen_vet2.json first")
    if a.stage == "dev":
        tasks, extra = tasks_dev(), True
    elif a.stage == "val":
        tasks, extra = tasks_val(), False
    else:
        tasks, extra = tasks_survey(a.rows, a.jobs), False
    out = Path(a.out) if a.out else V2 / f"vet2_{a.stage}.jsonl"
    if shard[1] > 1 and not a.out:
        out = V2 / f"vet2_{a.stage}.s{shard[0]}of{shard[1]}.jsonl"
    run(a.stage, tasks, out, extra_only=extra, shard=shard)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
