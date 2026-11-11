"""Validate the GPU coarse fold and full GPU search against the CPU search.

  coarse: per-period MES on the same exact grid, GPU vs CPU brute fold (search.fold_grid)
  full:   GPU search_star vs the CPU results already in controls_real.jsonl / a CPU rerun
Tolerance (fixed here, before comparing): top signal period agrees to |dP|/P < 1e-3 (or the
same integer alias), and MES agrees to within 2% (relative). Per-period coarse MES: median
|dMES| < 1e-3 and 99th percentile < 2% (float32 sums and bin-edge ties).

Usage: python -I validate_gpu.py coarse TIC | full TIC [TIC ...] | inj ID [ID ...]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from common import BIN_D, OUT  # noqa: E402
from gpu_search import GpuCoarse  # noqa: E402
from search import (DOMINANCE, DURATIONS, build_ses, fold_grid, period_grid,  # noqa: E402
                    period_range_for, preprocess, search_star)


def coarse_check(tic, mass=0.3, radius=0.3):
    from fetch import load

    lc, _ = load(tic)
    pp = preprocess(lc)
    T = (pp["idx"].max() - pp["idx"].min() + 1) * BIN_D
    g = GpuCoarse()
    for k in DURATIONS[:3]:
        s = max(1, k // 2)
        pr = period_range_for(k, mass, radius)
        dn = build_ses(pp, k, s, dense=True)
        binw = s * BIN_D
        lo, hi = pr[0], min(pr[1], pr[0] * 1.5)
        t = time.time()
        Pg, mg, tg, _ = g(dn, (lo, hi), binw, T)
        tg_s = time.time() - t
        pg = period_grid(lo, hi, binw, T)
        v = dn["cnt"] > 0
        t = time.time()
        mc, bc = fold_grid(np.flatnonzero(v) * binw, dn["N"][v], dn["D"][v], pg, binw, DOMINANCE)
        tc_s = time.time() - t
        full_g = np.zeros(len(pg))
        full_g[np.searchsorted(pg, Pg)] = mg
        mc = np.where(bc >= 0, mc, 0.0)
        d = np.abs(full_g - mc) / np.maximum(mc, 1e-9)
        nvalid_diff = int(((full_g > 0) != (mc > 0)).sum())
        print(f"tic {tic} k={k}: {len(pg)} periods x {int(v.sum())} samples; GPU {tg_s:.2f}s "
              f"CPU {tc_s:.2f}s; rel dMES median {np.median(d):.1e} p99 {np.percentile(d, 99):.1e}"
              f"; validity flips {nvalid_diff}; top P GPU {pg[np.argmax(full_g)]:.5f} "
              f"({full_g.max():.3f}) CPU {pg[np.argmax(mc)]:.5f} ({mc.max():.3f})", flush=True)


def full(jobs):
    from run_jobs import run_job

    import run_jobs
    g = GpuCoarse()
    orig = run_jobs.__dict__.get("search_star")
    import search as S

    def gpu_search_star(pp, m, r, ms):
        return S.search_star(pp, m, r, ms, coarse=g)

    out = []
    for job in jobs:
        S_ss = S.search_star
        S.search_star = lambda pp, m, r, ms, floor=S.MES_FLOOR: S_ss(pp, m, r, ms, floor, g)
        try:
            g.gpu_s = 0.0
            res = run_job(job)
            res["gpu_s"] = g.gpu_s
        finally:
            S.search_star = S_ss
        out.append(res)
    _ = orig
    return out


def compare():
    def rd(n):
        p = OUT / n
        return {r["id"]: r for r in map(json.loads, open(p))} if p.exists() else {}

    gpu = rd("val_gpu.jsonl")
    cpu = {**rd("controls_real.jsonl"), **rd("val_cpu.jsonl")}
    print("| job | n_sec | CPU top P / MES | GPU top P / MES | dP/P | dMES/MES | pass | "
          "CPU s | GPU-run CPU s + GPU s |")
    print("|---|---|---|---|---|---|---|---|---|")
    allpass = True
    for jid, g in gpu.items():
        c = cpu.get(jid)
        if c is None or "error" in g or "error" in c:
            print(f"| {jid} | | missing CPU reference or error | | | | | | |")
            continue
        cs = c["signals"][0] if c["signals"] else None
        gs = g["signals"][0] if g["signals"] else None
        if cs is None or gs is None:
            ok = cs is None and gs is None
            print(f"| {jid} | {c['n_sectors']} | {cs and round(cs['mes'], 2)} | "
                  f"{gs and round(gs['mes'], 2)} | | | {ok} | | |")
            allpass &= ok
            continue
        r = gs["P"] / cs["P"]
        alias = min(abs(r - round(r)) / max(round(r), 1),
                    abs(1 / r - round(1 / r)) / max(round(1 / r), 1))
        dp = abs(r - 1)
        dm = abs(gs["mes"] - cs["mes"]) / cs["mes"]
        ok = (dp < 1e-3 or alias < 1e-3) and dm < 0.02
        allpass &= ok
        print(f"| {jid} | {c['n_sectors']} | {cs['P']:.4f} / {cs['mes']:.2f} | "
              f"{gs['P']:.4f} / {gs['mes']:.2f} | {dp:.1e} | {dm:.3f} | {ok} | "
              f"{c['cpu_s']:.0f} | {g['cpu_s']:.0f} + {g.get('gpu_s', 0):.0f} |")
    print("all pass:", allpass)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "coarse":
        coarse_check(int(sys.argv[2]))
    elif mode == "compare":
        compare()
