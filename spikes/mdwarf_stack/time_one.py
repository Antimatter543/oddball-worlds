"""Timing probe on one CONTROL star (never a host): per-duration SES size, periods and fold time."""

from __future__ import annotations

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BIN_D, CAT_DIR
from fetch import load
from search import DURATIONS, build_ses, fold_grid, period_grid, period_range_for, preprocess

tic = int(sys.argv[1])
c = pd.read_csv(CAT_DIR / "controls.csv").set_index("tic").loc[tic]
lc, meta = load(tic)
t = time.time()
pp = preprocess(lc)
print(f"preprocess {time.time() - t:.1f}s bins {len(lc['idx'])} -> {len(pp['idx'])}")
T = (pp["idx"].max() - pp["idx"].min() + 1) * BIN_D
fold_grid(np.arange(10.0), np.ones(10), np.ones(10), np.array([2.0, 3.0]), 0.1, 0.8)  # compile
tot = 0
for k in DURATIONS:
    pr = period_range_for(k, c["mass"], c["rad"])
    if pr is None:
        continue
    s = max(1, round(k / 3))
    pg = period_grid(pr[0], pr[1], s * BIN_D, T)
    ses = build_ses(pp, k, s)
    t = time.time()
    sub = pg[:: max(1, len(pg) // 2000)]
    fold_grid(ses.t - ses.t.min(), ses.N, ses.D, sub, s * BIN_D, 0.8)
    dt = (time.time() - t) / len(sub) * len(pg)
    tot += dt
    print(f"k={k:2d} s={s} P {pr[0]:.2f}-{pr[1]:.2f} nP={len(pg):7d} nS={len(ses.t):6d} "
          f"est fold {dt:6.1f}s  ns/op {dt / len(pg) / len(ses.t) * 1e9:.2f}")
print(f"T_base {T:.0f} d, est total per iteration {tot:.0f}s")
