"""Allowed periods for a few-event eclipser: which P = (t_j - t_i)/n put every observed event on the same
ephemeris (or on a second, secondary-eclipse phase) without predicting an eclipse in covered data where
none is seen.

Given event times (same kind, e.g. three ~10% dips) and the eclipse width, for n = 1..nmax over the
shortest spacing (P >= pmin): an epoch is "covered" when >= 50% of its window has data; a covered
predicted epoch with mean depth < depth_ref/3 AND inconsistent with depth_ref/2 at 3 sigma excludes P.
Also requires every event to fall within width/2 of the ephemeris (timing tolerance 1.5 h by default).
Usage: python -I cool/aliases.py TIC --events t1 t2 [t3 ...] --width-h 2 [--depth 0.1] [--pmin 1]
Writes <OUT>/cool/aliases/<tic>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "survey"))
sys.path.insert(0, str(HERE))
from common import OUT
from ebscan import BIN, detrend, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tic", type=int)
    ap.add_argument("--events", type=float, nargs="+", required=True)
    ap.add_argument("--width-h", type=float, default=2.0)
    ap.add_argument("--depth", type=float, default=None)
    ap.add_argument("--pmin", type=float, default=1.0)
    ap.add_argument("--tol-h", type=float, default=1.5)
    a = ap.parse_args()
    t, f, sec, meta = load(a.tic)
    r, s = detrend(t, f)
    ev = np.sort(np.array(a.events))
    w = a.width_h / 24
    # measured depth of each event
    deps = []
    for e in ev:
        m = np.abs(t - e) <= w / 2
        deps.append(float(-r[m].mean()) if m.any() else np.nan)
    dref = a.depth or float(np.nanmin(deps))
    span = np.diff(ev).min() if len(ev) > 1 else None
    base = ev[-1] - ev[0]
    allowed = []
    tested = 0
    nmax = int(base / a.pmin) + 1
    for n in range(1, nmax + 1):
        P = base / n
        if P < a.pmin:
            break
        # every event on the ephemeris anchored at the first
        k = (ev - ev[0]) / P
        if np.max(np.abs(k - np.round(k))) * P > a.tol_h / 24:
            continue
        tested += 1
        kk = np.round((t - ev[0]) / P)
        dt = t - (ev[0] + kk * P)
        inw = np.abs(dt) <= w / 2
        bad, cov = [], 0
        for q in np.unique(kk[inw]):
            m = inw & (kk == q)
            if m.sum() < 0.5 * w / BIN:
                continue
            tq = ev[0] + q * P
            if np.min(np.abs(ev - tq)) < w:
                continue
            cov += 1
            d = -r[m].mean()
            e = s / np.sqrt(m.sum())
            if d < dref / 3 and (dref / 2 - d) / e > 3:
                bad.append(round(float(tq), 3))
        if not bad:
            allowed.append({"P": float(P), "n": n, "covered_other_epochs": cov})
    out = {"tic": a.tic, "events": ev.tolist(), "event_depths": deps, "depth_ref": dref, "width_h": a.width_h,
           "baseline_d": float(base), "n_tested": tested, "allowed": allowed}  # fmt: skip
    d = OUT / "cool" / "aliases"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{a.tic}.json").write_text(json.dumps(out, indent=1))
    print(a.tic, "event depths", np.round(deps, 4), "| tested", tested, "allowed", len(allowed))
    for x in allowed[:40]:
        print(f"  P={x['P']:.4f} (n={x['n']}, other covered epochs {x['covered_other_epochs']})")


if __name__ == "__main__":
    main()
