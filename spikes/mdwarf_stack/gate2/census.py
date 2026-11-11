"""Gate 2 rotation census on each star's REAL (uninjected) light curve. No search result is used.

Per sector: the harm Lomb-Scargle (search2.rot_sector, 0.1-5 d) peak period and normalised power.
Per star: n_sec, frac_strong = fraction of sectors whose peak power >= POW_STRONG, the median power,
and prot_med = median peak period over the strong sectors.

Why not the harm 'amp' field: for P_rot ~ 3-5 d the harmonic terms (spline amplitudes, 3 d knots)
are nearly collinear with the spline baseline, so the rotational part alone can be huge while the
total model is fine (DEV TIC 82355915: amp 292 in relative flux). Only the total model is used
for detrending, so this does not affect the search; it does make 'amp' useless for classifying.

The fast/slow rule itself (POW_STRONG, fast_min_frac) is fixed from DEV in frozen.json.

Usage: python -I gate2/census.py DEV|TEST|HOSTS|ALL
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import BIN_D, CAT_DIR, LC_DIR, OUT

POW_STRONG = 0.1
G2 = OUT / "gate2"


def star_census(tic: int) -> dict:
    from fetch import load
    from search2 import rot_sector

    lc, meta = load(tic)
    t = (lc["idx"] + 0.5) * BIN_D
    f = lc["flux"].astype(np.float64)
    secs = []
    for s in np.unique(lc["sector"]):
        m = lc["sector"] == s
        _, info = rot_sector(t[m], f[m])
        secs.append((int(s), info["P"], info["power"]))
    pw = np.array([x[2] for x in secs])
    strong = [x for x in secs if x[2] >= POW_STRONG]
    return {"tic": tic, "n_sec": len(secs), "frac_strong": float(np.mean(pw >= POW_STRONG)),
            "power_med": float(np.median(pw)),
            "prot_med": float(np.median([x[1] for x in strong])) if strong else np.nan,
            "per_sector": json.dumps(secs)}


def main() -> None:
    which = sys.argv[1]
    tics = []
    if which in ("DEV", "TEST", "ALL"):
        c = pd.read_csv(CAT_DIR / "controls_g2.csv")
        if which != "ALL":
            c = c[c["split"] == which]
        tics += [(int(t), s) for t, s in zip(c["tic"], c["split"])]
    if which in ("HOSTS", "ALL"):
        from make_jobs import HOSTS

        h = pd.read_csv(CAT_DIR / "holdout_g2.csv")
        h = h[h["in_range"] & h["tic"].notna()]
        tics += [(int(t), "g1host") for t in HOSTS] + [(int(t), "holdout")
                                                      for t in h["tic"].unique()]
    rows = []
    for t, s in tics:
        if not (LC_DIR / f"tic{t}.npz").exists():
            continue
        rows.append(star_census(t) | {"set": s})
    df = pd.DataFrame(rows).drop_duplicates("tic")
    out = G2 / f"rot_census_{which}.csv"
    df.to_csv(out, index=False)
    print(df.drop(columns="per_sector").sort_values("frac_strong").to_string(index=False))
    print("written", out)


if __name__ == "__main__":
    main()
