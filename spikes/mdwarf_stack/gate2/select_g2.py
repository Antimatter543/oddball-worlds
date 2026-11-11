"""Gate 2: enlarge the control set to 120 and split DEV / TEST by a fixed seed.

Same selection rules as gate 1 (select_controls.py: Teff 2700-3400 K, d < 50 pc, |ecl lat| > 60,
no TOI of any disposition, no exoplanet.eu object within 30", known hosts excluded), same
sector-count strata and the same 6:8:8:8 weights, scaled to 24/32/32/32. The 30 gate-1 controls
are kept and all go to DEV. The 90 new stars are drawn with a new fixed seed, without the old
ones; within each stratum one third of the new stars go to DEV and two thirds to TEST, by a
second fixed seed. Fast rotators are not filtered (natural rate).

Usage: python -I spikes/mdwarf_stack/gate2/select_g2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CAT_DIR
from select_controls import HOSTS, sectors_for

STRATA = [((2, 5), 24), ((6, 12), 32), ((13, 25), 32), ((26, 200), 32)]
SEED_DRAW, SEED_SPLIT = 20261121, 20261122


def main() -> None:
    old = pd.read_csv(CAT_DIR / "controls.csv")
    tic = pd.read_csv(CAT_DIR / "tic_cvz_mdwarfs.csv")
    tic = tic[tic["disposition"].isna()].drop_duplicates("ID")
    toi = pd.read_csv(CAT_DIR / "toi.csv", comment="#")
    tic = tic[~tic["ID"].isin(set(toi["TIC ID"])) & ~tic["ID"].isin(HOSTS)]
    eu = pd.read_csv(CAT_DIR / "exoplanet_eu.csv")
    eu = eu[np.isfinite(eu["ra"]) & np.isfinite(eu["dec"])]
    ra1, de1 = np.radians(tic["ra"].to_numpy())[:, None], np.radians(tic["dec"].to_numpy())[:, None]
    ra2, de2 = np.radians(eu["ra"].to_numpy())[None], np.radians(eu["dec"].to_numpy())[None]
    cosd = np.sin(de1) * np.sin(de2) + np.cos(de1) * np.cos(de2) * np.cos(ra1 - ra2)
    sep = np.degrees(np.arccos(np.clip(cosd, -1, 1))) * 3600
    tic = tic[sep.min(axis=1) > 30.0]
    secs = sectors_for(tic["ID"], tic["ra"], tic["dec"])
    tic = tic.assign(n_pred=[len(secs.get(int(i), [])) for i in tic["ID"]])
    pool_all = tic[~tic["ID"].isin(set(old["tic"]))]
    rng = np.random.default_rng(SEED_DRAW)
    rsp = np.random.default_rng(SEED_SPLIT)
    rows = []
    for (lo, hi), n in STRATA:
        n_old = int(((old["n_pred"] >= lo) & (old["n_pred"] <= hi)).sum())
        pool = pool_all[(pool_all["n_pred"] >= lo) & (pool_all["n_pred"] <= hi)]
        n_new = min(n - n_old, len(pool))
        print(f"stratum {lo}-{hi}: pool {len(pool)}, old {n_old}, new {n_new}")
        new = pool.iloc[rng.choice(len(pool), size=n_new, replace=False)].copy()
        perm = rsp.permutation(n_new)
        n_dev = n_new // 3
        new["split"] = np.where(np.isin(np.arange(n_new), perm[:n_dev]), "DEV", "TEST")
        new["origin"] = "gate2"
        rows.append(new)
    new = pd.concat(rows).rename(columns={"ID": "tic"})
    o = old.assign(split="DEV", origin="gate1")
    cols = ["tic", "ra", "dec", "Tmag", "Teff", "rad", "mass", "d", "eclat", "n_pred", "split",
            "origin"]
    sel = pd.concat([o, new])[cols].sort_values(["split", "n_pred"])
    sel.to_csv(CAT_DIR / "controls_g2.csv", index=False)
    print(sel.groupby(["split", pd.cut(sel["n_pred"], [1, 5, 12, 25, 200])], observed=True)
          .size().to_string())


if __name__ == "__main__":
    main()
