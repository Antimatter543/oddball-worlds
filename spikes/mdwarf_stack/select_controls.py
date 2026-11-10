"""Pick the control set: mid-to-late M dwarfs within 50 pc, no TOI, no known planet.

Inputs (cached under data/planets_spike/catalogs):
  tic_cvz_mdwarfs.csv  TIC v8 query: Teff 2700-3400 K, d 1-50 pc, Tmag 6-14.5, |ecl lat| > 60
  toi.csv              ExoFOP TOI table (all dispositions, including FPs: a TOI is excluded
                       whatever its disposition, so a control never has a known transit-like signal)
  exoplanet_eu.csv     exoplanet.eu catalogue (RV and imaging planets too), matched within 30"

Selection is stratified by predicted sector count (tess-point) so that the cost and
injection results span the survey's range, with a fixed seed. Hosts are excluded explicitly.

Usage: uv run python -I spikes/mdwarf_stack/select_controls.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CAT_DIR  # noqa: E402

HOSTS = {92226327, 150428135, 52005579}
# Strata of predicted sector count -> number of stars. Weighted to 6+ sectors (criterion b).
STRATA = [((2, 5), 6), ((6, 12), 8), ((13, 25), 8), ((26, 200), 8)]
SEED = 20261111


def sectors_for(tics, ra, dec) -> dict[int, list[int]]:
    from tess_stars2px import tess_stars2px_function_entry

    out = tess_stars2px_function_entry(np.asarray(tics), np.asarray(ra), np.asarray(dec))
    ids, sec = out[0], out[3]
    res: dict[int, set[int]] = {}
    for i, s in zip(ids, sec):
        if s > 0:
            res.setdefault(int(i), set()).add(int(s))
    return {k: sorted(v) for k, v in res.items()}


def main() -> None:
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
    print(f"eligible after exclusions: {len(tic)}; n_pred quantiles",
          np.percentile(tic["n_pred"], [10, 50, 90]))

    rng = np.random.default_rng(SEED)
    pick = []
    for (lo, hi), n in STRATA:
        pool = tic[(tic["n_pred"] >= lo) & (tic["n_pred"] <= hi)]
        pick.append(pool.iloc[rng.choice(len(pool), size=min(n, len(pool)), replace=False)])
    sel = pd.concat(pick)[["ID", "ra", "dec", "Tmag", "Teff", "rad", "mass", "d", "eclat", "n_pred"]]
    sel = sel.rename(columns={"ID": "tic"}).sort_values("n_pred")
    sel.to_csv(CAT_DIR / "controls.csv", index=False)
    print(sel.to_string(index=False))


if __name__ == "__main__":
    main()
