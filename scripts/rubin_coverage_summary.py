"""Summarise ALeRCE LSST coverage of young-star tiles by Galactic latitude (from tiles.csv).

    uv run python -I scripts/rubin_coverage_summary.py
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from oddball.rubin.alerce_api import data_root


def main() -> int:
    t = pd.read_csv(data_root() / "rubin" / "tiles.csv")
    c = SkyCoord(ra=t["ra_c"].to_numpy() * u.deg, dec=t["dec_c"].to_numpy() * u.deg).galactic
    t["absb"] = np.abs(c.b.deg)
    t["bbin"] = pd.cut(t["absb"], [0, 2, 5, 10, 20, 90], include_lowest=True)
    t["nonempty"] = t["has_lsst"].astype(bool) if "has_lsst" in t else t["n_obj"] > 0
    g = t.groupby("bbin", observed=True).agg(
        tiles=("tile_id", "size"), tiles_with_lsst=("nonempty", "sum"),
        yso_rows=("n_cat", "sum"),
        yso_rows_in_lsst_tiles=("n_cat", lambda s: int(s[t.loc[s.index, "nonempty"]].sum())),
        lsst_objects_ndet2=("n_obj", "sum"))
    g["frac_tiles_with_lsst"] = (g["tiles_with_lsst"] / g["tiles"]).round(3)
    print("| |b| (deg) | tiles | tiles with LSST objects | YSO catalogue rows | rows in those tiles"
          " | LSST objects (n_det>=2) |")
    print("|---|---|---|---|---|---|")
    for b, r in g.iterrows():
        print(f"| {b} | {int(r.tiles)} | {int(r.tiles_with_lsst)} ({r.frac_tiles_with_lsst:.1%}) | "
              f"{int(r.yso_rows)} | {int(r.yso_rows_in_lsst_tiles)} | {int(r.lsst_objects_ndet2)} |")
    print(f"\ntotal tiles {len(t)}, with LSST objects {int(t['nonempty'].sum())}, "
          f"truncated {int(t['truncated'].sum())}, errors {int((t['error'].fillna('') != '').sum())}")
    top = t[t["nonempty"]].sort_values("n_obj", ascending=False).head(8)
    top = top.assign(l=c.l.deg[top.index], b=c.b.deg[top.index])
    print(top[["tile_id", "ra_c", "dec_c", "l", "b", "n_cat", "n_obj"]].round(1).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
