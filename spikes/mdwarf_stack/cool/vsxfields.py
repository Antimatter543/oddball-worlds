"""AAVSO VSX new-variable fields for the solid, apparently uncatalogued eclipsing binaries (prepared only; nothing
is submitted). Position: Gaia DR3 (epoch 2016.0) moved to epoch J2000.0 with the Gaia proper motion (VizieR
I/355/gaiadr3 cone, cool/catgaia.py), equinox ICRS/J2000. Magnitudes: TESS band (TIC Tmag) at maximum, minimum
= Tmag - 2.5 log10(1 - primary depth) (depth from the trapezoid fit in cool/deep.py; dilution by unresolved
neighbours is not removed, so the range is a lower limit on the true amplitude). Gaia G also given.
Epoch: primary minimum from cool/ebscan.py / deep.py, BJD_TDB = BTJD + 2457000.

Usage: python -I cool/vsxfields.py TIC [TIC ...]   (writes <OUT>/cool/vsx_fields.csv and prints markdown rows)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import OUT

C = OUT / "cool"


def sexa(ra, de):
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    s = SkyCoord(ra * u.deg, de * u.deg)
    return s.to_string("hmsdms", sep=" ", precision=2)


def main():
    tics = [int(x) for x in sys.argv[1:]]
    gm = pd.read_csv(C / "cat" / "gaia_match.csv", dtype={"source_id": "Int64"})
    gs = pd.read_csv(C / "cat" / "gaia_tap_gaia_source.csv", dtype={"source_id": "Int64"})
    T = pd.concat([pd.read_csv(OUT / "gate2" / "survey_targets.csv"), pd.read_csv(OUT / "gate3" / "targets_100pc.csv")])
    T = T.drop_duplicates("tic").set_index("tic")
    rows = []
    for t in tics:
        d = json.loads((C / "deep" / f"{t}.json").read_text())
        m = gm[gm.tic == t].iloc[0]
        g = gs[gs.source_id == m.source_id]
        ra, de = float(m.ra_g), float(m.dec_g)
        pmra = float(g.pmra.iloc[0]) if len(g) and "pmra" in g else 0.0
        pmde = float(g.pmdec.iloc[0]) if len(g) and "pmdec" in g else 0.0
        dt = 2000.0 - 2016.0
        de2 = de + pmde * dt / 3.6e6
        ra2 = ra + pmra * dt / 3.6e6 / np.cos(np.radians(de))
        tm = float(T.loc[t, "Tmag"])
        d1 = d["fit_prim"]["depth"]
        d2 = d["fit_sec"]["depth"] if d.get("fit_sec") else np.nan
        rows.append({"tic": t, "gaia_dr3": int(m.source_id), "ra_dec_J2000": sexa(ra2, de2),
                     "pm_mas_yr": f"{pmra:.1f}, {pmde:.1f}", "type": "EA",
                     "max_T": round(tm, 2), "min_T": round(tm - 2.5 * np.log10(1 - d1), 2),
                     "min2_T": round(tm - 2.5 * np.log10(1 - d2), 2) if np.isfinite(d2) and d2 > 0 else None,
                     "G": round(float(m.G), 2), "period_d": round(d["P"], 5),
                     "epoch_BJD_TDB": round(d["tc"] + 2457000, 4), "sec_phase": round(d["ph2"], 3),
                     "T14_h": round(d["fit_prim"]["T14_h"], 2), "pixel": d.get("pixel", {}).get("status", "")})  # fmt: skip
    out = pd.DataFrame(rows)
    fp = C / "vsx_fields.csv"
    old = pd.read_csv(fp) if fp.exists() else pd.DataFrame()
    if len(old):
        old = old[~old.tic.isin(tics)]
    pd.concat([old, out]).to_csv(fp, index=False)
    pd.set_option("display.width", 250)
    print(out.to_string())


if __name__ == "__main__":
    main()
