"""Part 2.5 sample: TIC v8 M dwarfs at 50-100 pc (Teff 2700-3400 K, T < 15), in 30 random
sky cones of radius 6 deg (seed 20261113; scaled to the full sky by area), with tess-point
sectors through S107. Also the same cones at < 50 pc as a check of the scaling against the
gate-1 catalogue. Writes FU_OUT/tic100_sample.csv. Usage: python -I followup/tic100.py
"""

from __future__ import annotations

import math
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fu

N_CONE, R_DEG = 30, 6.0


def main():
    import download
    import tess_stars2px as t2p

    download._set_astroquery_ua()
    from astropy.coordinates import SkyCoord
    from astroquery.mast import Catalogs

    Catalogs._portal_api_connection.TIMEOUT = 180

    def _alarm(*_):
        raise TimeoutError("cone wall clock 240 s")

    signal.signal(signal.SIGALRM, _alarm)
    cache = fu.FU_DATA / "tic100_cones"
    cache.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20261113)
    ra = rng.uniform(0, 360, N_CONE)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, N_CONE)))
    rows, ok_cones = [], []
    for i, (a, d) in enumerate(zip(ra, dec)):
        cf = cache / f"cone{i:02d}.csv"
        if cf.exists():
            df = pd.read_csv(cf)
            rows.append(df)
            ok_cones.append(i)
            print("cone", i, "cached", len(df), flush=True)
            continue
        c = SkyCoord(a, d, unit="deg")
        for k in range(3):
            try:
                signal.alarm(240)
                r = Catalogs.query_criteria(catalog="Tic", coordinates=c, radius=R_DEG,
                                            Teff=[2700, 3400], d=[0, 100], Tmag=[-5, 15],
                                            objType="STAR")
                signal.alarm(0)
                break
            except Exception as ex:  # noqa: BLE001
                signal.alarm(0)
                print("retry", i, repr(ex)[:150], flush=True)
                time.sleep(10 * (k + 1))
        else:
            print("cone FAILED (not checked)", i, flush=True)
            continue
        df = r[["ID", "ra", "dec", "Tmag", "Teff", "rad", "mass", "d", "contratio"]].to_pandas()
        df.to_csv(cf, index=False)
        ok_cones.append(i)
        df["cone"] = i
        rows.append(df)
        print("cone", i, round(a, 1), round(d, 1), len(df), flush=True)
        time.sleep(0.5)
    df = pd.concat(rows).drop_duplicates("ID")
    secs = []
    for _, r in df.iterrows():
        out = t2p.tess_stars2px_function_entry(int(r["ID"]), float(r["ra"]), float(r["dec"]))
        s = sorted({int(x) for x in np.atleast_1d(out[3]) if 0 < int(x) <= 107})
        secs.append(" ".join(map(str, s)))
    df["sectors"] = secs
    df["n_sectors"] = [len(s.split()) for s in secs]
    cap = 2 * math.pi * (1 - math.cos(math.radians(R_DEG)))
    frac = len(ok_cones) * cap / (4 * math.pi)
    df.to_csv(fu.FU_OUT / "tic100_sample.csv", index=False)
    (fu.FU_OUT / "tic100_meta.json").write_text(
        f'{{"ok_cones": {ok_cones}, "n_cone": {N_CONE}, "r_deg": {R_DEG}, "area_frac": {frac}}}')
    print("cones ok", len(ok_cones), "area fraction (cones may overlap)", frac, "rows", len(df))


if __name__ == "__main__":
    main()
