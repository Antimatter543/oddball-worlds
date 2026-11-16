"""Cone-check a few stars against one VizieR catalogue (all its tables), CfA mirror then IDIA (an empty answer
is re-asked on the second mirror). Gaia DR3 positions from gaia_match.csv. Anonymous UA.
Usage: python -I cool/vizcheck.py CATALOGUE RADIUS_ARCSEC TIC [TIC ...]"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from catpass import ANON_USER_AGENT, CATD


def main():
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier

    cat, rad, tics = sys.argv[1], float(sys.argv[2]), [int(x) for x in sys.argv[3:]]
    gm = pd.read_csv(CATD / "gaia_match.csv")
    for t in tics:
        r = gm[gm.tic == t].iloc[0]
        c = SkyCoord(float(r.ra_g) * u.deg, float(r.dec_g) * u.deg)
        ans, n_empty, err = None, 0, ""
        for server in ("vizier.cfa.harvard.edu", "vizier.idia.ac.za"):
            v = Vizier(columns=["**"], row_limit=50, timeout=90)
            v.VIZIER_SERVER = server
            v._session.headers["User-Agent"] = ANON_USER_AGENT
            try:
                res = v.query_region(c, radius=rad * u.arcsec, catalog=cat)
            except Exception as e:  # noqa: BLE001
                err = repr(e)[:120]
                continue
            if len(res):
                ans = res
                break
            n_empty += 1
        if ans is None:
            print(t, "EMPTY on both mirrors" if n_empty >= 2 else f"NOT CHECKED ({err or 'one mirror empty, other failed'})")
            continue
        for name, tab in zip(ans.keys(), ans, strict=True):
            d = tab.to_pandas()
            keep = [x for x in d.columns if x not in ("_RAJ2000", "_DEJ2000")][:14]
            print(t, name, d[keep].to_dict("records"))


if __name__ == "__main__":
    main()
