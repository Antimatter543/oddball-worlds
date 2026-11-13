"""K2 / Kepler coverage at MAST by coordinate (anonymous). Usage: k2_kepler_mast.py OUTJSON"""
import json
import sys
import time

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Observations

Observations._session.headers["User-Agent"] = "astro-research-client/1.0"
STARS = {2102868: (79.33369680453, 32.12492877993), 234295610: (357.98397801044, -64.7931962242),
         146636926: (42.33950035414, -44.26805062609), 141816993: (205.44368664769, 58.25523811632),
         57830249: (174.89049075802, -30.66688311479)}
import os
out = json.load(open(sys.argv[1])) if os.path.exists(sys.argv[1]) else {}
sel = [int(x) for x in sys.argv[2:]] or list(STARS)
for tic, (ra, de) in [(t, STARS[t]) for t in sel]:
    tic = str(tic)
    try:
        t = Observations.query_criteria(coordinates=SkyCoord(ra * u.deg, de * u.deg),
                                        radius=0.01 * u.deg, obs_collection=["K2", "Kepler"])
        out[tic] = {"status": "ok", "n_obs": len(t),
                    "rows": [{k: str(r[k]) for k in ["obs_collection", "obs_id", "target_name",
                                                       "t_min", "t_max", "dataproduct_type"]}
                             for r in t]}
    except Exception as ex:  # noqa: BLE001
        out[tic] = {"status": f"failed {ex!r}"[:300]}
    time.sleep(1)
    # also K2 campaign footprint via the K2 FFI/ superstamp-independent catalogue (EPIC) cone
    try:
        from astroquery.vizier import Vizier
        v = Vizier(row_limit=5, vizier_server="vizier.idia.ac.za", timeout=60)
        r = v.query_region(SkyCoord(ra * u.deg, de * u.deg), radius=20 * u.arcsec,
                           catalog="IV/34")
        out[tic]["epic_rows"] = sum(len(x) for x in r)
    except Exception as ex:  # noqa: BLE001
        out[tic]["epic_rows"] = f"failed {ex!r}"[:200]
    time.sleep(1)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps({k: (v["status"], v.get("n_obs"), v.get("epic_rows")) for k, v in out.items()}))
