"""Re-ask empty RV-survey answers on the other VizieR mirror (CfA), 15" cone.
Usage: rv_recheck_cfa.py OUTJSON"""
import json
import sys
import time

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

STARS = {2102868: (79.33369680453, 32.12492877993), 234295610: (357.98397801044, -64.7931962242),
         146636926: (42.33950035414, -44.26805062609), 141816993: (205.44368664769, 58.25523811632),
         57830249: (174.89049075802, -30.66688311479)}
CATS = {"APOGEE DR17": "III/286", "RAVE DR6": "III/283", "GALAH DR3": "J/MNRAS/506/150",
        "GALAH DR4": "II/385"}
out = {}
for tic, (ra, de) in STARS.items():
    for name, cat in CATS.items():
        try:
            v = Vizier(columns=["_r", "*"], row_limit=50, vizier_server="vizier.cfa.harvard.edu",
                       timeout=90)
            tl = v.query_region(SkyCoord(ra * u.deg, de * u.deg), radius=15 * u.arcsec, catalog=cat)
            out[f"{tic}:{name}"] = {t.meta.get("name", cat): [float(x) for x in t["_r"]] for t in tl}
        except Exception as ex:  # noqa: BLE001
            out[f"{tic}:{name}"] = f"failed {ex!r}"[:200]
        time.sleep(1)
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out, indent=0))
