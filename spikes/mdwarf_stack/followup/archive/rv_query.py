"""Multi-epoch RV search (VizieR spectroscopic surveys + Gaia DR3 RV statistics from the ESA TAP).
Usage: rv_query.py OUTJSON"""
import json
import sys
import time

import astropy.units as u
import numpy as np
import requests
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier

outj = sys.argv[1]
UA = {"User-Agent": "astro-research-client/1.0"}
STARS = {2102868: (79.33369680453, 32.12492877993, 86.155, -296.162, 180614374211036416),
         234295610: (357.98397801044, -64.7931962242, 74.585, -59.845, 6485423779719878016),
         146636926: (42.33950035414, -44.26805062609, 69.578, 74.576, 4947259307983420928),
         141816993: (205.44368664769, 58.25523811632, 81.775, -59.357, 1658968059093995136),
         57830249: (174.89049075802, -30.66688311479, -86.422, -25.834, 3478940625208241920)}
SERVERS = ("vizier.idia.ac.za", "vizier.cfa.harvard.edu")
log = {"catalog_search": {}, "queries": {}, "gaia_tap": {}}

# find catalogue ids by keyword
cats = {}
for kw in ["LAMOST", "APOGEE DR17", "GALAH DR3", "GALAH DR4", "RAVE DR6"]:
    for server in SERVERS:
        try:
            v = Vizier(vizier_server=server, timeout=60)
            fc = v.find_catalogs(kw)
            log["catalog_search"][kw] = {k: str(val.description)[:120] for k, val in fc.items()}
            break
        except Exception as ex:  # noqa: BLE001
            log["catalog_search"][f"{kw}@{server}"] = f"failed {ex!r}"[:200]
        time.sleep(1)
    time.sleep(1)

CATS = ["III/286", "V/164", "V/156", "V/146", "V/153", "J/MNRAS/506/150", "III/283",
        "J/ApJS/266/35", "J/ApJS/245/34"]
for kw, d in log["catalog_search"].items():
    if isinstance(d, dict):
        for k, desc in d.items():
            if any(s in desc.lower() for s in ["radial", "velocit", "dr", "spectro", "medium", "low-res"]):
                if k.split("/")[0] in ("III", "V", "J") and k not in CATS:
                    CATS.append(k)
CATS = CATS[:24]
log["cats_queried"] = CATS

res = {}
for tic, (ra, de, pa, pd, gid) in STARS.items():
    res[tic] = {}
    c = SkyCoord(ra * u.deg, de * u.deg)
    for cat in CATS:
        got = None
        for server in SERVERS:
            try:
                v = Vizier(columns=["**", "_r"], row_limit=200, vizier_server=server, timeout=90)
                tl = v.query_region(c, radius=10 * u.arcsec, catalog=cat)
                got = {}
                for t in tl:
                    rows = []
                    for r in t:
                        rows.append({k: (None if np.ma.is_masked(r[k]) else
                                         (r[k].item() if hasattr(r[k], "item") else str(r[k])))
                                     for k in t.colnames})
                    got[t.meta.get("name", cat)] = rows
                log["queries"][f"{tic}:{cat}"] = f"ok {server} {sum(len(x) for x in got.values())} rows"
                break
            except Exception as ex:  # noqa: BLE001
                log["queries"][f"{tic}:{cat}@{server}"] = f"failed {ex!r}"[:200]
            time.sleep(0.5)
        if got:
            res[tic][cat] = got
        time.sleep(0.5)

# Gaia DR3 RV statistics (ESA archive TAP, anonymous sync)
ids = ",".join(str(v[4]) for v in STARS.values())
q = ("SELECT source_id, radial_velocity, radial_velocity_error, rv_nb_transits, "
     "rv_chisq_pvalue, rv_renormalised_gof, rv_amplitude_robust, rv_expected_sig_to_noise, "
     "rv_method_used, rv_template_teff, grvs_mag, ruwe, ipd_frac_multi_peak, "
     "ipd_gof_harmonic_amplitude, non_single_star, phot_variable_flag, has_epoch_photometry "
     f"FROM gaiadr3.gaia_source WHERE source_id IN ({ids})")
for url in ["https://gea.esac.esa.int/tap-server/tap/sync"]:
    try:
        r = requests.post(url, data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json",
                                     "QUERY": q}, headers=UA, timeout=120)
        r.raise_for_status()
        j = r.json()
        cols = [m["name"] for m in j["metadata"]]
        log["gaia_tap"]["rows"] = [dict(zip(cols, row)) for row in j["data"]]
        log["gaia_tap"]["status"] = f"ok {url}"
    except Exception as ex:  # noqa: BLE001
        log["gaia_tap"]["status"] = f"failed {url} {ex!r}"[:300]

json.dump({"log": log, "results": {str(k): v for k, v in res.items()}}, open(outj, "w"),
          indent=1, default=str)
print(json.dumps(log["queries"], indent=0)[:3000])
print(json.dumps(log["gaia_tap"], indent=1, default=str))
for tic, d in res.items():
    for cat, tabs in d.items():
        for tn, rows in tabs.items():
            print(tic, cat, tn, len(rows))
