"""Catalogue research for the vetted survey signals worth reading (the shortlist), per star.

Anonymous (oddball.common.http.ANON_USER_AGENT), one request at a time with pauses, cached as
OUT/survey/vet/research/<tic>.json. Each source records "ok" or its error; a failed source is
"not checked", never "not known".

  exofop     ExoFOP target JSON (TOIs, CTOIs, TFOP notes / tags / observation counts)
  nasa       NASA Exoplanet Archive TAP, pscomppars within 36" of the Gaia position
  simbad     SIMBAD TAP: objects within 30" (main id, type, reference count) and the target's
             identifiers
  gaia       Gaia DR3 gaia_source (VizieR I/355/gaiadr3) within 63" (3 TESS pixels): RUWE,
             non-single-star flag, RV and its error, neighbours with dG
  gaia_nss   Gaia DR3 non-single-star orbits (I/357/tbooc) within 5"
  prsa_eb    TESS EB catalogue (Prsa+2022, J/ApJS/258/16) within 42"

Usage: python -I survey/research_candidates.py TIC [TIC ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "src"))

from common import OUT

RES = OUT / "survey" / "vet" / "research"
MIRRORS = ("vizier.idia.ac.za", "vizier.cfa.harvard.edu")


def _ua():
    from oddball.common.http import ANON_USER_AGENT

    return ANON_USER_AGENT


def _get(url, params=None, timeout=90):
    import requests

    r = requests.get(url, params=params, headers={"User-Agent": _ua()}, timeout=timeout)
    r.raise_for_status()
    return r


def exofop(tic):
    r = _get("https://exofop.ipac.caltech.edu/tess/target.php", {"id": tic, "json": ""})
    d = r.json()
    keep = {}
    for k in ("basic_info", "tois", "ctois", "tags", "notes", "obs_notes", "imaging", "spectroscopy",
              "time_series", "planet_parameters", "stellar_parameters", "magnitudes"):
        if k in d:
            keep[k] = d[k]
    keep["keys"] = sorted(d)
    return keep


def nasa(ra, dec):
    # a box (the archive's firewall answers 403 to ADQL contains(point, circle) on 2026-11-12);
    # 36" in dec, 36"/cos(dec) in RA, then rows are kept within 36" of the position
    import math

    dd = 0.01
    dr = dd / max(math.cos(math.radians(dec)), 0.01)
    q = ("select pl_name,hostname,pl_orbper,pl_rade,disc_facility,tic_id,ra,dec from pscomppars "
         f"where ra between {ra - dr} and {ra + dr} and dec between {dec - dd} and {dec + dd}")
    r = _get("https://exoplanetarchive.ipac.caltech.edu/TAP/sync", {"query": q, "format": "json"})
    rows = r.json()
    if not isinstance(rows, list):
        raise TypeError(f"unexpected answer {str(rows)[:200]}")
    return rows


SIMBAD_TAP = "https://simbad.harvard.edu/simbad/sim-tap/sync"  # CDS host answered 404 on 2026-11-12


def simbad(ra, dec):
    q = ("select main_id, otype, ra, dec, nbref, "
         f"distance(point('icrs',ra,dec),point('icrs',{ra},{dec}))*3600 as sep "
         f"from basic where contains(point('icrs',ra,dec),circle('icrs',{ra},{dec},{30/3600}))=1 "
         "order by sep")
    r = _get(SIMBAD_TAP,
             {"request": "doQuery", "lang": "adql", "format": "json", "query": q})
    d = r.json()
    cols = [c["name"] for c in d["metadata"]]
    rows = [dict(zip(cols, x)) for x in d["data"]]
    ids = []
    stars = [x for x in rows if not str(x["otype"]).startswith("Pl")]
    if stars:
        q2 = ("select i.id from ident as i join basic as b on i.oidref = b.oid "
              f"where b.main_id = '{stars[0]['main_id'].replace(chr(39), chr(39) * 2)}'")
        r2 = _get(SIMBAD_TAP,
                  {"request": "doQuery", "lang": "adql", "format": "json", "query": q2})
        ids = [x[0] for x in r2.json()["data"]]
    return {"objects": rows, "target_ids": ids}


def vizier(cat, ra, dec, radius, cols=("**",)):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier

    last = None
    for server in MIRRORS:
        v = Vizier(columns=list(cols) + ["_r"], row_limit=-1, timeout=90)
        v.VIZIER_SERVER = server
        try:
            t = v.query_region(SkyCoord(ra, dec, unit="deg"), radius=radius * u.arcsec,
                               catalog=cat)
        except Exception as e:  # noqa: BLE001
            last = e
            continue
        rows = []
        for tab in t:
            df = tab.to_pandas()
            rows += json.loads(df.to_json(orient="records"))
        if rows:
            return {"server": server, "rows": rows}
        last = None  # empty answer: ask the other mirror before believing it
    if last is not None:
        raise last
    return {"server": "both", "rows": []}


def research(tic, ra, dec, refresh=False):
    RES.mkdir(parents=True, exist_ok=True)
    fp = RES / f"{tic}.json"
    out = json.loads(fp.read_text()) if fp.exists() and not refresh else {"tic": tic}
    out.update(ra=ra, dec=dec)
    jobs = {
        "exofop": lambda: exofop(tic),
        "nasa": lambda: nasa(ra, dec),
        "simbad": lambda: simbad(ra, dec),
        "gaia": lambda: vizier("I/355/gaiadr3", ra, dec, 63.0),
        "gaia_nss": lambda: vizier("I/357/tbooc", ra, dec, 5.0),
        "prsa_eb": lambda: vizier("J/ApJS/258/16", ra, dec, 42.0),
    }
    for k, f in jobs.items():
        if out.get(k, {}).get("status") == "ok":
            continue
        try:
            out[k] = {"status": "ok", "data": f()}
        except Exception as e:  # noqa: BLE001
            out[k] = {"status": f"not checked: {e!r}"[:300]}
        time.sleep(1.0)
    fp.write_text(json.dumps(out, indent=1, default=str))
    return out


def main():
    info = json.loads((OUT / "vetting" / "star_info.json").read_text())
    for t in sys.argv[1:]:
        s = info.get(str(t), {})
        g = (s.get("gaia") or {}).get("target")
        ra, dec = (g["ra"], g["dec"]) if g else (s["ra"], s["dec"])
        r = research(int(t), ra, dec)
        print(t, {k: v.get("status") for k, v in r.items() if isinstance(v, dict)}, flush=True)


if __name__ == "__main__":
    main()
