"""Quick Gaia DR3 look-up for a handful of kept stars through the VizieR CfA mirror (astroquery, anonymous UA):
eclipsing-binary table (I/358/veb), classifier (I/358/vclassre), two-body orbits (I/357/tbooc) within 3" of the Gaia
position, and the main table (I/355/gaiadr3) within 25" for neighbour parallaxes and proper motions.
Writes <OUT>/cool/gaiakept.json. Usage: python -I cool/gaiakept.py TIC [TIC ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from catpass import ANON_USER_AGENT, CATD, COOL


def main():
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier

    gm = pd.read_csv(CATD / "gaia_match.csv", dtype={"source_id": "Int64"})
    fp = COOL / "gaiakept.json"
    out = json.loads(fp.read_text()) if fp.exists() else {}
    for t in map(int, sys.argv[1:]):
        r = gm[gm.tic == t].iloc[0]
        c = SkyCoord(float(r.ra_g) * u.deg, float(r.dec_g) * u.deg)
        rec = {"source_id": int(r.source_id)}
        for cat, rad, cols in (("I/358/veb", 3, ["**"]), ("I/358/vclassre", 3, ["**"]), ("I/357/tbooc", 3, ["**"]),
                               ("I/355/gaiadr3", 25, ["Source", "RA_ICRS", "DE_ICRS", "Gmag", "BP-RP", "Plx", "e_Plx",
                                                      "pmRA", "pmDE", "RUWE", "RV", "NSS"])):  # fmt: skip
            # an empty answer is re-asked on the other mirror before it is believed (CfA sometimes answers empty)
            rec[cat] = {"ok": False, "error": "no mirror answered"}
            n_empty = 0
            for server in ("vizier.cfa.harvard.edu", "vizier.idia.ac.za"):
                v = Vizier(columns=cols, row_limit=200, timeout=120)
                v.VIZIER_SERVER = server
                v._session.headers["User-Agent"] = ANON_USER_AGENT
                try:
                    res = v.query_region(c, radius=rad * u.arcsec, catalog=cat)
                except Exception as e:  # noqa: BLE001
                    rec[cat] = {"ok": False, "error": repr(e)[:200]}
                    continue
                rows = []
                for tab in res:
                    rows += [{k: (None if pd.isna(x) else (x.item() if hasattr(x, "item") else x)) for k, x in d.items()}
                             for d in tab.to_pandas().to_dict("records")]  # fmt: skip
                if rows:
                    rec[cat] = {"ok": True, "rows": rows, "server": server}
                    break
                n_empty += 1
                rec[cat] = {"ok": n_empty >= 2, "rows": [], "server": server,
                            "note": "empty on both mirrors" if n_empty >= 2 else "empty on one mirror only"}
        out[str(t)] = rec
        veb = [x for x in rec["I/358/veb"].get("rows", []) if int(x.get("Source", 0) or 0) == rec["source_id"]]
        tb = [x for x in rec["I/357/tbooc"].get("rows", []) if int(x.get("Source", 0) or 0) == rec["source_id"]]
        cl = [(x.get("Class"), x.get("ClassSc")) for x in rec["I/358/vclassre"].get("rows", [])
              if int(x.get("Source", 0) or 0) == rec["source_id"]]  # fmt: skip
        nb = [(str(x.get("Source"))[-6:], round(c.separation(SkyCoord(x["RA_ICRS"] * u.deg, x["DE_ICRS"] * u.deg)).arcsec, 1),
               x.get("Gmag"), x.get("Plx"), x.get("pmRA"), x.get("pmDE")) for x in rec["I/355/gaiadr3"].get("rows", [])]  # fmt: skip
        print(t, "veb:", [(x.get("Freq"), x.get("Per") if "Per" in x else None) for x in veb] or "none",
              "| tbooc:", [(x.get("NSSmodel"), x.get("Per")) for x in tb] or "none", "| class:", cl or "none",
              "| ok:", {k: rec[k]["ok"] for k in rec if k != "source_id"})  # fmt: skip
        tgt = [x for x in nb if x[1] < 1.0]
        print("    target:", tgt, "| close/bright nb:", [x for x in nb if x[1] >= 1.0 and (x[1] < 12 or (x[2] or 99) < (tgt[0][2] if tgt else 99) + 2.5)])
    fp.write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
