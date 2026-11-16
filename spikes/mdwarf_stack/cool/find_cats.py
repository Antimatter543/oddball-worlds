"""Confirm VizieR catalogue ids and table names before querying (scratch)."""
import sys

from astroquery.vizier import Vizier

SRV = sys.argv[1]
sys.argv = sys.argv[1:]
for kw in sys.argv[1:]:
    if kw.startswith("id:"):
        try:
            v = Vizier(row_limit=1); v.VIZIER_SERVER = SRV; cats = v.get_catalogs(kw[3:])
            for t in cats:
                print(kw, t.meta.get("name"), len(t.colnames), t.colnames[:25], t.meta.get("description", "")[:100])
        except Exception as e:  # noqa: BLE001
            print(kw, "ERR", repr(e)[:200])
    else:
        try:
            v = Vizier(); v.VIZIER_SERVER = SRV; r = v.find_catalogs(kw)
            for k, v in list(r.items())[:12]:
                print(kw, "->", k, "|", v.description[:110])
        except Exception as e:  # noqa: BLE001
            print(kw, "ERR", repr(e)[:200])
