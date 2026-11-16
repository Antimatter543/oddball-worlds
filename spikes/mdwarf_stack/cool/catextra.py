"""Extra VizieR catalogues for the cool-objects pass (same machinery and caches as catpass.py):
young-star dip / complex-rotator catalogues found to matter once the light curves were looked at.

  capi_dip   J/ApJS/263/14  Capistrant+2022 TESS dipper stars, 10"
Bouma+2024 (complex periodic variables, AJ 167, 38), Tajiri+2020 (ApJS 251, 18), Zhan+2019 (ApJ 876, 127) and
Guenther+2022 (AJ 163, 144) have no VizieR tables on the CfA/IDIA mirrors (checked 16 Nov 2026: empty catalogue
lists); they are covered only through SIMBAD's reference lists (cool/simrefs.py), and recorded as such here.

Usage: python -I cool/catextra.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import catpass as C

C.R.CATALOGUES.update({
    "capi_dip": ("J/ApJS/263/14", 10.0, ["**"], {}),
})  # fmt: skip


def main():
    stars = C.star_list()
    sg = C.gpos(stars)
    for key, ref in (("bouma_cpv", "Bouma+2024 AJ 167 38"), ("tajiri_dip", "Tajiri+2020 ApJS 251 18")):
        C.status_update(key, {"ok": False, "error": f"{ref}: no VizieR table on the CfA/IDIA mirrors; SIMBAD references only"})
    for key in ["capi_dip"]:
        try:
            df = C.R.query_catalogue(key, sg, chunk=40, sleep=1.0)
            C.status_update(key, {"ok": True, "rows": len(df), "catalogue": C.R.CATALOGUES[key][0]})
            print(key, "ok", len(df), sorted(set(df.tic)) if "tic" in df else "")
        except Exception as e:  # noqa: BLE001
            C.status_update(key, {"ok": False, "error": repr(e)[:300]})
            print(key, "ERROR", repr(e)[:200])


if __name__ == "__main__":
    main()
