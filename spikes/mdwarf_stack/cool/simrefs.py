"""SIMBAD identifiers, object type and the full reference list (bibcode, year, title) for chosen stars,
from the Harvard SIMBAD TAP mirror (anonymous), by the Gaia DR3 position (5" cone, nearest non-planet).
Titles are scanned for keywords (eclips, binary, dipper, disk, variab, TESS, transit, young, brown dwarf).
Writes <OUT>/cool/simrefs/<tic>.json. Usage: python -I cool/simrefs.py TIC [TIC ...]
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))
from catpass import SIMBAD_TAP, _tap
from common import OUT

KEYS = re.compile(r"eclips|binar|dipper|dipping|disk|disc|variab|TESS|transit|young|brown dwarf|occult|period|"
                  r"rotation|flare|companion|orbit", re.IGNORECASE)  # fmt: skip


def main():
    g = pd.read_csv(OUT / "cool" / "cat" / "gaia_match.csv")
    d = OUT / "cool" / "simrefs"
    d.mkdir(parents=True, exist_ok=True)
    for tic in map(int, sys.argv[1:]):
        r = g[g.tic == tic].iloc[0]
        ra, de = float(r.ra_g), float(r.dec_g)
        q = ("select oid, main_id, otype, nbref, distance(point('icrs',ra,dec),point('icrs',"
             f"{ra},{de}))*3600 as sep from basic where contains(point('icrs',ra,dec),circle('icrs',{ra},{de},"
             f"{5 / 3600}))=1 order by sep")  # fmt: skip
        try:
            b = _tap(SIMBAD_TAP, q)
            b = b[~b.otype.astype(str).str.startswith("Pl")]
            if not len(b):
                out = {"tic": tic, "ok": True, "simbad": None}
            else:
                oid = int(b.iloc[0].oid)
                ids = _tap(SIMBAD_TAP, f"select id from ident where oidref = {oid}")
                refs = _tap(SIMBAD_TAP, "select r.bibcode, r.\"year\" as yr, r.title from has_ref as h join ref as r on "
                                         f"h.oidbibref = r.oidbib where h.oidref = {oid} order by yr")  # fmt: skip
                otypes = _tap(SIMBAD_TAP, f"select otype from otypes where oidref = {oid}")
                out = {"tic": tic, "ok": True, "main_id": b.iloc[0].main_id, "otype": b.iloc[0].otype,
                       "sep": float(b.iloc[0].sep), "all_otypes": sorted(set(otypes.otype.astype(str))),
                       "ids": list(ids.id.astype(str)),
                       "refs": refs.to_dict("records"),
                       "hits": [x for x in refs.to_dict("records") if KEYS.search(str(x["title"]))]}  # fmt: skip
        except Exception as e:  # noqa: BLE001
            out = {"tic": tic, "ok": False, "error": repr(e)[:300]}
        (d / f"{tic}.json").write_text(json.dumps(out, indent=1, default=str))
        if out.get("ok") and out.get("main_id"):
            print(tic, out["main_id"], out["otype"], out["all_otypes"], "refs", len(out["refs"]))
            for h in out["hits"]:
                print("   ", h["yr"], h["bibcode"], str(h["title"])[:140])
        else:
            print(tic, out)
        time.sleep(1)


if __name__ == "__main__":
    main()
