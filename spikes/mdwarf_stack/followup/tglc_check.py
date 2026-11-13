"""Does TGLC (S1-S55, bucket only for S12-S55) cover the survey's missing star-sectors?

Seeded sample (20261113) of 150 missing and 50 searched star-sectors in S1-S55. Gaia id from
TIC v8 (the GAIA column is DR2; DR3 ids are the same for most stars, a mismatch shows up as a
miss on the searched controls too). Camera/CCD from tess-point. One ranged GET per key
(1 byte). Writes FU_OUT/tglc_check.json. Usage: python -I followup/tglc_check.py
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fu
import products


def exists(key):
    for k in range(3):
        try:
            r = products._S.get(products.S3 + key, headers={"Range": "bytes=0-0"}, timeout=60)
            if r.status_code in (200, 206):
                return True
            if r.status_code in (403, 404):
                return False
        except Exception:  # noqa: BLE001
            time.sleep(3)
    return None


def main():
    import download

    download._set_astroquery_ua()
    import pandas as pd
    from astroquery.mast import Catalogs

    tg = pd.read_csv(fu.OUT / "gate2" / "survey_targets.csv").set_index("tic")
    miss, have = [], []
    with gzip.open(fu.OUT / "survey" / "merged" / "results.jsonl.gz", "rt") as fh:
        for line in fh:
            r = json.loads(line)
            found = {int(s) for s in (r.get("sectors_found") or [])}
            for s in r["sectors_pred"]:
                if s <= 55:
                    (have if s in found else miss).append((r["tic"], s))
    rng = np.random.default_rng(20261113)
    sm = [miss[i] for i in rng.choice(len(miss), 150, replace=False)]
    sh = [have[i] for i in rng.choice(len(have), 50, replace=False)]
    tics = sorted({t for t, _ in sm + sh})
    cat = Catalogs.query_criteria(catalog="Tic", ID=[str(t) for t in tics])
    gaia = {int(i): g for i, g in zip(cat["ID"], cat["GAIA"])}
    out = []
    for lab, lst in (("missing", sm), ("searched", sh)):
        for t, s in lst:
            g = gaia.get(int(t))
            row = {"set": lab, "tic": int(t), "sector": int(s), "tmag": float(tg.loc[t, "Tmag"])}
            try:
                g = int(g)
            except (TypeError, ValueError):
                row["result"] = "no gaia id"
                out.append(row)
                continue
            cc = products.cam_ccd(float(tg.loc[t, "ra"]), float(tg.loc[t, "dec"]), int(s))
            if cc is None:
                row["result"] = "tess-point off silicon"
            else:
                e = exists(products.tglc_key(g, int(s), *cc))
                row["result"] = {True: "present", False: "absent", None: "not checked"}[e]
            out.append(row)
            time.sleep(0.05)
    summ = {}
    for lab in ("missing", "searched"):
        rr = [r for r in out if r["set"] == lab]
        summ[lab] = {k: sum(r["result"] == k for r in rr) for k in
                     ("present", "absent", "not checked", "no gaia id", "tess-point off silicon")}
        for lo, hi in ((0, 13.5), (13.5, 15)):
            sel = [r for r in rr if lo <= r["tmag"] < hi]
            summ[lab][f"present T{lo}-{hi}"] = f"{sum(r['result'] == 'present' for r in sel)}/{len(sel)}"
    (fu.FU_OUT / "tglc_check.json").write_text(json.dumps({"summary": summ, "rows": out}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
