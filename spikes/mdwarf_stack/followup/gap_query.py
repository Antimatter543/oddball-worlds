"""Part 2.1 data: every MAST timeseries product (all collections and HLSPs) and the TIC
contamination ratio for the 4,681 survey targets, in batched anonymous queries.

Writes FU_OUT/gap_products.csv (tic, collection, provenance, sector, kind) and
FU_OUT/gap_tic.csv (tic, Tmag, contratio). Resumable (batches already written are skipped).
Usage: python -I followup/gap_query.py
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fu

B = 40


def kind_of(coll, prov, url):
    u = str(url)
    if coll == "TESS":
        if u.endswith("-s_lc.fits"):
            return "spoc120"
        if u.endswith("fast-lc.fits"):
            return "spoc20"
        if u.endswith("_dvt.fits"):
            return "dvt"
        return "tess_other"
    return str(prov).lower()


def main():
    import download

    download._set_astroquery_ua()
    from astroquery.mast import Catalogs, Observations

    tics = []
    with gzip.open(fu.OUT / "survey" / "merged" / "results.jsonl.gz", "rt") as fh:
        for line in fh:
            tics.append(json.loads(line)["tic"])
    tics = sorted(set(tics))
    pf = fu.FU_OUT / "gap_products.csv"
    donef = fu.FU_OUT / "gap_products.done"
    done = set(donef.read_text().split()) if donef.exists() else set()
    new = not pf.exists()
    with open(pf, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["tic", "collection", "provenance", "sector", "kind"])
        for i in range(0, len(tics), B):
            bid = str(i)
            if bid in done:
                continue
            batch = [str(t) for t in tics[i:i + B]]
            for k in range(4):
                try:
                    obs = Observations.query_criteria(target_name=batch,
                                                      dataproduct_type="timeseries")
                    break
                except Exception as ex:  # noqa: BLE001
                    print("retry", i, repr(ex)[:120], flush=True)
                    time.sleep(10 * (k + 1))
            else:
                print("FAILED batch", i, flush=True)
                continue
            for c, p, s, u, tn in zip(obs["obs_collection"], obs["provenance_name"],
                                      obs["sequence_number"], obs["dataURL"], obs["target_name"]):
                try:
                    s = int(s)
                except Exception:  # noqa: BLE001  (masked / non-numeric sequence numbers)
                    s = -1
                w.writerow([tn, c, p, s, kind_of(str(c), p, u)])
            fh.flush()
            with open(donef, "a") as d:
                d.write(bid + "\n")
            print("batch", i, len(obs), flush=True)
            time.sleep(0.5)
    tf = fu.FU_OUT / "gap_tic.csv"
    if not tf.exists():
        rows = []
        for i in range(0, len(tics), 200):
            batch = [str(t) for t in tics[i:i + 200]]
            for k in range(4):
                try:
                    r = Catalogs.query_criteria(catalog="Tic", ID=batch)
                    break
                except Exception as ex:  # noqa: BLE001
                    print("tic retry", i, repr(ex)[:120], flush=True)
                    time.sleep(10 * (k + 1))
            else:
                print("TIC FAILED batch", i, flush=True)
                continue
            for x in r:
                rows.append([x["ID"], x["Tmag"], x["contratio"], x["ra"], x["dec"]])
            print("tic batch", i, flush=True)
            time.sleep(0.5)
        with open(tf, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["tic", "Tmag", "contratio", "ra", "dec"])
            w.writerows(rows)


if __name__ == "__main__":
    main()
