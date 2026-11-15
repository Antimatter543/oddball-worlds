"""Seed the light-curve cache's product lists from the survey records, so the vetting rebuild
(survey/prep_vet.py -> get_lightcurve) downloads exactly the files the Kaggle search used without
asking MAST's observation query again (on 2026-11-16 it took > 1 min per star, which would have
cost ~7 h for 404 stars).

Each record's `provenance` lists the product and MAST URI of every sector the search used; the
cache file gets those, with queried_utc = the record's products_queried_utc (the download module
reuses lists younger than 30 days). Nothing about the conversion changes; prep_vet still compares
the rebuilt sector list with the record's `sectors_found`. Existing cache files are left alone.

Usage: python -I survey100/seed_products.py TICS_FILE [--merged M]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import DATA, OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tics")
    ap.add_argument("--merged", default=str(OUT / "survey100" / "merged" / "results.jsonl.gz"))
    a = ap.parse_args()
    want = {int(x) for x in Path(a.tics).read_text().split()}
    d = DATA / "lc_cache" / "products"
    d.mkdir(parents=True, exist_ok=True)
    n_new = n_old = 0
    with gzip.open(a.merged, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r["tic"] not in want:
                continue
            p = d / f"tic{r['tic']}.json"
            if p.exists():
                n_old += 1
                continue
            prods = {str(x["sector"]): [x["product"], x["uri"]] for x in r["provenance"]}
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps({"tic": r["tic"], "queried_utc": r["products_queried_utc"],
                                       "source": "survey_record", "products": prods}))
            tmp.replace(p)
            n_new += 1
    print(f"seeded {n_new}, already cached {n_old}, of {len(want)}")


if __name__ == "__main__":
    main()
