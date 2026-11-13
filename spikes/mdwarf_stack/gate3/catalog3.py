"""Star-level vetting information for 50-100 pc stars: the frozen vetting/catalog.py build
(Gaia DR3 neighbours, EB catalogues, TOI / CTOI / exoplanet.eu), with its star list (and
coordinates) taken from a CSV with tic, ra, dec, Tmag instead of gate2/survey_targets.csv, and its
output written to its own file instead of the shared <OUT>/vetting/star_info.json.

  python -I gate3/catalog3.py                       gate-3 sample -> <OUT>/gate3/star_info.json
  python -I gate3/catalog3.py --targets <OUT>/gate3/targets_100pc.csv --tics tics_long.txt \
      --out <OUT>/survey100/vet/star_info.json      (the extension's vetting step 3)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "vetting"))
sys.path.insert(0, str(HERE.parents[2] / "src"))
import catalog
from common import OUT

G3 = OUT / "gate3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=str(G3 / "sample.csv"))
    ap.add_argument("--tics", default=None, help="file with one TIC per line (default: all)")
    ap.add_argument("--out", default=str(G3 / "star_info.json"))
    a = ap.parse_args()
    s = pd.read_csv(a.targets)
    if a.tics:
        keep = {int(x) for x in open(a.tics).read().split()}
        s = s[s.tic.astype(int).isin(keep)]
    stars = pd.DataFrame({"tic": s.tic.astype(int), "ra": s.ra, "dec": s.dec, "tmag": s.Tmag})
    catalog.stars_for = lambda which: stars
    catalog.INFO = Path(a.out)
    catalog.V = Path(a.out).parent
    inf = catalog.build(["csv"])
    n_err = sum(any(v != "ok" for v in r["sources"].values()) for r in inf.values())
    print(len(inf), "stars;", n_err, "with a failed source")
    print(next(iter(inf.values()))["sources"])


if __name__ == "__main__":
    main()
