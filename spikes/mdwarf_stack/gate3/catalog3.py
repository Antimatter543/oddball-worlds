"""Star-level vetting information for the gate-3 sample: the frozen vetting/catalog.py build
(Gaia DR3 neighbours, EB catalogues, TOI / CTOI / exoplanet.eu), with its star list taken from
<OUT>/gate3/sample.csv and its output written to <OUT>/gate3/star_info.json instead of the
shared <OUT>/vetting/star_info.json. Usage: python -I gate3/catalog3.py
"""

from __future__ import annotations

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
    s = pd.read_csv(G3 / "sample.csv")
    stars = pd.DataFrame({"tic": s.tic.astype(int), "ra": s.ra, "dec": s.dec, "tmag": s.Tmag})
    catalog.stars_for = lambda which: stars
    catalog.INFO = G3 / "star_info.json"
    catalog.V = G3
    inf = catalog.build(["gate3"])
    n_err = sum(any(v != "ok" for v in r["sources"].values()) for r in inf.values())
    print(len(inf), "stars;", n_err, "with a failed source")
    print(next(iter(inf.values()))["sources"])


if __name__ == "__main__":
    main()
