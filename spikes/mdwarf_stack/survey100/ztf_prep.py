"""Ephemeris file for followup/archive/ztf_neighbours.py: linear ephemeris (fu.fit_ephem, with its
covariance) of each survivor key from <OUT>/survey100/vet/graded.jsonl.
Writes <OUT>/survey100/vet/ztf_ephem.json. Usage: python -I survey100/ztf_prep.py KEY [KEY ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "followup"))
sys.path.insert(0, str(HERE))
import fu
from common100 import VET


def main():
    g = {json.loads(x)["key"]: json.loads(x) for x in open(VET / "graded.jsonl") if x.strip()}
    fp = VET / "ztf_ephem.json"
    out = json.loads(fp.read_text()) if fp.exists() else {}
    for k in sys.argv[1:]:
        out[k] = {"ephem": fu.fit_ephem(g[k]["stats"])}
        e = out[k]["ephem"]
        print(k, round(e["P"], 5), round(e["T0"], 4), "sig_T0_h", round(24 * e["cov"][0][0] ** 0.5, 2),
              "sig_P_h", round(24 * e["cov"][1][1] ** 0.5, 3))
    fp.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
