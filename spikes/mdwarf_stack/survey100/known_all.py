"""Known TOIs with P 10-60 d on ALL searched 100 pc stars (not only the vetted ones): is the star
in the vetted list (a long signal >= tau_long), and was the object recovered (known.csv)?
The TOI table is <DATA>/catalogs/toi.csv (refreshed 2026-11-16).
Writes <OUT>/survey100/vet/known_all.csv. Usage: python -I survey100/known_all.py
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import DATA, OUT
from common100 import VET


def main():
    ok = {}
    with gzip.open(OUT / "survey100" / "merged" / "results.jsonl.gz", "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("status") == "ok":
                ok[int(r["tic"])] = r.get("n_sectors_found") or 0
    toi = pd.read_csv(DATA / "catalogs" / "toi.csv", low_memory=False)
    toi = toi[toi["TIC ID"].isin(ok) & toi["Period (days)"].between(10, 60)].copy()
    vetted = {int(x) for x in (VET / "tics_long.txt").read_text().split()}
    kn = pd.read_csv(VET / "known.csv")
    rec = {str(n): bool(r) for n, r in zip(kn.name, kn.recovered)}
    mes = {str(n): m for n, m in zip(kn.name, kn.mes_pred)}
    toi["name"] = "TOI-" + toi["TOI"].astype(str)
    toi["n_sectors"] = toi["TIC ID"].map(ok)
    toi["vetted_star"] = toi["TIC ID"].isin(vetted)
    toi["recovered"] = toi.name.map(rec).fillna(False)
    toi["mes_pred"] = toi.name.map(mes)
    cols = ["TIC ID", "name", "TFOPWG Disposition", "TESS Disposition", "Period (days)",
            "Planet Radius (R_Earth)", "Depth (ppm)", "Duration (hours)", "n_sectors", "vetted_star",
            "mes_pred", "recovered"]
    toi = toi[cols].sort_values("Period (days)")
    toi.to_csv(VET / "known_all.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"{len(toi)} TOIs with P 10-60 d on {toi['TIC ID'].nunique()} of {len(ok)} searched stars; "
          f"on a vetted star {int(toi.vetted_star.sum())}; recovered {int(toi.recovered.sum())}")
    print(toi.to_string(index=False))


if __name__ == "__main__":
    main()
