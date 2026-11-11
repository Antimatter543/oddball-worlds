"""Survey target list (written only if gate 2 passes; the survey itself is the owner's decision).

Source: data/planets_spike/catalogs/tic_mdwarfs_50pc.csv (TIC v8 via MAST, gate 1: Teff 2700-3400 K,
d < 50 pc, T < 15). Sectors: tess-point pointings up to S98 (taken as already observed on
2026-11-11; later pointings are listed separately). Kept: >= 2 observed sectors. Per star: sector
count, baseline, estimated CPU-s and GPU-s (cost model of analyze2 cost, GPU = CPU / 5) and the
sensitivity bucket, i.e. the TEST headline-cell recovery for its sector bucket.

Usage: python -I gate2/survey_targets.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from analyze2 import bucket
from common import CAT_DIR, OUT

LAST_OBSERVED = 98


def main() -> None:
    from select_controls import sectors_for

    S = json.loads((OUT / "gate2" / "summary2.json").read_text())
    c = S["cost"]
    s = pd.read_csv(CAT_DIR / "tic_mdwarfs_50pc.csv").drop_duplicates("ID")
    secs = sectors_for(s["ID"], s["ra"], s["dec"])
    rec = S.get("bucket_rec", {})
    rows = []
    for _, r in s.iterrows():
        ss = [x for x in secs.get(int(r["ID"]), []) if x <= LAST_OBSERVED]
        fut = [x for x in secs.get(int(r["ID"]), []) if x > LAST_OBSERVED]
        if len(ss) < 2:
            continue
        n = len(ss)
        T = (max(ss) - min(ss) + 1) * 27.4
        cpu = c["pre_per_sector_s"] * n
        for rg in ("long", "short"):
            a, b = c["model"][rg]
            cpu += c["iters"][rg] * math.exp(a) * (n * c["bins_per_sector"] * T) ** b
        bk = bucket(n)
        rows.append({"tic": int(r["ID"]), "ra": r["ra"], "dec": r["dec"], "Tmag": r["Tmag"],
                     "Teff": r["Teff"], "rad": r["rad"], "mass": r["mass"], "d_pc": r["d"],
                     "n_sectors": n, "sectors": " ".join(map(str, ss)),
                     "n_future": len(fut), "T_base_d": T, "cpu_s": cpu, "gpu_s": cpu / 5,
                     "bucket": bk, "rec_1_1.5Re_15_30d": rec.get(bk, np.nan)})
    df = pd.DataFrame(rows).sort_values("n_sectors", ascending=False)
    out = OUT / "gate2" / "survey_targets.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} stars with >= 2 sectors up to S{LAST_OBSERVED}; written {out}")
    print(df.groupby("bucket").agg(n=("tic", "size"), cpu_h=("cpu_s", lambda x: x.sum() / 3600),
                                   rec=("rec_1_1.5Re_15_30d", "first")).to_string())
    print(f"total: {df['cpu_s'].sum() / 3600:.0f} CPU-h ({df['cpu_s'].sum() / 3600 / 3 / 24:.1f} d on 3 "
          f"workers); GPU at 5x: {df['gpu_s'].sum() / 86400:.1f} d")


if __name__ == "__main__":
    main()
