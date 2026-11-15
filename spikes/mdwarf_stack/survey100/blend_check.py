"""Do searched neighbours show the same dip? For each survivor key, every star of the 100 pc and
50 pc target lists within 180" (about 8.5 TESS pixels) that the survey searched: its stored long
search signals (all 6 slots, any MES) are compared with the survivor by
  - shared epochs: a neighbour signal epoch within max(D, 2 h) of a survivor epoch;
  - period: P ratio within 1% of n or 1/n (n <= 3).
A neighbour signal sharing >= 2 epochs is a blend candidate (the same photons dip in both
apertures). Writes <OUT>/survey100/vet/blend_check.json. Usage: python -I survey100/blend_check.py KEY ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import OUT
from common100 import TARGETS, VET


def load_rows(p):
    return {json.loads(x)["tic"]: json.loads(x) for x in open(p) if x.strip()}


def main():
    keys = sys.argv[1:]
    g = {json.loads(x)["key"]: json.loads(x) for x in open(VET / "graded.jsonl") if x.strip()}
    t100 = pd.read_csv(TARGETS)
    t50 = pd.read_csv(OUT / "gate2" / "survey_targets.csv")
    tl = pd.concat([t100.assign(list="100pc"), t50.assign(list="50pc")], ignore_index=True)
    rows = {**load_rows(OUT / "survey" / "vet" / "rows_long.jsonl"), **load_rows(VET / "rows_long.jsonl")}
    out = {}
    for k in keys:
        r = g[k]
        s = r["stats"]
        me = tl[tl.tic == r["tic"]].iloc[0]
        sep = 3600 * np.hypot((tl.ra - me.ra) * np.cos(np.radians(me.dec)), tl.dec - me.dec)
        near = tl[(sep <= 180) & (tl.tic != r["tic"])].assign(sep=sep[(sep <= 180) & (tl.tic != r["tic"])])
        tol = max(s["dur_h"] / 24, 2 / 24)
        res = []
        for nb in near.itertuples():
            row = rows.get(int(nb.tic))
            ent = {"tic": int(nb.tic), "list": nb.list, "sep": float(nb.sep), "Tmag": float(nb.Tmag),
                   "searched": row is not None, "matches": []}
            if row:
                ent["sectors"] = row["sectors"]
                for j, sg in enumerate(row["signals"]):
                    ep = [e[0] for e in sg["epochs"]]
                    shared = sum(any(abs(a - b) <= tol for b in ep) for a in s["epochs"])
                    ratio = sg["P"] / s["P"]
                    pm = next((n for n in (1, 2, 3) for c in (ratio, 1 / ratio)
                               if abs(c - n) <= 0.01 * n), None)
                    if shared >= 1 or pm:
                        ent["matches"].append({"slot": j, "P": sg["P"], "mes": sg["mes"],
                                               "depth_ppt": 1e3 * sg["depth"], "shared_epochs": shared,
                                               "period_n": pm})
            res.append(ent)
        out[k] = res
        print(k, f"{len(res)} list stars within 180\":",
              "; ".join(f"TIC {e['tic']} ({e['list']}, {e['sep']:.0f}\", T {e['Tmag']:.1f}, "
                        f"{'searched' if e['searched'] else 'not searched'})"
                        + (f" MATCH {[(m['slot'], round(m['P'], 3), round(m['mes'], 1), m['shared_epochs'], m['period_n']) for m in e['matches']]}"
                           if e["matches"] else "") for e in res))
    fp = VET / "blend_check.json"
    old = json.loads(fp.read_text()) if fp.exists() else {}
    old.update(out)
    fp.write_text(json.dumps(old, indent=1, default=float))


if __name__ == "__main__":
    main()
