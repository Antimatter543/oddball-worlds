"""Table of ebscan results joined with triage (and the catalogue summary when present)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

C = Path("/home/antiraedus/Projects/oddball-worlds/outputs/planets_spike/cool")
pd.set_option("display.width", 260)
pd.set_option("display.max_rows", 400)
pd.set_option("display.max_colwidth", 70)


def build():
    rows = []
    for fp in sorted((C / "ebscan").glob("*.json")):
        o = json.loads(fp.read_text())
        if "error" in o:
            rows.append({"tic": o["tic"], "err": o["error"][:60]})
            continue
        sb = o.get("sec") or {}
        ev = o.get("events") or []
        rows.append({"tic": o["tic"], "P": o["P"], "d1": o["prim"].get("depth_peak", o["prim"]["depth"]),
                     "w1": o["prim"]["width_h"], "snr1": o["prim"]["snr"], "d2": sb.get("depth_peak", np.nan),
                     "snr2": sb.get("snr", np.nan), "ph2": sb.get("phase", np.nan), "oe": o.get("oe_z"),
                     "n_ep1": len(o.get("epochs_prim", [])), "n_ep2": len(o.get("epochs_sec", [])),
                     "twin": o.get("twin"), "ecc": o.get("ecc_flag"), "n_ev": len(ev), "src": o["src"],
                     "mult": round(o["mult"], 2), "nsec": len(o["sectors"])})  # fmt: skip
    E = pd.DataFrame(rows)
    T = pd.read_csv(C / "triage.csv")[["tic", "survey", "Tmag", "rad", "d_pc", "cls0", "known", "top_P", "top_mes"]]
    E = E.merge(T, on="tic", how="left")
    if (C / "cat_summary.csv").exists():
        K = pd.read_csv(C / "cat_summary.csv")
        K = K[[c for c in ["tic", "eb_on", "var_on", "nb_var", "simbad", "simbad_type", "toi", "ctoi", "young",
                           "not_checked", "ruwe", "rv_amp"] if c in K.columns]]
        E = E.merge(K, on="tic", how="left")
    return E


if __name__ == "__main__":
    E = build()
    E.to_csv(C / "ebscan_table.csv", index=False)
    cols = sys.argv[1].split(",") if len(sys.argv) > 1 else list(E.columns)
    q = sys.argv[2] if len(sys.argv) > 2 else None
    x = E.query(q) if q else E
    print(x.sort_values("P")[cols].round(4).to_string())
