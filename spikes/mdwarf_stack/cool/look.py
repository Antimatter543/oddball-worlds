"""Scratch viewer for the star-level tables."""
import sys

import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_colwidth", 60)
C = "/home/antiraedus/Projects/oddball-worlds/outputs/planets_spike/cool/"
L = pd.read_csv(C + "signals_long.csv")
S = pd.read_csv(C + "stars.csv")
mode = sys.argv[1]
if mode == "stars":
    S = S.sort_values("max_mes", ascending=False)
    print(S[["tic", "survey", "Tmag", "rad", "n_long", "max_mes", "max_depth", "tiers", "n_sec_fail",
             "n_oe_fail", "n_sap_fail", "n_gap_fail", "cat"]].head(int(sys.argv[2])).to_string())
elif mode == "tic":
    for t in sys.argv[2:]:
        x = L[L.tic == int(t)]
        print(x[["key", "P", "t0", "dur_h", "depth", "mes", "n_tr", "tier", "sec_z", "sec_phase_d", "sec_mes",
                 "oe_z", "d_odd", "d_even", "sap_status", "gap_status", "dur_ratio", "pixel_status"]].to_string())
        print(x.reasons.tolist(), x.cat_objects.iloc[0])
elif mode == "hist":
    import numpy as np
    print(pd.cut(S.max_depth, [0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 10]).value_counts().sort_index())
    print(pd.cut(S.max_mes, [8, 10, 15, 20, 30, 50, 100, 1e9]).value_counts().sort_index())
    print((S.n_sec_fail > 0).sum(), (S.n_oe_fail > 0).sum(), (S.cat.notna()).sum())
    print(S.cat.dropna().str.contains("eb:").sum())
    np.seterr(all="ignore")
