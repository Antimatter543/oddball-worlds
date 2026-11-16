"""Summary of the short-range (1-10 d) pile: signals at or above tau_short, by star, split into
TOI/CTOI hosts, catalogued variables (on target / neighbour), stars with a light-curve look (ebscan) and the rest.
Also lists the stars whose best short signal looks planet-sized (depth < 1%, MES >= 10) and that have no TOI/CTOI and
no catalogued variable on target: these are the ones that would need a closer look.

Usage: python -I cool/shortpile.py   (writes <OUT>/cool/shortpile.csv)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import OUT

C = OUT / "cool"


def main():
    S = pd.read_csv(C / "signals_short.csv")
    S = S[S.above_tau]
    best = S.sort_values("mes", ascending=False).drop_duplicates("tic")
    K = pd.read_csv(C / "cat_summary.csv")
    E = pd.read_csv(C / "ebscan_table.csv")[["tic", "P", "d1", "d2", "ph2", "snr1", "snr2"]] \
        if (C / "ebscan_table.csv").exists() else pd.DataFrame(columns=["tic"])  # fmt: skip
    b = best.merge(K, on="tic", how="left").merge(E, on="tic", how="left", suffixes=("", "_eb"))
    b["n_sig"] = b.tic.map(S.groupby("tic").size())
    b["toi_any"] = b.toi.notna() | b.ctoi.notna()
    b["eb_target"] = b.eb_on.notna()
    b["var_target"] = b.var_on.notna()
    b["var_nb"] = b.nb_var.notna()
    b["looked"] = b.P_eb.notna() if "P_eb" in b else b["P"].notna()
    cls = []
    for r in b.itertuples():
        if r.toi_any:
            cls.append("toi_or_ctoi")
        elif r.eb_target:
            cls.append("catalogued_eb_on_target")
        elif r.var_target:
            cls.append("catalogued_variable_on_target")
        elif r.var_nb:
            cls.append("catalogued_variable_neighbour")
        else:
            cls.append("not_in_catalogues_checked")
    b["cls"] = cls
    b["planet_sized"] = (b.depth < 0.01) & (b.mes >= 10)
    b.to_csv(C / "shortpile.csv", index=False)
    print("short-pile stars", len(b), "signals", len(S))
    print(b.cls.value_counts().to_string())
    print("\nby depth (best signal) x class")
    b["dbin"] = pd.cut(b.depth, [0, 0.003, 0.01, 0.03, 0.1, 10], labels=["<0.3%", "0.3-1%", "1-3%", "3-10%", ">10%"])
    print(pd.crosstab(b.dbin, b.cls).to_string())
    q = b[b.planet_sized & (b.cls == "not_in_catalogues_checked")]
    print("\nplanet-sized (depth<1%, MES>=10), not TOI, no catalogued variable on target or neighbour:", len(q))
    print(q.sort_values("mes", ascending=False)[["tic", "survey", "P", "depth", "dur_h", "mes", "n_ep", "n_sig",
                                                   "Tmag", "rad", "n_nb_21"]].head(40).to_string())  # fmt: skip


if __name__ == "__main__":
    main()
