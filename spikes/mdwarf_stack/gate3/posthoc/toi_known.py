"""TOIs already on the 50-100 pc list: how many sit in the projected cells (1-3 R_e, 15-50 d)?"""
import pandas as pd

T = pd.read_csv("/home/antiraedus/Projects/oddball-worlds/outputs/planets_spike/gate3/targets_100pc.csv")
toi = pd.read_csv("/home/antiraedus/Projects/oddball-worlds/data/planets_spike/catalogs/toi.csv",
                  low_memory=False)
tcol = [c for c in toi.columns if c.lower() in ("tic id", "tic_id", "ticid", "tid")][0]
pc = [c for c in toi.columns if "period" in c.lower() and "err" not in c.lower()][0]
rc = [c for c in toi.columns if "planet radius" in c.lower() and "err" not in c.lower()
      or c.lower() in ("pl_rade",)][0]
dc = [c for c in toi.columns if "disposition" in c.lower()][0]
m = toi[toi[tcol].isin(T.tic)]
print("columns used:", tcol, pc, rc, dc)
print("TOIs on the list:", len(m), "hosts", m[tcol].nunique())
print(m[[tcol, pc, rc, dc]].sort_values(pc).to_string())
sel = m[(m[pc] >= 15) & (m[pc] <= 50) & (m[rc] >= 1) & (m[rc] <= 3)]
print("in 1-3 R_e, 15-50 d:", len(sel))
