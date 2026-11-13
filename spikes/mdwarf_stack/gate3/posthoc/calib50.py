"""Post-hoc calibration (not part of the verdict): the 50 pc survey's own yield model vs what it found.
Known TOIs (PC/CP/KP) on the 50 pc survey list in 1-3 R_e, 15-50 d, and whether the survey's
long search returned them (K tier, period match)."""
import json
import sys
from pathlib import Path

import pandas as pd

OUT = Path("/home/antiraedus/Projects/oddball-worlds/outputs/planets_spike")
toi = pd.read_csv("/home/antiraedus/Projects/oddball-worlds/data/planets_spike/catalogs/toi.csv", low_memory=False)
cands = sorted(OUT.glob("**/survey_targets.csv"))
print("target files:", [str(c) for c in cands])
st = pd.read_csv(cands[0])
m = toi[toi["TIC ID"].isin(st.tic)]
for lo, hi in ((1.0, 1.5), (1.5, 2.0), (2.0, 3.0)):
    s = m[(m["Period (days)"] >= 15) & (m["Period (days)"] <= 50) & (m["Planet Radius (R_Earth)"] >= lo)
          & (m["Planet Radius (R_Earth)"] < hi)]
    print(f"TOIs on 50 pc list, {lo}-{hi} R_e, 15-50 d: {len(s)} {dict(s['TESS Disposition'].value_counts())}")
    for _, r in s.iterrows():
        print("   ", int(r["TIC ID"]), round(r["Period (days)"], 3), round(r["Planet Radius (R_Earth)"], 2),
              r["TESS Disposition"])
vk = []
for f in sorted((OUT / "vetting/v2").glob("vet2_survey.s*of2.jsonl")):
    for x in open(f):
        r = json.loads(x)
        if "stats" in r:
            vk.append((r["tic"], r["stats"]["P"], r["stats"]["mes"]))
sel = m[(m["Period (days)"] >= 15) & (m["Period (days)"] <= 50) & (m["Planet Radius (R_Earth)"] >= 1.0)
        & (m["Planet Radius (R_Earth)"] < 3.0)]
print("survey signals on those hosts (tic, P, MES):")
for t in sel["TIC ID"].unique():
    print("   ", int(t), [(round(p, 3), round(ms, 1)) for tt, p, ms in vk if tt == t])
print("stars on 50 pc list:", len(st))
