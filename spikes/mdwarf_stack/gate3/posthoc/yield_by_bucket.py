"""Post-hoc: share of the projected yield (central eta, detected; tiers not C are all of them here)
by predicted-sector bucket and Tmag stratum, plus the real-search CPU share of each bucket."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

O = Path("/home/antiraedus/Projects/oddball-worlds/outputs/planets_spike/gate3")
TAU = 8.054233220378304
G_, RS = 2.959122082855911e-4, 0.00465047
ETA = {"h": 0.134, "l": 0.098, "m": 0.258, "x": 0.323}
s = pd.read_csv(O / "sample.csv")
full = pd.read_csv(O / "targets_100pc.csv")
jobs, out, fetch = {}, {}, {}
for d in sorted((O / "kaggle_out").glob("*/run")):
    for x in open(d / "jobs.all.jsonl"):
        r = json.loads(x); jobs[r["id"]] = r
    for f in d.glob("out.h*.jsonl"):
        for x in open(f):
            r = json.loads(x); out[r["id"]] = r
    for x in open(d / "fetch_status.jsonl"):
        r = json.loads(x); fetch[r["tic"]] = r


def det(r):
    return bool(r.get("signals") and r["inj_match"] and r["inj_match"][0][0]
                and r["signals"][0]["mes"] >= TAU)


y = defaultdict(lambda: defaultdict(list))
cpu = {}
for i, r in out.items():
    j = jobs[i]
    if j["kind"] != "far":
        continue
    if r.get("inj"):
        a = (G_ * j["mass"] * r["inj"]["P"] ** 2 / (4 * np.pi ** 2)) ** (1 / 3)
        y[r["tic"]][j["cell"]].append(det(r) * j["radius"] * RS / a)
    if i.startswith("g3real-"):
        cpu[r["tic"]] = cpu.get(r["tic"], 0) + r.get("cpu_s", 0)
s["yv"] = [sum(ETA[c] * np.mean(v) for c, v in y[t].items()) if t in y else 0.0 for t in s.tic]
s["cpu"] = [cpu.get(t, 0.0) for t in s.tic]
Nh = None
from_strata = s.groupby("stratum").agg(n=("tic", "size"), y=("yv", "mean"), c=("cpu", "mean"))
full_sb = pd.cut(full.n_sectors, [6, 13, 26, 999], right=False, labels=["6-12", "13-25", "26+"]).astype(str)
full_tb = pd.cut(full.Tmag, [-5, 13, 14, 15], right=False, labels=["<13", "13-14", "14-15"]).astype(str)
Nh = (full_sb + "|" + full_tb).value_counts()
from_strata["N"] = Nh
from_strata["planets"] = from_strata.N * from_strata.y
from_strata["cpu_h"] = from_strata.N * from_strata.c / 3600
print(from_strata.round(4).to_string())
from_strata["sb"] = [i.split("|")[0] for i in from_strata.index]
g = from_strata.groupby("sb")[["N", "planets", "cpu_h"]].sum()
g["share_planets"] = g.planets / g.planets.sum()
g["share_cpu"] = g.cpu_h / g.cpu_h.sum()
print(g.round(3).to_string())
