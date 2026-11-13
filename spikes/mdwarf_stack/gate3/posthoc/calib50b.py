"""Post-hoc: apply the gate-3 yield model to the 50 pc survey using the 150 reference stars'
injections (cells m, x) run in the gate-3 kernels, and compare with what the survey returned."""
import json
from pathlib import Path

import numpy as np

G = Path("/home/antiraedus/Projects/oddball-worlds/outputs/planets_spike/gate3/kaggle_out")
TAU = 8.054233220378304
G_ = 2.959122082855911e-4  # AU^3 / (Msun d^2)
RS_AU = 0.00465047
ETA = {"m": (0.258, 0.200, 0.343), "x": (0.323, 0.233, 0.468)}
jobs, out = {}, {}
for d in sorted(G.glob("*/run")):
    for x in open(d / "jobs.all.jsonl"):
        r = json.loads(x); jobs[r["id"]] = r
    for f in d.glob("out.h*.jsonl"):
        for x in open(f):
            r = json.loads(x); out[r["id"]] = r


def det(r):
    return bool(r.get("signals") and r["inj_match"] and r["inj_match"][0][0]
                and r["signals"][0]["mes"] >= TAU)


def ptr(P, m, rad):
    m = m if m == m and m > 0 else 0.3
    rad = rad if rad == rad and rad > 0 else 0.3
    a = (G_ * m * P ** 2 / (4 * np.pi ** 2)) ** (1 / 3)
    return rad * RS_AU / a


for c in ("m", "x"):
    per = {}
    for i, r in out.items():
        j = jobs[i]
        if j["kind"] == "ref" and j["cell"] == c and r.get("inj") and "error" not in r:
            per.setdefault(r["tic"], []).append(det(r) * ptr(r["inj"]["P"], j["mass"], j["radius"]))
    y = np.mean([np.mean(v) for v in per.values()])
    # 50 pc survey: 4,350 searched of 4,681 (no-data stars count zero)
    E = 4350 * y
    print(f"cell {c}: ref stars {len(per)}, mean y {y:.5f}; 50 pc survey star-equivalents {E:.1f}; "
          f"planets {E * ETA[c][0]:.2f} (eta low {E * ETA[c][1]:.2f}, high {E * ETA[c][2]:.2f})")
