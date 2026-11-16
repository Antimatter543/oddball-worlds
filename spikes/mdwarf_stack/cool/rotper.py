"""Lomb-Scargle period per sector and overall on the median-normalised PDCSAP (rotation / CPV check),
and a per-sector phase plot at the overall best period. Usage: python -I cool/rotper.py TIC OUT.png [--pmax 10]"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "survey"))
sys.path.insert(0, str(HERE))
from ebscan import load

ap = argparse.ArgumentParser()
ap.add_argument("tic", type=int)
ap.add_argument("out")
ap.add_argument("--pmax", type=float, default=10)
ap.add_argument("--P", type=float)
a = ap.parse_args()
from astropy.timeseries import LombScargle

t, f, sec, meta = load(a.tic)
y = f - 1
fr = np.linspace(1 / a.pmax, 1 / 0.1, 20000)
per = {}
for s in sorted(set(sec.tolist())):
    m = sec == s
    p = LombScargle(t[m], y[m]).power(fr)
    per[s] = (1 / fr[np.argmax(p)], float(p.max()), float(np.nanpercentile(y[m], 99) - np.nanpercentile(y[m], 1)))
for s, (P, pw, amp) in per.items():
    print(f"S{s}: P={P:.4f} d power={pw:.2f} amp(1-99%)={amp*100:.1f}%")
P = a.P or float(np.median([v[0] for v in per.values()]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ss = sorted(per)
fig, axs = plt.subplots(1, len(ss), figsize=(2.2 * len(ss) + 1, 3), sharey=True, squeeze=False)
for ax, s in zip(axs[0], ss, strict=True):
    m = sec == s
    ax.plot(((t[m] - t[m].min()) / P) % 1, y[m], ".k", ms=1)
    ax.set_title(f"S{s} P={per[s][0]:.3f}", fontsize=8)
fig.suptitle(f"TIC {a.tic} folded at {P:.4f} d")
fig.tight_layout()
fig.savefig(a.out, dpi=60)
print(a.out, "P used", P)
