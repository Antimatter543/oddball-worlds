"""Scratch viewer: raw (median-normalised) PDCSAP per sector, one panel each, with optional SAP overlay.
Usage: python -I cool/lcview.py TIC OUT.png [--raw] [--tmin T --tmax T]"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "survey"))
sys.path.insert(0, str(HERE))
from ebscan import detrend, load

ap = argparse.ArgumentParser()
ap.add_argument("tic", type=int)
ap.add_argument("out")
ap.add_argument("--raw", action="store_true")
ap.add_argument("--tmin", type=float)
ap.add_argument("--tmax", type=float)
a = ap.parse_args()
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

t, f, sec, meta = load(a.tic)
r, s = detrend(t, f)
y = f - 1 if a.raw else r
if a.tmin is not None:
    m = (t >= a.tmin) & (t <= a.tmax)
    t, y, sec = t[m], y[m], sec[m]
ss = sorted(set(sec.tolist()))
n = len(ss)
fig, axs = plt.subplots(n, 1, figsize=(12, 1.6 * n + 0.5), squeeze=False)
for ax, s_ in zip(axs[:, 0], ss, strict=True):
    m = sec == s_
    ax.plot(t[m], y[m], ".k", ms=1)
    ax.set_ylabel(f"S{s_}")
fig.suptitle(f"TIC {a.tic} {'raw' if a.raw else 'detrended'}")
fig.tight_layout()
fig.savefig(a.out, dpi=60)
print(a.out)
