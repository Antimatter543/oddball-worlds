"""Phone plot for a fast-spinning star with dips that change shape: the light curve folded on the spin period,
one panel per chosen sector (same dark style as phone_eb.py). Raw median-normalised PDCSAP (no detrending,
since the detrend would eat a 0.3-0.5 d pattern), 10-min points + binned medians, x = fraction of one spin.

Usage: python -I cool/phone_cpv.py TIC P OUT.png --sectors 18 42 58 --labels "2019" "2021" "2022" --title T --sub "a|b"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "survey"))
sys.path.insert(0, str(HERE))
from ebscan import load
from phone_eb import BG, FG, GREY, ORANGE, binned, style


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("tic", type=int)
    ap.add_argument("P", type=float)
    ap.add_argument("out")
    ap.add_argument("--sectors", type=int, nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--sub", required=True)
    ap.add_argument("--t0", type=float, default=None)
    a = ap.parse_args()
    t, f, sec, meta = load(a.tic)
    t0 = a.t0 if a.t0 is not None else float(t.min())
    n = len(a.sectors)
    fig, axs = plt.subplots(1, n, figsize=(12.6, 8.1), dpi=100, sharey=True)
    axs = np.atleast_1d(axs)
    fig.patch.set_facecolor(BG)
    lo, hi = 0.0, 0.0
    for ax, s, lab in zip(axs, a.sectors, a.labels, strict=True):
        style(ax)
        m = sec == s
        y = (f[m] / np.nanmedian(f[m]) - 1) * 100
        ph = ((t[m] - t0) / a.P) % 1
        ph = np.concatenate([ph, ph + 1])
        yy = np.concatenate([y, y])
        ax.scatter(ph, yy, s=4, color=GREY, alpha=0.35, lw=0)
        xb, yb = binned(ph, yy, 0, 2, 0.02)
        ax.plot(xb, yb, "o", ms=4.5, color=ORANGE)
        lo, hi = min(lo, np.nanpercentile(yy, 0.5)), max(hi, np.nanpercentile(yy, 99.5))
        ax.set_xlim(0, 2)
        ax.set_title(lab, color=FG, fontsize=15)
        ax.set_xlabel("spins", color=FG, fontsize=14)
    axs[0].set_ylim(lo * 1.15, hi * 1.3)
    axs[0].set_ylabel("brightness change (%)", color=FG, fontsize=15)
    fig.text(0.07, 0.94, a.title, color=FG, fontsize=22, weight="bold", ha="left")
    fig.text(0.07, 0.83, a.sub.replace("|", "\n"), color="#c9d1d9", fontsize=14, ha="left")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.76, bottom=0.1, wspace=0.08)
    fig.savefig(a.out, facecolor=BG)
    print(a.out)


if __name__ == "__main__":
    main()
