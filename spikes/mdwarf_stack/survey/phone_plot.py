"""Phone-friendly, plain-language plot of one signal: the dip folded on its period, 10-minute points
plus binned averages, dark background (the style of outputs/planets_spike/visuals/lhs1140b.png).

The flux is the search input (frozen gate-2 preprocessing: rotation removal + biweight detrend) of
the star's rebuilt survey light curve. Bins of 20 min, inverse-variance weighted, 1-sigma errors.

Usage (python -I): survey/phone_plot.py TIC P T0 DUR_H OUT.png --title T --sub "line 1|line 2"
                                      [--window-h 10] [--ylim 1.5] [--label "average over N transits"]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "gate2"))
sys.path.insert(0, str(HERE))

from common import BIN_D, OUT

BG, FG, GRID, ORANGE, GREY = "#0d1117", "#e6e6e6", "#6e7681", "#ffb347", "#8b949e"


def fold(pp, P, t0, win_d):
    t = (pp["idx"] + 0.5) * BIN_D
    ph = (t - t0 + P / 2) % P - P / 2
    m = np.abs(ph) <= win_d
    return ph[m] * 24, (pp["flux"][m] - 1) * 100, pp["sigma"][m] * 100, t[m]


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from get_lightcurve import get_lightcurve
    from search2 import preprocess2

    ap = argparse.ArgumentParser()
    ap.add_argument("tic", type=int)
    ap.add_argument("P", type=float)
    ap.add_argument("t0", type=float)
    ap.add_argument("dur_h", type=float)
    ap.add_argument("out")
    ap.add_argument("--title", required=True)
    ap.add_argument("--sub", required=True)
    ap.add_argument("--window-h", type=float, default=10.0)
    ap.add_argument("--ylim", type=float, default=None)
    ap.add_argument("--bin-min", type=float, default=20.0)
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    rot = json.loads((OUT / "gate2" / "frozen.json").read_text())["rot"]
    lc, _ = get_lightcurve(a.tic, 2_000_000_000)
    pp, _ = preprocess2(lc, rot)
    x, y, s, t = fold(pp, a.P, a.t0, a.window_h / 24)
    n_tr = len(np.unique(np.round((t - a.t0) / a.P)[np.abs(x) < a.dur_h / 2]))
    edges = np.arange(-a.window_h, a.window_h + 1e-9, a.bin_min / 60)
    xb, yb, eb = [], [], []
    for lo, hi in itertools.pairwise(edges):
        m = (x >= lo) & (x < hi)
        if m.sum() < 2:
            continue
        w = 1 / s[m] ** 2
        xb.append((lo + hi) / 2)
        yb.append(np.sum(w * y[m]) / np.sum(w))
        eb.append(1 / np.sqrt(np.sum(w)))
    yb, eb = np.array(yb), np.array(eb)
    fig, ax = plt.subplots(figsize=(12.6, 8.1), dpi=100)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.scatter(x, y, s=4, color=GREY, alpha=0.35, lw=0, label="TESS brightness, 10-minute points")
    ax.errorbar(xb, yb, yerr=eb, fmt="o", ms=7, color=ORANGE, ecolor=ORANGE, elinewidth=1.5,
                label=a.label or f"average over {n_tr} transits")
    ax.axhline(0, color=GRID, lw=1)
    lim = a.ylim if a.ylim else max(0.5, 3.0 * float(np.nanmax(np.abs(yb))))
    ax.set_ylim(-lim, lim * 0.8)
    ax.set_xlim(-a.window_h * 1.05, a.window_h * 1.05)
    ax.set_xlabel("hours from mid-transit", color=FG, fontsize=15)
    ax.set_ylabel("brightness change (%)", color=FG, fontsize=15)
    ax.tick_params(colors=GREY, labelsize=14)
    for sp in ax.spines.values():
        sp.set_color("#30363d")
    leg = ax.legend(loc="lower right", frameon=False, fontsize=13)
    for txt in leg.get_texts():
        txt.set_color(FG)
    fig.text(0.07, 0.94, a.title, color=FG, fontsize=22, weight="bold", ha="left")
    ax.set_title(a.sub.replace("|", "\n"), color="#c9d1d9", fontsize=14, loc="left", pad=12)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.78, bottom=0.1)
    fig.savefig(a.out, facecolor=BG)
    print(a.out, "transits in window:", n_tr, "bins", len(xb))


if __name__ == "__main__":
    main()
