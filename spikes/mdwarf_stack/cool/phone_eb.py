"""Phone-friendly plots for the cool-objects write-up, in the dark style of survey/phone_plot.py.

Modes
  fold    an eclipsing pair folded on its period: two zoom panels (deep eclipse, shallow eclipse),
          10-min points + 20-min binned averages, x in hours from mid-eclipse.
  events  individual events side by side (one panel per event), x in hours from the event centre.
Light curve: the survey's rebuilt PDCSAP, detrended as in cool/ebscan.py (masked 1-d biweight).

Usage (python -I):
  cool/phone_eb.py fold TIC P TC PH2 OUT.png --title T --sub "a|b" [--win-h 8]
  cool/phone_eb.py events TIC OUT.png --times t1 t2 ... --title T --sub "a|b" [--win-h 12] [--labels ...]
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
from ebscan import detrend, load

BG, FG, GRID, ORANGE, GREY, BLUE = "#0d1117", "#e6e6e6", "#6e7681", "#ffb347", "#8b949e", "#79c0ff"


def binned(x, y, lo, hi, bw):
    edges = np.arange(lo, hi + 1e-9, bw)
    xb, yb = [], []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        m = (x >= a) & (x < b)
        if m.sum() >= 2:
            xb.append((a + b) / 2)
            yb.append(np.median(y[m]))
    return np.array(xb), np.array(yb)


def style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=GREY, labelsize=13)
    for sp in ax.spines.values():
        sp.set_color("#30363d")
    ax.axhline(0, color=GRID, lw=1)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fold", "events"])
    ap.add_argument("tic", type=int)
    ap.add_argument("rest", nargs="+")
    ap.add_argument("--title", required=True)
    ap.add_argument("--sub", required=True)
    ap.add_argument("--win-h", type=float, default=8.0)
    ap.add_argument("--times", type=float, nargs="*")
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--panel-titles", nargs="*")
    ap.add_argument("--ymin", type=float)
    a = ap.parse_args()
    t, f, sec, meta = load(a.tic)
    r, s = detrend(t, f)
    y = r * 100
    if a.mode == "fold":
        P, tc, ph2 = (float(v) for v in a.rest[:3])
        out = a.rest[3]
        centres = [tc, tc + ph2 * P]
        titles = a.panel_titles or ["deeper eclipse", "shallower eclipse"]
    else:
        out = a.rest[0]
        centres = list(a.times)
        titles = a.panel_titles or [f"dip {i + 1}" for i in range(len(centres))]
        P = None
    n = len(centres)
    fig, axs = plt.subplots(1, n, figsize=(12.6, 8.1), dpi=100, sharey=True)
    axs = np.atleast_1d(axs)
    fig.patch.set_facecolor(BG)
    lo_all = 0.0
    for ax, c, ttl in zip(axs, centres, titles, strict=True):
        style(ax)
        if P is not None:
            dt = (((t - c) / P + 0.5) % 1 - 0.5) * P * 24
        else:
            dt = (t - c) * 24
        m = np.abs(dt) <= a.win_h
        ax.scatter(dt[m], y[m], s=5, color=GREY, alpha=0.35, lw=0)
        xb, yb = binned(dt[m], y[m], -a.win_h, a.win_h, 20 / 60)
        ax.plot(xb, yb, "o", ms=6, color=ORANGE)
        if len(yb):
            lo_all = min(lo_all, np.nanmin(yb))
        ax.set_xlim(-a.win_h * 1.02, a.win_h * 1.02)
        ax.set_title(ttl, color=FG, fontsize=15)
        ax.set_xlabel("hours from mid-dip", color=FG, fontsize=14)
    ymin = a.ymin if a.ymin is not None else lo_all * 1.35
    axs[0].set_ylim(ymin, -ymin * 0.3)
    axs[0].set_ylabel("brightness change (%)", color=FG, fontsize=15)
    fig.text(0.07, 0.94, a.title, color=FG, fontsize=22, weight="bold", ha="left")
    fig.text(0.07, 0.83, a.sub.replace("|", "\n"), color="#c9d1d9", fontsize=14, ha="left")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.76, bottom=0.1, wspace=0.08)
    fig.savefig(out, facecolor=BG)
    print(out)


if __name__ == "__main__":
    main()
