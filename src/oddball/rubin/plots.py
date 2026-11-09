"""Light-curve plots for Rubin outburst candidates."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oddball.rubin.detect import ObjectResult

BAND_COLORS = {"u": "#7b3fbf", "g": "#2a9d4a", "r": "#d1342f", "i": "#e08a1e", "z": "#6b4a2b",
               "y": "#444444"}


def plot_candidate(lc: pd.DataFrame, res: ObjectResult, title: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax, axd) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.3]})
    for b in [b for b in "ugrizy" if b in set(lc["band"])]:
        g = lc[lc["band"] == b]
        c = BAND_COLORS.get(b, "k")
        good = ~g["bad"].astype(bool)
        det = g["is_det"].astype(bool)
        ax.errorbar(g.loc[good & ~det, "mjd"], g.loc[good & ~det, "mag"],
                    g.loc[good & ~det, "magerr"], fmt="o", ms=3, mfc="none", color=c, lw=0.6,
                    label=f"{b} forced")
        ax.errorbar(g.loc[good & det, "mjd"], g.loc[good & det, "mag"],
                    g.loc[good & det, "magerr"], fmt="o", ms=4, color=c, lw=0.6,
                    label=f"{b} alert det")
        ax.plot(g.loc[~good, "mjd"], g.loc[~good, "mag"], "x", color=c, ms=5, alpha=0.6)
        br = res.bands.get(b)
        if br is not None and np.isfinite(br.baseline_mag):
            ax.axhline(br.baseline_mag, color=c, ls=":", lw=0.8)
        axd.errorbar(g["mjd"], g["diff_flux"] / 1e3, g["diff_flux_err"] / 1e3, fmt=".", color=c,
                     lw=0.5, ms=3)
    p = res.primary
    if p is not None and np.isfinite(p.t_run_start):
        for a in (ax, axd):
            a.axvspan(p.t_run_start, p.t_run_end, color="0.85", zorder=0)
    ax.invert_yaxis()
    ax.set_ylabel("AB mag (direct-image PSF flux)")
    axd.set_ylabel("diff flux [uJy]")
    axd.axhline(0, color="k", lw=0.5)
    axd.set_xlabel("MJD")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, ncol=3, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
