"""Quick-look for vetting: every ingested sector of one star, plus a phase fold.

    uv run python -I scripts/fold_quicklook.py --tic 391813813 --out /path/plot.png
    uv run python -I scripts/fold_quicklook.py --tic 391813813 --period 2.345 --t0 2180.1 --out p.png

Top row: PDCSAP (black) and SAP (orange, offset) per sector. Bottom row: all sectors folded at
``--period`` (default: the strongest Lomb-Scargle peak of the best sector, and twice it, since
an EB's LS peak is often half its period), coloured by sector so changing depth or phase drift
shows. Prints the LS period, the fold's per-sector depth at phase 0 and the odd/even depths.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from oddball.tess.store import load_lightcurves, metadata

SECTORS = (32, 33, 38, 43, 44, 55)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.timeseries import LombScargle

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tic", type=int, required=True)
    ap.add_argument("--period", type=float, default=None)
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dpi", type=int, default=70)
    a = ap.parse_args()
    lcs = {}
    for s in SECTORS:
        m = metadata(s)
        if (m.tic == a.tic).any():
            lc = load_lightcurves(s, [a.tic]).get(a.tic)
            if lc is not None:
                lcs[s] = lc
    if not lcs:
        raise SystemExit(f"TIC {a.tic} is in none of {SECTORS}")
    T = np.concatenate([np.asarray(lc["time"], float) for lc in lcs.values()])
    F = np.concatenate([np.asarray(lc["flux"], float) / np.nanmedian(lc["flux"]) for lc in lcs.values()])
    S = np.concatenate([np.full(len(lc["time"]), s) for s, lc in lcs.items()])
    ok = np.isfinite(T) & np.isfinite(F)
    freq = np.linspace(1 / 13.0, 1 / 0.05, 40000)
    pw = LombScargle(T[ok], F[ok]).power(freq)
    p_ls = 1 / freq[np.argmax(pw)]
    periods = [a.period] if a.period else [p_ls, 2 * p_ls]
    print(f"TIC {a.tic}: sectors {list(lcs)}; LS peak P = {p_ls:.5f} d (power {pw.max():.2f})")
    n = len(lcs)
    fig, axes = plt.subplots(2, max(n, len(periods)), figsize=(4.2 * max(n, len(periods)), 6.5),
                             squeeze=False)  # fmt: skip
    for j, (s, lc) in enumerate(lcs.items()):
        ax = axes[0, j]
        t = np.asarray(lc["time"], float)
        f = np.asarray(lc["flux"], float) / np.nanmedian(lc["flux"])
        sp = np.asarray(lc["sap_flux"], float) / np.nanmedian(lc["sap_flux"])
        r = np.nanpercentile(f, 99) - np.nanpercentile(f, 1)
        ax.plot(t, f, ".", ms=1.2, color="k")
        ax.plot(t, sp - 1.3 * max(r, 0.002), ".", ms=1.0, color="tab:orange")
        ax.set_title(f"S{s} PDCSAP / SAP (offset)", fontsize=8)
        ax.tick_params(labelsize=6)
    for j in range(n, axes.shape[1]):
        axes[0, j].set_axis_off()
    cols = plt.cm.tab10(np.arange(len(SECTORS)))
    for j, P in enumerate(periods):
        ax = axes[1, j]
        t0 = a.t0 if a.t0 is not None else float(T[ok][np.nanargmin(F[ok])])
        ph = ((T - t0) / P + 0.5) % 1.0 - 0.5
        for k, s in enumerate(SECTORS):
            m = ok & (S == s)
            if m.any():
                ax.plot(ph[m], F[m], ".", ms=1.2, color=cols[k], label=f"S{s}")
                d0 = np.nanmedian(F[m & (np.abs(ph) < 0.02)]) if (m & (np.abs(ph) < 0.02)).any() else np.nan
                print(f"  P={P:.5f}: S{s} median flux at phase 0 (+-0.02) = {d0:.4f}")
        cyc = np.floor((T - t0) / P + 0.5)
        for name, sel in (("odd", cyc % 2 == 1), ("even", cyc % 2 == 0)):
            mm = ok & sel & (np.abs(ph) < 0.02)
            print(f"  P={P:.5f}: {name} depth {1 - np.nanmedian(F[mm]) if mm.any() else np.nan:.4f}")
        ax.set_title(f"fold P = {P:.5f} d, t0 = {t0:.3f}", fontsize=8)
        ax.legend(fontsize=6, markerscale=4)
        ax.tick_params(labelsize=6)
    for j in range(len(periods), axes.shape[1]):
        axes[1, j].set_axis_off()
    fig.suptitle(f"TIC {a.tic}", fontsize=9)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
