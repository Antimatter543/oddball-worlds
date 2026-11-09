"""Sanity plots for an ingested sector: N random light curves plus every known positive in it.

    uv run python -I scripts/plot_sanity.py 39 --n 6 --seed 0

Writes ``outputs/sanity_random_sNNNN.png`` and ``outputs/sanity_known_sNNNN.png``.
Each panel shows binned PDCSAP flux (black), SAP flux offset below (grey), momentum dumps
(red ticks, quality bit 32) and scattered-light-flagged bins (orange band, bits 2048/4096).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from oddball.common.paths import REPO_ROOT, outputs_root  # noqa: E402
from oddball.tess.store import load_lightcurves, metadata  # noqa: E402

KNOWN = REPO_ROOT / "docs" / "known_positives.csv"


def _panel(ax, lc, meta_row, label):
    t, f = lc["time"], lc["flux"]
    q = lc["qflags"]
    rms = meta_row.rms_bin
    sap_off = 1 - max(6 * rms, 0.01) if np.isfinite(rms) else 0.98
    for tt in t[(q & 32) != 0]:
        ax.axvline(tt, color="tab:red", lw=0.6, alpha=0.6)
    sl = (q & (2048 | 4096)) != 0
    if sl.any():
        ax.fill_between(t, 0, 1, where=sl, transform=ax.get_xaxis_transform(),
                        color="tab:orange", alpha=0.15, lw=0)  # fmt: skip
    sap = lc["sap_flux"]
    ax.plot(t, sap - np.nanmedian(sap) + sap_off, ".", ms=1.5, color="0.6", label="SAP (offset)")
    ax.plot(t, f, ".", ms=2, color="k", label="PDCSAP 30-min")
    lo, hi = np.nanpercentile(np.concatenate([f, sap - np.nanmedian(sap) + sap_off]), [0.5, 99.5])
    pad = 0.15 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(
        f"{label}TIC {lc['tic']}  T={meta_row.tmag:.1f}  cam{meta_row.camera}/ccd{meta_row.ccd}  "
        f"rms={rms * 1e3:.2f} ppt  bins={int(meta_row.n_bins_filled)}/{int(meta_row.n_bins)}",
        fontsize=8, loc="left",
    )  # fmt: skip
    ax.tick_params(labelsize=7)


def _figure(lcs, meta, labels, title, path):
    n = len(lcs)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.9 * n + 0.6), squeeze=False)
    for ax, (tic, lc) in zip(axes[:, 0], lcs.items(), strict=True):
        _panel(ax, lc, meta.loc[tic], labels.get(tic, ""))
    axes[-1, 0].set_xlabel("BTJD (BJD - 2457000)")
    axes[0, 0].legend(fontsize=7, loc="upper right", markerscale=4)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sector", type=int)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-known", type=int, default=12)
    a = ap.parse_args()

    meta = metadata(a.sector).set_index("tic")
    rng = np.random.default_rng(a.seed)
    rand = [int(x) for x in rng.choice(meta.index.to_numpy(), size=min(a.n, len(meta)), replace=False)]
    lcs = load_lightcurves(a.sector, rand)
    out = outputs_root()
    _figure({t: lcs[t] for t in rand}, meta, {},
            f"Sector {a.sector}: {len(rand)} random TESS-SPOC light curves (seed {a.seed})",
            out / f"sanity_random_s{a.sector:04d}.png")  # fmt: skip

    known = {}
    if KNOWN.exists():
        with KNOWN.open() as fh:
            for r in csv.DictReader(fh):
                if r["tic"] and int(r["tic"]) in meta.index:
                    known[int(r["tic"])] = f"{r['name']} [{r['class']}]  "
    if known:
        tics = list(known)[: a.max_known]
        klcs = load_lightcurves(a.sector, tics)
        _figure({t: klcs[t] for t in tics}, meta, known,
                f"Sector {a.sector}: known positives in the ingested set "
                f"({len(tics)} of {len(known)} shown)",
                out / f"sanity_known_s{a.sector:04d}.png")  # fmt: skip
    else:
        print("no known positives in this sector's ingested set")


if __name__ == "__main__":
    main()
