"""Quick-look PNG of each cutout with the target position at the plate epoch and at 2026.9.
Usage: s6_look.py DLDIR OUTPNG"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

warnings.filterwarnings("ignore")
dl, outpng = Path(sys.argv[1]), Path(sys.argv[2])
RA0, DE0, PMRA, PMDE = 79.3336968045, 32.1249287799, 86.155, -296.162


def pos(ep):
    dt = ep - 2016.0
    return (RA0 + PMRA * dt / 3.6e6 / np.cos(np.radians(DE0)), DE0 + PMDE * dt / 3.6e6)


EP = {"dss_poss1_red": 1955.81, "dss_poss1_blue": 1955.81, "dss_poss2ukstu_red": 1989.89,
      "dss_poss2ukstu_blue": 1996.77, "dss_poss2ukstu_ir": 1996.71, "2mass_J": 1998.09,
      "2mass_H": 1998.09, "2mass_K": 1998.09, "ps1_g": 2012.0, "ps1_r": 2012.0,
      "ps1_i": 2012.0, "ps1_z": 2012.0, "ps1_y": 2012.0}
fig, axs = plt.subplots(3, 5, figsize=(20, 12))
for ax, (k, ep) in zip(axs.flat, EP.items()):
    f = dl / f"{k}.fits"
    if not f.exists():
        ax.set_title(k + " missing")
        continue
    h = fits.getheader(f)
    d = fits.getdata(f).astype(float)
    w = WCS(h)
    p0 = pos(ep)
    p1 = pos(2026.9)
    x0, y0 = w.all_world2pix([[p0[0], p0[1]]], 0)[0]
    x1, y1 = w.all_world2pix([[p1[0], p1[1]]], 0)[0]
    sc = np.sqrt(np.abs(np.linalg.det(w.pixel_scale_matrix))) * 3600
    r = 40 / sc
    sub = d[int(y1 - r):int(y1 + r), int(x1 - r):int(x1 + r)]
    lo, hi = np.nanpercentile(sub, [5, 99.5])
    ax.imshow(d, origin="lower", cmap="gray_r", vmin=lo, vmax=hi)
    ax.set_xlim(x1 - r, x1 + r)
    ax.set_ylim(y1 - r, y1 + r)
    ax.plot(x0, y0, "b+", ms=12)
    ax.add_patch(plt.Circle((x1, y1), 2 / sc, fill=False, color="r"))
    ax.set_title(f"{k} ep {ep}  {sc:.2f}\"/px")
fig.tight_layout()
fig.savefig(outpng, dpi=70)
