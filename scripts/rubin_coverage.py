"""Map ALeRCE LSST object density on a Galactic grid + named star-forming regions.

Counts objects in 1-degree cones. Output: data/rubin/coverage.csv

    uv run python -I scripts/rubin_coverage.py [--step-l 10]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time

import astropy.units as u
from astropy.coordinates import SkyCoord

from oddball.rubin.alerce_api import AlerceLSST, data_root

SFRS = {
    "Orion": (83.8, -5.4), "Lupus3": (242.5, -39.1), "ChaI": (167.0, -77.0),
    "rhoOph": (246.8, -24.5), "CrA": (285.4, -37.0), "Carina": (161.0, -59.8),
    "Serpens": (277.5, 1.2), "Lagoon": (271.0, -24.4), "NGC2264": (100.25, 9.9),
    "VelaC": (135.0, -44.0), "NGC3603": (168.8, -61.3), "Taurus": (68.0, 25.0),
    "CMaR1": (106.0, -11.4), "Mon R2": (91.9, -6.4), "ExtragalField": (11.06, -43.33),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-l", type=float, default=10.0)
    ap.add_argument("--radius-deg", type=float, default=1.0)
    args = ap.parse_args()
    api = AlerceLSST()
    out = data_root() / "rubin" / "coverage.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    points = [(name, ra, dec) for name, (ra, dec) in SFRS.items()]
    l = 0.0
    while l < 360:
        for b in (-10.0, -4.0, 0.0, 4.0, 10.0, 30.0):
            c = SkyCoord(l=l * u.deg, b=b * u.deg, frame="galactic").icrs
            if c.dec.deg < 25:
                points.append((f"l{l:.0f}b{b:+.0f}", c.ra.deg, c.dec.deg))
        l += args.step_l
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "ra", "dec", "l", "b", "n_objects_1deg", "t_query_s"])
        for name, ra, dec in points:
            g = SkyCoord(ra=ra * u.deg, dec=dec * u.deg).galactic
            t = time.time()
            try:
                n = api.count_cone(ra, dec, args.radius_deg * 3600)
            except Exception as e:  # noqa: BLE001 - keep mapping; record failure
                print(name, "ERR", e, file=sys.stderr)
                n = -1
            dt = time.time() - t
            w.writerow([name, f"{ra:.4f}", f"{dec:.4f}", f"{g.l.deg:.2f}", f"{g.b.deg:.2f}", n,
                        f"{dt:.2f}"])
            f.flush()
            print(f"{name:14s} ra={ra:7.2f} dec={dec:6.2f} l={g.l.deg:6.1f} b={g.b.deg:5.1f} "
                  f"n={n} ({dt:.1f}s)", flush=True)
    print("requests", api.n_requests, "->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
