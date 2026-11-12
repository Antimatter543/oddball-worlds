"""What a vetted survey signal would mean if it were a planet on the target (for the shortlist).

All of it assumes the dip is on the target, circular orbit, TIC mass / radius / Teff. Each
number is a consequence of the signal, not a measurement of a planet.

  radius     R_p = R_* sqrt(depth / crowd_frac). `depth` is the search's box depth on PDCSAP (PDC
             already removes the CROWDSAP dilution), so crowd_frac = 1 unless the dip is shown to be
             diluted further; a box depth underestimates a limb-darkened transit by ~10-20%, so the
             radius is a slight underestimate.
  a          Kepler's third law from M_* and P.
  insolation S / S_earth = (R_*/R_sun)^2 (T_eff / 5772 K)^4 / (a / AU)^2.
  T_eq       278.6 K x S^0.25 x (1 - A)^0.25, full heat redistribution; quoted for A = 0.3 (Earth)
             and A = 0 (an upper bound).
  HZ         Kopparapu et al. (2013, ApJ 765, 131; erratum 2013) S_eff limits for 1 M_earth:
             conservative = runaway greenhouse .. maximum greenhouse; optimistic = recent Venus
             .. early Mars. T_eff is clipped to 2600 K (the fit's range).
"""

from __future__ import annotations

import math

G_SI = 6.674e-11
MSUN = 1.989e30
AU = 1.496e11
RSUN_RE = 109.08

_KOP = {  # S_eff_sun, a, b, c, d  (Kopparapu et al. 2013, corrected coefficients)
    "recent_venus": (1.7763, 1.4335e-4, 3.3954e-9, -7.6364e-12, -1.1950e-15),
    "runaway_greenhouse": (1.0385, 1.2456e-4, 1.4612e-8, -7.6345e-12, -1.7511e-15),
    "maximum_greenhouse": (0.3507, 5.9578e-5, 1.6707e-9, -3.0058e-12, -5.1925e-16),
    "early_mars": (0.3207, 5.4471e-5, 1.5275e-9, -2.1709e-12, -3.8282e-16),
}


def hz_limits(teff: float) -> dict:
    t = max(min(teff, 7200.0), 2600.0) - 5780.0
    return {k: s + a * t + b * t**2 + c * t**3 + d * t**4 for k, (s, a, b, c, d) in _KOP.items()}


def implied(P: float, depth: float, mass: float, radius: float, teff: float,
            crowd_frac: float = 1.0) -> dict:
    rp = radius * RSUN_RE * math.sqrt(max(depth, 0) / crowd_frac)
    a_au = (G_SI * mass * MSUN * (P * 86400) ** 2 / (4 * math.pi**2)) ** (1 / 3) / AU
    lum = radius**2 * (teff / 5772.0) ** 4
    s = lum / a_au**2
    hz = hz_limits(teff)
    if hz["runaway_greenhouse"] >= s >= hz["maximum_greenhouse"]:
        zone = "inside the conservative habitable zone"
    elif hz["recent_venus"] >= s >= hz["early_mars"]:
        zone = "inside the optimistic habitable zone only"
    elif s > hz["recent_venus"]:
        zone = f"hotter than the habitable zone ({s / hz['runaway_greenhouse']:.1f}x the inner edge)"
    else:
        zone = "colder than the habitable zone"
    return {"rp_re": rp, "a_au": a_au, "lum_lsun": lum, "s_earth": s,
            "teq_a03_k": 278.6 * s**0.25 * 0.7**0.25, "teq_a0_k": 278.6 * s**0.25,
            "hz": hz, "zone": zone}


def next_transits(t0: float, P: float, after: float, n: int = 3) -> list[float]:
    k = math.ceil((after - t0) / P)
    return [t0 + (k + i) * P for i in range(n)]


def tess_sectors(ra: float, dec: float, after_sector: int = 0) -> list[dict]:
    """Sectors (tess-point pointing table, through its last planned sector) in which the position
    falls on a camera, with approximate windows (BTJD) = sector mid-time +- 13.5 d. The window is
    a planning estimate: real sector boundaries differ by up to ~1-2 d and have a mid-sector
    gap."""
    import numpy as np
    import tess_stars2px as t2p

    out = t2p.tess_stars2px_function_entry(1, ra, dec)
    secs = sorted({int(s) for s in np.atleast_1d(out[3]) if int(s) > after_sector})
    sc = t2p.TESS_Spacecraft_Pointing_Data()
    mid = {int(s): float(m) - 2457000 for s, m in zip(sc.sectors, sc.midtimes)}
    return [{"sector": s, "mid": mid.get(s), "start": mid[s] - 13.5 if s in mid else None,
             "end": mid[s] + 13.5 if s in mid else None} for s in secs]


def timing_sigma(P: float, dur_d: float, epochs: list[float], t_pred: float) -> float:
    """Rough 1-sigma of a predicted mid-time (days): each epoch timed to ~D/4, a linear ephemeris
    fitted through the observed epochs, extrapolated to t_pred. Optimistic for low-SNR epochs."""
    import numpy as np

    e = np.asarray(epochs, float)
    n = np.round((e - e[0]) / P)
    if len(e) < 2 or np.ptp(n) == 0:
        return float("nan")
    st = dur_d / 4
    A = np.vstack([np.ones_like(n), n]).T
    cov = np.linalg.inv(A.T @ A) * st**2
    m = round((t_pred - e[0]) / P)
    v = np.array([1.0, m])
    return float(math.sqrt(v @ cov @ v))
