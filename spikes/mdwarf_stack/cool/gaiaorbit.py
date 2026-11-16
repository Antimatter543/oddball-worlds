"""Do TIC 238891168's two TESS eclipses fit the Gaia DR3 astrometric orbit (I/357/tbooc, NSSmodel 'Orbital')?

Gaia DR3 4633052007552688000: P = 152.361 +- 0.453 d, e = 0.557 +- 0.078, omega = 4.74 +- 10.29 deg,
Tperi = 46.807 +- 4.000 d after the Gaia reference epoch J2016.0 (JD 2457389.0), incl = 87.4 +- 2.7 deg
(values from <OUT>/cool/gaiakept.json, VizieR CfA mirror, 2026-11-16).

Conjunctions (where eclipses can happen) are at true anomaly f = 90 deg - omega and 270 deg - omega. The script
(1) gives the predicted spacing between the two kinds of conjunction (Monte Carlo over the Gaia errors) and compares it
with what the two eclipses need, (2) gives each eclipse's offset from the nearest predicted conjunction, and (3) lists
every predicted conjunction that falls inside TESS data, to check that no eclipse was missed where one should be.

Usage: python -I cool/gaiaorbit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "survey"))
sys.path.insert(0, str(HERE.parent))

TIC = 238891168
P, EP = 152.36103704962807, 0.45298234
E, EE = 0.5572486704226797, 0.07817516
W, EW = np.radians(4.7396898649454995), np.radians(10.2853298)
TP, ETP = 2457389.0 + 46.80716807138873 - 2457000.0, 4.000206  # BTJD (Gaia TCB ~ TDB to well under a minute)
ECL = np.array([1673.125, 3172.444])  # BTJD, cool/deep.py trapezoid centres


def mean_anom(f, e):
    ea = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(f / 2), np.sqrt(1 + e) * np.cos(f / 2))
    return (ea - e * np.sin(ea)) % (2 * np.pi)


def conj_times(p, e, w, tp):
    """Times (BTJD) of the first conjunction of each kind after periastron tp."""
    return [tp + mean_anom(fc, e) / (2 * np.pi) * p for fc in (np.pi / 2 - w, 3 * np.pi / 2 - w)]


def main():
    rng = np.random.default_rng(1)
    n = 20000
    p = P + EP * rng.standard_normal(n)
    e = np.clip(E + EE * rng.standard_normal(n), 0, 0.95)
    w = W + EW * rng.standard_normal(n)
    tp = TP + ETP * rng.standard_normal(n)
    sep = ((mean_anom(3 * np.pi / 2 - w, e) - mean_anom(np.pi / 2 - w, e)) % (2 * np.pi)) / (2 * np.pi) * p
    gap = ECL[1] - ECL[0]
    print(f"eclipse gap {gap:.3f} d = 9 P + {gap - 9 * P:.1f} +- {9 * EP:.1f} d  or  10 P - {10 * P - gap:.1f} +- {10 * EP:.1f} d")
    lo, med, hi = np.percentile(sep, [16, 50, 84])
    print(f"Gaia-predicted spacing from conjunction A (f=90-w) to B (f=270-w): {med:.1f} d (68%: {lo:.1f}-{hi:.1f})")
    # offsets of each eclipse from the nearest predicted conjunction of each kind, with Monte Carlo spread
    for i, t in enumerate(ECL):
        for k, lab in enumerate(("A", "B")):
            tc = np.array([conj_times(p[j], e[j], w[j], tp[j])[k] for j in range(2000)])
            cyc = np.round((t - tc) / p[:2000])
            off = t - (tc + cyc * p[:2000])
            print(f"  eclipse {i + 1} (BTJD {t:.3f}) vs conjunction {lab}: offset median {np.median(off):+.1f} d, "
                  f"68% {np.percentile(off, 16):+.1f} to {np.percentile(off, 84):+.1f} d")
    # predicted conjunctions inside the TESS data (nominal orbit, shifted to the observed eclipses)
    from ebscan import load

    t, f, sec, m = load(TIC)
    ta, tb = conj_times(P, E, W, TP)
    sa = ECL[0] - (ta + np.round((ECL[0] - ta) / P) * P)
    sb = ECL[1] - (tb + np.round((ECL[1] - tb) / P) * P)
    print(f"nominal-orbit shifts needed: A {sa:+.1f} d, B {sb:+.1f} d")
    for lab, t0 in (("A", ta + sa), ("B", tb + sb)):
        k = np.arange(np.floor((t.min() - t0) / P) - 1, np.ceil((t.max() - t0) / P) + 2)
        for tc in t0 + k * P:
            near = np.abs(t - tc) < 0.25
            if near.sum() > 5:
                s = sorted(set(sec[near].tolist()))
                y = f[near] / np.nanmedian(f[np.abs(t - tc) < 2]) - 1
                print(f"  conj {lab} at BTJD {tc:.2f}: in data (sector {s}), min flux within 6 h {np.nanmin(y):+.3f}")
            elif (np.abs(t - tc) < 15).sum() > 5:
                print(f"  conj {lab} at BTJD {tc:.2f}: within 15 d of data but not covered")
    # periods near the Gaia one that the TESS coverage allows: each eclipse repeats every P; a period is excluded if a
    # repeat of either eclipse lands on data (>= 4 points within 1 h of its centre) that shows no dip deeper than 5%
    grid = np.arange(P - 4 * EP, P + 4 * EP, 0.002)
    ok = []
    for pp in grid:
        bad = False
        for t0 in ECL:
            k = np.arange(np.floor((t.min() - t0) / pp) - 1, np.ceil((t.max() - t0) / pp) + 2)
            for tc in t0 + k[k != 0] * pp:
                near = np.abs(t - tc) < 1 / 24
                if near.sum() >= 4 and np.nanmin(f[near] / np.nanmedian(f[np.abs(t - tc) < 1]) - 1) > -0.05:
                    bad = True
                    break
            if bad:
                break
        if not bad:
            ok.append(pp)
    ok = np.array(ok)
    if len(ok):
        runs = np.split(ok, np.where(np.diff(ok) > 0.003)[0] + 1)
        print("periods within 4 sigma of Gaia's still allowed by the TESS coverage:",
              ", ".join(f"{r[0]:.3f}-{r[-1]:.3f}" for r in runs))
    else:
        print("no period within 4 sigma of Gaia's is allowed by the TESS coverage")


if __name__ == "__main__":
    main()
