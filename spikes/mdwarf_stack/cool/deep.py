"""Full-depth look at one kept object: trapezoid fits to the primary and secondary eclipses (integrated
over each sector's exposure), eclipse-by-eclipse list, the pixel check, and the numbers the write-up needs.

Inputs: <OUT>/cool/ebscan/<tic>.json (period, centre, secondary phase), optionally overridden with
--P --tc --ph2. The light curve is the survey's rebuilt PDCSAP (same bytes), detrended as in ebscan.
Model per eclipse: trapezoid with depth d, total duration T14 and flat fraction f = T23/T14, evaluated
on a 1-min grid and averaged over each point's exposure (30 min for S1-26 FFIs, 10 min otherwise, since
everything is binned to 10 min). Errors: residual-scaled least-squares covariance (indicative).
Pixel check: vetting/pixel_check.pixel_check on up to 4 sectors holding the deepest epochs.

Writes <OUT>/cool/deep/<tic>.json.  Usage: python -I cool/deep.py TIC [--P --tc --ph2] [--no-pixel]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "survey"))
sys.path.insert(0, str(HERE.parent / "vetting"))
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))
from common import OUT
from ebscan import detrend, epochs_at, load

COOL = OUT / "cool"
DD = COOL / "deep"


def trap(dt, d, T, f):
    """Trapezoid dip: depth d, total duration T (days), flat fraction f, centred at dt = 0."""
    a = np.abs(dt)
    hT, hF = T / 2, f * T / 2
    y = np.zeros_like(dt)
    y[a <= hF] = -d
    m = (a > hF) & (a < hT)
    y[m] = -d * (hT - a[m]) / max(hT - hF, 1e-6)
    return y


def model(dt, expo, d, T, f, c):
    off = np.linspace(-0.5, 0.5, 15)
    yy = np.mean([trap(dt + o * expo, d, T, f) for o in off], axis=0)
    return yy + c


def fit_eclipse(t, r, expo, P, tc, T0):
    from scipy.optimize import least_squares

    dt = (((t - tc) / P + 0.5) % 1 - 0.5) * P
    m = np.abs(dt) <= max(2.5 * T0, 0.25)
    if m.sum() < 20:
        return None
    x, y, e = dt[m], r[m], expo[m]
    d0 = max(-np.percentile(y[np.abs(x) < T0 / 3], 20) if (np.abs(x) < T0 / 3).any() else 0.01, 1e-4)

    def res(p):
        return model(x, e, p[0], p[1], p[2], p[3]) - y

    best = None
    for f0 in (0.05, 0.5, 0.85):
        for Tm in (0.7, 1.0, 1.5):
            p0 = [d0, T0 * Tm, f0, 0.0]
            try:
                s = least_squares(res, p0, bounds=([0, 0.01, 0.0, -0.05], [1.0, 1.0, 0.99, 0.05]))
            except ValueError:
                continue
            if best is None or s.cost < best.cost:
                best = s
    if best is None:
        return None
    J = best.jac
    dof = max(len(y) - 4, 1)
    s2 = 2 * best.cost / dof
    try:
        cov = np.linalg.inv(J.T @ J) * s2
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        err = np.full(4, np.nan)
    d, T, f, c = best.x
    return {"depth": float(d), "e_depth": float(err[0]), "T14_h": float(T * 24), "e_T14_h": float(err[1] * 24),
            "flat_frac": float(f), "e_flat": float(err[2]), "n": int(m.sum()), "rms": float(np.sqrt(s2))}


def star_block(tic):
    import pandas as pd

    g = pd.read_csv(COOL / "cat" / "gaia_match.csv")
    r = g[g.tic == tic]
    if not len(r) or not np.isfinite(r.iloc[0].ra_g):
        return None
    r = r.iloc[0]
    nb = json.loads(r.nb_list_63) if isinstance(r.nb_list_63, str) else []
    return {"gaia": {"target": {"ra": float(r.ra_g), "dec": float(r.dec_g), "source": str(r.source_id),
                                "G": float(r.G)}, "neighbours": nb}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tic", type=int)
    ap.add_argument("--P", type=float)
    ap.add_argument("--tc", type=float)
    ap.add_argument("--ph2", type=float)
    ap.add_argument("--w2", type=float, help="secondary box width (h)")
    ap.add_argument("--no-pixel", action="store_true")
    ap.add_argument("--pixel-on", choices=["prim", "sec"], default="prim")
    a = ap.parse_args()
    eb = json.loads((COOL / "ebscan" / f"{a.tic}.json").read_text())
    P = a.P or eb["P"]
    tc = a.tc or eb["tc"]
    w1 = eb["prim"]["width_h"] / 24
    ph2 = a.ph2 if a.ph2 is not None else (eb.get("sec") or {}).get("phase")
    w2 = (a.w2 or (eb.get("sec") or {}).get("width_h", eb["prim"]["width_h"])) / 24
    t, f, sec, meta = load(a.tic)
    r, s = detrend(t, f)
    cad = {int(x["sector"]): float(x.get("cadence_s") or 600) for x in meta["sectors"]}
    expo = np.array([max(cad.get(int(k), 600.0), 600.0) / 86400 for k in sec])
    out = {"tic": a.tic, "P": P, "tc": tc, "ph2": ph2, "sigma": s,
           "cadences": {str(k): v for k, v in sorted(cad.items())}}  # fmt: skip
    out["fit_prim"] = fit_eclipse(t, r, expo, P, tc, max(w1, 0.03))
    ep1 = epochs_at(t, r, s, P, tc, w1)
    out["epochs_prim"] = ep1
    fast = expo <= 600 / 86400 + 1e-9
    if fast.sum() > 100 and (~fast).any():
        out["fit_prim_fast"] = fit_eclipse(t[fast], r[fast], expo[fast], P, tc, max(w1, 0.03))
    if ph2 is not None:
        tc2 = tc + ph2 * P
        out["fit_sec"] = fit_eclipse(t, r, expo, P, tc2, max(w2, 0.03))
        out["epochs_sec"] = epochs_at(t, r, s, P, tc2, w2)
    if not a.no_pixel:
        from pixel_check import pixel_check

        star = star_block(a.tic)
        ep = ep1 if a.pixel_on == "prim" else out.get("epochs_sec", [])
        sec_of = {}
        for ti in (x[0] for x in ep):
            j = int(np.argmin(np.abs(t - ti)))
            sec_of[ti] = int(sec[j]) if abs(t[j] - ti) < 0.5 else None
        ep = [x for x in ep if sec_of.get(x[0])]
        stats = {"epochs": [x[0] for x in ep], "epoch_ses": [x[1] / x[2] for x in ep],
                 "epoch_sector": [sec_of[x[0]] for x in ep],
                 "dur_h": (w1 if a.pixel_on == "prim" else w2) * 24}  # fmt: skip
        out["pixel"] = pixel_check(stats, star) if star and ep else {"error": "no Gaia position or epochs"}
    DD.mkdir(parents=True, exist_ok=True)
    (DD / f"{a.tic}.json").write_text(json.dumps(out, default=float, indent=1))
    fp = out.get("fit_prim") or {}
    fs = out.get("fit_sec") or {}
    px = out.get("pixel") or {}
    print(a.tic, f"P={P:.5f}", f"prim d={fp.get('depth', np.nan):.4f} T14={fp.get('T14_h', np.nan):.2f}h f={fp.get('flat_frac', np.nan):.2f}",
          f"sec d={fs.get('depth', np.nan):.4f} T14={fs.get('T14_h', np.nan):.2f}h", "| pixel:", px.get("status", "err"),
          px.get("why", px.get("error", "")))  # fmt: skip


if __name__ == "__main__":
    main()
