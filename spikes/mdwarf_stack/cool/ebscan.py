"""Independent light-curve characterisation of flagged stars: true period, primary and secondary eclipse.

Not the frozen search. For each star: rebuild the survey's PDCSAP light curve (survey/get_lightcurve.py,
same bytes), detrend it with an iteratively masked biweight (1 d window, dips below -4 sigma masked),
then test candidate periods: every long- and short-search signal of the star and every catalogued period
(graded catalogue objects, VSX / Kostov / Gaia from the catalogue pass when present), each at
x1/3, 1/2, 1, 2, 3, locally refined. At each trial period a phase scan with box widths 0.5-8 h gives the
primary (highest SNR) and, outside +-1.5 widths of it, the secondary. Choice rule:
  * reject a trial whose primary odd/even depths differ at z > 4 (the period is half the true one);
  * keep the highest primary SNR, preferring the shortest period within 5% of it.
Then, at the chosen period, flags: twin (secondary at phase 0.5 +- 0.02 with depth ratio > 0.8, so P could
be half: equal eclipses), ecc (secondary phase off 0.5 by > 3 widths), and the per-epoch depths of both.

Also lists individual dip events (box SNR >= 8 at 1-6 h, one per 0.5 d) for single/double event review.

Writes <OUT>/cool/ebscan/<tic>.json (+ .png with --plot). Light curves stay in the capped lc_cache.
Usage: python -I cool/ebscan.py TIC [TIC ...] [--plot]   or   --file tics.txt [--shard i/n]
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "survey"))
from common import OUT

COOL = OUT / "cool"
ED = COOL / "ebscan"
BIN = 10.0 / 1440.0
WIDTHS_H = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
MAX_LC_BYTES = 1_200_000_000


def load(tic: int):
    from get_lightcurve import get_lightcurve

    lc, meta = get_lightcurve(int(tic), MAX_LC_BYTES)
    t = (lc["idx"].astype(float) + 0.5) * BIN
    f = lc["flux"].astype(float)
    sec = lc["sector"].astype(int)
    ok = np.isfinite(f) & (lc["q"] == 0)
    t, f, sec = t[ok], f[ok], sec[ok]
    for s in np.unique(sec):
        m = sec == s
        f[m] = f[m] / np.nanmedian(f[m])
    return t, f, sec, meta


def robust_sigma(x):
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def detrend(t, f, window=1.0):
    from wotan import flatten

    mask = np.zeros(len(t), bool)
    for _ in range(3):
        ff = np.where(mask, np.nan, f)
        tr = flatten(t, ff, method="biweight", window_length=window, break_tolerance=0.5,
                     edge_cutoff=0, return_trend=True)[1]  # fmt: skip
        bad = ~np.isfinite(tr)
        if bad.any():
            good = ~bad
            tr[bad] = np.interp(t[bad], t[good], tr[good])
        r = f / tr - 1
        s = robust_sigma(r[~mask])
        low = r < -4 * s
        new = np.zeros_like(mask)
        if low.any():
            tl = t[low]
            j = np.searchsorted(tl, t)
            d1 = np.abs(t - tl[np.clip(j - 1, 0, len(tl) - 1)])
            d2 = np.abs(t - tl[np.clip(j, 0, len(tl) - 1)])
            new = np.minimum(d1, d2) <= 0.15
        if (new == mask).all():
            break
        mask = new
    return r, s


def phase_scan(t, r, s, P, t0, width_d, step_frac=0.25):
    """Box depth and SNR at every phase (centres every step_frac*width)."""
    ph = ((t - t0) / P) % 1.0
    o = np.argsort(ph)
    ph, rr = ph[o], r[o]
    cs = np.concatenate([[0], np.cumsum(rr)])
    w = width_d / P
    centres = np.arange(0, 1, max(step_frac * w, 1e-5))
    lo = np.searchsorted(ph, centres - w / 2)
    hi = np.searchsorted(ph, centres + w / 2)
    n = hi - lo
    sm = cs[hi] - cs[lo]
    # wrap
    lo2 = np.searchsorted(ph, centres - w / 2 + 1)
    hi2 = np.searchsorted(ph, centres + w / 2 - 1)
    sm = sm + (cs[-1] - cs[lo2]) * (centres - w / 2 < 0) + cs[hi2] * (centres + w / 2 > 1)
    n = n + (len(ph) - lo2) * (centres - w / 2 < 0) + hi2 * (centres + w / 2 > 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        depth = -sm / n
        snr = depth * np.sqrt(n) / s
    snr[n < 3] = 0
    return centres, depth, snr, n


def best_at(t, r, s, P, t0, exclude=None):
    """Best (phase, width, depth, snr) over widths; exclude=(phase, halfwidth) masks the primary."""
    best = None
    for wh in WIDTHS_H:
        wd = wh / 24
        if wd > 0.2 * P:
            continue
        c, d, z, n = phase_scan(t, r, s, P, t0, wd)
        if exclude is not None:
            dph = np.abs(((c - exclude[0]) + 0.5) % 1 - 0.5)
            z = np.where(dph < exclude[1] + wd / P / 2, 0, z)
        i = int(np.nanargmax(z))
        # penalise width slightly so that the narrowest adequate box wins (SNR within 3%)
        if best is None or z[i] > best["snr"] * 1.03:
            best = {"phase": float(c[i]), "width_h": float(wh), "depth": float(d[i]), "snr": float(z[i]),
                    "n": int(n[i])}  # fmt: skip
    return best


def epochs_at(t, r, s, P, tc, width_d):
    """Per-epoch depth at centre times tc + k P (epochs with >= 50% coverage)."""
    k = np.round((t - tc) / P)
    dt = t - (tc + k * P)
    inw = np.abs(dt) <= width_d / 2
    need = 0.5 * width_d / BIN
    out = []
    for kk in np.unique(k[inw]):
        m = inw & (k == kk)
        if m.sum() >= max(2, need):
            out.append((float(tc + kk * P), float(-r[m].mean()), float(s / np.sqrt(m.sum())), int(m.sum())))
    return out


def oe_z(ep, P):
    """Odd/even depth difference. The error of each parity's mean uses the larger of the white-noise
    error and the empirical scatter of its epochs (deep eclipses scatter far more than white noise:
    10-30 min sampling of a 1-2 h eclipse, small period error), so z > 4 means a real alternation."""
    if len(ep) < 2:
        return np.nan, np.nan, np.nan
    e = np.array(ep)
    n = np.round((e[:, 0] - e[0, 0]) / P).astype(int) % 2
    if n.min() == n.max():
        return np.nan, np.nan, np.nan

    def wm(x, w):
        m = (x * w).sum() / w.sum()
        err = 1 / np.sqrt(w.sum())
        if len(x) >= 2:
            err = max(err, np.std(x, ddof=1) / np.sqrt(len(x)))
        else:
            err = max(err, 0.1 * abs(m))  # one epoch: allow 10% sampling scatter
        return m, err

    a, sa = wm(e[n == 0, 1], 1 / e[n == 0, 2] ** 2)
    b, sb = wm(e[n == 1, 1], 1 / e[n == 1, 2] ** 2)
    return float(abs(a - b) / np.hypot(sa, sb)), float(a), float(b)


def peak_depth(t, r, P, tc, wd):
    """Deepest point of the median-binned fold within +-wd of the centre (bins of max(wd/4, 10 min))."""
    dt = (((t - tc) / P + 0.5) % 1 - 0.5) * P
    m = np.abs(dt) <= wd
    if m.sum() < 5:
        return float("nan")
    bw = max(wd / 4, BIN)
    k = np.floor(dt[m] / bw).astype(int)
    meds = [np.median(r[m][k == j]) for j in np.unique(k) if (k == j).sum() >= 3]
    return float(-min(meds)) if meds else float("nan")


def refine(t, r, s, P, t0, wd):
    base = t.max() - t.min()
    step = max(wd * P / (4 * base), 1e-7)
    nstep = int(np.clip(np.ceil(max(1.2e-3 * P, 6 * step) / step), 6, 60))
    grid = P + step * np.arange(-nstep, nstep + 1)
    best = (P, -np.inf, t0)
    for p in grid:
        c, d, z, n = phase_scan(t, r, s, p, t0, wd)
        i = int(np.nanargmax(z))
        if z[i] > best[1]:
            best = (float(p), float(z[i]), float(t0 + c[i] * p))
    return best


def events(t, r, s, min_snr=8.0):
    """Individual dips: box SNR >= min_snr at widths 1-6 h on the 10-min grid; one per 0.5 d."""
    idx = np.round(t / BIN).astype(int)
    g0 = idx.min()
    N = idx.max() - g0 + 1
    v = np.zeros(N)
    c = np.zeros(N)
    v[idx - g0] = r
    c[idx - g0] = 1
    cv, cc = np.concatenate([[0], np.cumsum(v)]), np.concatenate([[0], np.cumsum(c)])
    found = []
    for wh in (1.0, 2.0, 3.0, 6.0):
        k = int(round(wh / 24 / BIN))
        sm = cv[k:] - cv[:-k]
        nn = cc[k:] - cc[:-k]
        ok = nn >= 0.6 * k
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.where(ok, -sm / nn * np.sqrt(nn) / s, 0)
        for j in np.flatnonzero(z >= min_snr):
            found.append(((g0 + j + k / 2) * BIN, float(z[j]), float(-sm[j] / nn[j]), wh))
    found.sort(key=lambda x: -x[1])
    keep = []
    for f in found:
        if all(abs(f[0] - k_[0]) > 0.5 for k_ in keep):
            keep.append(f)
    return sorted(keep)


def candidate_periods(tic):
    L = pd.read_csv(COOL / "signals_long.csv")
    S = pd.read_csv(COOL / "signals_short.csv")
    ps = [("long", p, t0) for p, t0 in L.loc[L.tic == tic, ["P", "t0"]].itertuples(index=False)]
    ps += [("short", p, t0) for p, t0 in S.loc[(S.tic == tic) & (S.mes >= 6), ["P", "t0"]].itertuples(index=False)]
    for v in L.loc[L.tic == tic, "cat_objects"].dropna():
        for o in str(v).split("; "):
            try:
                p = float(o.rsplit("@", 1)[1])
                ps.append((o.split("@")[0], p, None))
            except (IndexError, ValueError):
                pass
    extra = COOL / "cat_periods.csv"
    if extra.exists():
        E = pd.read_csv(extra)
        for p, src in E.loc[E.tic == tic, ["P", "src"]].itertuples(index=False):
            ps.append((src, float(p), None))
    return ps


def analyse(tic: int, plot=False):
    t, f, sec, meta = load(tic)
    r, s = detrend(t, f)
    base = []
    seen = []
    for src, p, t0 in candidate_periods(tic):
        if not np.isfinite(p) or p < 0.15:
            continue
        if any(abs(p / q - 1) < 1e-3 for q in seen):  # same base period from another source
            continue
        seen.append(p)
        for m in (1 / 3, 0.5, 1, 2, 3):
            pp = p * m
            if 0.15 <= pp <= 130:
                base.append((src, m, pp, t0 if t0 is not None else float(t.min())))
    # dedupe trial periods
    trials = []
    for b in sorted(base, key=lambda x: x[2]):
        if not trials or abs(b[2] / trials[-1][2] - 1) > 1e-3:
            trials.append(b)
    # stage 1: every trial unrefined; stage 2: refine the 4 best and the x1/2, x2 of the best
    coarse = []
    for src, m, p, t0 in trials:
        b0 = best_at(t, r, s, p, t0)
        if b0 is not None:
            coarse.append((b0["snr"], src, m, p, t0, b0))
    coarse.sort(key=lambda x: -x[0])
    pick = coarse[:4]
    if coarse:
        pb = coarse[0][3]
        for c in coarse[4:]:
            if any(abs(c[3] / (pb * k) - 1) < 2e-3 for k in (0.5, 2.0)):
                pick.append(c)
    res = []
    for _, src, m, p, t0, b0 in pick:
        wd = b0["width_h"] / 24
        tc0 = t0 + b0["phase"] * p
        p2, z2, tc = refine(t, r, s, p, tc0, wd)
        b1 = best_at(t, r, s, p2, tc)
        tc = tc + b1["phase"] * p2
        ep = epochs_at(t, r, s, p2, tc, b1["width_h"] / 24)
        z, da, db = oe_z(ep, p2)
        res.append({"src": src, "mult": m, "P": p2, "tc": tc, "prim": b1, "oe_z": z, "d_even": da, "d_odd": db,
                    "n_ep": len(ep)})  # fmt: skip
    if not res:
        return {"tic": tic, "error": "no trial periods"}
    ok = [x for x in res if not (np.isfinite(x["oe_z"]) and x["oe_z"] > 4)] or res
    top = max(x["prim"]["snr"] for x in ok)
    near = [x for x in ok if x["prim"]["snr"] >= 0.95 * top]
    ch = min(near, key=lambda x: x["P"])
    P, tc, pr = ch["P"], ch["tc"], ch["prim"]
    sec_b = best_at(t, r, s, P, tc, exclude=(0.0, 1.5 * pr["width_h"] / 24 / P))
    ep1 = epochs_at(t, r, s, P, tc, pr["width_h"] / 24)
    out = {"tic": int(tic), "sigma": float(s), "n_pts": len(t), "sectors": sorted({int(x) for x in sec}),
           "baseline_d": float(t.max() - t.min()), "P": P, "tc": tc, "src": ch["src"], "mult": ch["mult"],
           "prim": pr, "oe_z": ch["oe_z"], "d_even": ch["d_even"], "d_odd": ch["d_odd"], "epochs_prim": ep1,
           "coarse": [[c[1], c[2], c[3], c[0]] for c in coarse[:25]],
           "trials": [{k: x[k] for k in ("src", "mult", "P", "oe_z", "n_ep")} | {"snr": x["prim"]["snr"],
                       "depth": x["prim"]["depth"]} for x in res]}  # fmt: skip
    if sec_b is not None:
        tc2 = tc + sec_b["phase"] * P
        ep2 = epochs_at(t, r, s, P, tc2, sec_b["width_h"] / 24)
        ph2 = sec_b["phase"]
        out["sec"] = sec_b
        out["epochs_sec"] = ep2
        out["sec_ratio"] = sec_b["depth"] / pr["depth"] if pr["depth"] > 0 else np.nan
        out["twin"] = bool(abs(ph2 - 0.5) < 0.02 and out["sec_ratio"] > 0.8 and sec_b["snr"] >= 7)
        out["ecc_flag"] = bool(sec_b["snr"] >= 7 and abs(ph2 - 0.5) * P > 3 * pr["width_h"] / 24)
    out["prim"]["depth_peak"] = peak_depth(t, r, P, tc, pr["width_h"] / 24)
    if sec_b is not None:
        out["sec"]["depth_peak"] = peak_depth(t, r, P, tc + sec_b["phase"] * P, sec_b["width_h"] / 24)
    out["events"] = events(t, r, s)
    ED.mkdir(parents=True, exist_ok=True)
    (ED / f"{tic}.json").write_text(json.dumps(out, default=float))
    if plot:
        make_plot(tic, t, r, s, out)
    return out


def make_plot(tic, t, r, s, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P, tc = out["P"], out["tc"]
    ph = ((t - tc) / P + 0.25) % 1 - 0.25
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    ax[0, 0].plot(t, r, ",k")
    for e in out["events"][:30]:
        ax[0, 0].axvline(e[0], color="r", lw=0.4)
    ax[0, 0].set_title(f"TIC {tic} detrended ({len(out['sectors'])} sectors)")
    ax[0, 1].plot(ph, r, ",k", alpha=0.5)
    bins = np.linspace(-0.25, 0.75, 801)
    j = np.digitize(ph, bins)
    bm = [np.nanmedian(r[j == k]) if (j == k).sum() > 3 else np.nan for k in range(1, len(bins))]
    ax[0, 1].plot(0.5 * (bins[1:] + bins[:-1]), bm, "-r", lw=0.8)
    ax[0, 1].set_title(f"P={P:.5f} d  d1={out['prim']['depth']*1e3:.1f} ppt  "
                       f"d2={out.get('sec', {}).get('depth', np.nan)*1e3:.1f} ppt @ {out.get('sec', {}).get('phase', np.nan):.3f}")  # fmt: skip
    lo = np.nanpercentile(r, 0.05)
    ax[0, 1].set_ylim(min(lo, -3 * s) * 1.2, 6 * s)
    w = out["prim"]["width_h"] / 24
    for k, (ctr, lab) in enumerate([(0.0, "primary"), (out.get("sec", {}).get("phase", np.nan), "secondary")]):
        a = ax[1, k]
        if not np.isfinite(ctr):
            continue
        dph = (((t - tc) / P - ctr) + 0.5) % 1 - 0.5
        m = np.abs(dph * P) < 4 * w + 0.1
        a.plot(dph[m] * P * 24, r[m], ".k", ms=2, alpha=0.6)
        a.set_title(f"{lab}: hours from centre")
        a.set_ylim(min(np.nanpercentile(r[m], 0.5) if m.sum() else -0.01, -3 * s) * 1.3, 5 * s)
    fig.tight_layout()
    fig.savefig(ED / f"{tic}.png", dpi=70)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tics", nargs="*", type=int)
    ap.add_argument("--file")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args()
    tics = list(a.tics)
    if a.file:
        tics += [int(x) for x in open(a.file).read().split()]
    i, n = map(int, a.shard.split("/"))
    tics = tics[i::n]
    for tic in tics:
        if a.skip_done and (ED / f"{tic}.json").exists():
            continue
        try:
            o = analyse(tic, a.plot)
            sb = o.get("sec") or {}
            print(tic, f"P={o.get('P', np.nan):.5f}", f"d1={o['prim']['depth']:.4f}", f"snr1={o['prim']['snr']:.1f}",
                  f"d2={sb.get('depth', np.nan):.4f}", f"snr2={sb.get('snr', np.nan):.1f}",
                  f"ph2={sb.get('phase', np.nan):.3f}", f"oe={o.get('oe_z', np.nan):.1f}", "twin" if o.get("twin") else "",
                  "ecc" if o.get("ecc_flag") else "", flush=True)  # fmt: skip
        except Exception as e:  # noqa: BLE001
            ED.mkdir(parents=True, exist_ok=True)
            (ED / f"{tic}.json").write_text(json.dumps({"tic": tic, "error": repr(e)[:300],
                                                        "tb": traceback.format_exc()[-800:]}))  # fmt: skip
            print(tic, "ERROR", repr(e)[:200], flush=True)


if __name__ == "__main__":
    main()
