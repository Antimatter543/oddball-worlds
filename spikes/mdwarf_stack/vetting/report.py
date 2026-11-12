"""Plain-English summary and one-page diagnostic figure for one vetted signal.

The wording is tied to the tier: only tier A is called a "planet candidate", and even then the
summary lists what is still unproven. Tier B is "a possible planet signal", C "most likely a
false alarm", K "a known object". The word "planet" alone is never used for a signal.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import BIN_D  # noqa: E402

REARTH_RSUN = 0.0091577
BTJD_NOW = 4356.5  # 2026-11-12 00:00 UTC (JD 2461356.5 - 2457000)

QUESTION = {
    "odd_even": "an eclipsing binary at twice the period (alternate dips would differ)",
    "per_transit": "one unusual event carrying the detection, or dips that disagree",
    "sector_consistency": "a depth that changes between sectors (systematics or a blend)",
    "duration_density": "a dip too long for a planet around this small star",
    "sap": "an artefact of the pipeline's correction, or a background (sky) change",
    "systematics_window": "dips at data-gap edges, thruster firings or scattered light",
    "orbit_alias": "the spacecraft's 13.7-day orbit systematics",
    "secondary": "a second, shallower eclipse (an eclipsing binary)",
    "model_compare": "a step, a starspot wave or a single bad point instead of a dip",
    "uniqueness": "noise at the same period making dips (or bumps) as strong as this one",
    "rotation_residual": "the star's own spot modulation at the signal's period",
    "pixel": "a neighbouring star in the pixels being the one that dims",
    "contamination": "a nearby star bright enough to fake the dip",
    "catalogue": "an object that is already known (planet, TOI, CTOI, eclipsing binary)",
}
OUTCOME = {"pass": "ruled out by this test", "soft": "NOT ruled out (soft fail)",
           "fail": "FAILED: this explanation is favoured", "n/a": "not tested here",
           "error": "could not be checked (query or data problem)", "info": "matched"}

LACKS = {
    "A": ["reconnaissance spectroscopy (a double-lined binary or a giant star would end it)",
          "high-resolution imaging (a blend closer than a few arcseconds is invisible to TESS)",
          "a ground-based transit at the predicted time, on this star",
          "a statistical false-positive probability (TRICERATOPS-style)"],
}
LACKS["B"] = LACKS["A"] + ["the strike listed above resolved; for a two-transit signal the "
                           "true period (P, P/2, P/3, ...) is unknown"]


def planet_radius(depth, rstar, crowd=1.0):
    if not (depth and depth > 0 and rstar):
        return None
    return math.sqrt(depth) * rstar / REARTH_RSUN


def next_transit(stats):
    P, t0 = stats["P"], stats["t0"]
    ep = np.array(stats["epoch_n"])
    span = max(int(ep.max() - ep.min()), 1)
    dP = stats["dur_h"] / 24 / span  # a duration of drift over the baseline
    n = math.ceil((BTJD_NOW - t0) / P)
    tn = t0 + n * P
    sig = math.hypot(stats["dur_h"] / 24 / 2, (n - ep.max()) * dP)
    return tn, sig


def summary(stats, g, rstar=None, name=None, blind=False):
    name = name or f"TIC {stats.get('tic', '')}".strip()
    tier = g["tier"]
    rp = planet_radius(stats["depth"], rstar)
    size = f" (about {rp:.1f} Earth radii if it is on this star)" if rp else ""
    head = (f"{name}: a dip of {1e3 * stats['depth']:.2f} parts per thousand{size}, lasting "
            f"{stats['dur_h']:.1f} h, seen {stats['n_transits']} times, every {stats['P']:.3f} "
            f"days. Its strength (MES) is {stats['mes']:.1f}; the survey's false-alarm threshold "
            f"is 8.05.")
    verdict = {
        "A": "Tier A, a planet candidate: it passed every check below and has 3 or more dips.",
        "B": "Tier B, a possible planet signal with one open problem (listed below).",
        "C": "Tier C, most likely a false alarm.",
        "K": "Tier K: it matches an object that is already known.",
    }[tier]
    parts = [head, verdict]
    st = g["status"]
    checks = []
    for t, q in QUESTION.items():
        s = st.get(t, "n/a")
        if s == "n/a" and t in ("pixel",):
            checks.append(f"{q}: not checked (the pixel test was not run or is uninformative "
                          f"for a dip this shallow)")
            continue
        checks.append(f"{q}: {OUTCOME.get(s, s)}")
    parts.append("False-alarm explanations tested: " + "; ".join(checks) + ".")
    if g["reasons"]:
        parts.append("Details of the problems: " + "; ".join(g["reasons"]) + ".")
    if tier in ("A", "B"):
        tn, sig = next_transit(stats)
        unproven = list(LACKS[tier])
        if st.get("pixel") in ("n/a", "error"):
            unproven.insert(0, "that the dip is on this star at all (the pixel check is "
                               "missing or uninformative)")
        parts.append("Still unproven: " + "; ".join(unproven) + ".")
        how = (f"What would settle it: a ground-based observation of the next predicted dip "
               f"near BTJD {tn:.2f} (+-{24 * sig:.1f} h, roughly)")
        if stats["n_transits"] == 2:
            how += (f", and also at the times implied by P/2 and P/3 "
                    f"({stats['P'] / 2:.2f} and {stats['P'] / 3:.2f} d), since two dips cannot "
                    f"fix the period")
        how += "; more TESS sectors would also add dips and sharpen every test."
        parts.append(how)
    elif tier == "C":
        parts.append("Nothing to follow up unless the failed test is shown to be wrong.")
    else:
        parts.append("Its value is an independent recovery of a known object, not a discovery.")
    return " ".join(parts)


def render(ctx, stats, g, path, title=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from search import build_ses

    from vet import fold_phases

    pp = ctx.pp
    t = (pp["idx"] + 0.5) * BIN_D
    f = pp["flux"]
    P, t0, Dd = stats["P"], stats["t0"], stats["dur_h"] / 24
    ph = (t - t0 + 0.5 * P) % P - 0.5 * P
    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(5, 3, height_ratios=[1, 1, 0.9, 1, 1.2], hspace=0.45, wspace=0.25)

    ax = fig.add_subplot(gs[0, :])
    # compress gaps: plot against cumulative time with gaps > 2 d shrunk to 1 d
    gaps = np.diff(t)
    shrink = np.concatenate([[0], np.where(gaps > 2, gaps - 1, 0)]).cumsum()
    tc = t - shrink
    k = max(1, int(round(1 / 24 / BIN_D)))
    ax.plot(tc, f, ",", color="0.6")
    nb = len(f) // 6
    if nb > 10:
        ax.plot(tc[: nb * 6].reshape(-1, 6).mean(1), f[: nb * 6].reshape(-1, 6).mean(1), ".",
                ms=1.5, color="k")
    for e in stats["epochs"]:
        j = np.searchsorted(t, e)
        j = min(j, len(t) - 1)
        ax.axvline(e - shrink[j], color="tab:red", alpha=0.6, lw=1)
    ax.set_title(f"{title}  search-input flux, gaps > 2 d shrunk; red = dips (n = "
                 f"{stats['n_transits']})", fontsize=9)
    lo, hi = np.percentile(f, [0.5, 99.5])
    ax.set_ylim(lo - 0.3 * (hi - lo), hi + 0.3 * (hi - lo))
    del k

    def binned(x, y, w):
        b = np.floor(x / w).astype(int)
        u, inv = np.unique(b, return_inverse=True)
        return (u + 0.5) * w, np.bincount(inv, y) / np.bincount(inv)

    win = max(4 * Dd, 0.5)
    m = np.abs(ph) < win
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(ph[m] * 24, f[m], ".", ms=1, color="0.7")
    if m.sum() > 5:
        bx, by = binned(ph[m], f[m], Dd / 4)
        ax.plot(bx * 24, by, "o-", ms=3, color="k")
    ax.axhline(1 - stats["depth"], color="tab:red", lw=0.8)
    ax.set_title("phase fold (h from mid-transit)", fontsize=9)
    ax = fig.add_subplot(gs[1, 1])
    ep = np.round((t - t0) / P).astype(int)
    for par, col in ((0, "tab:blue"), (1, "tab:orange")):
        mm = m & (ep % 2 == par)
        if mm.sum() > 5:
            bx, by = binned(ph[mm], f[mm], Dd / 3)
            ax.plot(bx * 24, by, "o-", ms=3, color=col, label="even" if par == 0 else "odd")
    oe = stats.get("odd_even") or {}
    ax.set_title(f"odd / even (z = {oe.get('z', float('nan')):.1f})", fontsize=9)
    ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[1, 2])
    if ctx.sap is not None:
        ts = (ctx.sap["idx"] + 0.5) * BIN_D
        phs = (ts - t0 + 0.5 * P) % P - 0.5 * P
        ms_ = np.abs(phs) < win
        if ms_.sum() > 5:
            bx, by = binned(phs[ms_], ctx.sap["flux"][ms_], Dd / 3)
            ax.plot(bx * 24, by, "o-", ms=3, color="tab:green", label="SAP")
    if m.sum() > 5:
        bx, by = binned(ph[m], f[m], Dd / 3)
        ax.plot(bx * 24, by, "o-", ms=3, color="k", label="PDCSAP")
    sp = stats.get("sap") or {}
    ax.set_title(f"SAP vs PDCSAP (z_SAP = {-sp.get('z_neg', float('nan')):.1f})", fontsize=9)
    ax.legend(fontsize=7)

    sub = gs[2, :].subgridspec(1, min(8, max(1, stats["n_transits"])))
    order = np.argsort(-np.array(stats["epoch_ses"]))[:8]
    for i, j in enumerate(sorted(order, key=lambda x: stats["epochs"][x])):
        ax = fig.add_subplot(sub[0, i])
        c = stats["epochs"][j]
        mm = np.abs(t - c) < win
        ax.plot((t[mm] - c) * 24, f[mm], ".", ms=2, color="k")
        ax.axvspan(-Dd * 12, Dd * 12, color="tab:red", alpha=0.15)
        ax.set_title(f"{c:.2f}\nSES {stats['epoch_ses'][j]:.1f}", fontsize=7)
        ax.tick_params(labelsize=6)

    ax = fig.add_subplot(gs[3, :2])
    ses = build_ses(pp, stats["k"], 1)
    phase, fm, dok, bok = fold_phases(ses, P, t0)
    o = np.argsort(phase)
    ax.plot(phase[o], fm[o], lw=0.6, color="k")
    ax.axhline(0, color="0.5", lw=0.5)
    sec = stats.get("secondary") or {}
    un = stats.get("uniqueness") or {}
    ax.set_title(f"MES at every phase of the fold (primary at 0; best other dip "
                 f"{sec.get('mes_sec', 0):.1f}, best bump {un.get('mes_pos', 0):.1f}, "
                 f"scramble p = {un.get('p_scr', float('nan')):.2f})", fontsize=9)
    ax.set_xlabel("phase (d)")
    ax = fig.add_subplot(gs[3, 2])
    ax.axis("off")
    px = stats.get("pixel")
    star = (stats.get("star") or {})
    cont = star.get("contamination") or {}
    txt = ["pixel check: " + (px.get("why", px.get("status", "")) if px else "not run"),
           f"Gaia neighbours <= 21\": {cont.get('n_nb_1px', '?')}",
           f"min eclipse a neighbour needs: "
           + (f"{100 * cont['min_need']:.0f}%" if cont.get("min_need") else "none nearby")]
    ax.text(0, 1, "\n".join(txt), va="top", fontsize=8, family="monospace")

    ax = fig.add_subplot(gs[4, :])
    ax.axis("off")
    lines = [f"TIER {g['tier']}   strikes {g['strikes']}   hard fails {g['hard']}"]
    for tname, s in g["status"].items():
        lines.append(f"{tname:20s} {s}")
    lines.append("")
    lines += [r[:150] for r in g["reasons"][:8]]
    ax.text(0, 1, "\n".join(lines), va="top", fontsize=8, family="monospace")
    fig.savefig(path, dpi=80)
    plt.close(fig)
