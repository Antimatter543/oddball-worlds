"""Vetting v2 (docs/spikes/mdwarf_vetting.md, "Vetting v2: pre-registration", 2026-11-12).

v1 (`vet.py`) is frozen and untouched. v2 = v1's `measure` + three new statistics, graded by
`grade2` with a reduced v1 test set, contamination as info only, a revised pixel soft rule and
tier A+.

  measure2(ctx, sig, st=None) -> v1 stats (computed if st is None) + gap_edge, red_noise, n_class
  grade2(stats, thr, tau_det, blind=False) -> tier (A+/A/B/C/K), per-test status, strikes, reasons
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "gate2"))
sys.path.insert(0, str(HERE))
from common import BIN_D
from search import build_ses
from vet import _catalogue_status, _epoch_windows, _level, applicable, measure, stat_value

GAP_MIN_BINS = 3  # consecutive present bins >= 3 apart = >= 20 min without data
GAP_NEAR_D = 1.0 / 24  # an epoch window within 1 h of a gap is flagged
RED_WIN_D = 1.5  # local SES scatter from D < |t - c| <= 1.5 d
RED_MIN_N = 30
F_PMIN = 40.0  # few-transit class: n_tr <= 3, or P >= 40 d and n_tr <= 4

# v1 tests kept in v2, with v1's frozen thresholds (copied unchanged by freeze_vet2.py)
V1_KEPT = ("odd_even.z", "per_transit.dom", "per_transit.minses_neg", "duration_density.ratio",
           "secondary.excess", "model_compare.neg_dbic", "uniqueness.neg_margin", "sap.z_neg",
           "sap.bkg_stat", "systematics_window.share", "orbit_alias.neg_spread",
           "rotation_residual.neg_logp")
V1_DROPPED = ("uniqueness.p_scr", "per_transit.neg_logp", "sector_consistency.neg_logp",
              "contamination.need_neg")
# new statistics: (default soft, default fail); larger = worse
NEW_DEFAULT = {"gap_edge.share": (0.5, 0.8), "red_noise.neg_ratio": (-0.8, -0.6),
               "low_margin.neg_excess": (-1.0, None)}
NEW_TESTS = ("gap_edge", "red_noise", "low_margin")
TESTS2 = ("odd_even", "per_transit", "duration_density", "secondary", "model_compare",
          "uniqueness", "sap", "systematics_window", "orbit_alias", "rotation_residual",
          "gap_edge", "red_noise", "low_margin", "pixel", "catalogue")
INFO_TESTS = ("contamination",)


def applicable2(test: str, mode: str) -> bool:
    if test in NEW_TESTS:
        return True  # time sampling and search-input SES only: every mode
    return applicable(test, mode)


def in_class_f(n_tr: int, P: float) -> bool:
    return n_tr <= 3 or (P >= F_PMIN and n_tr <= 4)


# --------------------------------------------------------------------------------------------
# measurement


def gap_edge(pp_idx, centres, Dd, ses_e) -> dict:
    """Share of positive SES^2 in epochs whose box window lies within 1 h of a >= 20-min gap."""
    idx = np.asarray(pp_idx)
    j = np.flatnonzero(np.diff(idx) >= GAP_MIN_BINS)
    g0 = (idx[j] + 1.0) * BIN_D  # first missing time (end of the last present bin)
    g1 = idx[j + 1] * BIN_D  # first present time after the gap
    # the series' own start and end count as gaps too
    g0 = np.concatenate([[-np.inf], g0, [(idx[-1] + 1.0) * BIN_D]])
    g1 = np.concatenate([[idx[0] * BIN_D], g1, [np.inf]])
    flags, dist = [], []
    for c in centres:
        a, b = c - Dd / 2, c + Dd / 2
        d = np.maximum(0.0, np.maximum(g0 - b, a - g1))
        dm = float(d.min())
        dist.append(dm * 24)
        flags.append(dm < GAP_NEAR_D)
    s2 = np.maximum(np.asarray(ses_e, float), 0) ** 2
    fl = np.array(flags, bool)
    share = float(s2[fl].sum() / s2.sum()) if s2.sum() > 0 else 0.0
    return {"share": share, "n_flagged": int(fl.sum()), "dist_h": dist}


def red_noise(ses, wi, Dd) -> dict:
    t, sv = ses.t, ses.N / np.sqrt(np.maximum(ses.D, 1e-300))
    z, w, sig = [], [], []
    for i in wi:
        c = t[i]
        dt = np.abs(t - c)
        loc = sv[(dt > Dd) & (dt <= RED_WIN_D)]
        if len(loc) >= RED_MIN_N:
            s = 1.4826 * float(np.median(np.abs(loc - np.median(loc))))
            s = s if s > 0 else 1.0
        else:
            s = 1.0
        sig.append(s)
        z.append(sv[i] / s)
        w.append(math.sqrt(ses.D[i]))
    z, w = np.array(z), np.array(w)
    mes_red = float(np.sum(z * w) / math.sqrt(np.sum(w**2)))
    mes_vet = float(np.sum(sv[wi] * w) / math.sqrt(np.sum(w**2)))
    ratio = mes_red / mes_vet if mes_vet > 0 else 0.0
    return {"neg_ratio": float(-ratio), "mes_red": mes_red, "sigma_loc": [float(x) for x in sig]}


def measure_extra(ctx, sig: dict) -> dict:
    """The v2-only statistics for one signal."""
    P, t0, k = float(sig["P"]), float(sig["t0"]), int(sig["k"])
    Dd = k * BIN_D
    ses = build_ses(ctx.pp, k, 1)
    wi, ep = _epoch_windows(ses, P, t0)
    if len(wi) == 0:
        return {"error": "no epoch windows"}
    ses_e = ses.N[wi] / np.sqrt(ses.D[wi])
    centres = t0 + ep * P
    return {"gap_edge": gap_edge(ctx.pp["idx"], centres, Dd, ses_e),
            "red_noise": red_noise(ses, wi, Dd)}


def measure2(ctx, sig: dict, st: dict | None = None) -> dict:
    st = measure(ctx, sig) if st is None else st
    if st.get("error"):
        return st
    st["v2"] = measure_extra(ctx, sig)
    return st


# --------------------------------------------------------------------------------------------
# grading


def stat2(stats: dict, key: str, tau_det: float):
    test, name = key.split(".")
    if test == "low_margin":
        n_tr = int(stats.get("n_transits", 0))
        if not in_class_f(n_tr, float(stats["P"])):
            return -np.inf  # not in the few-transit class: always passes
        return -(float(stats["mes"]) - tau_det)
    if test in ("gap_edge", "red_noise"):
        d = (stats.get("v2") or {}).get(test)
        if not d:
            return None
        v = d.get(name)
        return None if v is None or not np.isfinite(v) else float(v)
    return stat_value(stats, key)


def pixel2(px: dict) -> tuple[str, str]:
    """Revised pixel rule: the off-target soft rule needs >= 3 strong epochs."""
    S_t, S_nb = px.get("S_t", float("nan")), px.get("S_nb", float("nan"))
    n_strong, n_off = int(px.get("n_strong", 0)), int(px.get("n_off", 0))
    nbf = S_nb is not None and np.isfinite(S_nb)
    smax = max(S_t if np.isfinite(S_t) else -np.inf, S_nb if nbf else -np.inf)
    if nbf and S_nb >= 3 and S_nb > S_t + 2:
        return "fail", f"the dip is stronger on a neighbour (SNR {S_nb:.1f} vs {S_t:.1f} on target)"
    if not np.isfinite(smax) or smax < 3:
        return "n/a", f"uninformative: pixel SNR {S_t:.1f} on target (< 3)"
    if n_strong >= 3 and n_off >= n_strong / 2:
        return "soft", f"difference centroid > 1 px off target in {n_off} of {n_strong} strong epochs"
    return "pass", f"dip on target (pixel SNR {S_t:.1f}, {n_strong} strong epochs)"


def grade2(stats: dict, thr: dict, tau_det: float, blind: bool = False) -> dict:
    mode = stats.get("mode", "real")
    res, reasons = {}, []
    order = {"pass": 0, "soft": 1, "fail": 2}

    def put(test, status, why=None):
        old = res.get(test)
        if old in order and status in order and order[old] >= order[status]:
            return
        res[test] = status
        if why and status in ("soft", "fail"):
            reasons.append(f"{test}: {why}")

    for key, (soft, fail) in thr.items():
        test = key.split(".")[0]
        if not applicable2(test, mode):
            res[test] = "n/a"
            continue
        if test in NEW_TESTS:
            d = (stats.get("v2") or {})
            if d.get("error"):
                res[test] = "error"
                reasons.append(f"{test}: could not be computed ({d['error']})")
                continue
        else:
            d = stats.get(test)
            if d is None:
                res.setdefault(test, "n/a")
                continue
            if d.get("error"):
                res[test] = "error"
                reasons.append(f"{test}: could not be computed ({d['error']})")
                continue
        v = stat2(stats, key, tau_det)
        lv = _level(v, soft, fail)
        if lv is None:
            res.setdefault(test, "n/a")
            continue
        if test == "orbit_alias" and lv == "soft" and stats["orbit_alias"].get("alias_n"):
            lv = "fail"
        if test == "low_margin":
            why = (f"only {stats['n_transits']} transits"
                   + (f" at P >= {F_PMIN:g} d" if stats["P"] >= F_PMIN else "")
                   + f" and MES {stats['mes']:.2f} is less than {-soft:.2g} above the "
                     f"detection threshold {tau_det:.3g}")
        else:
            why = (f"{key.split('.')[1]} = {v:.3g} (soft > {soft:g}"
                   + (f", fail > {fail:g})" if fail is not None else ")"))
        put(test, lv, why)
    known = False
    info = {}
    if mode == "real":
        star = stats.get("star")
        if star is None:
            res["catalogue"] = "error"
            reasons.append("catalogue: catalogue information missing")
        else:
            c = star.get("contamination") or {}
            if not c.get("error") and c.get("min_need") is not None:
                info["contamination"] = (f"a Gaia neighbour within 21\" could make this dip if it "
                                         f"were an eclipsing binary with a {100 * c['min_need']:.2g}%"
                                         f" eclipse")
            elif c.get("error"):
                info["contamination"] = f"not checked ({c['error']})"
            if blind:
                res["catalogue"] = "n/a"
            else:
                cat = _catalogue_status(star, stats["P"])
                res["catalogue"] = cat["status"]
                known = cat["known"]
                if cat["status"] in ("soft", "fail", "error"):
                    reasons.append(f"catalogue: {cat['why']}")
        px = stats.get("pixel")
        if px is None:
            res["pixel"] = "n/a"
        elif px.get("error"):
            res["pixel"] = "error"
            reasons.append(f"pixel: {px['error']}")
        else:
            s, why = pixel2(px)
            res["pixel"] = s
            if s in ("soft", "fail"):
                reasons.append(f"pixel: {why}")
    for t_ in TESTS2:
        res.setdefault(t_, "n/a")
    n_tr = int(stats.get("n_transits", 0))
    n_soft = sum(v == "soft" for v in res.values())
    hard = [k for k, v in res.items() if v == "fail"]
    strikes = n_soft + (1 if n_tr == 2 else 0)
    errors = [k for k, v in res.items() if v == "error" and k != "pixel"]
    if stats.get("error"):
        tier = "C"
        reasons.append(stats["error"])
    elif known:
        tier = "K"
    elif hard or strikes >= 2:
        tier = "C"
    elif strikes == 0 and n_tr >= 3 and not errors:
        px = stats.get("pixel") or {}
        tier = "A+" if res["pixel"] == "pass" and int(px.get("n_strong", 0)) >= 3 else "A"
    else:
        tier = "B"
    if n_tr == 2:
        reasons.append("only 2 transits: the period could be P/2, P/3, ... (strike)")
    return {"tier": tier, "status": res, "strikes": strikes, "hard": hard, "errors": errors,
            "reasons": reasons, "info": info, "tau_det": tau_det}


def load_frozen2(path=None):
    import json

    from common import OUT

    p = Path(path) if path else OUT / "vetting" / "v2" / "frozen_vet2.json"
    if not p.exists():
        p = HERE / "frozen_vet2.json"
    d = json.loads(p.read_text())
    return {k: tuple(v) for k, v in d["thresholds"].items()}
