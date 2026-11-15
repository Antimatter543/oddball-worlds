"""100 pc survey: tiers of every long signal >= tau_long next to the expectations for real planets
and for surviving false alarms.

Grading: frozen v2 (common100.graded: vet2.grade2, frozen_vet2.json, tau_det = tau_long, pixel
results merged). Expectations shown side by side (all computed here, none tuned on this run):

  pre-registered (gate 3, mdwarf_100pc.md R4/R6): planets 12.19 [9.41, 15.12] from the yield model,
      ~4.6 [~1.6, ~10] after the post-hoc 2.7x correction; surviving false alarms F = 5.5 [0, 16];
      vetting load ~195 signals.
  realised mixture (all false): far non-K signals in the bin x the 50 pc survey's survival rate
      (not C / non-K) in the same bin. The 50 pc survivors were all judged more likely false, so
      this is "as many as a 50 pc-like mix of noise, systematics and binaries would leave".
      Interval: Jeffreys on the 50 pc rate (Monte Carlo).
  noise only, far nulls: N_searched x r x s with r = the gate-3 far null rate per realisation at
      tau_long (9/940) and s = the share of far nulls >= 7 not in C (v2, tau_det 7.0) that fall in
      the bin (20/42 overall); Jeffreys on both. A 50 pc-method row (r = 1/118, s from the 50 pc
      fresh nulls) is shown beside it, as in docs/shortlist_planets_v1.md.
  real planets by bin: the gate-3 kept far injections (detected and not C), each weighted by
      DC15 eta for its cell / injections per star in the cell x R*/a, scaled to the pre-registered
      total (12.19) and to the corrected one (4.6).

Usage: python -I survey100/tables100.py -> <OUT>/survey100/vet/tables.txt, tables.json, graded.jsonl
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "gate3"))
from common import OUT
from common100 import VET, errors, graded

TIERS = ("K", "A+", "A", "B", "C")
N_SEARCHED = 15452
PLANETS_PRE, PLANETS_CORR = 12.19, 4.6
F_PRE = (5.48, 0.0, 16.0)
ETA = {"h": 0.134, "l": 0.098, "m": 0.258, "x": 0.323}   # analyze3.json eta central
PER_STAR = {"h": 2, "l": 1, "m": 1, "x": 1}
G, MSUN, RSUN, DAY = 6.674e-11, 1.989e30, 6.957e8, 86400.0


def ntr_bin(n):
    return "2" if n == 2 else "3" if n == 3 else "4" if n == 4 else "5+"


def p_bin(P):
    return "10-15" if P < 15 else "15-30" if P < 30 else "30-40" if P < 40 else "40+"


def sec_bin(n):
    return "1-3" if n <= 3 else "4-5" if n <= 5 else "6-12" if n <= 12 else "13+"


def rp_re(depth, rstar):
    return math.sqrt(max(depth, 0)) * rstar * 109.08


def a_over_r(P, m, r):
    a = (G * m * MSUN * (P * DAY) ** 2 / (4 * math.pi ** 2)) ** (1 / 3)
    return a / (r * RSUN)


def cell(inj):
    rp, P = inj["rp"], inj["P"]
    if rp < 1.5:
        return "h" if P < 30 else "l"
    return "m" if rp < 2.0 else "x"


def main():
    from analyze3 import grade_all, load_runs

    rows = graded()
    errs = errors()
    keys = {
        "ntr": lambda s, r: ntr_bin(int(s["n_transits"])),
        "P": lambda s, r: p_bin(s["P"]),
        "sec": lambda s, r: sec_bin(int(r["n_sectors"])),
        "pile": lambda s, r: "40+ & 2-4" if s["P"] >= 40 and 2 <= s["n_transits"] <= 4 else "other",
        "all": lambda s, r: "all",
    }
    labels = {"ntr": ["2", "3", "4", "5+"], "P": ["10-15", "15-30", "30-40", "40+"],
              "sec": ["1-3", "4-5", "6-12", "13+"], "pile": ["40+ & 2-4", "other"], "all": ["all"]}

    # 50 pc survey survival by bin (its graded rows, frozen v2 + its pixel results)
    g50 = [json.loads(x) for x in open(OUT / "survey" / "vet" / "graded.jsonl") if x.strip()]
    # gate-3 far material
    _, _, _, _, _, vet3 = load_runs()
    gg3 = grade_all(vet3)
    nulls = [(vet3[k], g) for k, g in gg3.items() if vet3[k].get("pop") == "g3_fa_null"
             and "stats" in vet3[k]]
    injk = [(vet3[k], g) for k, g in gg3.items() if vet3[k].get("pop") == "g3_true"
            and "stats" in vet3[k] and g["tier"] != "C"]
    # 50 pc fresh nulls >= 7 (as survey_tables.null_fa7)
    sys.path.insert(0, str(HERE.parent / "survey"))
    from survey_tables import null_fa7
    fa50 = null_fa7()

    rng = np.random.default_rng(20261216)
    size = 200_000
    r_far = rng.beta(9 + 0.5, 940 - 9 + 0.5, size)
    r_118 = rng.beta(1 + 0.5, 118 - 1 + 0.5, size)

    def q(x):
        return (float(x.mean()), float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975)))

    def fmt(e, d=1):
        return f"{e[0]:.{d}f} [{e[1]:.{d}f}, {e[2]:.{d}f}]"

    # injection weights for the planet expectation
    w = []
    for r, g in injk:
        c = cell(r["inj"])
        w.append(ETA[c] / PER_STAR[c] / a_over_r(r["inj"]["P"], r["mass"], r["radius"]))
    w = np.array(w)
    w = w / w.sum()

    out, js = [], {}
    P = out.append
    tiers = Counter(r["grade"]["tier"] for r in rows)
    best = {}
    for r in rows:
        t = r["grade"]["tier"]
        if r["tic"] not in best or TIERS.index(t) < TIERS.index(best[r["tic"]]):
            best[r["tic"]] = t
    bc = Counter(best.values())
    P(f"# 100 pc survey vetting v2: {len(rows)} long signals >= tau_long on {len(best)} stars "
      f"({len(errs)} measurement errors); pre-registered vetting load ~195")
    P("")
    P("| tier | " + " | ".join(TIERS) + " |")
    P("|---|" + "---|" * len(TIERS))
    P("| signals | " + " | ".join(str(tiers.get(t, 0)) for t in TIERS) + " |")
    P("| stars (best tier) | " + " | ".join(str(bc.get(t, 0)) for t in TIERS) + " |")
    js.update(n_signals=len(rows), n_stars=len(best), tiers=dict(tiers), star_tiers=dict(bc),
              n_errors=len(errs))
    nonk = [r for r in rows if r["grade"]["tier"] != "K"]
    P(f"non-K signals {len(nonk)}; pixel-checked {sum('pixel' in r['stats'] for r in rows)}")

    for name in ("all", "ntr", "P", "sec", "pile"):
        kf = keys[name]
        P(f"\n## By {name}: observed vs expectations (not C = A+/A/B)")
        P("| bin | far signals | K | A+ | A | B | C | not C observed | realised mix, 50 pc survival "
          "(all false) | noise only, far nulls | noise only, 50 pc method | real planets (model 12.2) "
          "| real planets (corrected 4.6) |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        res = {}
        for lab in labels[name]:
            sel = [r for r in rows if kf(r["stats"], r) == lab]
            c = Counter(r["grade"]["tier"] for r in sel)
            obs = sum(c.get(t, 0) for t in ("A+", "A", "B"))
            nk = sum(1 for r in sel if r["grade"]["tier"] != "K")
            s50 = [x for x in g50 if x["grade"]["tier"] != "K" and kf(x["stats"], x) == lab]
            k50 = sum(x["grade"]["tier"] != "C" for x in s50)
            sm = rng.beta(k50 + 0.5, len(s50) - k50 + 0.5, size) if s50 else np.zeros(size)
            e_mix = q(nk * sm)
            kn = sum(1 for r, g in nulls if g["tier"] != "C" and kf(r["stats"], r) == lab)
            sf = rng.beta(kn + 0.5, len(nulls) - kn + 0.5, size)
            e_far = q(N_SEARCHED * r_far * sf)
            kn50 = sum(1 for f in fa50 if f["g2"]["tier"] != "C"
                       and kf(f["stats"], {"n_sectors": f.get("n_sectors", 0)}) == lab) \
                if name != "sec" else None
            e_50 = q(N_SEARCHED * r_118 * rng.beta(kn50 + 0.5, len(fa50) - kn50 + 0.5, size)) \
                if kn50 is not None else None
            wp = float(sum(wi for wi, (r, g) in zip(w, injk) if kf(r["stats"], r) == lab))
            P(f"| {lab} | {len(sel)} | " + " | ".join(str(c.get(t, 0)) for t in TIERS)
              + f" | **{obs}** | {fmt(e_mix)} ({k50}/{len(s50)}) | {fmt(e_far)} ({kn}/{len(nulls)}) | "
              + (f"{fmt(e_50)} ({kn50}/{len(fa50)})" if e_50 else "n/a") + " | "
              + f"{PLANETS_PRE * wp:.1f} | {PLANETS_CORR * wp:.1f} |")
            res[lab] = {"n": len(sel), "tiers": dict(c), "notC": obs, "nonK": nk,
                        "mix": e_mix, "mix_k_n": [k50, len(s50)], "noise_far": e_far,
                        "noise_far_k": kn, "noise_50": e_50, "planet_share": wp}
        js[f"by_{name}"] = res
    P(f"\nPre-registered (gate 3): surviving false alarms F = {F_PRE[0]:.1f} [{F_PRE[1]:.0f}, "
      f"{F_PRE[2]:.0f}]; planets {PLANETS_PRE} [9.41, 15.12] (corrected ~{PLANETS_CORR} [~1.6, ~10]).")
    rnk = rng.beta(13 + 0.5, 462 - 13 + 0.5, size)
    js["F_realised_total"] = q(len(nonk) * rnk)
    P(f"Realised-count version of F (far non-K signals {len(nonk)} x s_50 = 13/462): "
      f"{fmt(js['F_realised_total'])}.")

    # property comparison
    def props(items, wts=None):
        if not items:
            return {}
        a = {"P": [], "ntr": [], "depth_ppt": [], "dur_ratio": [], "sectors": [], "rp": []}
        for r in items:
            s = r["stats"]
            a["P"].append(s["P"])
            a["ntr"].append(int(s["n_transits"]))
            a["depth_ppt"].append(1e3 * s["depth"])
            a["dur_ratio"].append((s.get("duration_density") or {}).get("ratio", np.nan))
            a["sectors"].append(r["n_sectors"])
            a["rp"].append(rp_re(s["depth"], r["radius"]))
        res = {}
        for k, v in a.items():
            v = np.array(v, float)
            ok = np.isfinite(v)
            if wts is None:
                res[k] = float(np.median(v[ok])) if ok.any() else np.nan
            else:
                o = np.argsort(v[ok])
                cw = np.cumsum(np.asarray(wts)[ok][o])
                res[k] = float(v[ok][o][np.searchsorted(cw, cw[-1] / 2)])
        res["n"] = len(items)
        res["frac_ge5_tr"] = float(np.average(np.array(a["ntr"]) >= 5, weights=wts))
        res["frac_P40"] = float(np.average(np.array(a["P"]) >= 40, weights=wts))
        res["frac_13sec"] = float(np.average(np.array(a["sectors"]) >= 13, weights=wts))
        return res

    surv = [r for r in rows if r["grade"]["tier"] in ("A+", "A", "B")]
    pops = {
        "far survivors (A+/A/B), this run": props(surv),
        "far kept injections (planet-like, weighted)": props([r for r, g in injk], w),
        "far nulls >= 7 not C (noise-like)": props([r for r, g in nulls if g["tier"] != "C"]),
        "50 pc survivors (13, all judged false)": props([x for x in g50 if x["grade"]["tier"] in ("A+", "A", "B")]),
    }
    P("\n## Properties: survivors vs the planet-like and the false-alarm-like populations (medians)")
    P("| population | n | P (d) | transits | share 5+ transits | share P >= 40 d | share on 13+ sector "
      "stars | depth (ppt) | duration / expected for density | real sectors | radius if real (R_e) |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, v in pops.items():
        if not v:
            P(f"| {k} | 0 | | | | | | | | | |")
            continue
        P(f"| {k} | {v['n']} | {v['P']:.1f} | {v['ntr']:.0f} | {v['frac_ge5_tr']:.2f} | "
          f"{v['frac_P40']:.2f} | {v['frac_13sec']:.2f} | {v['depth_ppt']:.2f} | {v['dur_ratio']:.2f} | "
          f"{v['sectors']:.0f} | {v['rp']:.2f} |")
    js["props"] = pops

    # measured false-alarm odds per (tier, transit bin), as the candidate paragraphs quote them
    odds_js = json.loads((HERE.parent / "vetting" / "analyze_val2.json").read_text())["odds"]
    P("\n## False alarms like each survivor, per (tier, transits), over the 15,452 searched stars")
    P("| tier | transits | survivors here | far nulls >= 7 in this cell (of 42), x r_far | 50 pc v2 "
      "validation cell (per 4,700 stars) x 15,452 / 4,700 | detected injections in this cell (50 pc "
      "validation) |")
    P("|---|---|---|---|---|---|")
    js["odds"] = {}
    for t in ("A+", "A", "B"):
        for b in ("2", "3", "4", "5+"):
            here = sum(1 for r in surv if r["grade"]["tier"] == t and ntr_bin(int(r["stats"]["n_transits"])) == b)
            if not here:
                continue
            kn = sum(1 for r, g in nulls if g["tier"] == t and ntr_bin(int(r["stats"]["n_transits"])) == b)
            e = q(N_SEARCHED * r_far * rng.beta(kn + 0.5, len(nulls) - kn + 0.5, size))
            o = odds_js.get(f"v2|{t if t != 'A+' else 'A'}|{b}", {})
            sc = N_SEARCHED / 4700
            P(f"| {t} | {b} | {here} | {fmt(e)} ({kn}) | {o.get('survey_fa_mean', float('nan')) * sc:.1f} "
              f"[{o.get('survey_fa_lo', float('nan')) * sc:.1f}, {o.get('survey_fa_hi', float('nan')) * sc:.1f}] "
              f"({o.get('fa_in_tier')}/{o.get('fa_total')}) | {o.get('inj_in_tier')} of {o.get('inj_bin')} |")
            js["odds"][f"{t}|{b}"] = {"far": e, "far_k": kn, "v50_scaled": [
                o.get("survey_fa_mean", np.nan) * sc, o.get("survey_fa_lo", np.nan) * sc,
                o.get("survey_fa_hi", np.nan) * sc]}
    P("(A+ uses the A cell: no null can reach A+, because the pixel test never runs on nulls.)")

    why = Counter()
    for r in rows:
        if r["grade"]["tier"] == "C":
            for k_, v in r["grade"]["status"].items():
                if v in ("soft", "fail"):
                    why[f"{k_} {v}"] += 1
            if int(r["stats"]["n_transits"]) == 2:
                why["duo strike"] += 1
    P("\n## Strikes among tier C signals")
    P(", ".join(f"{k} {v}" for k, v in why.most_common()))
    js["c_strikes"] = dict(why)

    P("\n## Every A+/A/B/K signal")
    P("| key | tier | P (d) | MES | n_tr | depth (ppt) | dur (h) | real sectors | R* | radius if real "
      "(R_e) | pixel | strikes / reasons |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|")
    keep = sorted([r for r in rows if r["grade"]["tier"] != "C"],
                  key=lambda r: (TIERS.index(r["grade"]["tier"]), -r["stats"]["mes"]))
    for r in keep:
        s = r["stats"]
        px = (s.get("pixel") or {}).get("status", "-")
        P(f"| {r['key']} | {r['grade']['tier']} | {s['P']:.4f} | {s['mes']:.2f} | {s['n_transits']} | "
          f"{1e3 * s['depth']:.2f} | {s['dur_h']:.1f} | {r['n_sectors']} | {r['radius']:.3f} | "
          f"{rp_re(s['depth'], r['radius']):.2f} | {px} | {'; '.join(r['grade']['reasons'])[:300]} |")
    (VET / "tables.txt").write_text("\n".join(out) + "\n")
    (VET / "tables.json").write_text(json.dumps(js, indent=1, default=float))
    with open(VET / "graded.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps({"key": r["key"], "tic": r["tic"], "sig_index": r["sig_index"],
                                 "n_sectors": r["n_sectors"], "mass": r["mass"], "radius": r["radius"],
                                 "grade": r["grade"], "stats": r["stats"]}, default=float) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
