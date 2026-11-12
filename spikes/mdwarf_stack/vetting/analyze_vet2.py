"""Vetting v2: paired v1-vs-v2 comparison on the FRESH validation set (pre-registered rule).

Grades every row of v2/vet2_val*.jsonl with v1 (frozen vet.grade, unchanged) and v2 (vet2.grade2,
frozen_vet2.json), restricted to the complete prefix of the draw order (stars whose search jobs all
finished), and writes v2/analyze_val.txt and v2/analyze_val.json (the json holds the false-alarm
odds by tier and transit count that the candidate summaries quote).

Usage: python -I vetting/analyze_vet2.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from analyze_vet import fmt
from common import OUT
from freeze_vet import load_frozen
from vet import grade
from vet2 import TESTS2, grade2, load_frozen2

V2 = OUT / "vetting" / "v2"
TAU = json.loads((OUT / "gate2" / "frozen.json").read_text())["tau_long"]
N_SURVEY = 4700
R_K, R_N = 1, 118  # real controls with a long signal >= tau_long (gate 2 DEV + TEST)
SEED_BOOT = 20261214
N_BOOT = 10_000
KEPT = ("A+", "A", "B", "K")


def ntr_bin(n):
    return "2" if n == 2 else "3" if n == 3 else "4" if n == 4 else "5+"


def p_bin(P):
    return "10-30" if P < 30 else "30-40" if P < 40 else "40+"


def complete_stars():
    """Stars (draw order) whose search jobs all finished; the set is the longest such prefix."""
    jobs = [json.loads(x) for x in open(V2 / "jobs" / "val2.jsonl") if x.strip()]
    done = {json.loads(x)["id"] for x in open(V2 / "val2.out.jsonl") if x.strip()}
    order = {}
    for j in jobs:
        order.setdefault(j["order"], []).append(j["id"])
    stars = []
    for o in sorted(order):
        if not all(i in done for i in order[o]):
            break
        stars.append(int(order[o][0].rsplit("-", 1)[1]))
    return stars, len(order)


def load_rows():
    rows = []
    for p in sorted(V2.glob("vet2_val*.jsonl")):
        rows += [json.loads(x) for x in open(p) if x.strip()]
    seen, out = set(), []
    for r in rows:
        if r["key"] in seen:
            continue
        seen.add(r["key"])
        out.append(r)
    return out


def boot_diff(stars, a, b, rng):
    """Cluster bootstrap over stars of mean(a) - mean(b) (a, b paired 0/1 arrays)."""
    stars = np.asarray(stars)
    u, inv = np.unique(stars, return_inverse=True)
    sa = np.bincount(inv, a, len(u))
    sb = np.bincount(inv, b, len(u))
    sn = np.bincount(inv, minlength=len(u)).astype(float)
    idx = rng.integers(0, len(u), (N_BOOT, len(u)))
    d = (sa[idx].sum(1) - sb[idx].sum(1)) / sn[idx].sum(1)
    return float((a - b).mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def jeffreys_mc(k, n, rng, size=200_000):
    return rng.beta(k + 0.5, n - k + 0.5, size)


def main():
    thr1 = load_frozen()
    thr2 = load_frozen2()
    stars, n_order = complete_stars()
    sset = set(stars)
    rows = load_rows()
    err = [r for r in rows if "stats" not in r or r["stats"].get("error")]
    rows = [r for r in rows if "stats" in r and not r["stats"].get("error") and r["tic"] in sset]
    for r in rows:
        r["is_inj"] = bool(r.get("inj"))
    inj = [r for r in rows if r["is_inj"]]
    nul = [r for r in rows if not r["is_inj"]]
    for r in inj:
        r["g1"] = grade(r["stats"], thr1)
        r["g2"] = grade2(r["stats"], thr2, TAU)
        i = r["inj"]
        r["head"] = 1.0 <= i["rp"] < 1.5 and 15.0 <= i["P"] < 30.0
    out, js = [], {}
    out.append(f"# Vetting v2 vs v1 on the fresh validation set "
               f"({len(stars)} of {n_order} stars complete in draw order)")
    out.append(f"rows used: {len(inj)} detected injections (matched, MES >= {TAU}), {len(nul)} null "
               f"top signals with MES >= 6; {len(err)} rows with a measurement error (excluded)")
    rng = np.random.default_rng(SEED_BOOT)

    def tiers_line(name, rs, gk):
        c = Counter(r[gk]["tier"] for r in rs)
        return f"| {name} | {len(rs)} | " + " | ".join(str(c.get(t, 0)) for t in
                                                      ("A+", "A", "B", "C", "K")) + " |"

    # ---------------- injections
    out.append("\n## Detected injections (kept = A+/A/B/K)")
    out.append("| set | version | n | A+ | A | B | C | K | kept (Wilson 95%) |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    ret = {}
    for name, sel in (("headline 1.0-1.5 R_e, 15-30 d", [r for r in inj if r["head"]]),
                      ("all detected", inj),
                      ("40 d+ with 2-4 transits", [r for r in inj if r["stats"]["P"] >= 40
                                                   and 2 <= r["stats"]["n_transits"] <= 4])):
        for v in ("g1", "g2"):
            c = Counter(r[v]["tier"] for r in sel)
            k = sum(c.get(t, 0) for t in KEPT)
            out.append(f"| {name} | {'v1' if v == 'g1' else 'v2'} | {len(sel)} | "
                       + " | ".join(str(c.get(t, 0)) for t in ("A+", "A", "B", "C", "K"))
                       + f" | {fmt(k, len(sel))} |")
            ret[(name, v)] = (k, len(sel))
        if sel:
            a = np.array([r["g2"]["tier"] in KEPT for r in sel], float)
            b = np.array([r["g1"]["tier"] in KEPT for r in sel], float)
            d = boot_diff([r["tic"] for r in sel], a, b, rng)
            out.append(f"| {name} | v2 - v1 | | | | | | | {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] |")
            js[f"inj_diff::{name}"] = d
    # ---------------- nulls
    pops = {}
    for lab, mmin, tdet in (("MES >= 7", 7.0, 7.0), ("MES >= tau_long", TAU, TAU),
                            ("MES >= 6", 6.0, 6.0)):
        sel = []
        for r in nul:
            if r["stats"]["mes"] < mmin:
                continue
            r2 = dict(r)
            r2["g1"] = grade(r["stats"], thr1)
            r2["g2"] = grade2(r["stats"], thr2, tdet)
            sel.append(r2)
        pops[lab] = sel
    out.append("\n## Null false alarms (tier C = rejected). v2 low_margin uses the selecting "
               "threshold (7, tau_long, 6)")
    out.append("| set | version | n | A+ | A | B | C | K | in C (Wilson 95%) |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for lab, sel in pops.items():
        for v in ("g1", "g2"):
            c = Counter(r[v]["tier"] for r in sel)
            out.append(f"| {lab} | {'v1' if v == 'g1' else 'v2'} | {len(sel)} | "
                       + " | ".join(str(c.get(t, 0)) for t in ("A+", "A", "B", "C", "K"))
                       + f" | {fmt(c.get('C', 0), len(sel))} |")
        if sel:
            a = np.array([r["g2"]["tier"] == "C" for r in sel], float)
            b = np.array([r["g1"]["tier"] == "C" for r in sel], float)
            d = boot_diff([r["tic"] for r in sel], a, b, rng)
            disc = (int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum()))
            out.append(f"| {lab} | v2 - v1 | | | | | | | {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] "
                       f"(only v2 C: {disc[0]}, only v1 C: {disc[1]}) |")
            js[f"null_diff::{lab}"] = d + disc
    fa7 = pops["MES >= 7"]
    # ---------------- by transit count and period
    out.append("\n## By transit count and period range")
    out.append("| bin | nulls >= 7: n | v1 C | v2 C | detected injections: n | v1 kept | v2 kept |")
    out.append("|---|---|---|---|---|---|---|")
    for kind, fn in (("n_tr", lambda r: ntr_bin(r["stats"]["n_transits"])),
                     ("P", lambda r: p_bin(r["stats"]["P"]))):
        bins = ("2", "3", "4", "5+") if kind == "n_tr" else ("10-30", "30-40", "40+")
        for b in bins:
            f = [r for r in fa7 if fn(r) == b]
            t = [r for r in inj if fn(r) == b]
            out.append(f"| {kind} {b} | {len(f)} | {sum(r['g1']['tier'] == 'C' for r in f)} | "
                       f"{sum(r['g2']['tier'] == 'C' for r in f)} | {len(t)} | "
                       f"{sum(r['g1']['tier'] in KEPT for r in t)} | "
                       f"{sum(r['g2']['tier'] in KEPT for r in t)} |")
    # ---------------- per-test power
    out.append("\n## Per-test outcome, v2 grading (soft-or-worse rate; n applicable)")
    out.append("| test | headline injections | all detected injections | nulls >= 7 | "
               "nulls >= 7, inv only |")
    out.append("|---|---|---|---|---|")
    power = {}
    for t in TESTS2:
        cells = []
        for sel in ([r for r in inj if r["head"]], inj, fa7, [r for r in fa7 if r["mode"] == "inv"]):
            st = [r["g2"]["status"].get(t) for r in sel]
            a = [s for s in st if s in ("pass", "soft", "fail")]
            bad = sum(s in ("soft", "fail") for s in a)
            cells.append(f"{bad / len(a):.2f} ({len(a)})" if a else "- (0)")
        power[t] = cells
        out.append(f"| {t} | " + " | ".join(cells) + " |")
    out.append("v1 per-test outcome on the same rows (v1 grading):")
    for t in ("per_transit", "sector_consistency", "uniqueness", "contamination"):
        cells = []
        for sel in ([r for r in inj if r["head"]], inj, fa7):
            st = [r["g1"]["status"].get(t) for r in sel]
            a = [s for s in st if s in ("pass", "soft", "fail")]
            bad = sum(s in ("soft", "fail") for s in a)
            cells.append(f"{bad / len(a):.2f} ({len(a)})" if a else "- (0)")
        out.append(f"- v1 {t}: headline {cells[0]}, all inj {cells[1]}, nulls >= 7 {cells[2]}")
    # what moved nulls to C under v2 only
    why = Counter()
    for r in fa7:
        if r["g2"]["tier"] == "C" and r["g1"]["tier"] != "C":
            for t, s in r["g2"]["status"].items():
                if s in ("soft", "fail"):
                    why[f"{t} {s}"] += 1
            if r["stats"]["n_transits"] == 2:
                why["duo strike"] += 1
    out.append(f"\nStrikes on the nulls >= 7 that only v2 sends to C: {dict(why.most_common())}")
    lost = Counter()
    for r in inj:
        if r["g2"]["tier"] not in KEPT and r["g1"]["tier"] in KEPT:
            for t, s in r["g2"]["status"].items():
                if s in ("soft", "fail"):
                    lost[f"{t} {s}"] += 1
    out.append(f"Strikes on injections that only v2 rejects: {dict(lost.most_common())}")
    # ---------------- decision
    d7 = js.get("null_diff::MES >= 7", (float("nan"),) * 5)
    k2, n2 = ret[("headline 1.0-1.5 R_e, 15-30 d", "g2")]
    k1, n1 = ret[("headline 1.0-1.5 R_e, 15-30 d", "g1")]
    r2, r1 = k2 / max(n2, 1), k1 / max(n1, 1)
    c1 = d7[0] > 0
    c2 = r2 >= 0.85
    c3 = (r1 - r2) <= 0.05
    adopt = c1 and c2 and c3
    out.append("\n## Pre-registered decision")
    out.append(f"(i) nulls >= 7 to C, v2 - v1 = {d7[0]:+.3f} [{d7[1]:+.3f}, {d7[2]:+.3f}]: "
               f"{'yes' if c1 else 'no'}" + ("; the interval spans zero, so the fresh set cannot "
                                            "tell them apart" if d7[1] <= 0 <= d7[2] else ""))
    out.append(f"(ii) v2 headline retention {fmt(k2, n2)} >= 0.85: {'yes' if c2 else 'no'}")
    out.append(f"(iii) v1 - v2 headline = {r1 - r2:+.3f} <= 0.05: {'yes' if c3 else 'no'}")
    win = "v2" if adopt else "v1"
    out.append(f"**Decision: {'ADOPT v2' if adopt else 'v1 STAYS'}** for the survey.")
    # ---------------- projection for the winner
    gk = "g2" if adopt else "g1"
    n = len(fa7)
    mc_r = jeffreys_mc(R_K, R_N, rng)
    proj = {}

    def proj_line(label, k):
        s = jeffreys_mc(k, n, rng)
        x = N_SURVEY * mc_r * s
        q = np.quantile(x, [0.025, 0.5, 0.975])
        mean = N_SURVEY * (R_K + 0.5) / (R_N + 1) * (k + 0.5) / (n + 1)
        proj[label] = {"k": k, "n": n, "mean": float(mean), "lo": float(q[0]), "hi": float(q[2])}
        out.append(f"| {label} | {k}/{n} | {mean:.1f} [{q[0]:.1f}, {q[2]:.1f}] |")

    out.append(f"\n## Projection with the winner ({win}): false alarms surviving per {N_SURVEY} "
               f"stars, N = {N_SURVEY} x r x s (r = {R_K}/{R_N}, Jeffreys; MC 2.5-97.5%)")
    out.append("| part (fresh nulls >= 7) | k / n | expected [95%] |")
    out.append("|---|---|---|")
    proj_line("before vetting (all)", n)
    proj_line("not C (any tier)", sum(r[gk]["tier"] != "C" for r in fa7))
    for t in ("A+", "A", "B"):
        proj_line(f"tier {t}", sum(r[gk]["tier"] == t for r in fa7))
    for b in ("2", "3", "4", "5+"):
        proj_line(f"not C, {b} transits",
                  sum(r[gk]["tier"] != "C" and ntr_bin(r["stats"]["n_transits"]) == b for r in fa7))
    proj_line("not C, 40 d+ with 2-4 transits",
              sum(r[gk]["tier"] != "C" and r["stats"]["P"] >= 40 and
                  2 <= r["stats"]["n_transits"] <= 4 for r in fa7))
    # ---------------- odds table for the summaries (both versions, by tier x transit bin)
    odds = {}
    for v, g in (("v1", "g1"), ("v2", "g2")):
        for t in ("A+", "A", "B"):
            for b in ("2", "3", "4", "5+"):
                kf = sum(r[g]["tier"] == t and ntr_bin(r["stats"]["n_transits"]) == b for r in fa7)
                nf = sum(ntr_bin(r["stats"]["n_transits"]) == b for r in fa7)
                ki = sum(r[g]["tier"] == t and ntr_bin(r["stats"]["n_transits"]) == b for r in inj)
                ni = sum(ntr_bin(r["stats"]["n_transits"]) == b for r in inj)
                s = jeffreys_mc(kf, n, rng)
                x = N_SURVEY * mc_r * s
                q = np.quantile(x, [0.025, 0.975])
                odds[f"{v}|{t}|{b}"] = {"fa_in_tier": kf, "fa_bin": nf, "fa_total": n,
                                        "inj_in_tier": ki, "inj_bin": ni,
                                        "survey_fa_mean": N_SURVEY * (R_K + 0.5) / (R_N + 1)
                                        * (kf + 0.5) / (n + 1),
                                        "survey_fa_lo": float(q[0]), "survey_fa_hi": float(q[1])}
    js.update({"decision": win, "adopt_v2": adopt, "n_stars": len(stars), "n_order": n_order,
               "projection": proj, "odds": odds, "power": power,
               "headline": {"v1": [k1, n1], "v2": [k2, n2]},
               "tiers": {lab: {v: dict(Counter(r[v]["tier"] for r in sel)) for v in ("g1", "g2")}
                         for lab, sel in pops.items()}})
    txt = "\n".join(out) + "\n"
    (V2 / "analyze_val.txt").write_text(txt)
    (V2 / "analyze_val.json").write_text(json.dumps(js, indent=1, default=float))
    print(txt)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
