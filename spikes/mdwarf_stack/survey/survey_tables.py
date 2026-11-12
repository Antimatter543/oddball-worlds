"""Survey vetting tables: tiers of every long signal >= tau_long next to the noise expectation.

Grades v2/vet2_survey*.jsonl with frozen_vet2.json via cards2.graded2 (pixel results merged; tau_det
= tau_long), and computes the noise expectation per bin exactly as the v2 projection does
(analyze_vet2: N = N_stars x r x s, r = 1/118 real controls with a long signal >= tau_long,
Jeffreys; s = fresh-null false alarms at MES >= 7 graded with tau_det = 7.0 that land in the bin;
Monte Carlo 2.5-97.5%), with N_stars = the 4,350 stars searched. A second "fat-tail" scenario
takes r from the fresh nulls' own rate above tau_long (16 of 600 realisations, the post-hoc
upper-side scenario). Both describe noise only: eclipsing binaries, known planets and stellar
variability are not in the null material, so they add to the observed counts.

Usage: python -I survey/survey_tables.py  -> OUT/survey/vet/tables.txt, tables.json, graded.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "gate2"))
sys.path.insert(0, str(HERE.parent / "vetting"))

from common import OUT

N_STARS = 4350
TIERS = ("K", "A+", "A", "B", "C")


def ntr_bin(n):
    return "2" if n == 2 else "3" if n == 3 else "4" if n == 4 else "5+"


def p_bin(P):
    return "10-30" if P < 30 else "30-40" if P < 40 else "40+"


def null_fa7():
    """Fresh-set null false alarms at MES >= 7, v2-graded with tau_det = 7.0 (as analyze_vet2)."""
    from analyze_vet2 import complete_stars, load_rows
    from vet2 import grade2, load_frozen2

    thr = load_frozen2()
    stars = set(complete_stars()[0])
    out = []
    for r in load_rows():
        if "stats" not in r or r["stats"].get("error") or r["tic"] not in stars or r.get("inj"):
            continue
        if r["stats"]["mes"] >= 7.0:
            r["g2"] = grade2(r["stats"], thr, 7.0)
            out.append(r)
    return out


def main() -> None:
    from cards2 import graded2

    rows = graded2("survey")
    seen, uniq = set(), []
    for r in rows:
        if r["key"] not in seen:
            seen.add(r["key"])
            uniq.append(r)
    rows = uniq
    errs = []
    for p in sorted((OUT / "vetting" / "v2").glob("vet2_survey*.jsonl")):
        errs += [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip() and '"error"' in x]
    errs = [e for e in errs if "stats" not in e]
    fa = null_fa7()
    n_fa = len(fa)
    rng = np.random.default_rng(20261215)
    size = 200_000
    r_base = rng.beta(1 + 0.5, 118 - 1 + 0.5, size)
    r_fat = rng.beta(16 + 0.5, 600 - 16 + 0.5, size)

    def expect(k, r_mc):
        s = rng.beta(k + 0.5, n_fa - k + 0.5, size)
        x = N_STARS * r_mc * s
        return float(x.mean()), float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))

    def fmt_e(e):
        return f"{e[0]:.1f} [{e[1]:.1f}, {e[2]:.1f}]"

    out, js = [], {"n_signals": len(rows), "n_measure_errors": len(errs), "n_fa7": n_fa}
    tiers = Counter(r["grade"]["tier"] for r in rows)
    out.append(f"# Survey vetting v2: {len(rows)} long signals >= tau_long on "
               f"{len({r['tic'] for r in rows})} stars ({len(errs)} measurement errors)")
    out.append("")
    out.append("## Tiers (signals)")
    out.append("| tier | " + " | ".join(TIERS) + " |")
    out.append("|---|" + "---|" * len(TIERS))
    out.append("| signals | " + " | ".join(str(tiers.get(t, 0)) for t in TIERS) + " |")
    best = {}
    for r in rows:
        t = r["grade"]["tier"]
        if r["tic"] not in best or TIERS.index(t) < TIERS.index(best[r["tic"]]):
            best[r["tic"]] = t
    bc = Counter(best.values())
    out.append("| stars (best tier) | " + " | ".join(str(bc.get(t, 0)) for t in TIERS) + " |")
    js["tiers"] = dict(tiers)
    js["star_tiers"] = dict(bc)

    def table(name, keyf, labels):
        out.append(f"\n## By {name}: observed tiers vs noise expectation (not C = A+/A/B)")
        out.append("| bin | n | K | A+ | A | B | C | not C (A+/A/B) observed | noise expectation, "
                   "r = 1/118 | fat-tail scenario |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        res = {}
        for lab in labels:
            sel = [r for r in rows if keyf(r["stats"]) == lab]
            c = Counter(r["grade"]["tier"] for r in sel)
            obs = sum(c.get(t, 0) for t in ("A+", "A", "B"))
            k = sum(1 for f in fa if keyf(f["stats"]) == lab and f["g2"]["tier"] != "C")
            e1, e2 = expect(k, r_base), expect(k, r_fat)
            out.append(f"| {lab} | {len(sel)} | " + " | ".join(str(c.get(t, 0)) for t in TIERS)
                       + f" | {obs} | {fmt_e(e1)} | {fmt_e(e2)} |")
            res[lab] = {"n": len(sel), "tiers": dict(c), "notC": obs, "fa_k": k,
                        "exp": e1, "exp_fat": e2}
        js[f"by_{name}"] = res

    table("all", lambda s: "all", ["all"])
    table("transit count", lambda s: ntr_bin(int(s["n_transits"])), ["2", "3", "4", "5+"])
    table("period", lambda s: p_bin(s["P"]), ["10-30", "30-40", "40+"])
    table("tier-scope 40 d+ with 2-4 transits",
          lambda s: "40+ & 2-4" if s["P"] >= 40 and 2 <= s["n_transits"] <= 4 else "other",
          ["40+ & 2-4", "other"])
    # per tier expectation
    out.append("\n## Per tier (signals) vs noise expectation")
    out.append("| tier | observed | noise expectation r = 1/118 | fat-tail |")
    out.append("|---|---|---|---|")
    for t in ("A+", "A", "B"):
        k = sum(1 for f in fa if f["g2"]["tier"] == t)
        out.append(f"| {t} | {tiers.get(t, 0)} | {fmt_e(expect(k, r_base))} | "
                   f"{fmt_e(expect(k, r_fat))} |")
    e_pre = (float(N_STARS * r_base.mean()), *[float(x) for x in
                                              np.quantile(N_STARS * r_base, [0.025, 0.975])])
    e_pre_f = (float(N_STARS * r_fat.mean()), *[float(x) for x in
                                               np.quantile(N_STARS * r_fat, [0.025, 0.975])])
    out.append(f"\nBefore vetting, noise expectation (stars with a signal >= tau_long): "
               f"{fmt_e(e_pre)} (r = 1/118); fat-tail {fmt_e(e_pre_f)}. Observed: "
               f"{len(best)} stars, {len(rows)} signals.")
    js["pre"] = {"exp": e_pre, "exp_fat": e_pre_f, "stars": len(best), "signals": len(rows)}
    # MES and slot and sector-count splits (no expectation: descriptive)
    for name, keyf, labs in (
            ("MES", lambda r: "8-10" if r["stats"]["mes"] < 10 else "10-15" if r["stats"]["mes"]
             < 15 else "15-30" if r["stats"]["mes"] < 30 else "30+", ["8-10", "10-15", "15-30", "30+"]),
            ("slot", lambda r: "slot 1" if r["sig_index"] == 0 else "slots 2-6", ["slot 1", "slots 2-6"]),
            ("real sectors", lambda r: "1-3" if r["n_sectors"] <= 3 else "4-5" if r["n_sectors"] <= 5
             else "6-12" if r["n_sectors"] <= 12 else "13+", ["1-3", "4-5", "6-12", "13+"])):
        out.append(f"\n## By {name} (descriptive)")
        out.append("| bin | n | " + " | ".join(TIERS) + " |")
        out.append("|---|---|" + "---|" * len(TIERS))
        for lab in labs:
            sel = [r for r in rows if keyf(r) == lab]
            c = Counter(r["grade"]["tier"] for r in sel)
            out.append(f"| {lab} | {len(sel)} | " + " | ".join(str(c.get(t, 0)) for t in TIERS) + " |")
    # what put signals in C
    why = Counter()
    for r in rows:
        if r["grade"]["tier"] == "C":
            for k_, v in r["grade"]["status"].items():
                if v in ("soft", "fail"):
                    why[f"{k_} {v}"] += 1
            if int(r["stats"]["n_transits"]) == 2:
                why["duo strike"] += 1
    out.append("\n## Strikes among tier C signals")
    out.append(", ".join(f"{k} {v}" for k, v in why.most_common()))
    # listing of non-C
    out.append("\n## Every A+/A/B/K signal")
    out.append("| key | tier | P (d) | MES | n_tr | depth (ppt) | dur (h) | sectors | strikes / reasons |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    keep = [r for r in rows if r["grade"]["tier"] != "C"]
    keep.sort(key=lambda r: (TIERS.index(r["grade"]["tier"]), -r["stats"]["mes"]))
    for r in keep:
        s = r["stats"]
        out.append(f"| {r['key']} | {r['grade']['tier']} | {s['P']:.4f} | {s['mes']:.2f} | "
                   f"{s['n_transits']} | {1e3 * s['depth']:.2f} | {s['dur_h']:.1f} | "
                   f"{r['n_sectors']} | {'; '.join(r['grade']['reasons'])[:300]} |")
    (OUT / "survey" / "vet" / "tables.txt").write_text("\n".join(out) + "\n")
    (OUT / "survey" / "vet" / "tables.json").write_text(json.dumps(js, indent=1, default=float))
    with open(OUT / "survey" / "vet" / "graded.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps({"key": r["key"], "tic": r["tic"], "sig_index": r["sig_index"],
                                 "n_sectors": r["n_sectors"], "mass": r["mass"],
                                 "radius": r["radius"], "grade": r["grade"],
                                 "stats": r["stats"]}, default=float) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
