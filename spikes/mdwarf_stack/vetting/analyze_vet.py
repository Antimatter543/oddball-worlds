"""Grade vetting statistics with (frozen) thresholds and write the tables.

Usage: python -I vetting/analyze_vet.py dev|test|all   -> <OUT>/vetting/analyze_<stage>.txt + .json
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import CAT_DIR, OUT
from vet import ALL_TESTS, DEFAULT_THR, grade

V = OUT / "vetting"
TAU = json.loads((OUT / "gate2" / "frozen.json").read_text())["tau_long"]
N_SURVEY = 4700


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def fmt(k, n):
    p, lo, hi = wilson(k, n)
    return f"{p:.2f} [{lo:.2f}, {hi:.2f}] ({k}/{n})" if n else "- (0)"


def load(stage):
    p = V / f"vet_{stage}.jsonl"
    return [json.loads(x) for x in open(p) if x.strip()] if p.exists() else []


def thresholds():
    p = V / "frozen_vet.json"
    if p.exists():
        return {k: tuple(v) for k, v in json.loads(p.read_text())["thresholds"].items()}, "frozen"
    return DEFAULT_THR, "DEFAULT (not frozen)"


def label_holdout(rows):
    """Mark which hold-out signals are the recovered planets (gate-2 holdout_result, rec)."""
    hr = pd.read_csv(OUT / "gate2" / "holdout_result.csv")
    hr = hr[hr["rec"] == True]
    for r in rows:
        r["label"] = "other"
        if "stats" not in r:
            continue
        P = r["stats"]["P"]
        m = hr[(hr.tic == r["tic"]) & (np.abs(hr.P_det / P - 1) < 1e-3)]
        if len(m):
            r["label"] = "planet_rec"
            r["planet"] = str(m.iloc[0].planet)
        elif r["id"].startswith("g1host"):
            r["label"] = "g1host"
    return rows


def table_tests(groups, out):
    out.append("| test | " + " | ".join(f"{g} (pass / soft / fail of n)" for g in groups) + " |")
    out.append("|---|" + "---|" * len(groups))
    for t in ALL_TESTS:
        cells = []
        for g, rs in groups.items():
            st = [r["grade"]["status"].get(t) for r in rs]
            a = [s for s in st if s in ("pass", "soft", "fail")]
            n = len(a)
            if n == 0:
                cells.append("n/a")
                continue
            c = Counter(a)
            cells.append(f"{c['pass'] / n:.2f} / {c['soft'] / n:.2f} / {c['fail'] / n:.2f} ({n})")
        out.append(f"| {t} | " + " | ".join(cells) + " |")


def table_tiers(groups, out, key="grade"):
    out.append("| set | n | A | B | C | K | A+B |")
    out.append("|---|---|---|---|---|---|---|")
    for g, rs in groups.items():
        c = Counter(r[key]["tier"] for r in rs)
        n = len(rs)
        if n == 0:
            out.append(f"| {g} | 0 | | | | | |")
            continue
        out.append(f"| {g} | {n} | {c['A']} | {c['B']} | {c['C']} | {c['K']} | "
                   f"{fmt(c['A'] + c['B'], n)} |")


def kill_reasons(rs):
    c = Counter()
    for r in rs:
        for t in r["grade"]["hard"]:
            c[f"fail:{t}"] += 1
        for t, s in r["grade"]["status"].items():
            if s == "soft":
                c[f"soft:{t}"] += 1
        if r["stats"]["n_transits"] == 2:
            c["duo"] += 1
    return ", ".join(f"{k} {v}" for k, v in c.most_common(10))


def project(fa_rows, out, n_real_fa, n_real_stars, tag):
    """N_surv = N x r x s, Jeffreys posteriors, MC."""
    rng = np.random.default_rng(1)
    n = len(fa_rows)
    k = sum(r["grade"]["tier"] != "C" for r in fa_rows)
    r_ = rng.beta(n_real_fa + 0.5, n_real_stars - n_real_fa + 0.5, 200000)
    s_ = rng.beta(k + 0.5, n - k + 0.5, 200000) if n else np.full(200000, np.nan)
    N = N_SURVEY * r_ * s_
    pre = N_SURVEY * r_
    q = np.percentile(N, [2.5, 50, 97.5])
    qp = np.percentile(pre, [2.5, 50, 97.5])
    out.append(f"- {tag}: survival s = {k}/{n} not tier C; r = {n_real_fa}/{n_real_stars} real "
               f"stars with a long signal >= tau_long. Before vetting: {qp[1]:.0f} "
               f"[{qp[0]:.0f}, {qp[2]:.0f}] false alarms per {N_SURVEY}; after: **{q[1]:.1f} "
               f"[{q[0]:.1f}, {q[2]:.1f}]**.")
    return {"k": k, "n": n, "median": q[1], "lo": q[0], "hi": q[2]}


def main(stage):
    thr, which = thresholds()
    out = [f"# Vetting analysis ({stage}); thresholds: {which}; tau_long = {TAU:.3f}", ""]
    res = {}
    stages = ["dev"] if stage == "dev" else ["dev", "test", "holdout"]
    rows = {s: [r for r in load(s) if "stats" in r] for s in stages}
    errs = {s: sum("error" in r for r in load(s)) for s in stages}
    out.append(f"rows with a measurement error (not graded): {errs}")
    for s, rs in rows.items():
        for r in rs:
            r["grade"] = grade(r["stats"], thr)
            r["blind"] = grade(r["stats"], thr, blind=True)
        if s == "holdout":
            label_holdout(rs)
    dev = rows["dev"]
    d_true = [r for r in dev if r["pop"] == "dev_true"]
    d_fa = [r for r in dev if r["pop"].startswith("dev_fa")]
    groups = {"DEV true (MES>=7)": d_true,
              "DEV true, MES 7-12": [r for r in d_true if r["stats"]["mes"] < 12],
              "DEV FA (MES>=6)": d_fa,
              "DEV FA MES>=7": [r for r in d_fa if r["stats"]["mes"] >= 7],
              "DEV FA >= tau": [r for r in d_fa if r["stats"]["mes"] >= TAU]}
    out += ["## DEV: per-test outcome", ""]
    table_tests(groups, out)
    out += ["", "## DEV: tiers", ""]
    table_tiers(groups, out)
    out += ["", f"DEV true, what costs strikes: {kill_reasons(d_true)}",
            f"DEV FA, what rejects: {kill_reasons(d_fa)}", ""]
    if stage != "dev":
        te = rows["test"]
        t_true = [r for r in te if r["pop"] == "test_true"]
        det = [r for r in t_true if r["stats"]["mes"] >= TAU]
        head = [r for r in det if r["inj"]["cell"] == 0]
        hard = [r for r in head if r["stats"]["mes"] < 12]
        t_fa = [r for r in te if r["pop"] in ("test_fa_real", "test_fa_null")]
        fa7 = [r for r in t_fa if r["stats"]["mes"] >= 7.0]
        fat = [r for r in t_fa if r["stats"]["mes"] >= TAU]
        ho = rows["holdout"]
        hp = [r for r in ho if r["label"] == "planet_rec"]
        g1 = [r for r in ho if r["label"] == "g1host"]
        groups = {"TEST headline detected": head, "headline MES<12": hard,
                  "TEST all detected": det,
                  "TEST FA >=6": t_fa, "TEST FA >=7": fa7, "TEST FA >=tau": fat,
                  "hold-out planets": hp}
        out += ["## TEST and hold-out: per-test outcome (pass / soft / fail of applicable n)", ""]
        table_tests(groups, out)
        out += ["", "## TEST and hold-out: tiers", ""]
        table_tiers(groups | {"TEST FA real >=6": [r for r in t_fa if r["pop"] == "test_fa_real"],
                              "TEST FA null >=6": [r for r in t_fa if r["pop"] == "test_fa_null"],
                              "gate-1 host signals >= tau": g1}, out)
        out += ["", "hold-out planets, blind tier (catalogue test off):", ""]
        table_tiers({"hold-out planets (blind)": hp}, out, key="blind")
        out += ["", f"headline, what costs strikes: {kill_reasons(head)}",
                f"TEST FA >=6, what rejects: {kill_reasons(t_fa)}",
                f"hold-out planets (blind), strikes: "
                f"{kill_reasons([dict(r, grade=r['blind']) for r in hp])}", ""]
        # by MES
        out += ["### headline-cell detected injections by MES", "",
                "| MES | n | A+B | A |", "|---|---|---|---|"]
        for lo, hi in ((TAU, 10), (10, 12), (12, 15), (15, 20), (20, 1e9)):
            sub = [r for r in head if lo <= r["stats"]["mes"] < hi]
            k = sum(r["grade"]["tier"] in "AB" for r in sub)
            out.append(f"| {lo:.1f}-{hi:g} | {len(sub)} | {fmt(k, len(sub))} | "
                       f"{sum(r['grade']['tier'] == 'A' for r in sub)} |")
        # targets
        k1 = sum(r["grade"]["tier"] in ("A", "B") for r in head)
        k2 = sum(r["grade"]["tier"] == "C" for r in fat)
        k2b = sum(r["grade"]["tier"] == "C" for r in fa7)
        k3 = sum(r["grade"]["tier"] in ("A", "B", "K") for r in hp)
        k3b = sum(r["blind"]["tier"] in ("A", "B") for r in hp)
        out += ["", "## Targets", "",
                f"1. headline detected in A or B: {fmt(k1, len(head))} (goal >= 0.80)",
                f"2. FA >= tau_long in C: {fmt(k2, len(fat))}; at MES >= 7.0: "
                f"{fmt(k2b, len(fa7))}; at MES >= 6: "
                f"{fmt(sum(r['grade']['tier'] == 'C' for r in t_fa), len(t_fa))} (goal >= 0.90)",
                f"3. hold-out planets in A, B or K: {fmt(k3, len(hp))}; blind (A or B): "
                f"{fmt(k3b, len(hp))} (goal >= 0.90)", ""]
        res["targets"] = {"t1": [k1, len(head)], "t2": [k2, len(fat)], "t2_7": [k2b, len(fa7)],
                          "t3": [k3, len(hp)], "t3_blind": [k3b, len(hp)]}
        # 40-55 d, 2-4 transits
        def p_sub(rs):
            return [r for r in rs if r["stats"]["P"] >= 40 and 2 <= r["stats"]["n_transits"] <= 4]

        tl, fl = p_sub(det) + p_sub([r for r in d_true if r["stats"]["mes"] >= TAU]), p_sub(t_fa)
        out += ["## P >= 40 d with 2-4 transits (the false-alarm pile-up region)", "",
                "(true = TEST detected injections plus DEV true >= tau_long; FA = TEST FA >= 6)", ""]
        table_tests({"true, P>=40, 2-4 tr": tl, "FA, P>=40, 2-4 tr": fl}, out)
        out.append("")
        table_tiers({"true, P>=40, 2-4 tr": tl, "FA, P>=40, 2-4 tr": fl}, out)
        out.append("")
        # projection
        real_fa = [r for r in te if r["pop"] == "test_fa_real"]
        g2real = [json.loads(x) for x in open(OUT / "gate2" / "test_real.out.jsonl")]
        devreal = [json.loads(x) for x in open(OUT / "gate2" / "dev_paired.out.jsonl")
                   if '"devreal-' in x]
        devreal = [r for r in devreal if r["rot"] == "harm"]
        n_star_fa = sum(any(s["mes"] >= TAU for s in r["signals"]) for r in g2real + devreal)
        n_stars = len(g2real) + len(devreal)
        out += ["## Projection: surviving false alarms per 4,700 survey stars", ""]
        res["proj7"] = project(fa7, out, n_star_fa, n_stars, "survival measured on TEST FA >= 7.0")
        res["projtau"] = project(fat, out, n_star_fa, n_stars, "survival on TEST FA >= tau_long")
        res["proj6"] = project(t_fa, out, n_star_fa, n_stars, "survival on TEST FA >= 6")
        out.append(f"- real-control FA signals at MES >= 6 vetted: {len(real_fa)}")
        # list the FA >= 7 with their tier
        out += ["", "### TEST false alarms at MES >= 7.0", "",
                "| id | MES | P | n_tr | tier | reasons |", "|---|---|---|---|---|---|"]
        for r in sorted(fa7, key=lambda r: -r["stats"]["mes"]):
            out.append(f"| {r['key']} | {r['stats']['mes']:.2f} | {r['stats']['P']:.3f} | "
                       f"{r['stats']['n_transits']} | {r['grade']['tier']} | "
                       f"{'; '.join(r['grade']['reasons'])[:160]} |")
        out += ["", "### hold-out planets", "", "| planet | MES | P | n_tr | tier | blind | reasons (blind) |",
                "|---|---|---|---|---|---|---|"]
        for r in sorted(hp, key=lambda r: r["stats"]["mes"]):
            out.append(f"| {r['planet']} | {r['stats']['mes']:.1f} | {r['stats']['P']:.3f} | "
                       f"{r['stats']['n_transits']} | {r['grade']['tier']} | {r['blind']['tier']} | "
                       f"{'; '.join(r['blind']['reasons'])[:160]} |")
    txt = "\n".join(out)
    (V / f"analyze_{stage}.txt").write_text(txt + "\n")
    (V / f"analyze_{stage}.json").write_text(json.dumps(res, indent=1, default=float))
    print(txt)


if __name__ == "__main__":
    _ = CAT_DIR
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
