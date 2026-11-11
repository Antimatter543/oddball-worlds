"""Threshold from nulls, real-control check, host recovery, injection table, cost model.

Usage: uv run python -I spikes/mdwarf_stack/analyze.py [threshold|hosts|inj|cost|all]
Writes outputs/planets_spike/summary.json and prints markdown tables.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CAT_DIR, OUT

KNOWN = {  # ExoFOP TOI id -> (host TIC, planet name); ephemerides read from toi.csv
    256.01: (92226327, "LHS 1140 b"), 256.02: (92226327, "LHS 1140 c"),
    700.01: (150428135, "TOI-700 c"), 700.02: (150428135, "TOI-700 d"),
    700.03: (150428135, "TOI-700 b"), 700.04: (150428135, "TOI-700 e"),
    6251.01: (52005579, "Gliese 12 b"),
}
TARGETS = ("LHS 1140 b", "TOI-700 d", "Gliese 12 b")
TESS_ORBIT = 13.7


def _rd(name):
    p = OUT / name
    if not p.exists():
        return []
    return [json.loads(x) for x in open(p) if x.strip()]


def rd(name):
    """hosts.jsonl / inj.jsonl also come from the combined hostinj.jsonl run (split by id)."""
    out = _rd(name)
    prefix = {"hosts.jsonl": "host-", "inj.jsonl": "inj"}.get(name)
    if prefix:
        out += [r for r in _rd("hostinj.jsonl") if r["id"].startswith(prefix)]
    if name == "inj.jsonl":  # GPU engine (validated) from the big-star end, CPU from the small end
        seen = {r["id"] for r in out}
        for r in _rd("inj_gpu.jsonl") + _rd("inj_cpu.jsonl"):
            if r["id"] not in seen and "error" not in r:
                out.append(r)
                seen.add(r["id"])
    return out


def stat(r):
    return max([s["mes"] for s in r.get("signals", [])], default=0.0)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def near_orbit(P, tol=0.03):
    for base in (TESS_ORBIT, 2 * TESS_ORBIT):
        for n in range(1, 5):
            for cand in (base * n, base / n):
                if abs(P / cand - 1) < tol:
                    return True
    return False


def threshold(summary):
    nulls = [r for r in rd("nulls.jsonl") if "error" not in r]
    errs = [r for r in rd("nulls.jsonl") if "error" in r]
    st = np.sort([stat(r) for r in nulls])[::-1]
    tau = float(st[1]) if len(st) >= 2 else float("nan")
    print(f"## Null realisations: {len(nulls)} (errors {len(errs)})")
    print(f"top 5 null stats: {np.round(st[:5], 2).tolist()}; tau (2nd highest) = {tau:.2f}; "
          f"max = {st[0]:.2f}")
    top = sorted(nulls, key=stat, reverse=True)[:10]
    print("| null id | n_sec | MES | P (d) | n_ep | near 13.7/27.4 multiple |")
    print("|---|---|---|---|---|---|")
    for r in top:
        s = max(r["signals"], key=lambda x: x["mes"]) if r["signals"] else None
        if s:
            print(f"| {r['id']} | {r['n_sectors']} | {s['mes']:.2f} | {s['P']:.3f} | "
                  f"{s['n_ep']} | {near_orbit(s['P'])} |")
    allP = [s["P"] for r in nulls for s in r["signals"]]
    frac_orb = float(np.mean([near_orbit(p) for p in allP])) if allP else float("nan")
    print(f"null signals above floor: {len(allP)}; share within 3% of a 13.7/27.4 d "
          f"multiple or fraction: {frac_orb:.2f}")
    real = [r for r in rd("controls_real.jsonl") if "error" not in r]
    rs = sorted(((stat(r), r) for r in real), key=lambda x: -x[0])
    n_above = sum(1 for v, _ in rs if v >= tau)
    print(f"\n## Real controls: {len(real)}; above tau: {n_above}")
    print("| control | n_sec | top MES | P (d) | n_ep | near orbit |")
    print("|---|---|---|---|---|---|")
    for v, r in rs[:8]:
        s = max(r["signals"], key=lambda x: x["mes"]) if r["signals"] else None
        if s:
            print(f"| {r['tic']} | {r['n_sectors']} | {v:.2f} | {s['P']:.3f} | {s['n_ep']} | "
                  f"{near_orbit(s['P'])} |")
    tau_cons = float(rs[0][0]) if rs else float("nan")
    flagged = n_above > 1
    summary.update(tau=tau, null_max=float(st[0]), n_null=len(nulls), real_above_tau=n_above,
                   tau_cons=tau_cons, tau_flagged=flagged,
                   null_top=[float(x) for x in st[:10]])
    print(f"threshold flagged as under-calibrated: {flagged}; tau_cons = {tau_cons:.2f}")
    # POST-HOC diagnostic (not pre-registered): nulls restricted to signals with P >= 10 d.
    # Biased low: a realisation's 3 signal slots may be used up by short-period rotation.
    sl = np.sort([max([s["mes"] for s in r["signals"] if s["P"] >= 10] or [0.0])
                  for r in nulls])[::-1]
    real_l = sorted(((max([s["mes"] for s in r["signals"] if s["P"] >= 10] or [0.0]), r["tic"])
                     for r in real), reverse=True)
    print(f"POST-HOC P>=10 d: top null stats {np.round(sl[:5], 2).tolist()}, tau_long (2nd) = "
          f"{sl[1]:.2f}; real controls top P>=10 d: "
          f"{[(round(v, 2), t) for v, t in real_l[:4]]}")
    summary.update(tau_long_posthoc=float(sl[1]), null_long_top=[float(x) for x in sl[:5]])
    return tau


def host_report(summary, tau):
    from fetch import load
    from inject import epochs_in_data, match

    toi = pd.read_csv(CAT_DIR / "toi.csv", comment="#").set_index("TOI")
    hosts = {r["tic"]: r for r in rd("hosts.jsonl") if "error" not in r}
    rows = []
    for toi_id, (tic, name) in KNOWN.items():
        if tic not in hosts:
            continue
        r = hosts[tic]
        k = toi.loc[toi_id]
        P, t0 = float(k["Period (days)"]), float(k["Epoch (BJD)"]) - 2457000.0
        dur = float(k["Duration (hours)"]) / 24
        lc, _ = load(tic)
        n_known = len(epochs_in_data(lc["idx"], P, t0, dur))
        hit = None
        for i, s in enumerate(r["signals"]):
            ok, n, fd, fk = match(s, P, t0, dur, lc["idx"])
            if ok:
                hit = (i, s, n, fd, fk)
                break
        row = {"planet": name, "P_true": P, "n_tr_in_data": n_known, "sectors": r["sectors"],
               "n_sectors": r["n_sectors"]}
        if hit:
            i, s, n, fd, fk = hit
            row.update(order=i + 1, P_det=s["P"], mes=s["mes"], above_tau=s["mes"] >= tau,
                       alias=n, frac_det=fd, frac_known=fk, n_ep=s["n_ep"], dur_h=s["dur_h"])
        rows.append(row)
    print(f"\n## Known planets (tau = {tau:.2f})")
    print("| planet | P_true | known transits in data | detected? | order | P_det | MES | "
          "≥tau | alias n | det/known epoch match |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for x in rows:
        if "P_det" in x:
            print(f"| {x['planet']} | {x['P_true']:.4f} | {x['n_tr_in_data']} | yes | "
                  f"{x['order']} | {x['P_det']:.4f} | {x['mes']:.2f} | {x['above_tau']} | "
                  f"{x['alias']} | {x['frac_det']:.2f}/{x['frac_known']:.2f} |")
        else:
            print(f"| {x['planet']} | {x['P_true']:.4f} | {x['n_tr_in_data']} | no | | | | | | |")
    for tic, r in hosts.items():
        print(f"host {tic} sectors {r['sectors']} signals: " + "; ".join(
            f"P={s['P']:.4f} MES={s['mes']:.2f} n_ep={s['n_ep']} D={s['dur_h']:.1f}h"
            for s in r["signals"]))
    n_rec = sum(1 for x in rows if x["planet"] in TARGETS and x.get("above_tau"))
    summary.update(hosts=rows, n_targets_recovered=n_rec)
    print(f"targets recovered above tau: {n_rec} of 3")


def bucket(n):
    return "1-5" if n <= 5 else "6-12" if n <= 12 else "13-25" if n <= 25 else "26+"


def inj_report(summary, tau, label="pre-registered tau"):
    res = [r for r in rd("inj.jsonl") if "error" not in r]
    errs = sum(1 for r in rd("inj.jsonl") if "error" in r)
    print(f"\n### Injection recovery at {label}")
    cells = {0: "1.0-1.5 Re, 15-30 d", 1: "1.0-1.5 Re, 30-50 d", 2: "1.5-2.0 Re, 15-30 d",
             3: "1.5-2.0 Re, 30-50 d"}
    rec = []
    for r in res:
        ok = any(m[0] and s["mes"] >= tau for m, s in zip(r["inj_match"], r["signals"]))
        rec.append({"cell": r["inj"]["cell"], "n_sec": r["n_sectors"],
                    "bucket": bucket(r["n_sectors"]), "rec": ok,
                    "ge2": r["inj"]["n_tr_data"] >= 2})
    df = pd.DataFrame(rec)
    print(f"\n## Injections: {len(df)} done (errors {errs}); tau = {tau:.2f}")
    if df.empty:
        return
    print("| cell | all stars | 6+ sectors | 6+ sectors, ≥2 transits in data | 1-5 | 6-12 | "
          "13-25 | 26+ |")
    print("|---|---|---|---|---|---|---|---|")
    out = {}
    for c, name in cells.items():
        d = df[df["cell"] == c]
        six = d[d["n_sec"] >= 6]

        def f(x):
            p, lo, hi = wilson(int(x["rec"].sum()), len(x))
            return f"{p:.2f} [{lo:.2f}, {hi:.2f}] (n={len(x)})"

        cols = [f(d), f(six), f(six[six["ge2"]])]
        cols += [f(d[d["bucket"] == b]) for b in ("1-5", "6-12", "13-25", "26+")]
        print(f"| {name} | " + " | ".join(cols) + " |")
        out[name] = {"all": wilson(int(d["rec"].sum()), len(d)),
                     "six": wilson(int(six["rec"].sum()), len(six)), "n_six": len(six)}
    # matched at all (any MES >= floor 6): where the misses go
    anym = [any(m[0] for m in r["inj_match"]) for r in res]
    print(f"matched by any signal (MES>=6 floor, no threshold): {np.mean(anym):.2f}; "
          f"with >=2 transits in data: {np.mean([r['inj']['n_tr_data'] >= 2 for r in res]):.2f}")
    summary["inj" if label.startswith("pre") else "inj_posthoc"] = out


def _fit(df, col):
    df = df.assign(per_it=(df[col] - df["pre"]).clip(lower=1e-3) / df["it"])
    x = np.log(df["bins"] * df["T"])
    A = np.vstack([np.ones(len(x)), x]).T
    coef, *_ = np.linalg.lstsq(A, np.log(df["per_it"]), rcond=None)
    return coef, float((np.log(df["per_it"]) - A @ coef).std()), df


def cost_report(summary):
    files = ("controls_real.jsonl", "nulls.jsonl", "hosts.jsonl", "inj.jsonl", "val_cpu.jsonl",
             "val_gpu.jsonl")
    runs = [r for f in files for r in (rd(f) if f.endswith("inj.jsonl") else _rd(f))
            if "error" not in r]
    df = pd.DataFrame([{"cpu": r["cpu_s"], "wall": r["wall_s"], "pre": r["cpu_pre"],
                        "it": r["n_iter"], "bins": r["n_bins"], "T": r["T_base"],
                        "nsec": r["n_sectors"], "kind": r["id"].split("-")[0],
                        "engine": r.get("engine", "cpu")} for r in runs])
    if df.empty:
        return
    dc, dg = df[df["engine"] == "cpu"], df[df["engine"] == "gpu"]
    real = dc[dc["kind"] == "real"]
    mean_it = float(real["it"].mean())
    bins_per_sec = float((df["bins"] / df["nsec"]).median())
    pre_per_sec = float((df["pre"] / df["nsec"]).median())
    models = {}
    for name, d, col in (("CPU", dc, "cpu"), ("GPU", dg, "wall")):
        if len(d) < 5:
            continue
        coef, sc, d2 = _fit(d, col)
        models[name] = coef
        print(f"\n## {name} cost ({len(d)} searches; {col} s): per-iteration = "
              f"exp({coef[0]:.2f}) * (bins*T_base)^{coef[1]:.3f}; log scatter {sc:.2f}")
        print(d2.groupby(pd.cut(d2["nsec"], [0, 5, 12, 25, 100]), observed=True)[
            [col, "per_it", "it"]].median().round(1).to_string())
    from select_controls import sectors_for

    s = pd.read_csv(CAT_DIR / "tic_mdwarfs_50pc.csv").drop_duplicates("ID")
    secs = sectors_for(s["ID"], s["ra"], s["dec"])
    rows = []
    for t in s["ID"]:
        ss = secs.get(int(t), [])
        if not ss:
            continue
        n = len(ss)
        T = (max(ss) - min(ss) + 1) * 27.4  # first-to-last baseline
        bins = n * bins_per_sec
        row = {"nsec": n, "T": T}
        for name, coef in models.items():
            row[name] = pre_per_sec * n + mean_it * math.exp(coef[0]) * (bins * T) ** coef[1]
        rows.append(row)
    sv = pd.DataFrame(rows)
    print(f"survey sample: {len(s)} TIC stars (Teff 2700-3400 K, d<50 pc, T<15), {len(sv)} with "
          f">=1 sector (tess-point, all pointings known to tess-point); median sectors "
          f"{sv['nsec'].median():.0f}, 90th pct {sv['nsec'].quantile(0.9):.0f}; mean signals "
          f"per star (real controls) {mean_it:.2f}; bins/sector {bins_per_sec:.0f}")
    rng = np.random.default_rng(1)
    out = {"mean_iter": mean_it}
    for name in models:
        v = sv[name].to_numpy()
        print(f"{name}-h per star: median {np.median(v) / 3600:.4f}, mean {v.mean() / 3600:.4f}, "
              f"max {v.max() / 3600:.3f}")
        for N in (3000, 8000):
            draws = [v[rng.integers(0, len(v), N)].sum() / 3600 for _ in range(200)]
            m, lo, hi = np.mean(draws), np.percentile(draws, 2.5), np.percentile(draws, 97.5)
            wall = m / (3 if name == "CPU" else 1) / 24
            print(f"  {N} stars: {m:.0f} {name}-h [{lo:.0f}, {hi:.0f}]; wall on "
                  f"{'3 CPU workers' if name == 'CPU' else '1 GPU'}: {wall:.1f} days")
            out[f"{name}_{N}"] = [float(m), float(lo), float(hi), float(wall)]
        out[f"{name}_per_star_mean_h"] = float(v.mean() / 3600)
    summary["cost"] = out


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    sp = OUT / "summary.json"
    summary = json.loads(sp.read_text()) if sp.exists() else {}
    tau = summary.get("tau", float("nan"))
    if what in ("threshold", "all"):
        tau = threshold(summary)
    if what in ("hosts", "all"):
        host_report(summary, tau)
    if what in ("inj", "all"):
        inj_report(summary, tau)
        if "tau_long_posthoc" in summary:
            inj_report(summary, summary["tau_long_posthoc"], "POST-HOC tau_long (not pre-reg)")
    if what in ("cost", "all"):
        cost_report(summary)
    sp.write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
