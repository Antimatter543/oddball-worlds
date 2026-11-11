"""Gate 2 analysis.

  dev      DEV rotation-arm comparison, fast/slow census, prediction efficiency
  tau      tau_long / tau_short from DEV nulls, DEV real controls, peak-period check
  test     criterion (a) on TEST real controls; injections (b) and (d)
  hosts    criterion (c)
  cost     criterion (e)
Writes outputs/planets_spike/gate2/summary2.json. Frozen choices are read from gate2/frozen.json.

Usage: python -I gate2/analyze2.py dev|tau|test|hosts|cost|all
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from analyze import near_orbit, wilson
from common import CAT_DIR, OUT

G2 = OUT / "gate2"
PROVISIONAL = 7.5


def rd(name):
    p = G2 / f"{name}.out.jsonl"
    if not p.exists():
        return []
    rows = {}
    for x in open(p):
        if x.strip():
            r = json.loads(x)
            rows[r["id"]] = r
    return list(rows.values())


def ok(rows):
    return [r for r in rows if "error" not in r]


def top(r):
    return max([s["mes"] for s in r.get("signals", [])], default=0.0)


def frozen():
    p = G2 / "frozen.json"
    return json.loads(p.read_text()) if p.exists() else {}


def is_fast(rinfo, fz):
    if not rinfo or not rinfo.get("applied"):
        return False
    frac = rinfo["n_sec_applied"] / max(rinfo["n_sec"], 1)
    return frac >= fz["fast_min_frac"] and (rinfo.get("amp") or 0) >= fz["fast_min_amp"]


def rec_top(r, tau):
    """Top signal (slot 1) matches the injection with MES >= tau."""
    if not r.get("signals"):
        return False
    return bool(r["inj_match"][0][0]) and r["signals"][0]["mes"] >= tau


def fmt(k, n):
    p, lo, hi = wilson(k, n)
    return f"{p:.2f} [{lo:.2f}, {hi:.2f}] ({k}/{n})"


def bucket(n):
    return "2-5" if n <= 5 else "6-12" if n <= 12 else "13-25" if n <= 25 else "26+"


def dev(S):
    fz = frozen()
    rows = ok(rd("dev_paired"))
    real = {(r["tic"], r["rot"]): r for r in rows if r["id"].startswith("devreal")}
    print("## DEV real controls, long search (top MES; rotation info from harm arm)")
    print("| tic | n_sec | none: top MES (P) | harm: top MES (P) | harm applied sectors | P_rot | amp | max LS power |")
    print("|---|---|---|---|---|---|---|---|")
    census = {}
    for (t, arm), r in sorted(real.items()):
        if arm != "harm":
            continue
        n = real.get((t, "none"))
        ri = r["rot_info"]
        census[t] = ri

        def tp(x):
            if x is None or not x["signals"]:
                return "-"
            s = max(x["signals"], key=lambda s: s["mes"])
            return f"{s['mes']:.2f} ({s['P']:.2f})"

        print(f"| {t} | {r['n_sectors']} | {tp(n)} | {tp(r)} | {ri['n_sec_applied']}/{ri['n_sec']} | "
              f"{ri['prot'] if ri['prot'] is None else round(ri['prot'], 3)} | "
              f"{(ri['amp'] or 0) * 1e3:.2f} ppt | {ri['max_power']:.3f} |")
    inj = [r for r in rows if r["id"].startswith("devinj")]
    pairs = {}
    for r in inj:
        key = r["id"].rsplit("-", 1)[0]
        pairs.setdefault(key, {})[r["rot"]] = r
    if fz:
        print(f"\nfast rule: applied in >= {fz['fast_min_frac']} of sectors and amp >= "
              f"{fz['fast_min_amp'] * 1e3:.1f} ppt")
    for grp in ("slow", "fast"):
        ks = [k for k, v in pairs.items() if len(v) == 2 and fz
              and (is_fast(census.get(v["harm"]["tic"]), fz) == (grp == "fast"))]
        for arm in ("none", "harm"):
            k = sum(rec_top(pairs[x][arm], PROVISIONAL) for x in ks)
            anym = sum(bool(pairs[x][arm]["inj_match"] and pairs[x][arm]["inj_match"][0][0])
                       for x in ks)
            print(f"DEV {grp} rotators, {arm}: recovered at MES>={PROVISIONAL}: {fmt(k, len(ks))}; "
                  f"top signal matched at any MES: {anym}/{len(ks)}")
        d = [(pairs[x]["harm"]["signals"][0]["mes"] if pairs[x]["harm"]["inj_match"]
              and pairs[x]["harm"]["inj_match"][0][0] else 0)
             - (pairs[x]["none"]["signals"][0]["mes"] if pairs[x]["none"]["inj_match"]
                and pairs[x]["none"]["inj_match"][0][0] else 0) for x in ks]
        if d:
            print(f"   matched-MES difference harm-none: median {np.median(d):.3f}, "
                  f"mean {np.mean(d):.3f}, min {np.min(d):.2f}, max {np.max(d):.2f}")
    # prediction efficiency
    h = [r for r in inj if r["rot"] == "harm" and r["inj"]["snr_white"] >= 12]
    m = [r for r in h if r["inj_match"] and r["inj_match"][0][0]]
    if m:
        ratio = [r["signals"][0]["mes"] / r["inj"]["snr_white"] for r in m]
        print(f"\nprediction efficiency: DEV harm injections with SNR_white >= 12: {len(h)}, matched "
              f"{len(m)}; median MES/SNR_white {np.median(ratio):.3f} "
              f"(16-84%: {np.percentile(ratio, 16):.3f}-{np.percentile(ratio, 84):.3f})")
        S["eff"] = float(np.median(ratio))
    S["dev_census"] = {str(k): v for k, v in census.items()}


def _tau(stats):
    st = np.sort(stats)[::-1]
    k = int(math.floor(0.005 * len(st)))
    return float(st[k]), st


def tau(S):
    for rg, name in (("long", "dev_nulls_long_harm"), ("short", "dev_nulls_short_harm")):
        rows = [r for r in ok(rd(name)) if r["id"].startswith("null")]
        errs = len(rd(name)) - len(ok(rd(name)))
        if not rows:
            continue
        t, st = _tau([top(r) for r in rows])
        S[f"tau_{rg}"] = t
        S[f"n_null_{rg}"] = len(rows)
        print(f"\n## {rg} nulls: {len(rows)} (errors {errs}); top 6 {np.round(st[:6], 2).tolist()}; "
              f"tau_{rg} = {t:.2f} (rank {int(math.floor(0.005 * len(st))) + 1})")
        print("| null | n_sec | MES | P (d) | n_ep | near 13.7/27.4 |")
        print("|---|---|---|---|---|---|")
        for r in sorted(rows, key=top, reverse=True)[:8]:
            s = r["signals"][0]
            print(f"| {r['id']} | {r['n_sectors']} | {s['mes']:.2f} | {s['P']:.3f} | {s['n_ep']} | "
                  f"{near_orbit(s['P'])} |")
        allP = [r["signals"][0]["P"] for r in rows if r["signals"]]
        print(f"null top signals: {len(allP)}; within 3% of a 13.7/27.4 d multiple or fraction: "
              f"{np.mean([near_orbit(p) for p in allP]):.2f}")
        df = pd.DataFrame({"b": [bucket(r["n_sectors"]) for r in rows], "s": [top(r) for r in rows]})
        print("per-bucket null max / 99th pct / n: " + "; ".join(
            f"{b}: {g['s'].max():.2f}/{g['s'].quantile(0.99):.2f}/{len(g)}"
            for b, g in df.groupby("b")))
        S[f"null_top_{rg}"] = [float(x) for x in st[:10]]
    # DEV real controls against tau
    if "tau_long" in S:
        real = [r for r in ok(rd("dev_paired")) if r["id"].startswith("devreal") and r["rot"] == "harm"]
        above = [(top(r), r["tic"]) for r in real if top(r) >= S["tau_long"]]
        print(f"DEV real controls (harm, long) above tau_long: {len(above)}/{len(real)} {above}")
    if "tau_short" in S:
        rs = [r for r in ok(rd("dev_nulls_short_harm")) if r["id"].startswith("devrealS")]
        above = [(round(top(r), 2), r["tic"], round(r["signals"][0]["P"], 3)) for r in rs
                 if top(r) >= S["tau_short"]]
        print(f"DEV real controls (harm, short) above tau_short: {len(above)}/{len(rs)} {above}")


def test(S):
    fz = frozen()
    tl, ts = fz["tau_long"], fz["tau_short"]
    real = ok(rd("test_real"))
    census = {r["tic"]: r["rot_info"] for r in real}
    print(f"\n## (a) TEST real controls; tau_long = {tl:.2f}, tau_short = {ts:.2f}")
    for rg, t in (("long", tl), ("short", ts)):
        rr = [r for r in real if r["range"] == rg]
        above = [r for r in rr if top(r) >= t]
        print(f"{rg}: {len(rr)} searched, {len(above)} above tau ({len(above) / max(len(rr), 1):.3f})")
        print("| tic | n_sec | top MES | P (d) | n_ep | dur h | near orbit | fast |")
        print("|---|---|---|---|---|---|---|---|")
        for r in sorted(rr, key=top, reverse=True)[:6]:
            if not r["signals"]:
                continue
            s = max(r["signals"], key=lambda s: s["mes"])
            print(f"| {r['tic']} | {r['n_sectors']} | {s['mes']:.2f} | {s['P']:.3f} | {s['n_ep']} | "
                  f"{s['dur_h']:.1f} | {near_orbit(s['P'])} | {is_fast(r['rot_info'], fz)} |")
        S[f"test_real_{rg}_above"] = [len(above), len(rr)]
        S[f"test_real_{rg}_top"] = [[r["tic"], top(r), max(r["signals"], key=lambda s: s["mes"])["P"]]
                                    for r in sorted(rr, key=top, reverse=True)[:6] if r["signals"]]
    allP = [s["P"] for r in real if r["range"] == "long" for s in r["signals"]]
    if allP:
        print(f"TEST long signals (all slots, MES>=6): {len(allP)}, near 13.7/27.4: "
              f"{np.mean([near_orbit(p) for p in allP]):.2f}")
    inj = ok(rd("test_inj"))
    none = {r["id"][:-5]: r for r in ok(rd("test_inj_none"))}
    rows = []
    for r in inj:
        n = none.get(r["id"])
        if n is not None and n.get("identical_to_harm"):
            n = r  # harm changed no sector of this injected light curve: identical arms
        rows.append({"cell": r["inj"]["cell"], "nsec": r["n_sectors"], "b": bucket(r["n_sectors"]),
                     "fast": is_fast(census.get(r["tic"]), fz), "rec": rec_top(r, tl),
                     "rec_none": rec_top(n, tl) if n else np.nan, "has_none": n is not None,
                     "ge2": r["inj"]["n_tr_data"] >= 2, "snr": r["inj"]["snr_white"],
                     "matched_any": bool(r["inj_match"] and r["inj_match"][0][0]),
                     "applied_inj": bool(r["rot_info"]["applied"])})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    names = {0: "1.0-1.5 R_e, 15-30 d (6+ sec)", 1: "1.0-1.5 R_e, 30-50 d (6+)",
             2: "1.5-2.0 R_e, 15-30 d (6+)", 3: "1.5-2.0 R_e, 30-50 d (6+)",
             4: "1.0-1.5 R_e, 15-30 d (2-5 sec)"}
    print(f"\n## (b) TEST injections at tau_long = {tl:.2f} (top signal)")
    print("| cell | all | 6-12 | 13-25 | 26+ | slow | fast | >=2 transits in data |")
    print("|---|---|---|---|---|---|---|---|")
    S["inj"] = {}
    for c, nm in names.items():
        d = df[df["cell"] == c]
        if d.empty:
            continue
        cols = [fmt(int(d["rec"].sum()), len(d))]
        for b in ("6-12", "13-25", "26+"):
            x = d[d["b"] == b]
            cols.append(fmt(int(x["rec"].sum()), len(x)) if len(x) else "-")
        for f in (False, True):
            x = d[d["fast"] == f]
            cols.append(fmt(int(x["rec"].sum()), len(x)) if len(x) else "-")
        cols.append(f"{d['ge2'].mean():.2f}")
        print(f"| {nm} | " + " | ".join(cols) + " |")
        S["inj"][nm] = [int(d["rec"].sum()), len(d), *wilson(int(d["rec"].sum()), len(d))]
    h = df[df["cell"] == 0]
    p, lo, hi = wilson(int(h["rec"].sum()), len(h))
    S["b"] = {"k": int(h["rec"].sum()), "n": len(h), "p": p, "lo": lo, "hi": hi,
              "pass": bool(lo >= 0.20 and len(h) >= 200)}
    print(f"(b) headline: {fmt(int(h['rec'].sum()), len(h))} -> lower bound {lo:.3f}; "
          f"{'PASS' if S['b']['pass'] else 'FAIL'}")
    print(f"misses: top signal matched below tau {int((h['matched_any'] & ~h['rec']).sum())}, "
          f"top signal unmatched or none {int((~h['matched_any']).sum())}")
    for lo_s, hi_s in ((0, 8), (8, 12), (12, 20), (20, 1e9)):
        x = h[(h["snr"] >= lo_s) & (h["snr"] < hi_s)]
        if len(x):
            print(f"   headline SNR_white {lo_s}-{hi_s}: {fmt(int(x['rec'].sum()), len(x))}")
    print(f"\n## (d) paired rot=none arm")
    dd = df[df["has_none"]]
    for f, nm in ((False, "slow"), (True, "fast")):
        x = dd[dd["fast"] == f]
        if x.empty:
            continue
        kh, kn = int(x["rec"].sum()), int(x["rec_none"].sum())
        diff = (kh - kn) / len(x)
        print(f"{nm}: harm {fmt(kh, len(x))}; none {fmt(kn, len(x))}; diff {diff:+.3f}; "
              f"rotation removal applied to the injected lc: {x['applied_inj'].mean():.2f}")
        S[f"d_{nm}"] = [kh, kn, len(x), diff]


def hosts(S):
    from fetch import load
    from inject import match

    fz = frozen()
    tl = fz["tau_long"]
    rows = {r["id"]: r for r in ok(rd("hosts"))}
    known = {"Gliese 12 b": (52005579, 6251.01), "LHS 1140 b": (92226327, 256.01),
             "TOI-700 d": (150428135, 700.02)}
    toi = pd.read_csv(CAT_DIR / "toi.csv", comment="#").set_index("TOI")
    print(f"\n## (c) gate-1 hosts (non-blind), long search, tau_long = {tl:.2f}")
    g1 = {}
    for nm, (tic, tid) in known.items():
        r = rows.get(f"g1host-long-{tic}")
        if r is None:
            continue
        k = toi.loc[tid]
        P, t0, dur = (float(k["Period (days)"]), float(k["Epoch (BJD)"]) - 2457000.0,
                      float(k["Duration (hours)"]) / 24)
        lc, _ = load(tic)
        hit = [(i, s, match(s, P, t0, dur, lc["idx"])) for i, s in enumerate(r["signals"])]
        hit = [h for h in hit if h[2][0]]
        if hit:
            i, s, m = hit[0]
            g1[nm] = s["mes"]
            print(f"{nm}: slot {i + 1}, P_det {s['P']:.4f}, MES {s['mes']:.2f}, alias {m[1]}, "
                  f">= tau {s['mes'] >= tl}")
        else:
            g1[nm] = 0.0
            print(f"{nm}: no matching signal; signals: "
                  + "; ".join(f"{s['P']:.3f}/{s['mes']:.2f}" for s in r["signals"]))
    S["g1_hosts"] = g1
    pred = pd.read_csv(G2 / "holdout_pred.csv")
    out = []
    for _, p in pred[pred["in_range"]].iterrows():
        row = {"planet": p["planet"], "tic": p["tic"], "P": p["P"], "rp": p["rp_re"],
               "tmag": p["tmag"], "n_sec": p.get("n_sectors", 0), "n_tr": p.get("n_tr", 0),
               "mes_pred": p.get("mes_pred", 0.0), "eph": p["eph"]}
        row["pred_det"] = bool(row["mes_pred"] >= 1.25 * tl)
        r = rows.get(f"holdout-long-{int(p['tic'])}") if np.isfinite(p["tic"]) else None
        row.update(searched=r is not None, mes=np.nan, alias=None, rec=False, slot=None)
        if r is not None and np.isfinite(p["t0"]):
            lc, _ = load(int(p["tic"]))
            dur = p["dur_used_h"] / 24 if np.isfinite(p.get("dur_used_h", np.nan)) else 0.1
            for i, s in enumerate(r["signals"]):
                m = match(s, p["P"], p["t0"], dur, lc["idx"])
                if m[0]:
                    row.update(mes=s["mes"], alias=m[1], slot=i + 1, rec=bool(s["mes"] >= tl),
                               P_det=s["P"])
                    break
            if row["slot"] is None:
                # period-only check (ephemeris may be stale): report, never counted
                for i, s in enumerate(r["signals"]):
                    rat = s["P"] / p["P"]
                    for c in (rat, 1 / rat):
                        if abs(c - round(c)) <= 0.01 * max(round(c), 1) and 1 <= round(c) <= 6:
                            row.setdefault("period_only", f"slot {i + 1} P {s['P']:.3f} MES {s['mes']:.2f}")
        out.append(row)
    o = pd.DataFrame(out).sort_values("mes_pred", ascending=False)
    o.to_csv(G2 / "holdout_result.csv", index=False)
    cols = ["planet", "tic", "P", "rp", "tmag", "n_sec", "n_tr", "mes_pred", "pred_det", "mes",
            "alias", "slot", "rec"] + (["period_only"] if "period_only" in o else [])
    print(o[cols].to_string(index=False))
    pdet = o[o["pred_det"]]
    k = int(pdet["rec"].sum())
    S["c_holdout"] = [k, len(pdet)]
    S["c_pass"] = bool(all(v >= tl for v in g1.values()) and len(g1) == 3 and len(pdet) >= 3
                       and k >= len(pdet) / 2)
    print(f"(c) gate-1 hosts above tau: {sum(v >= tl for v in g1.values())}/3; hold-out predicted "
          f"detectable {len(pdet)}, recovered {k}; {'PASS' if S['c_pass'] else 'FAIL'}")
    extra = o[~o["pred_det"] & o["rec"]]
    print(f"recovered although not predicted detectable: {list(extra['planet'])}")


def cost(S):
    from select_controls import sectors_for

    names = ["dev_paired", "dev_nulls_long_harm", "dev_nulls_short_harm", "test_real", "test_inj",
             "test_inj_none", "hosts"]
    runs = [r for n in names for r in ok(rd(n)) if not r.get("identical_to_harm")]
    df = pd.DataFrame([{"cpu": r["cpu_s"], "pre": r["cpu_pre"], "it": r["n_iter"],
                        "bins": r["n_bins"], "T": r["T_base"], "nsec": r["n_sectors"],
                        "rg": r["range"], "rot": r["rot"], "real": r["mode"] == "real"
                        and "inj" not in r} for r in runs])
    print(f"\n## (e) cost: {len(df)} CPU searches")
    models = {}
    for rg in ("long", "short"):
        d = df[(df["rg"] == rg)]
        d = d.assign(per_it=(d["cpu"] - d["pre"]).clip(lower=1e-3) / d["it"])
        x = np.log(d["bins"] * d["T"])
        A = np.vstack([np.ones(len(x)), x]).T
        coef, *_ = np.linalg.lstsq(A, np.log(d["per_it"]), rcond=None)
        models[rg] = coef
        print(f"{rg}: per-iteration CPU-s = exp({coef[0]:.2f}) (bins*T)^{coef[1]:.3f}; n={len(d)}; "
              f"median per-iteration {d['per_it'].median():.1f} s; median pre (incl. rotation) "
              f"{d['pre'].median():.2f} s")
    harm = df[(df["rot"] == "harm") & (df["real"])]
    pre_per_sec = float((harm["pre"] / harm["nsec"]).median())
    bins_per_sec = float((df["bins"] / df["nsec"]).median())
    it = {rg: float(df[(df["rg"] == rg) & df["real"]]["it"].mean()) for rg in ("long", "short")}
    s = pd.read_csv(CAT_DIR / "tic_mdwarfs_50pc.csv").drop_duplicates("ID")
    secs = sectors_for(s["ID"], s["ra"], s["dec"])
    v = []
    for t in s["ID"]:
        ss = secs.get(int(t), [])
        if not ss:
            continue
        n = len(ss)
        T = (max(ss) - min(ss) + 1) * 27.4
        c = pre_per_sec * n
        for rg in ("long", "short"):
            c += it[rg] * math.exp(models[rg][0]) * (n * bins_per_sec * T) ** models[rg][1]
        v.append(c)
    v = np.array(v)
    rng = np.random.default_rng(1)
    draws = [v[rng.integers(0, len(v), 8000)].sum() / 3600 for _ in range(200)]
    m = float(np.mean(draws))
    S["cost"] = {"cpu_h_8000": m, "cpu_wall_days_3w": m / 3 / 24, "gpu_days": m / 5 / 24,
                 "per_star_mean_h": float(v.mean() / 3600), "iters": it,
                 "pre_per_sector_s": pre_per_sec}
    print(f"8,000 stars: {m:.0f} CPU-h [{np.percentile(draws, 2.5):.0f}, "
          f"{np.percentile(draws, 97.5):.0f}]; 3 niced workers: {m / 3 / 24:.1f} days; GPU at 5x: "
          f"{m / 5 / 24:.1f} days -> {'PASS' if m / 5 / 24 < 14 else 'FAIL'}; mean iterations per "
          f"real star {it}; rotation+preprocessing {pre_per_sec:.2f} CPU-s per sector")


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    sp = G2 / "summary2.json"
    S = json.loads(sp.read_text()) if sp.exists() else {}
    for k, f in (("dev", dev), ("tau", tau), ("test", test), ("hosts", hosts), ("cost", cost)):
        if what in (k, "all"):
            f(S)
    sp.write_text(json.dumps(S, indent=1, default=str))


if __name__ == "__main__":
    main()
