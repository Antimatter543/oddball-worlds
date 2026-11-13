"""Gate 3 analysis (pre-registered rules, docs/spikes/mdwarf_100pc.md sections 3-5).

Inputs: <OUT>/gate3/{targets_100pc.csv, sample.csv, ref.csv, star_info.json, pixel.jsonl} and the
unpacked kernel outputs <OUT>/gate3/kaggle_out/*/run/. 50 pc references: v2 validation
(<OUT>/vetting/v2/val2.out.jsonl), survey vetting rows (v2/vet2_survey.s*.jsonl + vetting/
pixel.jsonl). Writes <OUT>/gate3/analyze3.txt and analyze3.json.
Usage: python -I gate3/analyze3.py [--list-ab]   (--list-ab: print real A/B keys for pixel3.py)
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "vetting"))
sys.path.insert(0, str(HERE))
from common import OUT
from sample3 import strata

G3 = OUT / "gate3"
V2 = OUT / "vetting" / "v2"
TAU = json.loads((OUT / "gate2" / "frozen.json").read_text())["tau_long"]
RNG = np.random.default_rng(20261119)
NBOOT = 2000
G, RS_AU = 2.959122082855911e-4, 0.00465047
CELLS = {"h": "1.0-1.5 R_e, 15-30 d", "l": "1.0-1.5 R_e, 30-50 d", "m": "1.5-2.0 R_e, 15-50 d",
         "x": "2.0-3.0 R_e, 15-50 d"}
SECB = ((1, 3), (4, 5), (6, 12), (13, 999))
TMB = ((-5, 12), (12, 13), (13, 14), (14, 15))
S50 = {"all": (13, 462)}  # 50 pc survey: not-C survivors / non-K signals (regraded below)


# ----------------------------------------------------------------------------- occurrence
def eta_cells():
    """DC15 Table 4 (planets per star), bins 5.5-18.2 d (b3) and 18.2-60.3 d (b4), log-uniform
    inside each; 1.0-1.5 R_e uses the b4 density for 15-18.2 d (b3 is an upper limit)."""
    b3 = {"1.0-1.5": None, "1.5-2.0": (0.2006, 0.0404, 0.0545),
          "2.0-2.5": (0.1569, 0.0355, 0.0491), "2.5-3.0": (0.0654, 0.0218, 0.0351)}
    b4 = {"1.0-1.5": (0.2308, 0.0602, 0.0938), "1.5-2.0": (0.2673, 0.0608, 0.0899),
          "2.0-2.5": (0.2365, 0.0583, 0.0881), "2.5-3.0": (0.1042, 0.0374, 0.0668)}
    w3, w4 = math.log(18.2 / 5.5), math.log(60.3 / 18.2)

    def piece(rbins, p0, p1):
        out = np.zeros(3)
        for rb in rbins:
            for k, sgn in ((0, 0), (1, -1), (2, 1)):
                c4 = b4[rb][0] + (sgn * b4[rb][k] if k else 0)
                d4 = c4 / w4
                if b3[rb] is None:
                    d3 = d4
                else:
                    c3 = b3[rb][0] + (sgn * b3[rb][k] if k else 0)
                    d3 = c3 / w3
                lo3, hi3 = max(p0, 5.5), min(p1, 18.2)
                lo4, hi4 = max(p0, 18.2), min(p1, 60.3)
                v = (d3 * math.log(hi3 / lo3) if hi3 > lo3 else 0) + \
                    (d4 * math.log(hi4 / lo4) if hi4 > lo4 else 0)
                out[k] += v
        return out  # central, low, high

    return {"h": piece(["1.0-1.5"], 15, 30), "l": piece(["1.0-1.5"], 30, 50),
            "m": piece(["1.5-2.0"], 15, 50), "x": piece(["2.0-2.5", "2.5-3.0"], 15, 50)}


# ----------------------------------------------------------------------------- helpers
def rows(p):
    p = Path(p)
    return [json.loads(x) for x in open(p) if x.strip()] if p.exists() else []


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def fw(k, n):
    p, lo, hi = wilson(k, n)
    return f"{k}/{n} = {p:.3f} [{lo:.3f}, {hi:.3f}]" if n else "0/0"


def ptr(P, m, r):
    a = (G * m * P**2 / (4 * np.pi**2)) ** (1 / 3)
    return r * RS_AU / a


def load_runs():
    runs = sorted(G3.glob("kaggle_out/*/run"))
    fetch, noise, jobs, out, vet = {}, {}, {}, {}, {}
    for d in runs:
        for r in rows(d / "fetch_status.jsonl"):
            fetch[r["tic"]] = r
        for r in rows(d / "noise.jsonl"):
            noise[(r["tic"], r["kind"])] = r
        for r in rows(d / "jobs.all.jsonl"):
            jobs[r["id"]] = r
        for f in sorted(d.glob("out.h*.jsonl")):
            for r in rows(f):
                out[r["id"]] = r
        for f in sorted(d.glob("vet3.s*.jsonl")):
            for r in rows(f):
                vet[r["key"]] = r
    return runs, fetch, noise, jobs, out, vet


def grade_all(vet: dict) -> dict:
    from vet2 import grade2, load_frozen2

    thr = load_frozen2()
    pix = {r["key"]: r["pixel"] for r in rows(G3 / "pixel.jsonl")}
    g = {}
    for k, r in vet.items():
        if "error" in r or "stats" not in r:
            g[k] = {"tier": "C", "error": r.get("error"), "n_tr": 0}
            continue
        st = dict(r["stats"])
        if k in pix:
            st["pixel"] = pix[k]
        tdet = 7.0 if r["pop"] == "g3_fa_null" else TAU
        gg = grade2(st, thr, tdet)
        gg["n_tr"] = int(st.get("n_transits", 0))
        gg["pixel_done"] = k in pix
        g[k] = gg
    return g


def survey_s50() -> dict:
    """Regrade the 50 pc survey's vetting rows (frozen v2 + its pixel results): survival per cut."""
    from vet2 import grade2, load_frozen2

    thr = load_frozen2()
    pix = {r["key"]: r["pixel"] for r in rows(OUT / "vetting" / "pixel.jsonl")}
    tiers = Counter()
    cut = Counter()
    nonk = 0
    for f in sorted(V2.glob("vet2_survey.s*of2.jsonl")):
        for r in rows(f):
            if "stats" not in r:
                tiers["error"] += 1
                continue
            st = dict(r["stats"])
            if r["key"] in pix:
                st["pixel"] = pix[r["key"]]
            g = grade2(st, thr, TAU)
            tiers[g["tier"]] += 1
            if g["tier"] == "K":
                continue
            nonk += 1
            ntr = int(st.get("n_transits", 0))
            if g["tier"] != "C":
                cut["all"] += 1
                cut["ge4"] += ntr >= 4
                if g["tier"] in ("A", "A+"):
                    cut["A"] += 1
                    cut["A_ge4"] += ntr >= 4
    return {"tiers": dict(tiers), "nonK": nonk, "surv": dict(cut)}


# ----------------------------------------------------------------------------- main
def main():
    L = []
    P = L.append
    full = pd.read_csv(G3 / "targets_100pc.csv")
    full["stratum"] = strata(full)
    meta = json.loads((G3 / "targets_meta.json").read_text())
    sample = pd.read_csv(G3 / "sample.csv").sort_values("order").reset_index(drop=True)
    ref = pd.read_csv(G3 / "ref.csv")
    runs, fetch, noise, jobs, out, vet = load_runs()
    grades = grade_all(vet)
    if "--list-ab" in sys.argv:
        for k, g in sorted(grades.items()):
            if vet[k]["pop"] == "g3_real" and g["tier"] in ("A+", "A", "B") \
                    and not g.get("pixel_done"):
                print(k)
        return
    eta = eta_cells()
    Nfull = len(full)
    Nh = full["stratum"].value_counts().to_dict()
    P(f"# Gate 3 analysis (tau_long {TAU}); kernel runs: {len(runs)}")
    P(f"\n## 1. Target list\n{json.dumps(meta)}")
    full["tb"] = pd.cut(full.Tmag, [-5, 12, 13, 14, 15], right=False,
                        labels=["<12", "12-13", "13-14", "14-15"])
    full["sb"] = pd.cut(full.n_sectors, [6, 13, 26, 999], right=False,
                        labels=["6-12", "13-25", "26+"])
    P(pd.crosstab(full.tb, full.sb, margins=True).to_string())
    P(f"Tmag median {full.Tmag.median():.2f}; predicted sectors median "
      f"{full.n_sectors.median():.0f}, mean {full.n_sectors.mean():.1f}")

    # ------------------------------------------------ measured set (prefix rule)
    by_tic = defaultdict(list)
    for j in jobs.values():
        if j["kind"] == "far":
            by_tic[j["tic"]].append(j["id"])
    status, done_flags = {}, []
    for t in sample.tic:
        f = fetch.get(int(t))
        if f is None:
            status[t] = "missing"
        elif f["status"] == "no_data":
            status[t] = "no_data"
        elif f["status"] != "ok":
            status[t] = "fetch_error"
        else:
            ids = by_tic.get(int(t), [])
            status[t] = "ok" if ids and all(i in out for i in ids) else "incomplete"
        done_flags.append(status[t] in ("ok", "no_data", "fetch_error"))
    n_meas = len(done_flags) if all(done_flags) else done_flags.index(False)
    ms = sample.iloc[:n_meas].copy()
    ms["status"] = [status[t] for t in ms.tic]
    P(f"\n## 2. Sample\n300 drawn; measured set (prefix rule) {n_meas}; status "
      f"{dict(Counter(ms.status))}; finished beyond the prefix "
      f"{sum(1 for t in sample.tic[n_meas:] if status[t] == 'ok')}")
    from scipy.stats import ks_2samp

    P(f"KS vs full list: Tmag p = {ks_2samp(ms.Tmag, full.Tmag).pvalue:.2f}, predicted sectors "
      f"p = {ks_2samp(ms.n_sectors, full.n_sectors).pvalue:.2f}; medians sample/full Tmag "
      f"{ms.Tmag.median():.2f}/{full.Tmag.median():.2f}, sectors {ms.n_sectors.median():.0f}/"
      f"{full.n_sectors.median():.0f}")
    st_tab = pd.DataFrame({"full": pd.Series(Nh), "sample": ms.stratum.value_counts()}).fillna(0)
    st_tab["full_share"] = st_tab.full / st_tab.full.sum()
    st_tab["sample_share"] = st_tab["sample"] / max(1, st_tab["sample"].sum())
    P(st_tab.sort_index().to_string())

    # ------------------------------------------------ per-star tables
    real_long = {t: out.get(f"g3real-long-{t}") for t in ms.tic}
    real_short = {t: out.get(f"g3real-short-{t}") for t in ms.tic}
    ok = ms[ms.status == "ok"]

    # coverage
    P("\n## 3.1 Coverage (real vs predicted sectors through S107)")
    pred = ms.n_sectors.to_numpy()
    found = np.array([len([s for s in (real_long[t] or {}).get("sectors", []) if s <= 107])
                      if status[t] == "ok" else 0 for t in ms.tic])
    found_all = np.array([len((real_long[t] or {}).get("sectors", [])) if status[t] == "ok"
                          else 0 for t in ms.tic])
    nd = (ms.status == "no_data").to_numpy()
    okm = (ms.status == "ok").to_numpy()
    P(f"- no product at all: {fw(int(nd.sum()), int(nd.sum() + okm.sum()))} (50 pc: 331/4,681 = "
      f"0.071; follow-up far check 34/200)")
    P(f"- real/predicted over stars with data: {found[okm].sum() / pred[okm].sum():.3f} "
      f"(50 pc survey 0.778); over all measured stars {found.sum() / pred[nd | okm].sum():.3f} "
      f"(50 pc 0.734; follow-up far check 0.647); incl. sectors > 107: "
      f"{found_all[okm].sum() / pred[okm].sum():.3f}")
    P(f"- real sectors per star with data: median {np.median(found_all[okm]):.0f}; < 6: "
      f"{int((found_all[okm] < 6).sum())} of {int(okm.sum())}")
    for lo, hi in TMB:
        m = (ms.Tmag >= lo).to_numpy() & (ms.Tmag < hi).to_numpy() & (okm | nd)
        if m.sum():
            P(f"  T {lo}-{hi}: n {int(m.sum())}, no data {int((nd & m).sum())}, real/pred "
              f"{found[m].sum() / pred[m].sum():.3f}")
    prod = Counter()
    for t in ok.tic:
        nz = noise.get((int(t), "far"))
        if nz and "products" in nz:
            prod.update(nz["products"].values())
    P(f"- products of far sectors: {dict(prod)}")

    # noise
    P("\n## 3.2 Noise: robust scatter of 2-h bins, frozen preprocessing (ppm; per star = median "
      "over sectors)")

    def star_noise(kind, tics):
        o = {}
        for t in tics:
            nz = noise.get((int(t), kind))
            if nz and "per_sector" in nz:
                v = [x[0] for x in nz["per_sector"].values() if x[0] == x[0]]
                if v:
                    o[int(t)] = float(np.median(v))
        return o

    nf = star_noise("far", ok.tic)
    nr = star_noise("ref", ref.tic)
    tm_f = dict(zip(sample.tic.astype(int), sample.Tmag))
    tm_r = dict(zip(ref.tic.astype(int), ref.Tmag))
    vf, vr = np.array(list(nf.values())), np.array(list(nr.values()))

    def bmed_ratio(a, b):
        bs = [np.median(RNG.choice(a, len(a))) / np.median(RNG.choice(b, len(b)))
              for _ in range(NBOOT)]
        return np.median(a) / np.median(b), *np.percentile(bs, [2.5, 97.5])

    if len(vf) and len(vr):
        r_, lo, hi = bmed_ratio(vf, vr)
        P(f"- all: far median {np.median(vf):.0f} (n {len(vf)}), 50 pc reference median "
          f"{np.median(vr):.0f} (n {len(vr)}); ratio {r_:.2f} [{lo:.2f}, {hi:.2f}]; Tmag medians "
          f"{np.median([tm_f[t] for t in nf]):.2f} vs {np.median([tm_r[t] for t in nr]):.2f}")
        for lo_, hi_ in TMB:
            a = np.array([v for t, v in nf.items() if lo_ <= tm_f[t] < hi_])
            b = np.array([v for t, v in nr.items() if lo_ <= tm_r[t] < hi_])
            if len(a) >= 3 and len(b) >= 3:
                r_, lo, hi = bmed_ratio(a, b)
                P(f"  T {lo_}-{hi_}: far {np.median(a):.0f} (n {len(a)}), 50 pc {np.median(b):.0f} "
                  f"(n {len(b)}), ratio {r_:.2f} [{lo:.2f}, {hi:.2f}]")
            else:
                P(f"  T {lo_}-{hi_}: far n {len(a)}, 50 pc n {len(b)} (too few)")

    # nulls
    P("\n## 3.3 Null false alarms at tau_long (top signal, long range)")
    nul = [r for r in out.values() if jobs.get(r["id"], {}).get("kind") == "far"
           and r.get("mode") in ("inv", "invroll", "roll") and int(r["tic"]) in set(ok.tic)
           and "error" not in r]
    top = np.array([r["signals"][0]["mes"] if r.get("signals") else 0.0 for r in nul])
    k = int((top >= TAU).sum())
    pn = wilson(k, len(nul))
    P(f"- per realisation: {fw(k, len(nul))}; 50 pc 16/600 = 0.027; limit (b) 0.0533")
    P(f"- >= 7: {fw(int((top >= 7).sum()), len(nul))} (50 pc 48/600); by mode "
      f"{dict(Counter(r['mode'] + str(r['seed']) for r, t in zip(nul, top) if t >= TAU))}")
    star_any = Counter()
    for r, t in zip(nul, top):
        star_any[r["tic"]] |= bool(t >= TAU)
    P(f"- per star (any of 4): {fw(sum(star_any.values()), len(star_any))} (50 pc 9/150)")
    hi_ep = Counter(r["signals"][0]["n_ep"] for r, t in zip(nul, top) if t >= TAU)
    P(f"- epochs of nulls >= tau_long: {dict(sorted(hi_ep.items()))}")
    nsec_n = np.array([r["n_sectors"] for r in nul])
    for lo, hi in SECB:
        m = (nsec_n >= lo) & (nsec_n <= hi)
        P(f"  real sectors {lo}-{hi}: {fw(int((top[m] >= TAU).sum()), int(m.sum()))}")
    vn = [g for kk, g in grades.items() if vet[kk]["pop"] == "g3_fa_null"]
    P(f"- nulls >= 7 vetted with v2 (tau_det 7.0 proxy): {len(vn)}, not C "
      f"{fw(sum(g['tier'] != 'C' for g in vn), len(vn))} (50 pc 23/48)")
    b_pass = pn[0] <= 2 * 16 / 600

    # injections
    P("\n## 3.4 Injection recovery (detected = top signal matched, MES >= tau_long; kept = "
      "detected and v2 tier not C)")
    inj = [r for r in out.values() if r.get("inj") and jobs.get(r["id"], {}).get("kind") == "far"
           and int(r["tic"]) in set(ok.tic) and "error" not in r]
    def det(r):
        return bool(r.get("signals") and r["inj_match"][0][0] and r["signals"][0]["mes"] >= TAU)

    def kept(r, cut="all"):
        if not det(r):
            return False
        g = grades.get(f"{r['id']}#0")
        if g is None:
            return False
        if cut == "all":
            return g["tier"] != "C"
        if cut == "A":
            return g["tier"] in ("A", "A+", "K")
        if cut == "ge4":
            return g["tier"] != "C" and g["n_tr"] >= 4
        if cut == "A_ge4":
            return g["tier"] in ("A", "A+", "K") and g["n_tr"] >= 4
        raise ValueError(cut)

    missing_vet = sum(1 for r in inj if det(r) and f"{r['id']}#0" not in grades)
    P(f"- far injections {len(inj)}; detected without a vetting row: {missing_vet} (counted as not "
      f"kept)")
    ref_inj = [r for r in out.values() if r.get("inj") and jobs.get(r["id"], {}).get("kind") == "ref"
               and "error" not in r]
    v2 = rows(V2 / "val2.out.jsonl")
    v2h = [r for r in v2 if r.get("inj") and r["id"].startswith("v2injhead")]
    for c, lab in CELLS.items():
        s = [r for r in inj if jobs[r["id"]]["cell"] == c]
        d = [r for r in s if det(r)]
        kp = [r for r in s if kept(r)]
        line = f"- {lab}: detected {fw(len(d), len(s))}; kept {fw(len(kp), len(s))}"
        if c == "h":
            line += f" | 50 pc (v2 validation, detection): {fw(sum(det(r) for r in v2h), len(v2h))}"
        if c in ("m", "x"):
            rs = [r for r in ref_inj if jobs[r["id"]]["cell"] == c]
            line += f" | 50 pc reference (detection): {fw(sum(det(r) for r in rs), len(rs))}"
        P(line)
        for lo, hi in SECB:
            sb = [r for r in s if lo <= r["n_sectors"] <= hi]
            ln = f"    real sectors {lo}-{hi if hi < 999 else '+'}: det {fw(sum(det(r) for r in sb), len(sb))}"
            if c == "h":
                vb = [r for r in v2h if lo <= r["n_sectors"] <= hi]
                ln += f" | 50 pc {fw(sum(det(r) for r in vb), len(vb))}"
            if c in ("m", "x"):
                rb = [r for r in ref_inj if jobs[r["id"]]["cell"] == c and lo <= r["n_sectors"] <= hi]
                ln += f" | 50 pc ref {fw(sum(det(r) for r in rb), len(rb))}"
            P(ln)
    dets = [r for r in inj if det(r)]
    tiers_d = Counter(grades[f"{r['id']}#0"]["tier"] for r in dets if f"{r['id']}#0" in grades)
    P(f"- tiers of detected injections: {dict(tiers_d)}")

    # real search
    P("\n## 3.5 Real search on the sample (long range, 6 slots)")
    nsig = {t: sum(s["mes"] >= TAU for s in (real_long[t] or {}).get("signals", []))
            if status[t] == "ok" else 0 for t in ms.tic}
    n_s = sum(nsig.values())
    n_st = sum(v > 0 for v in nsig.values())
    nsearch = int(okm.sum())
    P(f"- signals >= tau_long: {n_s} on {n_st} of {nsearch} searched stars: "
      f"{n_s / max(1, nsearch):.3f} per searched star (50 pc 510/4,350 = 0.117); stars with any "
      f"{fw(n_st, nsearch)} (50 pc 252/4,350 = 0.058)")
    realg = {kk: g for kk, g in grades.items() if vet[kk]["pop"] == "g3_real"}
    P(f"- vetted {len(realg)}; tiers {dict(Counter(g['tier'] for g in realg.values()))} "
      f"(50 pc: K 48, A+ 0, A 3, B 10, C 449); pixel checked "
      f"{sum(g.get('pixel_done', False) for g in realg.values())}")
    for kk, g in sorted(realg.items()):
        if g["tier"] != "C":
            s = vet[kk]["stats"]
            P(f"    {kk}: tier {g['tier']}, P {s['P']:.3f} d, MES {s['mes']:.2f}, "
              f"{g['n_tr']} transits; {'; '.join(g['reasons'][:3])}")
    surv = {cut: defaultdict(int) for cut in ("all", "A", "ge4", "A_ge4")}
    nonk = defaultdict(int)
    for kk, g in realg.items():
        t = int(vet[kk]["tic"])
        if g["tier"] == "K":
            continue
        nonk[t] += 1
        if g["tier"] != "C":
            surv["all"][t] += 1
            surv["ge4"][t] += g["n_tr"] >= 4
            if g["tier"] in ("A", "A+"):
                surv["A"][t] += 1
                surv["A_ge4"][t] += g["n_tr"] >= 4
    s50 = survey_s50()
    P(f"- 50 pc survey regraded (frozen v2 + its pixel results): tiers {s50['tiers']}, non-K "
      f"{s50['nonK']}, survivors by cut {s50['surv']}")

    # cost
    P("\n## 3.6 Cost (Kaggle CPU, real search long + short, preprocessing included)")
    cpu = np.array([(real_long[t] or {}).get("cpu_s", 0) + (real_short[t] or {}).get("cpu_s", 0)
                    for t in ok.tic])
    dlw = np.array([fetch[int(t)].get("wall_s", 0) for t in ok.tic])
    P(f"- per searched star: mean {cpu.mean():.1f} CPU-s (median {np.median(cpu):.1f}, max "
      f"{cpu.max():.0f}); survey (50 pc, Kaggle) 89.9 CPU-h / 4,350 = 74.4 s; download wall "
      f"mean {dlw.mean():.1f} s (sequential, PDCSAP+SAP)")
    share_ok = okm.sum() / max(1, (okm | nd).sum())
    cpu_h = cpu.mean() * Nfull * share_ok / 3600
    n_shards = int(math.ceil(cpu_h / (3 * 3.9)))
    waves = int(math.ceil(n_shards / 5))
    wall_h = cpu_h / (5 * 3) + waves * 0.3
    P(f"- full list ({Nfull} stars, {share_ok:.2f} with data): {cpu_h:.0f} CPU-h; at 3 workers "
      f"and ~3.9 h per shard: {n_shards} shards, {waves} waves of 5 -> about "
      f"{wall_h:.0f} h wall ({wall_h / 24:.1f} d); limit 5 d")
    disk_mb = Nfull * share_ok * 16.6 / 1e3
    P(f"- local disk: result records {disk_mb:.0f} MB at 16.6 kB per star (survey value); "
      f"limit 500 MB")
    d_pass = wall_h < 120 and disk_mb < 500

    # ------------------------------------------------ projection
    P("\n## 4. Projection to the full list")
    P("occurrence per star (DC15 Table 4; central [low, high]): " + "; ".join(
        f"{CELLS[c]} {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]" for c, v in eta.items()))
    strata_list = sorted(Nh)
    ms_idx = {h: ms.index[ms.stratum == h].to_numpy() for h in strata_list}
    tic_arr = ms.tic.astype(int).to_numpy()
    inj_by = defaultdict(list)
    for r in inj:
        inj_by[(int(r["tic"]), jobs[r["id"]]["cell"])].append(r)

    def ymat(cut):
        Y = np.full((len(ms), 4), np.nan)
        for i, t in enumerate(tic_arr):
            if status[t] == "no_data" or status[t] == "fetch_error":
                Y[i] = 0.0 if status[t] == "no_data" else np.nan
                continue
            for ci, c in enumerate(CELLS):
                rr = inj_by.get((t, c), [])
                if rr:
                    Y[i, ci] = np.mean([kept(r, cut) * ptr(r["inj"]["P"], jobs[r["id"]]["mass"],
                                                           jobs[r["id"]]["radius"]) for r in rr])
        return Y

    def kvec(dct):
        return np.array([dct.get(int(t), 0) if status[t] == "ok" else 0 for t in tic_arr], float)

    def strat_total(v, idx_map):
        tot = 0.0
        for h in strata_list:
            ii = idx_map.get(h, np.array([], int))
            vals = v[ii] if len(ii) else np.array([])
            vals = vals[np.isfinite(vals)] if len(vals) else vals
            if not len(vals):  # stratum with no measured star: its predicted-sector bucket mean
                sbk = h.split("|")[0]
                pool_ = np.concatenate([idx_map[g] for g in strata_list
                                        if g.startswith(sbk) and len(idx_map[g])] or [[]]).astype(int)
                vals = v[pool_][np.isfinite(v[pool_])] if len(pool_) else np.array([0.0])
            tot += Nh[h] * float(np.mean(vals))
        return tot

    def boot_idx():
        return {h: (RNG.choice(ii, len(ii)) if len(ii) else ii) for h, ii in ms_idx.items()}

    res = {}
    for cut in ("all", "A", "ge4", "A_ge4"):
        Y = ymat(cut)
        E = np.array([strat_total(Y[:, ci], ms_idx) for ci in range(4)])
        mF = kvec(nonk)
        kF = kvec(surv[cut])
        s_k = s50["surv"].get(cut, 0)
        s_50 = s_k / s50["nonK"]
        Fd = strat_total(kF, ms_idx)
        Fs = strat_total(mF, ms_idx) * s_50
        bE, bFd, bFs = [], [], []
        for _ in range(NBOOT):
            bi = boot_idx()
            bE.append([strat_total(Y[:, ci], bi) for ci in range(4)])
            bFd.append(strat_total(kF, bi))
            bFs.append(strat_total(mF, bi) * RNG.beta(s_k + 0.5, s50["nonK"] - s_k + 0.5))
        bE = np.array(bE)
        res[cut] = {"E_cell_strstar": E.tolist(), "Fd": Fd, "Fs": Fs, "s_50": s_50,
                    "bE": bE, "bFd": np.array(bFd), "bFs": np.array(bFs)}
    etac = np.array([eta[c][0] for c in CELLS])
    etal = np.array([eta[c][1] for c in CELLS])
    etah = np.array([eta[c][2] for c in CELLS])
    r0 = res["all"]
    E0 = np.array(r0["E_cell_strstar"])
    P("expected detectable planets (detected AND not C), full list:")
    for ci, c in enumerate(CELLS):
        lo, hi = np.percentile(r0["bE"][:, ci] * etac[ci], [2.5, 97.5])
        P(f"- {CELLS[c]}: {E0[ci] * etac[ci]:.2f} [{lo:.2f}, {hi:.2f}] (eta low {E0[ci] * etal[ci]:.2f}"
          f", high {E0[ci] * etah[ci]:.2f}); 'kept transiting star-equivalents' {E0[ci]:.1f}")
    tot_c = float(E0 @ etac)
    tot_b = r0["bE"] @ etac
    lo, hi = np.percentile(tot_b, [2.5, 97.5])
    P(f"- **sum 1.0-3.0 R_e, 15-50 d: {tot_c:.2f} [{lo:.2f}, {hi:.2f}]**; eta low "
      f"{float(E0 @ etal):.2f}, high {float(E0 @ etah):.2f}; without the 2.0-3.0 R_e cell "
      f"{float(E0[:3] @ etac[:3]):.2f} (low eta {float(E0[:3] @ etal[:3]):.2f})")
    hc = E0[0] * etac[0]
    P(f"- headline cell alone (1.0-1.5 R_e, 15-30 d): {hc:.2f}; follow-up's 50 pc-rate projection "
      f"for the same list size class: 0.65-1.73 (with its 0.084-0.14 occurrence)")
    F = max(r0["Fd"], r0["Fs"])
    P(f"expected false alarms surviving v2 over the full list: F_direct {r0['Fd']:.1f} "
      f"[{np.percentile(r0['bFd'], 2.5):.1f}, {np.percentile(r0['bFd'], 97.5):.1f}], F_scaled "
      f"{r0['Fs']:.1f} [{np.percentile(r0['bFs'], 2.5):.1f}, {np.percentile(r0['bFs'], 97.5):.1f}]"
      f" (s_50 {r0['s_50']:.4f}); F = {F:.1f}; ratio F / planets = {F / max(tot_c, 1e-9):.1f}")
    allsig = kvec({t: v for t, v in nsig.items()})
    load = strat_total(allsig, ms_idx)
    P(f"- vetting load (signals >= tau_long over the full list): {load:.0f}")
    P("\nstricter cuts (for (c); not part of the verdict):")
    for cut in ("A", "ge4", "A_ge4"):
        rr = res[cut]
        Ec = float(np.array(rr["E_cell_strstar"]) @ etac)
        Fc = max(rr["Fd"], rr["Fs"])
        P(f"- {cut}: planets {Ec:.2f} ({Ec / max(tot_c, 1e-9):.0%} of the yield), F {Fc:.1f} "
          f"(direct {rr['Fd']:.1f}, scaled {rr['Fs']:.1f}), ratio {Fc / max(Ec, 1e-9):.1f}")

    # ------------------------------------------------ criteria
    a_pass = tot_c >= 1.0
    c_pass = F <= 10 * tot_c
    P("\n## 5. Criteria")
    P(f"- (a) planets {tot_c:.2f} >= 1.0: {'PASS' if a_pass else 'FAIL'}")
    P(f"- (b) null rate {pn[0]:.4f} <= 0.0533: {'PASS' if b_pass else 'FAIL'}")
    P(f"- (c) F {F:.1f} <= 10 x {tot_c:.2f} = {10 * tot_c:.1f}: {'PASS' if c_pass else 'FAIL'}")
    P(f"- (d) wall {wall_h:.0f} h < 120 h and disk {disk_mb:.0f} MB < 500 MB: "
      f"{'PASS' if d_pass else 'FAIL'}; vetting load ~{load:.0f} signals")
    verdict = ("recommend the extension" if a_pass and b_pass and c_pass and d_pass else
               "stop: the extension cannot find enough" if not a_pass else
               "do not recommend as specified (fails " + ",".join(
                   k for k, v in (("b", b_pass), ("c", c_pass), ("d", d_pass)) if not v) + ")")
    P(f"- **verdict: {verdict}**")

    # ------------------------------------------------ confirmation difficulty
    P("\n## 6. Typical far candidate")
    dt_ = [r for r in inj if kept(r)]
    if dt_:
        tm = np.array([tm_f[int(r["tic"])] for r in dt_])
        rs = np.array([jobs[r["id"]]["radius"] for r in dt_])
        ms_ = np.array([jobs[r["id"]]["mass"] for r in dt_])
        dep = np.array([r["inj"]["depth0"] for r in dt_])
        rp = np.array([r["inj"]["rp"] for r in dt_])
        Pp = np.array([r["inj"]["P"] for r in dt_])
        mp = np.where(rp < 1.5, rp ** 3.7, 2.7 * rp ** 1.3)  # rough rocky / Chen-Kipping-like
        K = 28.4329 / np.sqrt(1) * mp / 317.8 * (ms_) ** (-2 / 3) * (Pp / 365.25) ** (-1 / 3)
        P(f"- kept injections: host Tmag median {np.median(tm):.2f} [16-84% {np.percentile(tm, 16):.2f}"
          f"-{np.percentile(tm, 84):.2f}], R* median {np.median(rs):.2f}, depth median "
          f"{1e3 * np.median(dep):.2f} ppt, RV semi-amplitude for a rough mass-radius relation "
          f"median {np.median(K):.1f} m/s")
    txt = "\n".join(L) + "\n"
    (G3 / "analyze3.txt").write_text(txt)
    (G3 / "analyze3.json").write_text(json.dumps({
        "n_measured": n_meas, "planets_sum": tot_c, "F": F, "Fd": r0["Fd"], "Fs": r0["Fs"],
        "null_rate": pn, "criteria": {"a": a_pass, "b": b_pass, "c": c_pass, "d": d_pass},
        "verdict": verdict, "cpu_h": cpu_h, "n_shards": n_shards, "wall_h": wall_h,
        "vet_load": load, "eta": {c: list(v) for c, v in eta.items()}}, indent=1, default=float))
    print(txt)


if __name__ == "__main__":
    main()
