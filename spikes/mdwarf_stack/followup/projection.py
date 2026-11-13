"""Part 2.3 / 2.5: projected headline-cell recovery (1.0-1.5 R_e, 15-30 d, detection at
tau_long, frozen gate-2 search) for (A) the survey as run, (B) the survey with gap-filled
sectors, (C) the 331 no-data stars made searchable, (D) 50-100 pc M dwarfs with existing
products only.

Method ("transplant" of the v2 validation injections, outputs/planets_spike/vetting/v2/
val2.out.jsonl, 300 headline injections on 150 survey stars):
  * detection probability as a function of white-noise SNR, logistic in log SNR, fitted on the
    300 headline injections (detected = top signal matched with MES >= tau_long, as in
    posthoc_val2.py);
  * an injection on a star with N real sectors moved to N_eff equivalent SPOC sectors has
    SNR x sqrt(N_eff / N) (transit count scales with sectors on average);
  * an added sector from a product whose 2-h noise is r x SPOC's counts as 1 / r^2 sectors
    (r measured in noise.py, per product, median over the paired sample);
  * a target star with (Tmag, N_eff) gets the mean detection probability of the validation
    injections within +-0.5 mag of its Tmag, each moved to N_eff.
Intervals: bootstrap over validation stars (refit) x over the noise sample, 2,000 draws.
Usage: python -I followup/projection.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fu

TAU = 8.054
RNG = np.random.default_rng(20261114)
NBOOT = 2000


def val_injections():
    rows = [json.loads(x) for x in (fu.OUT / "vetting" / "v2" / "val2.out.jsonl").read_text()
            .splitlines() if x.strip()]
    inj = [r for r in rows if r.get("inj") and r["id"].startswith("v2injhead")]
    tg = pd.read_csv(fu.OUT / "gate2" / "survey_targets.csv").set_index("tic")
    out = []
    for r in inj:
        det = bool(r.get("signals") and r["inj_match"][0][0] and r["signals"][0]["mes"] >= TAU)
        out.append({"tic": r["tic"], "N": r["n_sectors"], "snr": max(r["inj"]["snr_white"], 1e-3),
                    "det": det, "tmag": float(tg.loc[r["tic"], "Tmag"])})
    return pd.DataFrame(out)


def fit_logistic(x, y, iters=60):
    X = np.vstack([np.ones_like(x), x]).T
    b = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = p * (1 - p) + 1e-9
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(2)
        b = b + np.linalg.solve(H, X.T @ (y - p))
    return b


def pdet(b, snr):
    return 1 / (1 + np.exp(-(b[0] + b[1] * np.log(np.maximum(snr, 1e-3)))))


def recovery(b, vi, targets):
    """Mean P(det) over targets (Tmag, N_eff) via transplanted validation injections."""
    tm = vi["tmag"].to_numpy()
    snr = vi["snr"].to_numpy()
    N = vi["N"].to_numpy()
    res = np.zeros(len(targets))
    for j, (t, ne) in enumerate(targets):
        m = np.abs(tm - t) <= 0.5
        if m.sum() < 5:
            m = np.argsort(np.abs(tm - t))[:5]
        res[j] = 0.0 if ne <= 0 else float(np.mean(pdet(b, snr[m] * np.sqrt(ne / N[m]))))
    return res


def main():
    vi = val_injections()
    b = fit_logistic(np.log(vi["snr"].to_numpy()), vi["det"].to_numpy().astype(float))
    L = [f"validation headline injections {len(vi)}, detected {int(vi['det'].sum())}; logistic "
         f"in log SNR: b0 {b[0]:.2f}, b1 {b[1]:.2f}; P(det) = 0.5 at SNR {np.exp(-b[0] / b[1]):.1f}"]
    for lo, hi in ((1, 3), (4, 5), (6, 12), (13, 999)):
        m = (vi["N"] >= lo) & (vi["N"] <= hi)
        L.append(f"  N {lo}-{hi}: measured {vi['det'][m].mean():.2f} ({int(vi['det'][m].sum())}/"
                 f"{int(m.sum())}), model {np.mean(pdet(b, vi['snr'][m])):.2f}")
    noise = [json.loads(x) for x in (fu.FU_OUT / "noise.jsonl").read_text().splitlines()]
    nz = pd.DataFrame([r for r in noise if r.get("status") == "ok"])
    piv = nz.pivot_table(index="tic", columns="product", values="sigma2h_ppm")
    ratio = {p: (piv[p] / piv["survey"]).dropna() for p in piv.columns if p != "survey"}
    L.append("noise ratio vs survey product (2-h bins), median [16-84%] (n): " + "; ".join(
        f"{p} {np.median(v):.2f} [{np.percentile(v, 16):.2f}, {np.percentile(v, 84):.2f}] "
        f"({len(v)})" for p, v in ratio.items()))
    gs = pd.read_csv(fu.FU_OUT / "gap_stars.csv").fillna("")

    def nadd(row, prods, last_tesscut=False):
        """Sectors added per product class, best (lowest-noise) product first."""
        used = set()
        out = {}
        for p in prods:
            col = {"qlp": "add_qlp", "tars": "add_tars", "tglc": "add_tglc_any",
                   "tesscut": "missing"}[p]
            ss = set(str(row[col]).split()) - used
            out[p] = len(ss)
            used |= ss
        return out

    # TGLC coverage: from the bucket check (tglc_check.json: present for 140/150 missing
    # star-sectors sampled), applied as "every missing sector <= 55" (slightly optimistic)
    gs["add_tglc_any"] = gs["missing"].astype(str).apply(
        lambda s: " ".join(x for x in s.split() if int(x) <= 55))
    order = sorted([p for p in ("qlp", "tars", "tglc") if p in ratio],
                   key=lambda p: np.median(ratio[p]))

    _cache = {}

    def counts(prods):
        key = tuple(prods)
        if key not in _cache:
            _cache[key] = {p: np.array([nadd(row, prods)[p] for _, row in gs.iterrows()])
                           for p in prods}
        return _cache[key]

    def neff(r_med, prods):
        c = counts(prods)
        return gs["n_found"].to_numpy() + sum(c[p] / r_med[p] ** 2 for p in prods)

    # (C) TESS-SPOC catches up: missing S87-S107 sectors filled by the survey's own product at
    # the rate it covered the same star's Tmag bin in S56-S86 (no noise penalty, no re-validation)
    import gzip

    cov = {}
    late = {}
    with gzip.open(fu.OUT / "survey" / "merged" / "results.jsonl.gz", "rt") as fh:
        for line in fh:
            r = json.loads(line)
            pr = [s for s in r["sectors_pred"] if 56 <= s <= 86]
            fo = {int(s) for s in (r.get("sectors_found") or [])}
            cov[r["tic"]] = (len(pr), sum(s in fo for s in pr))
            late[r["tic"]] = sum(1 for s in r["sectors_pred"] if 87 <= s <= 107 and s not in fo)
    gs["tb"] = np.floor(gs["tmag"] * 2) / 2
    q = {}
    for b_, sub in gs.groupby("tb"):
        a = sum(cov[t][0] for t in sub["tic"])
        q[b_] = sum(cov[t][1] for t in sub["tic"]) / max(1, a)
    n_c = gs["n_found"].to_numpy() + np.array([late[t] * q[b_] for t, b_ in
                                              zip(gs["tic"], gs["tb"])])

    def scen(r_med, bb):
        base = recovery(bb, vi, list(zip(gs["tmag"], gs["n_found"])))
        out = {"A_survey": base}
        out["C_tess_spoc_catch_up"] = recovery(bb, vi, list(zip(gs["tmag"], n_c)))
        prods1 = order
        out["B_fill_hlsp"] = recovery(bb, vi, list(zip(gs["tmag"], neff(r_med, prods1))))
        if "tesscut" in ratio:
            out["B2_fill_hlsp_tesscut"] = recovery(
                bb, vi, list(zip(gs["tmag"], neff(r_med, prods1 + ["tesscut"]))))
        return out

    r_med = {p: float(np.median(v)) for p, v in ratio.items()}
    pt = scen(r_med, b)
    ok = (gs["status"] == "ok").to_numpy()
    nd = ~ok
    boots = {k: [] for k in pt}
    boots_nd = {k: [] for k in pt}
    stars = vi["tic"].unique()
    boots_tot = {}
    for _ in range(NBOOT // 10):  # each draw re-runs every target (slow); 200 draws
        sel = RNG.choice(stars, len(stars), replace=True)
        vb = pd.concat([vi[vi["tic"] == s] for s in sel])
        bb = fit_logistic(np.log(vb["snr"].to_numpy()), vb["det"].to_numpy().astype(float))
        rb = {p: float(np.median(RNG.choice(v.to_numpy(), len(v), replace=True)))
              for p, v in ratio.items()}
        sc = scen(rb, bb)
        for k in sc:
            boots[k].append(sc[k][ok].mean())
            boots_nd[k].append(sc[k][nd].sum())
            boots_tot.setdefault(k, []).append(sc[k].sum() - sc["A_survey"].sum())
    L.append("\nheadline-cell recovery per star (transiting 1.0-1.5 R_e, 15-30 d), point "
             "[95% bootstrap]")
    for k in pt:
        lo, hi = np.percentile(boots[k], [2.5, 97.5])
        nlo, nhi = np.percentile(boots_nd[k], [2.5, 97.5])
        L.append(f"- {k}: searched stars mean {pt[k][ok].mean():.3f} [{lo:.3f}, {hi:.3f}] "
                 f"(x {ok.sum()} = {pt[k][ok].sum():.0f}); no-data stars sum {pt[k][nd].sum():.1f}"
                 f" [{nlo:.1f}, {nhi:.1f}] star-equivalents; all {pt[k].sum():.0f}")
    # planets: star-equivalents x geometric transit probability x occurrence in the cell
    tg = pd.read_csv(fu.OUT / "gate2" / "survey_targets.csv").set_index("tic")
    G, RS_AU = 2.959122082855911e-4, 0.00465047
    Ps = np.exp(np.linspace(np.log(15), np.log(30), 50))
    ptr = []
    for t in gs["tic"]:
        m = float(tg.loc[t, "mass"]) if np.isfinite(tg.loc[t, "mass"]) else 0.3
        r_ = float(tg.loc[t, "rad"]) if np.isfinite(tg.loc[t, "rad"]) else 0.3
        a = (G * m * Ps**2 / (4 * np.pi**2)) ** (1 / 3)
        ptr.append(float(np.mean(r_ * RS_AU / a)))
    ptr = np.array(ptr)
    eta = (0.56 * 0.15, 0.56 * 0.25)  # DC15 1-1.5 R_e, P < 50 d; share in 15-30 d, log-uniform
    L.append(f"\nmean transit probability 15-30 d: {ptr.mean():.4f}; occurrence in the cell "
             f"assumed {eta[0]:.3f}-{eta[1]:.3f} per star (Dressing & Charbonneau 2015: 0.56 "
             f"1.0-1.5 R_e planets per M dwarf at P < 50 d; 15-30 d share 0.15-0.25 for "
             f"log-uniform periods from 0.5 or 3 d to 50 d)")
    for k in pt:
        x = float(np.sum(pt[k] * ptr))
        L.append(f"- {k}: expected headline-cell detections if planets exist at that rate "
                 f"{x * eta[0]:.2f}-{x * eta[1]:.2f}")
        if k != "A_survey":
            d = float(np.sum((pt[k] - pt["A_survey"]) * ptr))
            lo, hi = np.percentile(boots_tot[k], [2.5, 97.5])
            L.append(f"  gain over the survey: {pt[k].sum() - pt['A_survey'].sum():.0f} "
                     f"[{lo:.0f}, {hi:.0f}] star-equivalents -> {d * eta[0]:.2f}-{d * eta[1]:.2f}"
                     f" planets")
    nef = neff(r_med, order)
    L.append(f"\nsectors per searched star now: median {np.median(gs['n_found'][ok]):.0f}; "
             f"< 6: {(gs['n_found'][ok] < 6).sum()}")
    L.append(f"N_eff with HLSP fill ({'+'.join(order)}): median {np.median(nef[ok]):.1f}; "
             f"< 6: {(nef[ok] < 6).sum()}; raw added sectors per star median "
             f"{np.median([sum(nadd(r, order).values()) for _, r in gs[ok].iterrows()]):.0f}")
    np.save(fu.FU_OUT / "projection_boot.npy", {k: np.array(v) for k, v in boots.items()})
    # (D) 50-100 pc with existing products
    tf = fu.FU_OUT / "tic100_sample.csv"
    if tf.exists():
        t1 = pd.read_csv(tf)
        gt = pd.read_csv(fu.FU_OUT / "gap_tables_frac.csv") if \
            (fu.FU_OUT / "gap_tables_frac.csv").exists() else None
        L.append(project_100(t1, b, vi, gt, eta))
    txt = "\n".join(L) + "\n"
    (fu.FU_OUT / "projection.txt").write_text(txt)
    print(txt)


def project_100(t1, b, vi, frac, eta):
    """50-100 pc stars, same selection as the survey (Teff 2700-3400 K, T < 15, >= 6 predicted
    sectors through S107): expected real sectors = predicted x (searched / predicted for the
    survey at the same Tmag), i.e. the same SPOC / TESS-SPOC coverage as the survey had."""

    meta = json.loads((fu.FU_OUT / "tic100_meta.json").read_text())
    area = float(meta["area_frac"])
    fr = dict(zip(frac["bin_lo"], frac["frac"])) if frac is not None else {}

    def f_of(t):
        ks = sorted(fr)
        k = max([x for x in ks if x <= t], default=ks[0]) if ks else None
        return fr.get(k, 0.78)

    t1 = t1[t1["n_sectors"] >= 6].copy()
    far = t1[(t1["d"] >= 50) & (t1["d"] < 100)].copy()
    near = t1[t1["d"] < 50]
    far["n_real"] = [n * f_of(t) for n, t in zip(far["n_sectors"], far["Tmag"])]
    near_real = [n * f_of(t) for n, t in zip(near["n_sectors"], near["Tmag"])]
    rec = recovery(b, vi, list(zip(far["Tmag"], far["n_real"])))
    rec_n = recovery(b, vi, list(zip(near["Tmag"], near_real)))
    G, RS_AU = 2.959122082855911e-4, 0.00465047
    Ps = np.exp(np.linspace(np.log(15), np.log(30), 50))

    def ptr(df):
        out = []
        for m, r_ in zip(df["mass"], df["rad"]):
            m = m if np.isfinite(m) else 0.3
            r_ = r_ if np.isfinite(r_) else 0.3
            a = (G * m * Ps**2 / (4 * np.pi**2)) ** (1 / 3)
            out.append(float(np.mean(r_ * RS_AU / a)))
        return np.array(out)

    x = float(np.sum(rec * ptr(far))) / area
    xn = float(np.sum(rec_n * ptr(near))) / area
    # second scaling: calibrate the cones on the survey itself (4,681 targets with >= 6
    # predicted sectors < 50 pc), since >= 6 sectors depends on ecliptic latitude and 30 cones
    # sample that unevenly
    cal = 4681 / max(1, len(near))
    calib = (f"\n- survey-calibrated scaling (x 4,681 / {len(near)} = {cal:.1f} per cone star, "
             f"vs {1 / area:.1f} by area): 50-100 pc {len(far) * cal:.0f} stars, "
             f"{rec.sum() * cal:.0f} star-equivalents, expected headline-cell detections "
             f"{float(np.sum(rec * ptr(far))) * cal * eta[0]:.2f}-"
             f"{float(np.sum(rec * ptr(far))) * cal * eta[1]:.2f}")
    return (f"\n(D) 50-100 pc, same selection (>= 6 predicted sectors), {len(meta['ok_cones'])} of "
            f"{meta['n_cone']} cones answered, covering {area:.4f} of the sky (overlaps ignored)\n"
            f"- < 50 pc in the same cones: {len(near)} -> {len(near) / area:.0f} all-sky (survey: "
            f"4,681 targets; scaling check); their projected recovery {rec_n.mean():.3f} -> "
            f"{rec_n.sum() / area:.0f} star-equivalents, {xn * eta[0]:.2f}-{xn * eta[1]:.2f} "
            f"planets (survey model above: compare A_survey)\n"
            f"- 50-100 pc: {len(far)} in cones -> {len(far) / area:.0f} all-sky; Tmag median "
            f"{far['Tmag'].median():.2f}; mean predicted sectors {far['n_sectors'].mean():.1f}; "
            f"mean headline recovery {rec.mean():.3f} -> {rec.sum() / area:.0f} star-equivalents; "
            f"mean transit probability {ptr(far).mean():.4f}; expected headline-cell detections "
            f"{x * eta[0]:.2f}-{x * eta[1]:.2f}\n"
            f"- 50-100 pc recovery by Tmag: " + "; ".join(
                f"T {lo}-{hi}: n {int(m.sum())}, rec {rec[m].mean():.3f}" for lo, hi in
                ((0, 12), (12, 13), (13, 14), (14, 15))
                for m in [((far["Tmag"] >= lo) & (far["Tmag"] < hi)).to_numpy()] if m.sum())
            + calib)


if __name__ == "__main__":
    main()
