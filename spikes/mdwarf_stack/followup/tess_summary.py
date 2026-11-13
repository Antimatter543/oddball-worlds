"""Tables (and one figure per signal) from FU_OUT/tess_epochs.jsonl, with the pre-registered
per-source rules: new-photon epochs (gaps of the survey light curve, unsearched sectors) decide
supported / refuted / no information; detected epochs re-reduced by another pipeline are the
separate artefact test. Usage: python -I followup/tess_summary.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fu

NEW = ("gap_in_searched_sector", "sector_not_searched")


def verdict(outs):
    s = sum(o == "supported" for o in outs)
    r = sum(o == "refuted" for o in outs)
    if s and r:
        return "conflicting (no information)"
    if s:
        return "supported"
    if r:
        return "refuted"
    return "no information"


def fmt(r):
    if r.get("outcome") in ("not checked", "not available"):
        return f"{r['outcome']} ({r.get('why', '')[:70]})"
    if not r.get("n_cov"):
        return f"no data in window ({r.get('cad_min', '?')} min cadence)"
    ap = r.get("at_pred") or {}
    return (f"cover {r['win_cover']:.0%}, exp SES {r['exp_ses']:.1f}, at pred "
            f"{1e3 * ap['depth']:.2f}+-{1e3 * ap['sigma']:.2f} ppt" if ap else
            f"cover {r['win_cover']:.0%}, exp SES {r['exp_ses']:.1f}, pred time uncovered") + \
        (f"; max SES {r['best_ses']:.1f} ({r['best_depth_ppt']:.2f} ppt, "
         f"{24 * (r['best_t'] - r['t_pred']):+.1f} h; thr {r['support_thr']:.1f}); inj "
         f"{sum(v is not None and v >= 3 for v in r['inj_ses'])}/5")


def main():
    rows = [json.loads(x) for x in (fu.FU_OUT / "tess_epochs.jsonl").read_text().splitlines()]
    eph = json.loads((fu.FU_OUT / "ephem.json").read_text())
    L = []
    for key in fu.KEYS:
        rr = [r for r in rows if r["key"] == key]
        e = eph[key]["ephem"]
        L.append(f"\n### {key} (depth {1e3 * e['depth']:.2f} ppt, D {24 * e['D']:.1f} h)")
        L.append("| epoch n | BTJD | sector | kind | product | outcome | evidence |")
        L.append("|---|---|---|---|---|---|---|")
        for r in sorted(rr, key=lambda r: (r["t_pred"], r["product"])):
            kind = "re-reduction" if r["detected"] else (
                "new photons" if r["status"] in NEW else r["status"])
            L.append(f"| {r['n']} | {r['t_pred']:.3f} | {r['sector']} | {kind} | {r['product']} | "
                     f"{r.get('outcome')} | {fmt(r)} |")
        L.append("")
        by_src_new, by_src_re = defaultdict(list), defaultdict(list)
        for r in rr:
            if r["product"] == "survey":
                continue
            (by_src_re if r["detected"] else by_src_new)[r["product"]].append(r)
        for src in sorted(set(by_src_new) | set(by_src_re)):
            nw = by_src_new.get(src, [])
            re_ = by_src_re.get(src, [])
            j = joint([r for r in nw if r.get("at_pred")], e["depth"])
            L.append(f"- {src}: new-photon epochs {len(nw)} -> **{verdict([r['outcome'] for r in nw])}**"
                     f" (usable windows {sum(bool(r.get('n_cov')) for r in nw)}); joint {j}; "
                     f"re-reductions {len(re_)}: "
                     + ", ".join(f"n={r['n']} {r['outcome']}" for r in re_))
    txt = "\n".join(L) + "\n"
    (fu.FU_OUT / "tess_tables.txt").write_text(txt)
    print(txt)
    figures(rows, eph)


def joint(rows, depth):
    if not rows:
        return "none"
    w = np.array([1 / r["at_pred"]["sigma"] ** 2 for r in rows])
    d = np.array([r["at_pred"]["depth"] for r in rows])
    dj, sj = float(np.sum(w * d) / np.sum(w)), float(1 / np.sqrt(np.sum(w)))
    exp, obs = depth / sj, dj / sj
    # the owner's rule: a non-detection refutes only if the injected dip is recovered there
    inj_ok = all(r.get("inj_all_rec") for r in rows)
    out = "refuted" if (exp >= 4 and obs < exp - 3 and inj_ok) else (
        "supported" if (obs >= 3 and abs(dj - depth) < 2 * sj) else "no information")
    return (f"{1e3 * dj:.2f}+-{1e3 * sj:.2f} ppt (exp SES {exp:.1f}, obs {obs:.1f}, injections "
            f"{'recovered' if inj_ok else 'NOT all recovered'}) -> {out}")


def figures(rows, eph):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tess_test import LCD

    figd = fu.FU_OUT / "figures"
    figd.mkdir(exist_ok=True)
    for key in fu.KEYS:
        rr = [r for r in rows if r["key"] == key and r.get("n_cov")]
        if not rr:
            continue
        eps = sorted({(r["n"], r["t_pred"], r["sector"]) for r in rr})
        prods = sorted({r["product"] for r in rr}, key=lambda p: (p != "survey", p))
        e = eph[key]["ephem"]
        fig, axs = plt.subplots(len(eps), len(prods), figsize=(2.6 * len(prods), 1.9 * len(eps)),
                                squeeze=False, sharex=True)
        for i, (n, tp, s) in enumerate(eps):
            for j, p in enumerate(prods):
                ax = axs[i][j]
                f = LCD / f"tic{eph[key]['tic']}_s{s}_{p}.npz"
                r = next((x for x in rr if x["n"] == n and x["product"] == p
                          and x["sector"] == s), None)
                if r is None or not f.exists():
                    ax.axis("off")
                    continue
                z = np.load(f, allow_pickle=False)
                lc = {"idx": z["idx"], "flux": z["flux"].astype(float),
                      "sector": np.full(len(z["idx"]), s, np.int16)}
                pp = fu.prep(lc)
                t = (pp["idx"] + 0.5) * fu.BIN_D - tp
                m = np.abs(t) < 0.6
                ax.plot(24 * t[m], 1e3 * (pp["flux"][m] - 1), ".", ms=1.5, color="0.6")
                tb = np.floor(t[m] * 48) / 48
                if m.sum():
                    u, inv = np.unique(tb, return_inverse=True)
                    ax.plot(24 * (u + 1 / 96), 1e3 * (np.bincount(inv, pp["flux"][m]) /
                                                      np.bincount(inv) - 1), "k-", lw=0.8)
                h = 24 * fu.NSIG_WIN * r["sig_h"] / 24
                ax.axvspan(-h, h, color="tab:orange", alpha=0.15)
                ax.axhline(-1e3 * e["depth"], color="tab:red", lw=0.6, ls="--")
                ax.set_ylim(-4e3 * e["depth"], 3e3 * e["depth"])
                ax.set_title(f"n={n} S{s} {p}: {r['outcome']}", fontsize=7)
                ax.tick_params(labelsize=6)
        for ax in axs[-1]:
            ax.set_xlabel("hours from prediction", fontsize=7)
        fig.suptitle(f"{key}: frozen-detrended flux (ppt), 30-min bins; orange = +-3 sigma "
                     f"window, red = signal depth", fontsize=8)
        fig.tight_layout()
        fig.savefig(figd / f"tess_{key.replace('#', '_')}.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
