"""POST-HOC diagnostics on the fresh validation set (after the v2 decision; nothing here is credited).

Nulls and matched injections only (the blinding of vetting/run_vet2.py holds: no real unaltered
search and no unmatched injection signal is read).
  1. The null tail above tau_long on survey stars: per realisation and per star, by mode, transit
     count and real sector count, versus the gate-2 DEV nulls (tau_long was set at 0.5% of those).
  2. Injection detection on survey stars: why far fewer were detected than planned.
  3. A projection with r from the fresh null rate instead of the real-control rate.

Usage: python -I vetting/posthoc_val2.py  -> v2/posthoc_val.txt
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

V2 = OUT / "vetting" / "v2"
TAU = json.loads((OUT / "gate2" / "frozen.json").read_text())["tau_long"]


def rows(p):
    with open(p) as fh:
        return [json.loads(x) for x in fh if x.strip()]


def main():
    out = []
    srch = rows(V2 / "val2.out.jsonl")
    nul = [r for r in srch if not r.get("inj")]
    inj = [r for r in srch if r.get("inj")]
    top = np.array([r["signals"][0]["mes"] if r.get("signals") else 0.0 for r in nul])
    dev = rows(OUT / "gate2" / "dev_nulls_long_harm.out.jsonl")
    dtop = np.array([r["signals"][0]["mes"] if r.get("signals") else 0.0 for r in dev])
    out.append("# POST-HOC (not credited): fresh-set nulls and injections")
    out.append("\n## 1. Null tail on survey stars vs gate-2 DEV controls")
    for lab, m in (("MES >= 6", 6.0), ("MES >= 7", 7.0), ("MES >= tau_long", TAU)):
        k, n = int((top >= m).sum()), len(top)
        kd, nd = int((dtop >= m).sum()), len(dtop)
        out.append(f"- {lab}: fresh {fmt(k, n)} per realisation; DEV {fmt(kd, nd)}")
    hi = [r for r, t in zip(nul, top, strict=True) if t >= TAU]
    stars = {r["tic"] for r in nul}
    out.append(f"- realisations >= tau_long: {len(hi)} on {len({r['tic'] for r in hi})} of "
               f"{len(stars)} stars; by mode {dict(Counter(r['mode'] for r in hi))}; by epoch "
               f"count {dict(sorted(Counter(r['signals'][0]['n_ep'] for r in hi).items()))}")
    ns_all = np.array([r["n_sectors"] for r in nul])
    ns_hi = np.array([r["n_sectors"] for r in hi])
    out.append(f"- real sectors: all nulls median {np.median(ns_all):.0f} (10-90% "
               f"{np.percentile(ns_all, 10):.0f}-{np.percentile(ns_all, 90):.0f}); >= tau_long "
               f"median {np.median(ns_hi) if len(ns_hi) else float('nan'):.0f}")
    for lo, hi_ in ((1, 3), (4, 5), (6, 12), (13, 999)):
        sel = (ns_all >= lo) & (ns_all <= hi_)
        k = int((top[sel] >= TAU).sum())
        out.append(f"  - {lo}-{hi_ if hi_ < 999 else '+'} real sectors: {fmt(k, int(sel.sum()))}")
    out.append("\n## 2. Injection detection on survey stars")
    det = [r for r in inj if r.get("signals") and r["inj_match"][0][0]
           and r["signals"][0]["mes"] >= TAU]
    for tag in ("head", "long"):
        s = [r for r in inj if r["id"].startswith(f"v2inj{tag}")]
        d = [r for r in det if r["id"].startswith(f"v2inj{tag}")]
        out.append(f"- {tag}: detected {fmt(len(d), len(s))}")
    ns_i = np.array([r["n_sectors"] for r in inj])
    for lo, hi_ in ((1, 3), (4, 5), (6, 12), (13, 999)):
        s = [r for r in inj if r["id"].startswith("v2injhead") and lo <= r["n_sectors"] <= hi_]
        d = [r for r in s if r in det]
        out.append(f"  - headline, {lo}-{hi_ if hi_ < 999 else '+'} real sectors: {fmt(len(d), len(s))}")
    out.append(f"- real sectors of the 150 stars: median {np.median(ns_i):.0f}; "
               f"injections with < 2 transits in the data: "
               f"{sum(r['inj'].get('n_tr_data', 0) < 2 for r in inj)} of {len(inj)}")
    out.append("\n## 3. Projection with r from the fresh nulls (per star: any of its 4 nulls >= tau_long)")
    by_star = {}
    for r, t in zip(nul, top, strict=True):
        by_star[r["tic"]] = max(by_star.get(r["tic"], 0.0), t)
    k = sum(v >= TAU for v in by_star.values())
    out.append(f"- stars with a null >= tau_long: {fmt(k, len(by_star))} (4 realisations each, so this "
               f"over-states a 1-realisation rate; per realisation {fmt(len(hi), len(nul))})")
    js = json.loads((V2 / "analyze_val.json").read_text())
    pr = js["projection"]
    s_notc = (pr["not C (any tier)"]["k"] + 0.5) / (pr["not C (any tier)"]["n"] + 1)
    rate = (len(hi) + 0.5) / (len(nul) + 1)
    out.append(f"- with r = per-realisation fresh rate {rate:.3f} and the v2 survival {s_notc:.2f}: "
               f"{4700 * rate * s_notc:.0f} surviving false alarms per 4,700 stars (point "
               f"estimate), vs {pr['not C (any tier)']['mean']:.0f} with r = 1/118")
    t = js["tiers"]["MES >= tau_long"]["g2"]
    n_t = sum(t.values())
    s_t = (n_t - t.get("C", 0) + 0.5) / (n_t + 1)
    out.append(f"- same r, survival from the >= tau_long nulls themselves ({n_t - t.get('C', 0)} of "
               f"{n_t} not C under v2): {4700 * rate * s_t:.0f} per 4,700 stars")
    txt = "\n".join(out) + "\n"
    (V2 / "posthoc_val.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
