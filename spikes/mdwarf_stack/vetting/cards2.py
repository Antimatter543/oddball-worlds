"""Candidate cards graded with vetting v2: one PNG + plain-English summary per signal.

Reads v2/vet2_<stage>.jsonl (run_vet2.py), merges pixel results (vetting/pixel.jsonl), grades with
frozen_vet2.json (tau_det = tau_long), and writes <OUT>/vetting/v2/cards/<stage>/ with cards.md.
Each summary quotes the measured false-alarm odds for its own tier and transit count
(v2/analyze_val.json, the fresh validation set).

Usage: python -I vetting/cards2.py STAGE [--tiers A+,A,B,K] [--keys K1 ...] [--max 40] [--text-only]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "gate2"))
sys.path.insert(0, str(HERE))
from common import OUT

V2 = OUT / "vetting" / "v2"
TAU = json.loads((OUT / "gate2" / "frozen.json").read_text())["tau_long"]
ORDER = ("K", "A+", "A", "B", "C")


def graded2(stage):
    from cards import load_pixels
    from vet2 import grade2, load_frozen2

    thr = load_frozen2()
    pix = load_pixels()
    rows = []
    for p in sorted(V2.glob(f"vet2_{stage}*.jsonl")):
        rows += [json.loads(x) for x in open(p) if x.strip()]
    rows = [r for r in rows if "stats" in r]
    for r in rows:
        if r["key"] in pix:
            r["stats"]["pixel"] = pix[r["key"]]
        r["grade"] = grade2(r["stats"], thr, TAU)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--tiers", default="A+,A,B,K")
    ap.add_argument("--keys", nargs="*")
    ap.add_argument("--max", type=int, default=40)
    ap.add_argument("--text-only", action="store_true")
    a = ap.parse_args()
    from report2 import load_odds, summary2

    odds = load_odds()
    rows = graded2(a.stage)
    if a.keys:
        rows = [r for r in rows if r["key"] in set(a.keys)]
    else:
        rows = [r for r in rows if r["grade"]["tier"] in a.tiers.split(",")]
    rows.sort(key=lambda r: (ORDER.index(r["grade"]["tier"]), -r["stats"]["mes"]))
    rows = rows[: a.max]
    d = V2 / "cards" / a.stage
    d.mkdir(parents=True, exist_ok=True)
    md = [f"# Vetting v2 cards: {a.stage} ({len(rows)} signals)", ""]
    for r in rows:
        name = f"TIC {r['tic']}" + (" (injection)" if r.get("inj") else "") + \
               (f" ({r['mode']} null)" if r["mode"] != "real" else "")
        md += [f"## {r['key']}  tier {r['grade']['tier']}", ""]
        if not a.text_only:
            from report import render
            from run_vet import _load_star
            from vet import base_series, make_ctx

            lc, meta, sap, crowd, info = _load_star(r["tic"])
            base = base_series(lc, sap, crowd, inj=r.get("inj"), mass=r["mass"],
                               radius=r["radius"])
            ctx = make_ctx(base, r["tic"], r["mass"], r["radius"], r["mode"], r["seed"], crowd,
                           info)
            png = d / f"{r['key'].replace('#', '_')}.png"
            g1 = dict(r["grade"], tier="A" if r["grade"]["tier"] == "A+" else r["grade"]["tier"])
            render(ctx, r["stats"], g1, png, title=f"{name} {r['key']} (v2 tier "
                                                   f"{r['grade']['tier']})")
            md += [f"![]({png.name})", ""]
        md += [summary2(r["stats"], r["grade"], rstar=r["radius"], name=name, odds=odds), ""]
        print(r["key"], r["grade"]["tier"], flush=True)
    (d / "cards.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
