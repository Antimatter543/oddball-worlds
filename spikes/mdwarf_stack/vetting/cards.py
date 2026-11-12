"""Candidate cards: one-page diagnostic figure + plain-English summary per vetted signal.

Grades with the frozen thresholds (frozen_vet.json), optionally merges pixel results
(pixel.jsonl), and writes <OUT>/vetting/cards/<stage>/<key>.png and a cards.md with every
summary, sorted by tier then MES.

Usage: python -I vetting/cards.py STAGE [--tiers A,B,K] [--keys K1 K2 ...] [--max 40]
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

V = OUT / "vetting"


def load_pixels():
    p = V / "pixel.jsonl"
    if not p.exists():
        return {}
    return {json.loads(x)["key"]: json.loads(x)["pixel"] for x in open(p) if x.strip()}


def graded(stage, thr=None):
    from freeze_vet import load_frozen
    from vet import grade

    thr = thr or load_frozen()
    pix = load_pixels()
    rows = [json.loads(x) for x in open(V / f"vet_{stage}.jsonl") if x.strip()]
    rows = [r for r in rows if "stats" in r]
    for r in rows:
        if r["key"] in pix:
            r["stats"]["pixel"] = pix[r["key"]]
        r["grade"] = grade(r["stats"], thr)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--tiers", default="A,B,K")
    ap.add_argument("--keys", nargs="*")
    ap.add_argument("--max", type=int, default=40)
    a = ap.parse_args()
    from report import render, summary
    from run_vet import _load_star
    from vet import base_series, make_ctx

    rows = graded(a.stage)
    if a.keys:
        rows = [r for r in rows if r["key"] in set(a.keys)]
    else:
        rows = [r for r in rows if r["grade"]["tier"] in a.tiers.split(",")]
    rows.sort(key=lambda r: ("KABC".index(r["grade"]["tier"]), -r["stats"]["mes"]))
    rows = rows[: a.max]
    d = V / "cards" / a.stage
    d.mkdir(parents=True, exist_ok=True)
    md = [f"# Vetting cards: {a.stage} ({len(rows)} signals)", ""]
    for r in rows:
        lc, meta, sap, crowd, info = _load_star(r["tic"])
        base = base_series(lc, sap, crowd, inj=r.get("inj"), mass=r["mass"], radius=r["radius"])
        ctx = make_ctx(base, r["tic"], r["mass"], r["radius"], r["mode"], r["seed"], crowd, info)
        name = f"TIC {r['tic']}" + (" (injection)" if r.get("inj") else "") + \
               (f" ({r['mode']} null)" if r["mode"] != "real" else "")
        png = d / f"{r['key'].replace('#', '_')}.png"
        render(ctx, r["stats"], r["grade"], png, title=f"{name} {r['key']}")
        text = summary(r["stats"], r["grade"], rstar=r["radius"], name=name)
        md += [f"## {r['key']}  tier {r['grade']['tier']}", "", f"![]({png.name})", "", text, ""]
        print(r["key"], r["grade"]["tier"], flush=True)
    (d / "cards.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
