"""Paths and grading for the 50-100 pc survey's vetting (frozen v2, unchanged).

Everything lives under <OUT>/survey100/vet/, apart from the 50 pc survey's files:
  vet2_survey100.s{0,1}of2.jsonl  v2 measurement rows (gate3/vet100.py)
  pixel.jsonl                     pixel check results (gate3/pixel3.py with PIXEL3_OUT)
  star_info.json                  catalogue / Gaia info (gate3/catalog3.py)
Grading is cards2.graded2's: vet2.grade2 with frozen_vet2.json at tau_det = tau_long, pixel merged.
`patch()` points the shared vetting modules (cards, cards2, catalog) at these files, so the
frozen card and grading code runs unchanged on them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "gate2"))
sys.path.insert(0, str(HERE.parent / "vetting"))
sys.path.insert(0, str(HERE.parents[2] / "src"))
from common import OUT

S100 = OUT / "survey100"
VET = S100 / "vet"
STAGE = "survey100"
TARGETS = OUT / "gate3" / "targets_100pc.csv"


def patch():
    import cards
    import cards2
    import catalog

    cards.V = VET          # load_pixels -> VET / pixel.jsonl
    cards2.V2 = VET        # graded2 -> VET / vet2_survey100*.jsonl ; cards -> VET / cards / survey100
    catalog.INFO = VET / "star_info.json"


def graded():
    """Every measured 100 pc signal, graded (dict rows with 'grade'), one per key."""
    patch()
    from cards2 import graded2

    seen, out = set(), []
    for r in graded2(STAGE):
        if r["key"] not in seen:
            seen.add(r["key"])
            out.append(r)
    return out


def errors():
    out = []
    for p in sorted(VET.glob(f"vet2_{STAGE}*.jsonl")):
        out += [json.loads(x) for x in open(p) if x.strip() and '"error"' in x]
    return [e for e in out if "stats" not in e]


if __name__ == "__main__":
    # survey100/common100.py [--ab-without-pixel]: list keys (tier) for the pixel check
    from collections import Counter

    rows = graded()
    pix = {json.loads(x)["key"] for x in open(VET / "pixel.jsonl")} if (VET / "pixel.jsonl").exists() else set()
    print("tiers", dict(Counter(r["grade"]["tier"] for r in rows)), "errors", len(errors()),
          file=sys.stderr)
    if "--ab-without-pixel" in sys.argv:
        for r in rows:
            if r["grade"]["tier"] in ("A+", "A", "B") and r["key"] not in pix:
                print(r["key"])
