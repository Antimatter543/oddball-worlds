"""Cards for the 100 pc survey: the frozen vetting/cards2.py, pointed at <OUT>/survey100/vet/.

Only the batch's stars are drawn (their light curves must still be on disk). cards2 writes one
cards.md per call, so each batch's summaries go to cards_<batch>.md and are joined at the end.

Usage: VET_LC_DIR=... python -I survey100/cards100.py --batch BATCH_FILE [--tiers A+,A,B,K]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common100 import STAGE, VET, graded, patch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--tiers", default="A+,A,B,K")
    a = ap.parse_args()
    patch()
    import cards2

    tics = {int(x) for x in open(a.batch).read().split()}
    keys = [r["key"] for r in graded() if r["tic"] in tics and r["grade"]["tier"] in a.tiers.split(",")]
    d = VET / "cards" / STAGE
    if not keys:
        print("no cards in this batch")
        return
    sys.argv = ["cards2.py", STAGE, "--keys", *keys, "--max", "100000"]
    cards2.main()
    (d / "cards.md").rename(d / f"cards_{Path(a.batch).name}.md")


if __name__ == "__main__":
    main()
