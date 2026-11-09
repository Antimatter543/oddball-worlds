"""Seed stability (ML_DESIGN §6): top-100 overlap of real-star rankings across scorer runs.

    uv run python -I scripts/ml_stability.py --scorers ssl_v2_seed0 ssl_v2_seed1 ssl_v2_seed2

For the combined ``score`` and each sub-score, the top ``n`` real light curves of every pair of
scorers are intersected (same rule as ``ml_quicklook.py``: ``nlargest`` by that column, over
all real stars of the sector). Reports pairwise overlaps and their mean. Writes JSON next to the
first scorer's ML outputs.
"""

from __future__ import annotations

import argparse
import itertools
import json

import pandas as pd

from oddball.common.paths import outputs_root

COLS = ("score", "score_knn", "score_recon", "score_iforest", "score_recon_mean")
BAR = 0.5  # minimum mean top-100 overlap for a ranking to be reported (our reading of §6)


def top_overlap(a: pd.DataFrame, b: pd.DataFrame, col: str, n: int = 100) -> float:
    return len(set(a.nlargest(n, col).tic) & set(b.nlargest(n, col).tic)) / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorers", nargs="+", required=True)
    ap.add_argument("--sector", type=int, default=32)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sdir = f"s{a.sector:04d}"
    files = {s: pd.read_parquet(outputs_root() / "scores" / s / sdir / "real.parquet")
             for s in a.scorers}  # fmt: skip
    res = {"scorers": a.scorers, "n": a.n, "bar": BAR}
    for c in COLS:
        if not all(c in f for f in files.values()):
            continue
        pairs = {f"{x}|{y}": top_overlap(files[x], files[y], c, a.n)
                 for x, y in itertools.combinations(a.scorers, 2)}  # fmt: skip
        mean = sum(pairs.values()) / max(len(pairs), 1)
        res[c] = {"pairs": pairs, "mean": round(mean, 3), "passes": mean >= BAR}
        print(f"{c:18s} mean top-{a.n} overlap {mean:.3f}  " +
              "  ".join(f"{v:.2f}" for v in pairs.values()))  # fmt: skip
    out = outputs_root() / "ml" / (a.out or a.scorers[0]) / f"stability_{sdir}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
