"""Evaluate scorers from their score files (docs/ML_DESIGN.md §6).

    ODDBALL_DATA=... uv run python -I scripts/evaluate.py --scorer classical
    ODDBALL_DATA=... uv run python -I scripts/evaluate.py --compare classical classical_cm my_encoder

Single mode writes ``outputs/eval/<scorer>/sNNNN/report.md`` + CSVs + plots + summary.json.
Compare mode writes ``outputs/eval/compare/sNNNN/<a>__<b>.../report.md`` and a plot.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oddball.common.paths import outputs_root
from oddball.eval.harness import load
from oddball.eval.report import write_compare, write_report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sector", type=int, default=32)
    ap.add_argument("--inj", default="v1")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scorer")
    g.add_argument("--compare", nargs="+")
    ap.add_argument("--split", default="test", help="split for compare mode headline")
    a = ap.parse_args()
    cmd = "python scripts/evaluate.py " + " ".join(shlex.quote(x) for x in sys.argv[1:])

    if a.scorer:
        ed = load(a.scorer, a.sector, a.inj)
        out = outputs_root() / "eval" / a.scorer / f"s{a.sector:04d}"
        s = write_report(ed, out, cmd)
        print(json.dumps({"gate": s["gate"]["test"], "null_fpr": s["null_fpr"]["test"]},
                         indent=1, default=float))  # fmt: skip
        print(f"report: {out / 'report.md'}")
    else:
        eds = [load(n, a.sector, a.inj) for n in a.compare]
        out = outputs_root() / "eval" / "compare" / f"s{a.sector:04d}" / "__".join(a.compare)
        write_compare(eds, out, cmd, split=a.split)
        print(f"report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
