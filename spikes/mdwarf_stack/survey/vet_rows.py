"""Survey records -> the row format the vetting reads (gate2/run2.py output), plus jobs and TICs.

The survey runner stores one record per star with both searches under `search.long` /
`search.short`. `vetting/run_vet2.py survey --rows R --jobs J` expects one run2-style row per
(star, range) with `id`, `tic`, `mode`, `seed`, `rot`, `range`, `max_sig`, `n_sectors`,
`sectors` and `signals`. This only reshapes; no signal is changed or filtered here (the vetting
selects MES >= tau_long itself). The job file carries the mass and radius the search actually
used (the survey record's), so the vetting uses the same stellar parameters.

Usage: python -I survey/vet_rows.py [--merged OUT/survey/merged/results.jsonl.gz]
  -> OUT/survey/vet/rows_long.jsonl, jobs_long.jsonl, tics_long.txt (stars with a long signal
     >= tau_long, the v1 procedure's step-1 list)
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import OUT


def convert(rec: dict, rng: str = "long") -> dict:
    s = rec["search"][rng]
    return {"id": f"survey-{rec['tic']}-{rng}", "tic": int(rec["tic"]), "mode": "real", "seed": 0,
            "rot": "harm", "range": rng, "max_sig": s["max_sig"],
            "n_sectors": rec["n_sectors_found"], "sectors": rec["sectors_found"],
            "signals": s["signals"], "incomplete": bool(rec.get("incomplete"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default=str(OUT / "survey" / "merged" / "results.jsonl.gz"))
    ap.add_argument("--out", default=str(OUT / "survey" / "vet"))
    a = ap.parse_args()
    tau = json.loads((OUT / "gate2" / "frozen.json").read_text())["tau_long"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n_ok = n_sig = 0
    tics = set()
    with gzip.open(a.merged, "rt") as fin, open(out / "rows_long.jsonl", "w") as fr, \
            open(out / "jobs_long.jsonl", "w") as fj:
        for line in fin:
            rec = json.loads(line)
            if rec.get("status") != "ok":
                continue
            n_ok += 1
            row = convert(rec)
            fr.write(json.dumps(row) + "\n")
            fj.write(json.dumps({"id": row["id"], "tic": row["tic"], "mass": rec["mass"],
                                 "radius": rec["radius"], "mr_source": rec.get("mr_source")})
                     + "\n")
            k = sum(sg["mes"] >= tau for sg in row["signals"])
            if k:
                n_sig += k
                tics.add(row["tic"])
    (out / "tics_long.txt").write_text("\n".join(str(t) for t in sorted(tics)) + "\n")
    print(f"{n_ok} searched stars; {n_sig} long signals >= tau_long {tau} on {len(tics)} stars")


if __name__ == "__main__":
    main()
