"""Append one validated per-star triage row (so eyeballing labels always carry their evidence).

    uv run python -I scripts/triage_append.py --out labels/triage_allsectors_v1.csv --json '{...}'
    uv run python -I scripts/triage_append.py --merge labels/triage_parts --out labels/triage_allsectors_v1.csv

A row is refused unless the class is in ``CLASSES``, the confidence is high/med/low, and the
reason, catalogue evidence, already-known answer and confirm/kill test are filled in. A star
classed ``unexplained`` must also say why each boring explanation fails (``ruled_out``: EB,
systematic, blend). Rows are appended (one ``write`` per row), so the file survives a crash.
``--merge`` concatenates part files (one per vetter) into ``--out``, keeping the last row per TIC.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

FIELDS = ["tic", "sectors_obs", "best_sector", "cls", "subclass", "confidence", "reason",
          "catalogue_evidence", "already_known", "known_where", "confirm_or_kill", "ruled_out",
          "period_d", "depth_pct", "research", "vetter", "card", "utc"]  # fmt: skip
CLASSES = {
    "EB",  # eclipsing / ellipsoidal binary (subclass: detached, contact, ellipsoidal, single_eclipse)
    "rotation",  # starspot modulation
    "pulsator",  # dsct_gdor, rr_lyrae, cepheid, bcep_spb, sdbv, other
    "stochastic",  # massive-star SLF, red-giant granulation, semi-regular / LPV
    "be_outburst",  # Be star / gamma Cas outbursts
    "flare_star",
    "yso_known",  # catalogued young star whose variability is already explained by youth
    "asteroid",
    "scattered_light",
    "momentum_dump",
    "systematic_other",  # PDC / crowding / background artefacts not covered above
    "blend",  # the signal belongs to a neighbour (subclass: the neighbour's class)
    "quiet",  # no convincing astrophysical signal
    "unexplained",  # subclass: dipper, boyajian, single_dip, exocomet, changing_transit, other
}
CONF = {"high", "med", "low"}
REQUIRED = ["tic", "cls", "confidence", "reason", "catalogue_evidence", "already_known",
            "confirm_or_kill", "research", "vetter", "card"]  # fmt: skip


def validate(row: dict) -> list[str]:
    errs = [f"missing {k}" for k in REQUIRED if not str(row.get(k, "")).strip()]
    if row.get("cls") not in CLASSES:
        errs.append(f"class {row.get('cls')!r} not in {sorted(CLASSES)}")
    if row.get("confidence") not in CONF:
        errs.append(f"confidence {row.get('confidence')!r} not in {sorted(CONF)}")
    if str(row.get("already_known", "")).split(":")[0].strip() not in ("yes", "no", "partly"):
        errs.append("already_known must start with yes / no / partly")
    if row.get("research") not in ("full", "visual"):
        errs.append("research must be full or visual")
    if row.get("cls") == "unexplained":
        ro = str(row.get("ruled_out", "")).lower()
        for k in ("eb", "systematic", "blend"):
            if k not in ro:
                errs.append(f"unexplained needs ruled_out to address '{k}'")
    return errs


def append(path: Path, row: dict) -> None:
    row = {k: row.get(k, "") for k in FIELDS} | {"utc": dt.datetime.now(dt.UTC).isoformat(
        timespec="seconds")}  # fmt: skip
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    import io

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS)
    if new:
        w.writeheader()
    w.writerow(row)
    with path.open("a", newline="") as f:
        f.write(buf.getvalue())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json", default=None, help="one row as JSON, or a JSON list of rows")
    ap.add_argument("--merge", type=Path, default=None, help="directory of part CSVs")
    a = ap.parse_args()
    if a.merge:
        parts = sorted(a.merge.glob("*.csv"))
        df = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False) for p in parts])
        bad = [(r["tic"], validate(r)) for r in df.to_dict("records") if validate(r)]
        if bad:
            sys.exit(f"invalid rows: {bad[:5]}")
        df = df.drop_duplicates("tic", keep="last")[FIELDS]
        df.to_csv(a.out, index=False)
        print(f"merged {len(parts)} parts -> {a.out}: {len(df)} stars")
        print(df.cls.value_counts().to_string())
        return
    rows = json.loads(a.json)
    rows = rows if isinstance(rows, list) else [rows]
    for r in rows:
        errs = validate(r)
        if errs:
            sys.exit(f"TIC {r.get('tic')}: refused: {'; '.join(errs)}")
    for r in rows:
        append(a.out, r)
    print(f"appended {len(rows)} row(s) to {a.out}")


if __name__ == "__main__":
    main()
