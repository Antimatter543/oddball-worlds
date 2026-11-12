"""Steps 1-2 of the v2 survey vetting procedure for a TIC list: rebuild each star's survey NPZ
(`get_lightcurve`) and fetch its SAP + background (`vetting/fetch_sap.fetch_sap`), one star at a
time, in ONE process.

Importing `download` first replaces `fetch._get` with `download.polite_get` (the survey's
transport: the identical file from the public STScI bucket, MAST fallback, anonymous
User-Agent), so `fetch_sap` reads the same bytes as before, faster. Nothing in the conversion
changes. Each rebuilt NPZ's sector list is compared with the survey record's `sectors_found`; a
mismatch is printed and logged (the vetting would then see different data than the search).

Usage: VET_LC_DIR is set here to DATA/lc_cache/lc.
  python -I survey/prep_vet.py OUT/survey/vet/tics_long.txt [--max-mb 2000]
  -> OUT/survey/vet/prep.jsonl (one line per star)
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "vetting"))
sys.path.insert(0, str(HERE))

import download  # noqa: F401  (patches fetch._get before fetch_sap imports it)
from common import DATA, OUT

os.environ.setdefault("VET_LC_DIR", str(DATA / "lc_cache" / "lc"))


def main() -> None:
    from fetch_sap import fetch_sap
    from get_lightcurve import get_lightcurve

    ap = argparse.ArgumentParser()
    ap.add_argument("tics")
    ap.add_argument("--max-mb", type=float, default=2000)
    ap.add_argument("--merged", default=str(OUT / "survey" / "merged" / "results.jsonl.gz"))
    a = ap.parse_args()
    tics = [int(x) for x in Path(a.tics).read_text().split()]
    want = set(tics)
    found = {}
    with gzip.open(a.merged, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r["tic"] in want:
                found[r["tic"]] = sorted(int(s) for s in r.get("sectors_found", []))
    logp = OUT / "survey" / "vet" / "prep.jsonl"
    done = {json.loads(x)["tic"] for x in logp.read_text().splitlines()} if logp.exists() else set()
    with open(logp, "a") as log:
        for i, t in enumerate(tics):
            if t in done:
                continue
            t0 = time.time()
            rec = {"tic": t}
            try:
                _, meta = get_lightcurve(t, int(a.max_mb * 1e6))
                secs = sorted(int(s["sector"]) for s in meta["sectors"])
                rec.update(n_sectors=len(secs), same_sectors=secs == found.get(t),
                           lc_failures=len(meta["failures"]))
                if secs != found.get(t):
                    rec["sectors"] = secs
                    rec["survey_sectors"] = found.get(t)
            except Exception as e:  # noqa: BLE001
                rec["lc_error"] = repr(e)[:300]
            if "lc_error" not in rec:
                try:
                    rec["sap"] = fetch_sap(t, threads=2)
                except Exception as e:  # noqa: BLE001
                    rec["sap"] = "ERROR " + repr(e)[:200]
            rec["wall_s"] = round(time.time() - t0, 1)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(i, json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
