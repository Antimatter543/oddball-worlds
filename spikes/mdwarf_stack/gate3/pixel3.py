"""Pixel check (frozen vetting/pixel_check.pixel_check, unchanged) for gate-3 real signals,
with star info from <OUT>/gate3/star_info.json and results in <OUT>/gate3/pixel.jsonl.
Usage: python -I gate3/pixel3.py VET_ROWS.jsonl KEY [KEY ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "vetting"))
sys.path.insert(0, str(HERE.parents[2] / "src"))
from common import OUT

G3 = OUT / "gate3"


def main():
    from pixel_check import pixel_check

    info = json.loads((G3 / "star_info.json").read_text())
    rows = {}
    for f in sys.argv[1].split(","):
        for x in open(f):
            if x.strip():
                r = json.loads(x)
                rows[r["key"]] = r
    outp = G3 / "pixel.jsonl"
    done = {json.loads(x)["key"] for x in open(outp)} if outp.exists() else set()
    with open(outp, "a") as fh:
        for k in sys.argv[2:]:
            if k in done:
                continue
            r = rows[k]
            t0 = time.time()
            try:
                res = pixel_check(r["stats"], info.get(str(r["tic"])))
            except Exception as e:  # noqa: BLE001
                res = {"error": repr(e)[:200]}
            fh.write(json.dumps({"key": k, "tic": r["tic"], "pixel": res}, default=float) + "\n")
            fh.flush()
            print(k, res.get("status", "error"), res.get("why", res.get("error")),
                  f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
