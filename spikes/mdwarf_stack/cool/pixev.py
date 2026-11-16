"""Pixel check (vetting/pixel_check.pixel_check, unchanged) at explicit event times.

Usage: python -I cool/pixev.py TIC --times t1 t2 ... --width-h W [--tag NAME]
Writes <OUT>/cool/pixel/<tic>[_tag].json. The sector of each time is taken from the rebuilt light curve.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "survey"))
sys.path.insert(0, str(HERE.parent / "vetting"))
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))
from common import OUT
from deep import star_block
from ebscan import detrend, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tic", type=int)
    ap.add_argument("--times", type=float, nargs="+", required=True)
    ap.add_argument("--width-h", type=float, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-sectors", type=int, default=4)
    a = ap.parse_args()
    from pixel_check import pixel_check

    t, f, sec, meta = load(a.tic)
    r, s = detrend(t, f)
    ep, ses, secs = [], [], []
    for ti in a.times:
        j = int(np.argmin(np.abs(t - ti)))
        if abs(t[j] - ti) > 0.3:
            print("no data at", ti)
            continue
        m = np.abs(t - ti) <= a.width_h / 48
        ep.append(ti)
        ses.append(float(-r[m].mean() / (s / np.sqrt(max(m.sum(), 1)))))
        secs.append(int(sec[j]))
    stats = {"epochs": ep, "epoch_ses": ses, "epoch_sector": secs, "dur_h": a.width_h}
    res = pixel_check(stats, star_block(a.tic), max_sectors=a.max_sectors)
    d = OUT / "cool" / "pixel"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{a.tic}{'_' + a.tag if a.tag else ''}.json").write_text(json.dumps({"tic": a.tic, "times": ep, "ses": ses,
                                                                          "sectors": secs, "pixel": res}, default=float, indent=1))  # fmt: skip
    print(a.tic, a.tag, res.get("status", "error"), res.get("why", res.get("error")))
    for p in res.get("per_epoch", []):
        print("  ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in p.items()})


if __name__ == "__main__":
    main()
