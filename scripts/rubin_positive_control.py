"""Real-data positive control for the outburst detector.

Runs assemble -> detect on well-sampled LSST objects that ALeRCE's stamp classifier calls SN
(sustained multi-night brightenings over a template). The young-star backtest returning
zero candidates only means something if the detector demonstrably fires on real Rubin
light curves.

    uv run python -I scripts/rubin_positive_control.py [--n 25]

Outputs: outputs/rubin/positive_control/results.csv and plots/ for flagged objects.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import pandas as pd

from oddball.rubin.alerce_api import AlerceLSST
from oddball.rubin.detect import detect_object
from oddball.rubin.lightcurve import assemble
from oddball.rubin.plots import plot_candidate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--min-ndet", type=int, default=12)
    args = ap.parse_args()
    out = Path("outputs/rubin/positive_control")
    api = AlerceLSST()
    d = api.list_objects(page_size=200, class_name="SN", classifier="stamp_classifier_rubin_beta",
                         n_det=[args.min_ndet, 100000], probability=0.5, order_by="n_det",
                         order_mode="DESC")
    objs = pd.DataFrame(d["items"]).drop_duplicates("oid").head(args.n)
    rows = []
    for o in objs.itertuples():
        lc = assemble(api.detections(o.oid), api.forced_photometry(o.oid))
        res = detect_object(lc)
        s = res.summary()
        s.update({"oid": o.oid, "meanra": o.meanra, "meandec": o.meandec, "n_lc": len(lc),
                  "name": f"LSST {o.oid}", "plot": ""})
        if res.flagged:
            p = res.primary
            s["plot"] = str(plot_candidate(
                lc, res, f"positive control (ALeRCE SN class) LSST {o.oid}\n"
                f"peak d={p.peak_dmag:.2f} ({p.band}) {s['flags']}",
                out / "plots" / f"{o.oid}.png"))
        rows.append(s)
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results.csv", index=False)
    print(collections.Counter(zip(df["flagged"], df["reasons"].str.split(":").str[0])))
    print(df[df["flagged"]][["oid", "meanra", "meandec", "p_band", "p_peak_dmag", "flags",
                             "plot"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
