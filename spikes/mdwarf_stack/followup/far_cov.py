"""Part 2.5 check: for 50-100 pc M dwarfs, how many predicted sectors actually have a SPOC 2-min
or TESS-SPOC light curve at MAST? (The projection otherwise assumes the survey's own
searched / predicted fraction at the same Tmag, which may not hold for more distant stars.)
200 random stars (seed 20261115) from tic100_sample.csv with d 50-100 pc and >= 6 predicted
sectors; batched anonymous Observations queries. Writes FU_OUT/far_cov.csv.
Usage: python -I followup/far_cov.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fu
from gap_query import kind_of

N, B = 200, 40


def main():
    import download

    download._set_astroquery_ua()
    from astroquery.mast import Observations

    t = pd.read_csv(fu.FU_OUT / "tic100_sample.csv")
    far = t[(t["d"] >= 50) & (t["d"] < 100) & (t["n_sectors"] >= 6)]
    far = far.sample(n=min(N, len(far)), random_state=20261115)
    got = {}
    status = {}
    for i in range(0, len(far), B):
        batch = [str(int(x)) for x in far["ID"].iloc[i:i + B]]
        for k in range(3):
            try:
                obs = Observations.query_criteria(target_name=batch,
                                                  dataproduct_type="timeseries")
                break
            except Exception as ex:  # noqa: BLE001
                print("retry", i, repr(ex)[:120], flush=True)
                time.sleep(10 * (k + 1))
        else:
            print("FAILED batch (not checked)", i, flush=True)
            for b in batch:
                status[int(b)] = "not checked"
            continue
        for b in batch:
            status[int(b)] = "ok"
        for c, p, s, u, tn in zip(obs["obs_collection"], obs["provenance_name"],
                                  obs["sequence_number"], obs["dataURL"], obs["target_name"]):
            kd = kind_of(str(c), p, u)
            if kd in ("spoc120", "tess-spoc"):
                try:
                    got.setdefault(int(tn), set()).add(int(s))
                except (TypeError, ValueError):
                    pass
        print("batch", i, len(obs), flush=True)
        time.sleep(0.5)
    rows = []
    for _, r in far.iterrows():
        tic = int(r["ID"])
        pred = {int(x) for x in str(r["sectors"]).split()}
        have = got.get(tic, set()) & pred
        rows.append({"tic": tic, "Tmag": r["Tmag"], "d": r["d"], "n_pred": len(pred),
                     "n_have": len(have) if status.get(tic) == "ok" else np.nan,
                     "status": status.get(tic, "not checked")})
    df = pd.DataFrame(rows)
    df.to_csv(fu.FU_OUT / "far_cov.csv", index=False)
    ok = df[df["status"] == "ok"]
    print(f"checked {len(ok)} of {len(df)}; have / predicted {ok['n_have'].sum() / ok['n_pred'].sum():.3f}"
          f"; no product at all {(ok['n_have'] == 0).sum()}")
    for lo, hi in ((0, 12), (12, 13), (13, 13.5), (13.5, 14), (14, 15)):
        m = (ok["Tmag"] >= lo) & (ok["Tmag"] < hi)
        if m.sum():
            print(f"T {lo}-{hi}: n {int(m.sum())}, have/pred "
                  f"{ok['n_have'][m].sum() / ok['n_pred'][m].sum():.3f}, none "
                  f"{int((ok['n_have'][m] == 0).sum())}")


if __name__ == "__main__":
    main()
