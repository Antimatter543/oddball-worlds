"""Gate 3 target list: TIC v8 M dwarfs at 50-100 pc with the survey's cuts (pre-registered in
docs/spikes/mdwarf_100pc.md section 1).

Route: MAST TIC criteria query (no cones), Teff 2700-3400 K, d 50-100 pc, Tmag -5..15,
objType STAR, in 36 RA slices of 10 deg (each under a 180 s alarm, 3 tries, one more pass at
the end for failures). Duplicates and any gate-1 (< 50 pc) TIC dropped; tess-point sectors
through S107; kept >= 6 predicted sectors.

Writes <OUT>/gate3/tic100_raw.csv (all rows), targets_100pc.csv (the frozen list) and
targets_meta.json. Usage: python -I gate3/targets3.py
"""

from __future__ import annotations

import hashlib
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "survey"))
from common import CAT_DIR, OUT

G3 = OUT / "gate3"
CACHE = G3 / "tic_slices"
LAST = 107
COLS = ["ID", "ra", "dec", "Tmag", "Teff", "rad", "mass", "d", "contratio", "disposition",
        "objType"]


def _alarm(*_):
    raise TimeoutError("slice wall clock 180 s")


def query_slice(Catalogs, lo, hi):
    last = None
    for k in range(3):
        try:
            signal.alarm(180)
            r = Catalogs.query_criteria(catalog="Tic", Teff=[2700, 3400], d=[50, 100],
                                        Tmag=[-5, 15], objType="STAR", ra=[lo, hi])
            signal.alarm(0)
            return r
        except Exception as ex:  # noqa: BLE001
            signal.alarm(0)
            last = ex
            print("retry", lo, repr(ex)[:150], flush=True)
            time.sleep(15 * (k + 1))
    raise RuntimeError(repr(last)[:200])


def main():
    import download

    download._set_astroquery_ua()
    from astroquery.mast import Catalogs

    Catalogs._portal_api_connection.TIMEOUT = 150
    signal.signal(signal.SIGALRM, _alarm)
    CACHE.mkdir(parents=True, exist_ok=True)
    slices = [(10 * i, 10 * (i + 1)) for i in range(36)]
    # "rev": a second (polite: 2 concurrent requests in all) process fetching slices from the
    # other end into the same cache; it only fetches, the normal run assembles the list
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    rev = mode in ("rev", "mid")  # fetch-only helpers
    order = {"rev": slices[::-1], "mid": slices[18:] + slices[:18]}.get(mode, slices)
    failed = []
    for p in (0, 1):
        todo = order if p == 0 else failed
        failed = []
        for lo, hi in todo:
            cf = CACHE / f"ra{lo:03d}.csv"
            if cf.exists():
                continue
            t0 = time.time()
            try:
                r = query_slice(Catalogs, lo, hi)
            except Exception as ex:  # noqa: BLE001
                print("slice FAILED", lo, ex, flush=True)
                failed.append((lo, hi))
                continue
            df = r[[c for c in COLS if c in r.colnames]].to_pandas()
            df.to_csv(cf, index=False)
            print(f"slice {lo}-{hi}: {len(df)} rows, {time.time() - t0:.0f} s", flush=True)
            time.sleep(1.0)
    if rev:
        return
    ok =[lo for lo, _ in slices if (CACHE / f"ra{lo:03d}.csv").exists()]
    raw = pd.concat([pd.read_csv(CACHE / f"ra{lo:03d}.csv") for lo in ok], ignore_index=True)
    n_raw = len(raw)
    raw = raw.drop_duplicates("ID")
    near = set(pd.read_csv(CAT_DIR / "tic_mdwarfs_50pc.csv")["ID"].astype(int))
    n_dup = n_raw - len(raw)
    raw = raw[~raw["ID"].isin(near) & (raw["d"] >= 50)]
    raw.to_csv(G3 / "tic100_raw.csv", index=False)
    from select_controls import sectors_for

    secs = sectors_for(raw["ID"].to_numpy(), raw["ra"].to_numpy(), raw["dec"].to_numpy())
    rows = []
    for r in raw.itertuples():
        ss = [x for x in secs.get(int(r.ID), []) if x <= LAST]
        fut = [x for x in secs.get(int(r.ID), []) if x > LAST]
        rows.append({"tic": int(r.ID), "ra": r.ra, "dec": r.dec, "Tmag": r.Tmag, "Teff": r.Teff,
                     "rad": r.rad, "mass": r.mass, "d_pc": r.d, "contratio": r.contratio,
                     "n_sectors": len(ss), "sectors": " ".join(map(str, ss)),
                     "n_future": len(fut)})
    allf = pd.DataFrame(rows)
    allf.to_csv(G3 / "tic100_allsectors.csv", index=False)
    t = allf[allf["n_sectors"] >= 6].sort_values("tic").reset_index(drop=True)
    out = G3 / "targets_100pc.csv"
    t.to_csv(out, index=False)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    meta = {"slices_ok": len(ok), "slices_failed": [lo for lo, _ in slices if lo not in ok],
            "rows_raw": n_raw, "duplicates": n_dup, "after_dedupe_and_50pc_cut": len(raw),
            "with_ge6_sectors": len(t), "sha256": sha, "last_sector": LAST,
            "queried": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    (G3 / "targets_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    t["tb"] = pd.cut(t["Tmag"], [-5, 12, 13, 14, 15], right=False,
                     labels=["<12", "12-13", "13-14", "14-15"])
    t["sb"] = pd.cut(t["n_sectors"], [6, 13, 26, 999], right=False,
                     labels=["6-12", "13-25", "26+"])
    print(pd.crosstab(t["tb"], t["sb"], margins=True).to_string())
    print("all sectors (>= 0):", len(allf), "; >= 2:", int((allf.n_sectors >= 2).sum()),
          "; Tmag median of list", float(np.median(t["Tmag"])))


if __name__ == "__main__":
    main()
