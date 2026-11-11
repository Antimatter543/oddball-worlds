"""Fetch the SAP flux (and SAP background) for the exact products a star's gate-2 NPZ used.

The gate-2 NPZ keeps PDCSAP only. The vetting's SAP-vs-PDCSAP test needs SAP on the same
10-min grid, so each sector's FITS (same MAST URI as in the NPZ provenance) is downloaded again,
converted in memory by the same `convert_fits_bytes`, and only the binned SAP / background are
kept (FITS never touches disk). Output: <OUT>/vetting/sap/tic<T>.npz with idx, sap, bkg, sector.

Usage: python -I vetting/fetch_sap.py dev|test|hosts|survey-list FILE  [--threads 2]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "src"))
from common import BIN_MIN, CAT_DIR, LC_DIR, OUT  # noqa: E402
from fetch import _get, load  # noqa: E402

SAP_DIR = OUT / "vetting" / "sap"


def sap_path(tic: int) -> Path:
    return SAP_DIR / f"tic{tic}.npz"


def _one(tic: int, sector: int, uri: str):
    from oddball.tess.lightcurve import convert_fits_bytes

    data = _get(uri)
    try:
        meta, b = convert_fits_bytes(data, bin_minutes=BIN_MIN)
    finally:
        del data
    if int(meta["tic"]) != tic:
        raise ValueError(f"TIC mismatch {meta['tic']} != {tic}")
    sap = b.extras["sap_flux"]
    bkg = b.extras["bkg"]
    good = np.isfinite(sap)
    idx = (b.bin0 + np.arange(b.nbins))[good]
    return {"idx": idx.astype(np.int64), "sap": sap[good].astype(np.float32),
            "bkg": bkg[good].astype(np.float32),
            "sector": np.full(good.sum(), sector, np.int16)}, {
        "sector": sector, "crowdsap": meta.get("crowdsap"), "flfrcsap": meta.get("flfrcsap"),
        "uri": uri}


def fetch_sap(tic: int, threads: int = 2) -> str:
    out = sap_path(tic)
    if out.exists():
        return "cached"
    if not (LC_DIR / f"tic{tic}.npz").exists():
        return "no-npz"
    _, meta = load(tic)
    recs, metas, fails = [], [], []
    with ThreadPoolExecutor(threads) as ex:
        futs = {s["sector"]: ex.submit(_one, tic, int(s["sector"]), s["uri"])
                for s in meta["sectors"]}
        for s, f in futs.items():
            try:
                r, m = f.result()
                recs.append(r)
                metas.append(m)
            except Exception as e:  # noqa: BLE001 - logged per sector
                fails.append({"sector": s, "error": repr(e)[:200]})
    if not recs:
        return f"failed {fails[:1]}"
    cat = {k: np.concatenate([r[k] for r in recs]) for k in recs[0]}
    o = np.argsort(cat["idx"], kind="stable")
    cat = {k: v[o] for k, v in cat.items()}
    prov = {"tic": tic, "downloaded_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "sectors": metas, "failures": fails}
    SAP_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(prov, default=str)), **cat)
    tmp.rename(out)
    return f"ok {len(recs)} sectors, {len(fails)} failed"


def load_sap(tic: int):
    p = sap_path(tic)
    if not p.exists():
        return None, None
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files if k != "meta"}, json.loads(str(z["meta"]))


def star_list(which: str, file: str | None):
    if which in ("dev", "test"):
        c = pd.read_csv(CAT_DIR / "controls_g2.csv")
        return [int(t) for t in c[c["split"] == which.upper()]["tic"]]
    if which == "hosts":
        from make_jobs import HOSTS

        h = pd.read_csv(CAT_DIR / "holdout_g2.csv")
        h = h[h["in_range"] & h["tic"].notna()]
        return list(HOSTS) + sorted({int(t) for t in h["tic"]})
    return [int(x) for x in open(file).read().split()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="+")
    ap.add_argument("--file")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    for w in a.which:
        for t in star_list(w, a.file):
            t0 = time.time()
            try:
                msg = fetch_sap(t, a.threads)
            except Exception as e:  # noqa: BLE001
                msg = "ERROR " + repr(e)[:200]
            print(w, t, msg, f"{time.time() - t0:.0f}s", flush=True)
