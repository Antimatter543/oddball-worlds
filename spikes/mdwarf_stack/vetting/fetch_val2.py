"""Vetting v2: download the fresh validation stars (PDCSAP NPZ + SAP NPZ from ONE read per sector).

Same product rule and conversion as the survey (survey/download.py: MAST observation query for the
product list, bytes from the public STScI bucket with MAST fallback, polite and anonymous), and the
same per-sector records as fetch.fetch (PDCSAP) and vetting/fetch_sap.py (SAP + background), so
gate2/run2.py and vetting/run_vet.py read them unchanged:
  PDCSAP -> data/planets_spike/lc/tic<T>.npz      SAP -> outputs/planets_spike/vetting/sap/tic<T>.npz
Every file written is appended to <OUT>/vetting/v2/created_files.txt; `clean` deletes exactly those.
Validation stars are disjoint from every gate-2 star, so nothing existing is overwritten (checked).

Usage: python -I vetting/fetch_val2.py fetch | clean
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "survey"))
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))
from common import BIN_MIN, DATA, LC_DIR, OUT

V2 = OUT / "vetting" / "v2"
SAP_DIR = OUT / "vetting" / "sap"
CREATED = V2 / "created_files.txt"
STATUS = V2 / "fetch_status.jsonl"


def _one(tic, s, kind, uri):
    import download as D

    from oddball.tess.lightcurve import convert_fits_bytes

    data = D.polite_get(uri)
    try:
        meta, b = convert_fits_bytes(data, bin_minutes=BIN_MIN)
    finally:
        del data
    if int(meta["tic"]) != tic:
        raise ValueError(f"TIC mismatch {meta['tic']} != {tic}")
    good = b.npts > 0
    idx = (b.bin0 + np.arange(b.nbins))[good]
    rec = {"idx": idx.astype(np.int64), "flux": b.flux[good].astype(np.float32),
           "err": b.flux_err[good].astype(np.float32),
           "q": b.extras["qflags"][good].astype(np.int32),
           "sector": np.full(good.sum(), s, np.int16)}
    m = {k: meta.get(k) for k in ("sector", "camera", "ccd", "tmag", "teff", "radius",
                                   "crowdsap", "procver", "data_rel", "cadence_s", "rms_bin")}
    m.update(product=kind, uri=uri,
             downloaded_utc=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
             source=D.pop_source(uri))
    sap = b.extras["sap_flux"]
    bkg = b.extras["bkg"]
    gs = np.isfinite(sap)
    sidx = (b.bin0 + np.arange(b.nbins))[gs]
    srec = {"idx": sidx.astype(np.int64), "sap": sap[gs].astype(np.float32),
            "bkg": bkg[gs].astype(np.float32), "sector": np.full(gs.sum(), s, np.int16)}
    sm = {"sector": s, "crowdsap": meta.get("crowdsap"), "flfrcsap": meta.get("flfrcsap"),
          "uri": uri}
    return rec, m, srec, sm


def _save(path: Path, recs, prov):
    cat = {k: np.concatenate([r[k] for r in recs]) for k in recs[0]}
    o = np.argsort(cat["idx"], kind="stable")
    cat = {k: v[o] for k, v in cat.items()}
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(prov, default=str)), **cat)
    tmp.replace(path)
    with open(CREATED, "a") as fh:
        fh.write(str(path) + "\n")


def fetch_one(dl, tic, pool):
    lcp, sp = LC_DIR / f"tic{tic}.npz", SAP_DIR / f"tic{tic}.npz"
    created = set(CREATED.read_text().split()) if CREATED.exists() else set()
    for p in (lcp, sp):
        if p.exists() and str(p) not in created:
            raise SystemExit(f"{p} exists and was not made by this script: refusing to touch it")
    if lcp.exists() and sp.exists():
        return "ok", "cached"
    prods, _ = dl.products(tic)
    if not prods:
        return "no_data", "no SPOC 2-min or TESS-SPOC product"
    futs = {s: pool.submit(_one, tic, s, k, u) for s, (k, u) in prods.items()}
    recs, metas, srecs, smetas, fails = [], [], [], [], []
    for s in sorted(futs):
        try:
            r, m, sr, sm = futs[s].result()
            recs.append(r)
            metas.append(m)
            if len(sr["idx"]):
                srecs.append(sr)
                smetas.append(sm)
        except Exception as e:  # noqa: BLE001
            fails.append({"sector": s, "error": repr(e)[:200]})
    if not recs:
        return "no_data", f"every sector failed: {fails[:2]}"
    now = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    _save(lcp, recs, {"tic": tic, "downloaded_utc": now, "bin_minutes": BIN_MIN,
                      "sectors": metas, "failures": fails})
    if srecs:
        SAP_DIR.mkdir(parents=True, exist_ok=True)
        _save(sp, srecs, {"tic": tic, "downloaded_utc": now, "sectors": smetas,
                          "failures": fails})
    return "ok", f"{len(recs)} sectors, {len(fails)} failed"


def fetch():
    import pandas as pd

    import download as D

    V2.mkdir(parents=True, exist_ok=True)
    log = V2 / "fetch_errors.jsonl"

    def errlog(r):
        with open(log, "a") as fh:
            fh.write(json.dumps(r, default=str) + "\n")

    dl = D.Downloader(DATA / "v2val", errlog)
    done = {}
    if STATUS.exists():
        for x in open(STATUS):
            if x.strip():
                r = json.loads(x)
                done[int(r["tic"])] = r["status"]
    s = pd.read_csv(V2 / "val_stars.csv")
    pool = ThreadPoolExecutor(4, thread_name_prefix="v2dl")
    t0 = time.time()
    n = 0
    for b, g in s.groupby("bucket", sort=False):
        quota = int(g["quota"].iloc[0])
        have = 0
        for tic in g.sort_values("rank")["tic"].astype(int):
            if have >= quota:
                break
            st = done.get(tic)
            if st is None:
                try:
                    st, why = fetch_one(dl, tic, pool)
                except SystemExit:
                    raise
                except Exception as e:  # noqa: BLE001
                    st, why = "error", repr(e)[:200]
                with open(STATUS, "a") as fh:
                    fh.write(json.dumps({"tic": tic, "status": st, "why": why}) + "\n")
                n += 1
                if n % 10 == 0:
                    print(f"{n} stars fetched, {time.time() - t0:.0f}s", flush=True)
            have += st == "ok"
        print(f"bucket {b}: {have} of {quota} with data", flush=True)
    pool.shutdown()


def clean():
    if not CREATED.exists():
        print("nothing to clean")
        return
    n = 0
    for p in sorted(set(CREATED.read_text().split())):
        q = Path(p)
        if q.exists():
            q.unlink()
            n += 1
    import shutil

    shutil.rmtree(DATA / "v2val", ignore_errors=True)
    print(f"deleted {n} files and {DATA / 'v2val'}")


if __name__ == "__main__":
    {"fetch": fetch, "clean": clean}[sys.argv[1]]()
