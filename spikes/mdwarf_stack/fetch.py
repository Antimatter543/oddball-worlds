"""Fetch every TESS sector of one star, bin to 10 min, store one compact NPZ per star.

Product choice per sector (fixed before any search): SPOC 2-min PDCSAP (`*-s_lc.fits`) if it
exists, else the TESS-SPOC FFI HLSP light curve. 20-s data are skipped (the 2-min file always
exists alongside). QLP is not used (different flux definition); sectors with neither product are
simply absent. FITS bytes are converted in memory and never written to disk.

Usage: uv run python -I spikes/mdwarf_stack/fetch.py TIC [TIC ...]
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common import BIN_MIN, LC_DIR, USER_AGENT

from oddball.tess.lightcurve import convert_fits_bytes

DL = "https://mast.stsci.edu/api/v0.1/Download/file?uri="
_SEC = re.compile(r"-s(\d{4})")


def list_products(tic: int) -> dict[int, tuple[str, str]]:
    """sector -> (product, dataURI), applying the SPOC-2min > TESS-SPOC preference."""
    from astroquery.mast import Observations

    obs = Observations.query_criteria(
        target_name=[str(tic)], obs_collection=["TESS", "HLSP"], dataproduct_type="timeseries"
    )
    if len(obs) == 0:
        return {}
    keep = [
        i for i, (c, p) in enumerate(zip(obs["obs_collection"], obs["provenance_name"]))
        if c == "TESS" or str(p).upper() == "TESS-SPOC"
    ]
    if not keep:
        return {}
    prods = Observations.get_product_list(obs[keep])
    best: dict[int, tuple[str, str]] = {}
    for uri in prods["dataURI"]:
        uri = str(uri)
        if uri.endswith("-s_lc.fits") and uri.startswith("mast:TESS/product/"):
            kind = "spoc120"
        elif "hlsp_tess-spoc" in uri and uri.endswith("_lc.fits"):
            kind = "tess-spoc"
        else:
            continue
        m = _SEC.search(uri.split("/")[-1])
        if not m:
            continue
        s = int(m.group(1))
        if s not in best or (best[s][0] == "tess-spoc" and kind == "spoc120"):
            best[s] = (kind, uri)
    return dict(sorted(best.items()))


def _get(uri: str, tries: int = 4) -> bytes:
    for k in range(tries):
        try:
            r = requests.get(DL + uri, headers={"User-Agent": USER_AGENT}, timeout=120)
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if k == tries - 1:
                raise
            time.sleep(2 * 2**k)
    raise RuntimeError("unreachable")


def _one(tic: int, sector: int, kind: str, uri: str):
    data = _get(uri)
    try:
        meta, b = convert_fits_bytes(data, bin_minutes=BIN_MIN)
    finally:
        del data
    if int(meta["tic"]) != tic:
        raise ValueError(f"TIC mismatch {meta['tic']} != {tic}")
    good = b.npts > 0
    idx = (b.bin0 + np.arange(b.nbins))[good]
    rec = {
        "idx": idx.astype(np.int64),
        "flux": b.flux[good].astype(np.float32),
        "err": b.flux_err[good].astype(np.float32),
        "q": b.extras["qflags"][good].astype(np.int32),
        "sector": np.full(good.sum(), sector, np.int16),
    }
    m = {k: meta.get(k) for k in ("sector", "camera", "ccd", "tmag", "teff", "radius",
                                   "crowdsap", "procver", "data_rel", "cadence_s", "rms_bin")}
    m.update(product=kind, uri=uri)
    return rec, m


def fetch(tic: int, force: bool = False, workers: int = 4) -> Path | None:
    out = LC_DIR / f"tic{tic}.npz"
    if out.exists() and not force:
        return out
    prods = list_products(tic)
    if not prods:
        return None
    recs, metas, fails = [], [], []
    with ThreadPoolExecutor(workers) as ex:
        futs = {s: ex.submit(_one, tic, s, k, u) for s, (k, u) in prods.items()}
        for s, f in futs.items():
            try:
                r, m = f.result()
                recs.append(r)
                metas.append(m)
            except Exception as e:  # noqa: BLE001 - logged per sector
                fails.append({"sector": s, "error": repr(e)[:200]})
    if not recs:
        return None
    cat = {k: np.concatenate([r[k] for r in recs]) for k in recs[0]}
    o = np.argsort(cat["idx"], kind="stable")
    cat = {k: v[o] for k, v in cat.items()}
    prov = {"tic": tic, "downloaded_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "bin_minutes": BIN_MIN, "sectors": metas, "failures": fails}
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(prov, default=str)), **cat)
    tmp.rename(out)
    return out


def load(tic: int) -> tuple[dict, dict]:
    z = np.load(LC_DIR / f"tic{tic}.npz", allow_pickle=False)
    d = {k: z[k] for k in z.files if k != "meta"}
    return d, json.loads(str(z["meta"]))


if __name__ == "__main__":
    for a in sys.argv[1:]:
        t0 = time.time()
        try:
            p = fetch(int(a))
        except Exception as e:  # noqa: BLE001
            print(a, "ERROR", repr(e)[:200], flush=True)
            continue
        if p is None:
            print(a, "no products", flush=True)
            continue
        _, m = load(int(a))
        secs = [s["sector"] for s in m["sectors"]]
        kinds = sorted({s["product"] for s in m["sectors"]})
        print(a, len(secs), secs, kinds, "fails", len(m["failures"]),
              f"{time.time() - t0:.0f}s", flush=True)
