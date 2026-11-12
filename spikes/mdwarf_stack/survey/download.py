"""Survey downloader: every TESS sector of one star, exactly as gate 2 fetched it, politely.

What is reused unchanged from gate 2 (`spikes/mdwarf_stack/fetch.py`):
  * The product rule of `fetch.list_products` (SPOC 2-min > TESS-SPOC per sector), applied by
    `list_products_obs` to the same observation query's `dataURL`s instead of a
    get_product_list call (that MAST service hung for minutes on 2026-11-12).
  * `fetch._one`: FITS bytes -> 10-min binned record (`convert_fits_bytes`), TIC check, and the
    per-sector metadata. FITS bytes stay in memory and are never written to disk.
  * The NPZ layout of `fetch.fetch` (sectors concatenated in sector order, stable sort by bin
    index, `meta` JSON with tic / downloaded_utc / bin_minutes / sectors / failures), so
    `searchjob.load_npz` reads exactly what `fetch.load` read in gate 2.

What is wrapped (behaviour of the conversion is untouched):
  * `fetch._get` is replaced by `polite_get`. ROUTE "s3" (default) reads the same file name
    from the public STScI bucket (s3://stpubdata over anonymous HTTPS: SPOC 2-min for every
    sector, TESS-SPOC HLSP up to s0081) and falls back to MAST when the bucket lacks it.
    MAST requests: anonymous User-Agent (`oddball.common.http`), a process-wide semaphore of
    MAX_CONCURRENT (2), bounded retries with exponential backoff + jitter (Retry-After
    honoured), no retry on permanent 4xx, error classification, and a circuit breaker that
    pauses all MAST requests after a run of exhausted retries. Gate 2 used 4 threads per star
    and 4 tries with 2-16 s backoff. Each sector's source ("s3"/"mast") is recorded.
  * The astroquery product query runs under the same semaphore with the anonymous User-Agent
    and a socket timeout (astroquery sends timeout=None).
  * Per-sector partial cache (`partial/tic<T>/sNNNN.npz`, write-then-rename), so a killed
    40-sector star does not re-download the sectors it already has.
"""

from __future__ import annotations

import datetime as dt
import json
import random
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "src"))

import fetch  # noqa: E402  (gate-2 module, used unchanged except for _get)
from common import BIN_MIN  # noqa: E402

from oddball.common.http import ANON_USER_AGENT  # noqa: E402

MAX_CONCURRENT = 2  # hard cap on simultaneous MAST requests, process-wide
S3_CONCURRENT = 4  # simultaneous requests to the public AWS bucket (not MAST)
SECTOR_THREADS = 4  # sector download threads per process (MAST share still capped at 2)
S3_BASE = "https://stpubdata.s3.amazonaws.com/"
S3_HLSP_MAX_SECTOR = 81  # mast/hlsp/tess-spoc/ in the bucket stops at s0081 (2026-11-12)
ROUTE = "s3"  # "s3": bytes from the bucket, MAST fallback; "mast": gate-2 transport only
TRIES = 5
BACKOFF_S = (5.0, 15.0, 45.0, 120.0)  # sleeps between the TRIES attempts
TIMEOUT = (30.0, 180.0)  # connect, read
BREAKER_N = 8  # consecutive exhausted requests -> pause everything
BREAKER_PAUSE_S = (600.0, 1200.0, 2400.0, 3600.0)
PRODUCT_CACHE_DAYS = 30  # product lists newer than this are reused without a query

_SEM = threading.BoundedSemaphore(MAX_CONCURRENT)
_S3_SEM = threading.BoundedSemaphore(S3_CONCURRENT)
_tls = threading.local()
_TID = re.compile(r"-s(\d{4})-(\d{16})-")
SOURCE: dict[str, str] = {}  # uri -> "s3" | "mast", read back by Downloader._sector
_SRC_LOCK = threading.Lock()


class DownloadError(Exception):
    def __init__(self, kind: str, msg: str):
        super().__init__(f"{kind}: {msg}")
        self.kind = kind
        self.msg = msg


def classify(e: BaseException) -> str:
    if isinstance(e, DownloadError):
        return e.kind
    if isinstance(e, requests.HTTPError) and e.response is not None:
        c = e.response.status_code
        return "http_429" if c == 429 else f"http_{c // 100}xx"
    if isinstance(e, requests.Timeout):
        return "timeout"
    if isinstance(e, requests.ConnectionError):
        return "connection"
    if isinstance(e, requests.RequestException):
        return "request_other"
    if isinstance(e, ValueError) and "TIC mismatch" in str(e):
        return "tic_mismatch"
    if isinstance(e, (OSError, KeyError, ValueError, TypeError, IndexError)):
        return "convert"
    return "other"


class Breaker:
    """Counts requests that exhausted their retries. After BREAKER_N in a row (no success in
    between) every request waits out a pause, which grows on repeat trips."""

    def __init__(self, log):
        self.lock = threading.Lock()
        self.consec = 0
        self.trips = 0
        self.until = 0.0
        self.log = log

    def wait(self):
        while True:
            with self.lock:
                left = self.until - time.time()
            if left <= 0:
                return
            time.sleep(min(left, 30.0))

    def ok(self):
        with self.lock:
            self.consec = 0
            self.trips = 0

    def fail(self):
        with self.lock:
            self.consec += 1
            if self.consec >= BREAKER_N and time.time() >= self.until:
                pause = BREAKER_PAUSE_S[min(self.trips, len(BREAKER_PAUSE_S) - 1)]
                self.trips += 1
                self.consec = 0
                self.until = time.time() + pause
                self.log({"kind": "breaker_pause", "pause_s": pause, "trips": self.trips})


BREAKER: Breaker | None = None
ERRLOG = None  # callable(dict) set by Downloader


def _session() -> requests.Session:
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = ANON_USER_AGENT
        _tls.s = s
    return s


def _sleep_before_retry(k: int, e: BaseException):
    base = BACKOFF_S[min(k, len(BACKOFF_S) - 1)]
    ra = None
    if isinstance(e, requests.HTTPError) and e.response is not None:
        try:
            ra = float(e.response.headers.get("Retry-After", ""))
        except ValueError:
            ra = None
    time.sleep(max(base, min(ra or 0.0, 600.0)) * (1 + 0.25 * random.random()))


def _permanent(e: BaseException) -> bool:
    if isinstance(e, requests.HTTPError) and e.response is not None:
        c = e.response.status_code
        return 400 <= c < 500 and c not in (408, 429)
    return False


class S3Missing(Exception):
    pass


def s3_key(uri: str) -> str | None:
    """MAST data URI -> key in the public STScI bucket (s3://stpubdata), or None.

    SPOC 2-min: mast:TESS/product/<file> -> tess/public/tid/sSSSS/tttt/tttt/tttt/tttt/<file>
    TESS-SPOC:  mast:HLSP/tess-spoc/<path> -> mast/hlsp/tess-spoc/<path> (bucket holds s1-s81)"""
    if uri.startswith("mast:TESS/product/"):
        f = uri.rsplit("/", 1)[-1]
        m = _TID.search(f)
        if not m:
            return None
        s, t = int(m.group(1)), m.group(2)
        return f"tess/public/tid/s{s:04d}/{t[0:4]}/{t[4:8]}/{t[8:12]}/{t[12:16]}/{f}"
    if uri.startswith("mast:HLSP/tess-spoc/"):
        return "mast/hlsp/" + uri[len("mast:HLSP/"):]
    return None


def s3_get(key: str, tries: int = 4) -> bytes:
    last = None
    for k in range(tries):
        try:
            with _S3_SEM:
                r = _session().get(S3_BASE + key, timeout=TIMEOUT)
                if r.status_code in (403, 404):  # the bucket answers 403 for absent keys too
                    raise S3Missing(f"{r.status_code} {key}")
                r.raise_for_status()
                data = r.content
            n = r.headers.get("Content-Length")
            if n is not None and n.isdigit() and int(n) != len(data):
                raise DownloadError("truncated", f"{len(data)} of {n} bytes")
            return data
        except S3Missing:
            raise
        except Exception as e:  # noqa: BLE001
            last = e
            ERRLOG({"kind": "s3_" + classify(e), "key": key, "attempt": k + 1, "final": False,
                    "msg": repr(e)[:200]})
            if k < tries - 1:
                time.sleep(2.0 * 3**k * (1 + 0.25 * random.random()))
    raise last


def s3_list(prefix: str) -> list[tuple[str, int]]:
    """Keys and sizes under a prefix (anonymous ListObjectsV2; one page is enough per star)."""
    import xml.etree.ElementTree as ET

    for k in range(4):
        try:
            with _S3_SEM:
                r = _session().get(S3_BASE, params={"list-type": "2", "prefix": prefix,
                                                    "max-keys": "1000"}, timeout=TIMEOUT)
                r.raise_for_status()
            ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
            root = ET.fromstring(r.content)
            return [(c.find("s:Key", ns).text, int(c.find("s:Size", ns).text))
                    for c in root.findall("s:Contents", ns)]
        except requests.RequestException as e:
            if k == 3:
                raise
            ERRLOG({"kind": "s3_list_" + classify(e), "prefix": prefix, "attempt": k + 1,
                    "final": False, "msg": repr(e)[:200]})
            time.sleep(2.0 * 3**k)
    return []


def list_products_s3(tic: int, sectors) -> tuple[dict[int, tuple[str, str]], list[int]]:
    """The frozen product rule from bucket listings alone, over candidate sectors (tess-point).
    Returns (sector -> (kind, MAST-style uri), sectors it cannot decide: no SPOC 2-min file and
    a sector above the bucket's TESS-SPOC range, so only MAST knows). Used when the MAST
    observation query fails, and by the pilot to compare the two listings."""
    t = f"{tic:016d}"
    split = f"{t[0:4]}/{t[4:8]}/{t[8:12]}/{t[12:16]}/"
    best, undecided = {}, []
    for s in sorted(set(int(x) for x in sectors)):
        keys = [k for k, _ in s3_list(f"tess/public/tid/s{s:04d}/{split}")]
        lc = [k for k in keys if k.endswith("-s_lc.fits")]
        if lc:
            best[s] = ("spoc120", "mast:TESS/product/" + sorted(lc)[-1].rsplit("/", 1)[-1])
            continue
        if s > S3_HLSP_MAX_SECTOR:
            undecided.append(s)
            continue
        keys = [k for k, _ in s3_list(f"mast/hlsp/tess-spoc/s{s:04d}/target/{split}")]
        lc = [k for k in keys if "hlsp_tess-spoc" in k and k.endswith("_lc.fits")]
        if lc:
            best[s] = ("tess-spoc", "mast:HLSP/" + sorted(lc)[-1][len("mast/hlsp/"):])
    return best, undecided


def _note_source(uri: str, src: str):
    with _SRC_LOCK:
        SOURCE[uri] = src


def pop_source(uri: str) -> str | None:
    with _SRC_LOCK:
        return SOURCE.pop(uri, None)


def polite_get(uri: str, tries: int = TRIES) -> bytes:
    """Drop-in for fetch._get: same URI in, same bytes out. ROUTE "s3" reads the identical file
    from the public bucket and falls back to MAST if the bucket lacks it or keeps failing."""
    if ROUTE == "s3":
        key = s3_key(uri)
        if key is not None:
            try:
                data = s3_get(key)
                _note_source(uri, "s3")
                return data
            except S3Missing as e:
                ERRLOG({"kind": "s3_missing", "uri": uri, "final": False, "msg": str(e)[:200]})
            except Exception as e:  # noqa: BLE001
                ERRLOG({"kind": "s3_gave_up", "uri": uri, "final": False, "msg": repr(e)[:200]})
    data = mast_get(uri, tries)
    _note_source(uri, "mast")
    return data


def mast_get(uri: str, tries: int = TRIES) -> bytes:
    """The MAST download route (gate 2's URL), at most MAX_CONCURRENT at once."""
    last = None
    for k in range(tries):
        BREAKER.wait()
        try:
            with _SEM:
                r = _session().get(fetch.DL + uri, timeout=TIMEOUT)
                r.raise_for_status()
                data = r.content
            n = r.headers.get("Content-Length")
            if n is not None and n.isdigit() and int(n) != len(data):
                raise DownloadError("truncated", f"{len(data)} of {n} bytes")
            BREAKER.ok()
            return data
        except Exception as e:  # noqa: BLE001 - classified and logged
            last = e
            ERRLOG({"kind": classify(e), "uri": uri, "attempt": k + 1, "final": False,
                    "msg": repr(e)[:200]})
            if _permanent(e) or k == tries - 1:
                break
            _sleep_before_retry(k, e)
    BREAKER.fail()
    raise last


fetch._get = polite_get  # the one wrapped piece of fetch.py; _one resolves _get at call time


def _set_astroquery_ua():
    """Anonymous User-Agent, and a socket timeout on astroquery's session: its portal requests
    go out with timeout=None, and one dead connection hung a pilot query for > 6 min."""
    from astroquery.mast import Observations

    sessions = {id(s): s for s in (Observations._session,
                                   Observations._portal_api_connection._session)}
    for s in sessions.values():
        s.headers["User-Agent"] = ANON_USER_AGENT
        if getattr(s, "_survey_timeout", False):
            continue
        orig = s.request

        def request(method, url, *args, _orig=orig, **kw):
            if kw.get("timeout") is None:
                kw["timeout"] = TIMEOUT
            return _orig(method, url, *args, **kw)

        s.request = request
        s._survey_timeout = True


def list_products_obs(tic: int) -> dict[int, tuple[str, str]]:
    """fetch.list_products without the get_product_list call.

    Same observation query, same collection/provenance filter and the same URI rule
    (SPOC `-s_lc.fits` under mast:TESS/product/ beats `hlsp_tess-spoc ... _lc.fits`), applied to
    each observation's own `dataURL`, which for TESS timeseries observations is the light-curve
    file itself. Reason: on 2026-11-12 MAST's product-list service (Mast.Caom.Products) hung for
    minutes per call while the observation query answered in 1.5 s. Checked against the URIs
    gate 2 stored in its NPZ metadata for every gate-2 star (pilot.py routecheck).
    If no dataURL matches although there are observations, the gate-2 route is used."""
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
    best: dict[int, tuple[str, str]] = {}
    for uri in obs["dataURL"][keep]:
        uri = str(uri)
        if uri.endswith("-s_lc.fits") and uri.startswith("mast:TESS/product/"):
            kind = "spoc120"
        elif "hlsp_tess-spoc" in uri and uri.endswith("_lc.fits"):
            kind = "tess-spoc"
        else:
            continue
        m = fetch._SEC.search(uri.split("/")[-1])
        if not m:
            continue
        s = int(m.group(1))
        if s not in best or (best[s][0] == "tess-spoc" and kind == "spoc120"):
            best[s] = (kind, uri)
    if not best:
        ERRLOG({"kind": "route_fallback", "tic": tic, "final": False,
                "msg": f"{len(keep)} observations but no dataURL light curve"})
        return fetch.list_products(tic)
    return dict(sorted(best.items()))


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, default=str))
    tmp.replace(path)


class Downloader:
    """Per-star download into data_dir. Thread-safe for one star-loop thread plus its own
    pool of MAX_CONCURRENT sector threads."""

    def __init__(self, data_dir: Path, errlog):
        global BREAKER, ERRLOG
        self.data_dir = data_dir
        self.prod_dir = data_dir / "products"
        self.part_dir = data_dir / "partial"
        self.lc_dir = data_dir / "lc"
        for d in (self.prod_dir, self.part_dir, self.lc_dir):
            d.mkdir(parents=True, exist_ok=True)
        ERRLOG = errlog
        BREAKER = Breaker(errlog)
        self.pool = ThreadPoolExecutor(SECTOR_THREADS, thread_name_prefix="dl")
        _set_astroquery_ua()

    def npz_path(self, tic: int) -> Path:
        return self.lc_dir / f"tic{tic}.npz"

    # ---------------------------------------------------------------- product list (cached)
    def products(self, tic: int, pred_sectors=()) -> tuple[dict[int, tuple[str, str]], str]:
        """Product list: MAST observation query (cached). If MAST keeps failing and there are
        predicted sectors, the bucket listing over them is used instead; its source is then
        "s3_listing" and any sector only MAST could decide is logged."""
        cp = self.prod_dir / f"tic{tic}.json"
        if cp.exists():
            c = json.loads(cp.read_text())
            age = dt.datetime.now(dt.UTC) - dt.datetime.fromisoformat(c["queried_utc"])
            if age.days < PRODUCT_CACHE_DAYS and c["products"]:
                self.list_source = c.get("source", "mast_obs")
                return {int(s): tuple(v) for s, v in c["products"].items()}, c["queried_utc"]
        self.list_source, self.undecided = "mast_obs", []
        try:
            prods = self._query(tic)
        except DownloadError:
            if ROUTE != "s3" or not pred_sectors:
                raise
            prods, self.undecided = list_products_s3(tic, pred_sectors)
            self.list_source = "s3_listing"
            ERRLOG({"kind": "listing_from_s3", "tic": tic, "final": False,
                    "msg": f"MAST query failed; bucket listing, undecided {self.undecided}"})
            q = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
            return prods, q
        if not prods:
            # an empty answer is re-asked once after a pause before it is believed
            time.sleep(20.0)
            prods = self._query(tic)
        q = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        if prods:
            _atomic_json(cp, {"tic": tic, "queried_utc": q, "source": "mast_obs",
                              "products": {str(s): list(v) for s, v in prods.items()}})
        return prods, q

    def _query(self, tic: int):
        last = None
        for k in range(TRIES):
            BREAKER.wait()
            try:
                with _SEM:
                    p = list_products_obs(tic)
                BREAKER.ok()
                return p
            except Exception as e:  # noqa: BLE001
                last = e
                ERRLOG({"kind": "query_" + classify(e), "tic": tic, "attempt": k + 1,
                        "final": False, "msg": repr(e)[:200]})
                if k < TRIES - 1:
                    _sleep_before_retry(k, e)
        BREAKER.fail()
        raise DownloadError("query", repr(last)[:200])

    # ---------------------------------------------------------------- one sector (cached)
    def _sector(self, tic: int, s: int, kind: str, uri: str):
        pdir = self.part_dir / f"tic{tic}"
        pf = pdir / f"s{s:04d}.npz"
        if pf.exists():
            z = np.load(pf, allow_pickle=False)
            m = json.loads(str(z["meta"]))
            if m.get("uri") == uri:
                return {k: z[k] for k in z.files if k != "meta"}, m
        last = None
        for k in range(2):  # a conversion failure (e.g. a corrupt body) gets one fresh download
            try:
                rec, m = fetch._one(tic, s, kind, uri)
                break
            except requests.RequestException:
                raise
            except DownloadError:
                raise
            except Exception as e:  # noqa: BLE001
                last = e
                ERRLOG({"kind": classify(e), "tic": tic, "sector": s, "attempt": k + 1,
                        "final": False, "msg": repr(e)[:200]})
                if classify(e) == "tic_mismatch":
                    raise
        else:
            raise last
        m["downloaded_utc"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        m["source"] = pop_source(uri)
        pdir.mkdir(parents=True, exist_ok=True)
        tmp = pdir / f"s{s:04d}.tmp.npz"
        np.savez(tmp, meta=np.array(json.dumps(m, default=str)), **rec)
        tmp.replace(pf)
        return rec, m

    # ---------------------------------------------------------------- one star
    def fetch_star(self, tic: int, prods: dict[int, tuple[str, str]]):
        """Download every listed sector (sector order). Returns (recs, metas, fails)."""
        futs = {s: self.pool.submit(self._sector, tic, s, k, u) for s, (k, u) in prods.items()}
        recs, metas, fails = [], [], []
        for s in sorted(futs):
            try:
                r, m = futs[s].result()
                recs.append(r)
                metas.append(m)
            except Exception as e:  # noqa: BLE001
                fails.append({"sector": s, "kind": classify(e), "error": repr(e)[:200]})
        return recs, metas, fails

    def write_npz(self, tic: int, recs, metas, fails) -> Path:
        """Same layout as fetch.fetch, write-then-rename; then drop the partial cache."""
        out = self.npz_path(tic)
        cat = {k: np.concatenate([r[k] for r in recs]) for k in recs[0]}
        o = np.argsort(cat["idx"], kind="stable")
        cat = {k: v[o] for k, v in cat.items()}
        prov = {"tic": tic, "downloaded_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "bin_minutes": BIN_MIN, "sectors": metas, "failures": fails}
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez_compressed(tmp, meta=np.array(json.dumps(prov, default=str)), **cat)
        tmp.replace(out)
        self.drop_partial(tic)
        return out

    def drop_partial(self, tic: int):
        shutil.rmtree(self.part_dir / f"tic{tic}", ignore_errors=True)
