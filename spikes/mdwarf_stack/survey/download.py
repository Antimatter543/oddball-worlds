"""Survey downloader: every TESS sector of one star, exactly as gate 2 fetched it, politely.

What is reused unchanged from gate 2 (`spikes/mdwarf_stack/fetch.py`):
  * `fetch.list_products`: the SPOC 2-min > TESS-SPOC product choice per sector.
  * `fetch._one`: FITS bytes -> 10-min binned record (`convert_fits_bytes`), TIC check, and the
    per-sector metadata. FITS bytes stay in memory and are never written to disk.
  * The NPZ layout of `fetch.fetch` (sectors concatenated in sector order, stable sort by bin
    index, `meta` JSON with tic / downloaded_utc / bin_minutes / sectors / failures), so
    `searchjob.load_npz` reads exactly what `fetch.load` read in gate 2.

What is wrapped (behaviour of the conversion is untouched):
  * `fetch._get` is replaced by `polite_get`: anonymous User-Agent (`oddball.common.http`),
    a process-wide semaphore of MAX_CONCURRENT (2) requests, bounded retries with exponential
    backoff + jitter (Retry-After honoured), no retry on permanent 4xx, error classification,
    and a circuit breaker that pauses all requests after a run of exhausted retries (MAST down).
    Gate 2 used 4 threads per star and 4 tries with 2-16 s backoff.
  * The astroquery product query runs under the same semaphore with the anonymous User-Agent.
  * Per-sector partial cache (`partial/tic<T>/sNNNN.npz`, write-then-rename), so a killed
    40-sector star does not re-download the sectors it already has.
"""

from __future__ import annotations

import datetime as dt
import json
import random
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
TRIES = 5
BACKOFF_S = (5.0, 15.0, 45.0, 120.0)  # sleeps between the TRIES attempts
TIMEOUT = (30.0, 180.0)  # connect, read
BREAKER_N = 8  # consecutive exhausted requests -> pause everything
BREAKER_PAUSE_S = (600.0, 1200.0, 2400.0, 3600.0)
PRODUCT_CACHE_DAYS = 30  # product lists newer than this are reused without a query

_SEM = threading.BoundedSemaphore(MAX_CONCURRENT)
_tls = threading.local()


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


def polite_get(uri: str, tries: int = TRIES) -> bytes:
    """Drop-in for fetch._get: same URL and return value, polite transport."""
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
    from astroquery.mast import Observations

    Observations._session.headers["User-Agent"] = ANON_USER_AGENT
    Observations._portal_api_connection._session.headers["User-Agent"] = ANON_USER_AGENT


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
        self.pool = ThreadPoolExecutor(MAX_CONCURRENT, thread_name_prefix="dl")
        _set_astroquery_ua()

    def npz_path(self, tic: int) -> Path:
        return self.lc_dir / f"tic{tic}.npz"

    # ---------------------------------------------------------------- product list (cached)
    def products(self, tic: int) -> tuple[dict[int, tuple[str, str]], str]:
        cp = self.prod_dir / f"tic{tic}.json"
        if cp.exists():
            c = json.loads(cp.read_text())
            age = dt.datetime.now(dt.UTC) - dt.datetime.fromisoformat(c["queried_utc"])
            if age.days < PRODUCT_CACHE_DAYS and c["products"]:
                return {int(s): tuple(v) for s, v in c["products"].items()}, c["queried_utc"]
        prods = self._query(tic)
        if not prods:
            # an empty answer is re-asked once after a pause before it is believed
            time.sleep(20.0)
            prods = self._query(tic)
        q = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        if prods:
            _atomic_json(cp, {"tic": tic, "queried_utc": q,
                              "products": {str(s): list(v) for s, v in prods.items()}})
        return prods, q

    def _query(self, tic: int):
        last = None
        for k in range(TRIES):
            BREAKER.wait()
            try:
                with _SEM:
                    p = fetch.list_products(tic)
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
