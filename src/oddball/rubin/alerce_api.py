"""Thin anonymous client for the ALeRCE LSST (Rubin) REST API.

Production host verified 2026-11-09: https://api-lsst.alerce.online/ (no auth, no
rate-limit headers). The ZTF host (api.alerce.online) only serves LSST through these
multisurvey routes; ``survey_id=lsst`` must be sent or the object API searches ZTF only.

Fluxes are in nJy (LSST convention): mag_AB = -2.5 log10(flux_nJy) + 31.4.

Responses are cached as gzipped JSON under ``data/rubin/cache/`` so a backtest can be
re-run without hitting the service again. See docs/rubin_access.md.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import requests

from oddball.common.http import ANON_USER_AGENT

BASE_URL = os.environ.get("ODDBALL_ALERCE_LSST_URL", "https://api-lsst.alerce.online/")
SURVEY = "lsst"
MAX_PAGE_SIZE = 200  # enforced server side (HTTP 422 above this)
BAND_MAP = {1: "g", 2: "r", 3: "i", 4: "z", 5: "y", 6: "u"}


def data_root() -> Path:
    return Path(os.environ.get("ODDBALL_DATA", "data"))


class AlerceLSST:
    """Polite client: minimum interval between requests, retries with backoff, disk cache."""

    def __init__(self, cache_dir: Path | None = None, min_interval: float = 0.25,
                 max_retries: int = 5, timeout: float = 120.0):
        self.cache_dir = Path(cache_dir) if cache_dir else data_root() / "rubin" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = ANON_USER_AGENT
        self._last = 0.0
        self.n_requests = 0
        self.n_cache_hits = 0

    # -- transport ---------------------------------------------------------------------
    def _cache_path(self, route: str, params: dict) -> Path:
        key = json.dumps([route, sorted((k, str(v)) for k, v in params.items())])
        h = hashlib.sha1(key.encode()).hexdigest()[:20]
        return self.cache_dir / route.replace("/", "_") / f"{h}.json.gz"

    def get(self, route: str, params: dict, cache: bool = True) -> Any:
        path = self._cache_path(route, params)
        if cache and path.exists():
            self.n_cache_hits += 1
            with gzip.open(path, "rt") as f:
                return json.load(f)
        delay = 2.0
        for attempt in range(self.max_retries):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.session.get(BASE_URL + route, params=params, timeout=self.timeout)
            except requests.RequestException:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
                continue
            self.n_requests += 1
            if r.status_code == 200:
                data = r.json()
                if cache:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with gzip.open(path, "wt") as f:
                        json.dump(data, f)
                return data
            if r.status_code in (429, 500, 502, 503, 504):
                ra = r.headers.get("Retry-After")
                time.sleep(float(ra) if ra and ra.isdigit() else delay)
                delay *= 2
                continue
            # A 404 on per-object routes means "no data"; anything else is a caller bug.
            if r.status_code == 404:
                return None
            raise RuntimeError(f"ALeRCE {route} HTTP {r.status_code}: {r.text[:300]}")
        raise RuntimeError(f"ALeRCE {route}: gave up after {self.max_retries} attempts")

    # -- object search -----------------------------------------------------------------
    def list_objects(self, page: int = 1, page_size: int = MAX_PAGE_SIZE, count: bool = False,
                     cache: bool = True, **filters) -> dict:
        params = {"survey_id": SURVEY, "page": page, "page_size": page_size,
                  "count": str(count).lower(), **filters}
        return self.get("object_api/list_objects", params, cache=cache)

    def cone_objects(self, ra: float, dec: float, radius_arcsec: float, max_pages: int = 50,
                     **filters) -> list[dict]:
        """All objects within ``radius_arcsec`` (object_api radius is in arcsec)."""
        out: list[dict] = []
        for page in range(1, max_pages + 1):
            d = self.list_objects(page=page, ra=ra, dec=dec, radius=radius_arcsec, **filters)
            items = (d or {}).get("items") or []
            out.extend(items)
            if not (d or {}).get("has_next"):
                break
        return out

    def count_cone(self, ra: float, dec: float, radius_arcsec: float) -> int:
        d = self.list_objects(page=1, page_size=1, count=True, ra=ra, dec=dec,
                              radius=radius_arcsec)
        return int((d or {}).get("total") or 0)

    def objects_by_oid(self, oids: list[int | str]) -> list[dict]:
        out = []
        for i in range(0, len(oids), MAX_PAGE_SIZE):
            chunk = [str(o) for o in oids[i:i + MAX_PAGE_SIZE]]
            d = self.list_objects(oid=chunk)
            out.extend((d or {}).get("items") or [])
        return out

    # -- per-object products -----------------------------------------------------------
    def obj(self, oid) -> dict | None:
        return self.get("object_api/object",
                        {"survey_id": SURVEY, "oid": oid, "return_survey_extra": "true"})

    def detections(self, oid) -> list[dict]:
        return self.get("lightcurve_api/detections", {"survey_id": SURVEY, "oid": oid}) or []

    def forced_photometry(self, oid) -> list[dict]:
        return self.get("lightcurve_api/forced-photometry",
                        {"survey_id": SURVEY, "oid": oid}) or []

    def probabilities(self, oid) -> list[dict]:
        return self.get("probability_api/probability", {"survey_id": SURVEY, "oid": oid}) or []
