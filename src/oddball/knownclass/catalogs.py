"""Anonymous catalogue queries for the known-class labels, cached and resumable.

Sources (no accounts):

* ``vsx``   AAVSO VSX (VizieR ``B/vsx/vsx``) on the CfA VizieR mirror: a multi-position cone
            query per chunk of target stars, radius ``VSX_RADIUS`` (one TESS pixel, so blended
            variables are seen too; the label itself needs a much closer match).
* ``prsa``  Prsa et al. 2022, ApJS 258, 16, TESS eclipsing binaries from 2-min sectors 1-26
            (VizieR ``J/ApJS/258/16``), matched by TIC id.
* ``gaia``  Gaia DR3 via the ESA Gaia archive TAP (anonymous upload): every DR3 source within
            ``GAIA_RADIUS`` of each target (and not > 3 mag fainter than its Tmag), with
            ``vari_classifier_result`` (best class + score) and the ``vari_summary`` SOS flags.

Each chunk is written as ``<data>/knownclass/raw/<source>/chunk_NNNN.parquet``, so an
interrupted download resumes. ``provenance.json`` in each source directory records the service,
catalogue id, radius and query date.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from oddball.common.paths import data_root

VIZIER_SERVER = os.environ.get("ODDBALL_VIZIER_SERVER", "vizier.cfa.harvard.edu")
GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap"
VSX_CAT = "B/vsx/vsx"
PRSA_CAT = "J/ApJS/258/16"
VSX_RADIUS = 21.0  # arcsec
GAIA_RADIUS = 3.0  # arcsec


def raw_dir(source: str) -> Path:
    return data_root() / "knownclass" / "raw" / source


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _write_prov(source: str, info: dict) -> None:
    p = raw_dir(source) / "provenance.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(info)
    old.setdefault("first_query_utc", _now())
    old["last_query_utc"] = _now()
    p.write_text(json.dumps(old, indent=1))


def _chunks(targets: pd.DataFrame, size: int):
    for i, s in enumerate(range(0, len(targets), size)):
        yield i, targets.iloc[s : s + size]


def _retry(fn, tries: int = 5, wait: float = 20.0, log=print):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # network services: retry anything, then re-raise
            if k == tries - 1:
                raise
            log(f"  retry {k + 1}/{tries - 1} after {type(e).__name__}: {str(e)[:200]}")
            time.sleep(wait * (k + 1))


def _cone_multi(v, part: pd.DataFrame, catalog: str, radius: float, log, failed: list,
                depth: int = 0) -> pd.DataFrame:  # fmt: skip
    """Multi-position cone query; on a broken response, bisect the position list.

    The CfA mirror sometimes returns a truncated VOTable for a whole list (deterministically, so
    retrying the same list does not help); halves usually go through. A single position that
    still fails is recorded in ``failed`` and skipped.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    c = SkyCoord(part.ra.to_numpy() * u.deg, part.dec.to_numpy() * u.deg)
    try:
        res = _retry(lambda: v.query_region(c, radius=radius * u.arcsec, catalog=catalog),
                     tries=2, wait=5.0, log=log)  # fmt: skip
    except Exception as e:  # noqa: BLE001 (any broken response: bisect)
        if len(part) == 1:
            log(f"  {catalog}: giving up on TIC {int(part.tic.iloc[0])}: {type(e).__name__}")
            failed.append(int(part.tic.iloc[0]))
            return pd.DataFrame()
        h = len(part) // 2
        return pd.concat([_cone_multi(v, part.iloc[:h], catalog, radius, log, failed, depth + 1),
                          _cone_multi(v, part.iloc[h:], catalog, radius, log, failed, depth + 1)],
                         ignore_index=True)  # fmt: skip
    if not len(res):
        return pd.DataFrame()
    df = res[0].to_pandas()
    q = df.pop("_q").to_numpy().astype(np.int64) - 1  # 1-based index into the list
    df.insert(0, "tic", part.tic.to_numpy()[q])
    ra = next(k for k in ("RAJ2000", "RA_ICRS") if k in df)
    de = next(k for k in ("DEJ2000", "DE_ICRS") if k in df)
    df["sep_arcsec"] = _sep(part.ra.to_numpy()[q], part.dec.to_numpy()[q],
                            df[ra].to_numpy(), df[de].to_numpy())  # fmt: skip
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    return df


def fetch_cone(source: str, catalog: str, columns: list[str], radius: float,
               targets: pd.DataFrame, chunk: int = 5000, log=print) -> Path:  # fmt: skip
    """Cone-match every target against a VizieR table; resumable chunk files."""
    from astroquery.vizier import Vizier

    out = raw_dir(source)
    out.mkdir(parents=True, exist_ok=True)
    _write_prov(source, {"service": f"VizieR ({VIZIER_SERVER}) multi-position cone",
                         "catalog": catalog, "columns": columns, "radius_arcsec": radius,
                         "chunk": chunk, "n_targets": len(targets)})  # fmt: skip
    v = Vizier(columns=columns, row_limit=-1, vizier_server=VIZIER_SERVER, timeout=900)
    n_chunks = -(-len(targets) // chunk)
    for i, part in _chunks(targets, chunk):
        f = out / f"chunk_{i:04d}.parquet"
        if f.exists():
            continue
        failed: list[int] = []
        df = _cone_multi(v, part, catalog, radius, log, failed)
        if df.empty:
            df = pd.DataFrame({"tic": pd.Series(dtype=np.int64)})
        df.to_parquet(f, index=False)
        if failed:
            ff = out / "failed_tics.txt"
            with ff.open("a") as fh:
                fh.write("".join(f"{t}\n" for t in failed))
        log(f"  {source} chunk {i + 1}/{n_chunks}: {len(df)} rows"
            + (f", {len(failed)} positions failed" if failed else ""))
    return out


def fetch_vsx(targets: pd.DataFrame, chunk: int = 5000, log=print) -> Path:
    """VSX entries within VSX_RADIUS of each target. ``targets``: tic, ra, dec (deg, J2000)."""
    return fetch_cone("vsx", VSX_CAT,
                      ["OID", "Name", "Type", "max", "min", "Period", "RAJ2000", "DEJ2000"],
                      VSX_RADIUS, targets, chunk, log)  # fmt: skip


GAIA_VCLASS_CAT = "I/358/vclassre"


def fetch_gaia_vclass(targets: pd.DataFrame, chunk: int = 5000, log=print) -> Path:
    """Gaia DR3 ``vari_classifier_result`` (VizieR I/358/vclassre) within GAIA_RADIUS.

    The fallback for ``fetch_gaia``: the ESA archive's TAP upload timed out (HTTP 408 for 500
    positions, 2026-11-09). VizieR's copy has no G magnitude and no SOS flags; it carries the
    DR3 (epoch 2016) position, so the match tolerance absorbs the proper motion since J2000.
    """
    return fetch_cone("gaia_vclass", GAIA_VCLASS_CAT,
                      ["Source", "Classifier", "Class", "ClassSc", "RA_ICRS", "DE_ICRS"],
                      GAIA_RADIUS, targets, chunk, log)  # fmt: skip


def gaia_from_vclass(df: pd.DataFrame) -> pd.DataFrame:
    """VizieR vclassre rows -> the ``fetch_gaia`` schema used by ``labels.build_labels``."""
    out = pd.DataFrame({"tic": df.tic.astype(np.int64), "source_id": df.Source,
                        "gmag": np.nan, "tmag": np.nan, "varflag": "VARIABLE",
                        "sep_arcsec": df.sep_arcsec, "best_class_name": df.Class,
                        "best_class_score": pd.to_numeric(df.ClassSc, errors="coerce")})
    return out


def fetch_prsa(log=print) -> Path:
    from astroquery.vizier import Vizier

    out = raw_dir("prsa")
    out.mkdir(parents=True, exist_ok=True)
    f = out / "tess_ebs.parquet"
    if not f.exists():
        v = Vizier(columns=["TIC", "Per", "Morph", "Tmag"],
                   row_limit=-1, vizier_server=VIZIER_SERVER, timeout=600)  # fmt: skip
        t = _retry(lambda: v.get_catalogs(PRSA_CAT), log=log)[0]
        df = t.to_pandas()
        df.to_parquet(f, index=False)
        _write_prov("prsa", {"service": f"VizieR ({VIZIER_SERVER})", "catalog": PRSA_CAT,
                             "table": t.meta.get("name", ""), "rows": len(df),
                             "reference": "Prsa et al. 2022, ApJS 258, 16"})  # fmt: skip
        log(f"  prsa: {len(df)} rows")
    return out


GAIA_QUERY = """
SELECT u.tic, u.tmag, g.source_id, g.phot_g_mean_mag AS gmag, g.phot_variable_flag AS varflag,
  DISTANCE(POINT(u.ra, u.dec), POINT(g.ra, g.dec)) * 3600.0 AS sep_arcsec,
  c.best_class_name, c.best_class_score,
  s.in_vari_eclipsing_binary, s.in_vari_rrlyrae, s.in_vari_cepheid,
  s.in_vari_long_period_variable, s.in_vari_rotation_modulation, s.in_vari_ms_oscillator,
  s.in_vari_short_timescale, s.in_vari_planetary_transit
FROM tap_upload.t AS u
JOIN gaiadr3.gaia_source AS g
  ON 1 = CONTAINS(POINT(g.ra, g.dec), CIRCLE(u.ra, u.dec, {r} / 3600.0))
LEFT JOIN gaiadr3.vari_classifier_result AS c ON c.source_id = g.source_id
LEFT JOIN gaiadr3.vari_summary AS s ON s.source_id = g.source_id
WHERE g.phot_g_mean_mag < u.tmag + 3.0
"""


def _gaia_sync(part: pd.DataFrame, timeout: float) -> pd.DataFrame:
    import requests
    from astropy.table import Table

    buf = io.BytesIO()
    t = Table.from_pandas(part[["tic", "ra", "dec", "tmag"]].reset_index(drop=True))
    t.write(buf, format="votable", tabledata_format="binary2")
    r = requests.post(f"{GAIA_TAP}/sync",
                      data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv",
                            "QUERY": GAIA_QUERY.format(r=GAIA_RADIUS), "UPLOAD": "t,param:t"},
                      files={"t": ("t.xml", buf.getvalue(), "application/x-votable+xml")}, timeout=timeout)  # fmt: skip
    if r.status_code != 200:
        raise RuntimeError(f"Gaia TAP HTTP {r.status_code}: {r.text[:300]}")
    return pd.read_csv(io.StringIO(r.text))


def fetch_gaia(targets: pd.DataFrame, chunk: int = 20000, timeout: float = 1800,
               log=print) -> Path:  # fmt: skip
    out = raw_dir("gaia")
    out.mkdir(parents=True, exist_ok=True)
    _write_prov("gaia", {"service": f"ESA Gaia archive TAP sync with upload ({GAIA_TAP})",
                         "tables": ["gaiadr3.gaia_source", "gaiadr3.vari_classifier_result",
                                    "gaiadr3.vari_summary"],
                         "radius_arcsec": GAIA_RADIUS, "chunk": chunk,
                         "query": GAIA_QUERY.format(r=GAIA_RADIUS),
                         "n_targets": len(targets)})  # fmt: skip
    n_chunks = -(-len(targets) // chunk)
    for i, part in _chunks(targets, chunk):
        f = out / f"chunk_{i:04d}.parquet"
        if f.exists():
            continue
        t0 = time.time()
        df = _retry(lambda part=part: _gaia_sync(part, timeout), wait=60.0, log=log)
        df.to_parquet(f, index=False)
        log(f"  gaia chunk {i + 1}/{n_chunks}: {len(df)} rows in {time.time() - t0:.0f} s")
    return out


def load_raw(source: str) -> pd.DataFrame:
    files = sorted(raw_dir(source).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no cached {source} query results under {raw_dir(source)}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _sep(ra1, dec1, ra2, dec2) -> np.ndarray:
    """Angular separation in arcsec (haversine)."""
    r1, d1, r2, d2 = (np.radians(np.asarray(a, np.float64)) for a in (ra1, dec1, ra2, dec2))
    h = np.sin((d2 - d1) / 2) ** 2 + np.cos(d1) * np.cos(d2) * np.sin((r2 - r1) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))) * 3600.0
