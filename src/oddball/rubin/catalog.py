"""Young-star catalogues from VizieR -> compact Parquet under data/rubin/catalogs/.

Catalogues (all anonymous VizieR):
- ``spicy``: SPICY, Kuhn+2021 ApJS 254, 33 (J/ApJS/254/33/table1). ~117k Spitzer/GLIMPSE YSO
  candidates, |b| <~ 1 deg, 255 < l < 110 deg. Mid-IR selected: many are optically faint.
- ``gaia_yso``: Gaia DR3 variability classifier, best class YSO (I/358/vclassre, Class=YSO).
  ~79k optically visible variable YSO candidates, all-sky. Best overlap with Rubin's
  optical depth (G <~ 20.5).
- ``known_yso``: Marton+2023 A&A 674, A21 compilation of literature YSOs matched to
  Gaia DR3 (J/A+A/674/A21/kyso). Carries names and literature variability types.

The CDS VizieR host (vizier.cds.unistra.fr) timed out on 2026-11-09; the CfA mirror is the
default here. Override with ODDBALL_VIZIER_SERVER.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from oddball.rubin.alerce_api import data_root

VIZIER_SERVER = os.environ.get("ODDBALL_VIZIER_SERVER", "vizier.cfa.harvard.edu")

CATALOGS = {
    "spicy": {
        "vizier": "J/ApJS/254/33/table1",
        "columns": ["SPICY", "RA_ICRS", "DE_ICRS", "GLON", "GLAT", "class", "var", "ZTFrmag",
                    "GaiaDR2"],
        "filters": {},
        "ra": "RA_ICRS", "dec": "DE_ICRS", "id": "SPICY", "prefix": "SPICY ",
    },
    "gaia_yso": {
        "vizier": "I/358/vclassre",
        "columns": ["Source", "Class", "ClassSc", "RA_ICRS", "DE_ICRS"],
        "filters": {"Class": "YSO"},
        "ra": "RA_ICRS", "dec": "DE_ICRS", "id": "Source", "prefix": "Gaia DR3 ",
    },
    "known_yso": {
        "vizier": "J/A+A/674/A21/kyso",
        "columns": ["RAJ2000", "DEJ2000", "SFR", "Name", "YSOtype", "vartype", "GaiaDR3",
                    "Gmag"],
        "filters": {},
        "ra": "RAJ2000", "dec": "DEJ2000", "id": "Name", "prefix": "",
    },
}


def catalog_path(name: str) -> Path:
    return data_root() / "rubin" / "catalogs" / f"{name}.parquet"


def fetch(name: str, force: bool = False) -> Path:
    """Download a catalogue from VizieR into Parquet (+ provenance JSON sidecar)."""
    from astroquery.vizier import Vizier

    spec = CATALOGS[name]
    out = catalog_path(name)
    if out.exists() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    v = Vizier(columns=spec["columns"], column_filters=spec["filters"], row_limit=-1,
               vizier_server=VIZIER_SERVER, timeout=600)
    tl = v.get_catalogs(spec["vizier"])
    t = tl[0]
    df = t.to_pandas()
    df = standardise(df, name)
    df.to_parquet(out, index=False)
    meta = {"catalog": name, "vizier": spec["vizier"], "server": VIZIER_SERVER,
            "filters": spec["filters"], "rows": len(df),
            "downloaded_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return out


def standardise(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Add uniform ``cat``, ``cat_id``, ``ra``, ``dec`` columns."""
    spec = CATALOGS[name]
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    df["cat"] = name
    df["cat_id"] = spec["prefix"] + df[spec["id"]].astype(str)
    df["ra"] = df[spec["ra"]].astype(float)
    df["dec"] = df[spec["dec"]].astype(float)
    return df[np.isfinite(df["ra"]) & np.isfinite(df["dec"])].reset_index(drop=True)


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(catalog_path(name))


def load_union() -> pd.DataFrame:
    """All available catalogues stacked (one row per catalogue entry; duplicates possible)."""
    frames = [load(n)[["cat", "cat_id", "ra", "dec"]] for n in CATALOGS if catalog_path(n).exists()]
    return pd.concat(frames, ignore_index=True)
