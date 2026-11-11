"""Catalogue cross-match for gallery cards: SIMBAD, VSX, Gaia DR3 variability, TOI/CTOI.

Anonymous public services only, every answer cached under ``<data>/gallery/xmatch/`` with its
provenance (service URL, query, UTC time):

* **SIMBAD** TAP (``basic`` + ``ident``): the object whose identifiers include ``TIC <id>``,
  plus every object within ``SIMBAD_RADIUS`` (one TESS pixel) for blends. Servers are tried in
  ``SIMBAD_TAP`` order: the CfA mirror first, since CDS Strasbourg often times out from here.
* **VSX** and **Gaia DR3 vari_classifier_result**: read from the known-class cache
  (``<data>/knownclass/raw/{vsx,gaia_vclass}``, VizieR ``B/vsx/vsx`` and ``I/358/vclassre``,
  cone 21″ / 3″ around every ingested TIC; see ``oddball.knownclass.catalogs``). A TIC that the
  cache never covered is queried live on the CfA, then IDIA VizieR mirrors (not CDS).
* **TOI / CTOI**: the ExoFOP CSV downloads, cached and refreshed after ``EXOFOP_MAX_AGE_DAYS``.

``verdict()`` turns the matches into a one-line, plain-language summary for the card.
"""

from __future__ import annotations

import datetime as dt
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from oddball.common.http import ANON_USER_AGENT
from oddball.common.paths import data_root
from oddball.knownclass import catalogs
from oddball.knownclass.labels import GAIA_EXCLUDE, VSX_LABEL_RADIUS, vsx_class, vsx_excluded

SIMBAD_TAP = (
    "https://simbad.cfa.harvard.edu/simbad/sim-tap/sync",
    "https://simbad.cds.unistra.fr/simbad/sim-tap/sync",
)
VIZIER_MIRRORS = ("vizier.cfa.harvard.edu", "vizier.idia.ac.za")
TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
CTOI_URL = "https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=csv"
EXOFOP_MAX_AGE_DAYS = 7.0
SIMBAD_RADIUS = 21.0  # arcsec, one TESS pixel
MATCH_RADIUS = 5.0  # arcsec: "this star" (VSX labels use the same)
TIMEOUT = 30.0
USER_AGENT = ANON_USER_AGENT

# SIMBAD otype codes (otypedef). Common, well-understood variable types: "boring" here.
SIMBAD_COMMON = {
    "EB*", "EB?", "El*", "El?", "RR*", "RR?", "dS*", "gD*", "SX*", "Ce*", "Ce?", "cC*", "WV*",
    "WV?", "Mi*", "Mi?", "LP*", "LP?", "RS*", "RS?", "BY*", "BY?", "Ro*", "Ro?", "a2*", "a2?",
    "bC*", "bC?", "Pu*", "Pu?", "RV*", "RV?",
}  # fmt: skip
# Young stars and irregular dimmers: exactly the kind of thing we hunt, never "boring".
SIMBAD_YOUNG = {"Y*O", "Y*?", "TT*", "TT?", "Or*", "Ae*", "Ae?", "Ir*", "RC*", "RC?", "FU*",
                "Er*", "Er?"}  # fmt: skip


def xmatch_dir() -> Path:
    return data_root() / "gallery" / "xmatch"


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


# ---------------------------------------------------------------------------------------------
# SIMBAD


def _tap_csv(sess: requests.Session, query: str) -> tuple[pd.DataFrame, str]:
    err = None
    for url in SIMBAD_TAP:
        try:
            r = sess.post(url, data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv",
                                     "QUERY": query}, timeout=TIMEOUT)  # fmt: skip
            if r.status_code == 200:
                return pd.read_csv(io.StringIO(r.text), dtype=str, keep_default_na=False), url
            err = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            err = type(e).__name__
    raise RuntimeError(f"SIMBAD TAP failed on every server ({err})")


def _simbad_cone_query(ra: float, dec: float, radius: float) -> str:
    return (
        "SELECT b.main_id, b.otype, b.ra, b.dec, b.sp_type, "
        f"DISTANCE(POINT('ICRS', b.ra, b.dec), POINT('ICRS', {ra:.7f}, {dec:.7f})) * 3600.0 "
        "AS sep_arcsec FROM basic AS b WHERE CONTAINS(POINT('ICRS', b.ra, b.dec), "
        f"CIRCLE('ICRS', {ra:.7f}, {dec:.7f}, {radius / 3600.0:.8f})) = 1 ORDER BY sep_arcsec"
    )


def _simbad_id_query(tic: int) -> str:
    return ("SELECT b.main_id, b.otype FROM ident AS i JOIN basic AS b ON i.oidref = b.oid "
            f"WHERE i.id = 'TIC {int(tic)}'")  # fmt: skip


def simbad(tic: int, ra: float, dec: float, refresh: bool = False,
           sess: requests.Session | None = None) -> dict:  # fmt: skip
    """SIMBAD matches for one star, cached as ``simbad/<tic>.json``."""
    tic, ra, dec = int(tic), float(ra), float(dec)
    if not (np.isfinite(ra) and np.isfinite(dec)):
        return {"ok": False, "error": "no coordinates"}
    f = xmatch_dir() / "simbad" / f"{tic}.json"
    if f.exists() and not refresh:
        return json.loads(f.read_text())
    sess = sess or _session()
    try:
        idq, cq = _simbad_id_query(tic), _simbad_cone_query(ra, dec, SIMBAD_RADIUS)
        ids, url = _tap_csv(sess, idq)
        cone, url2 = _tap_csv(sess, cq)
    except RuntimeError as e:
        return {"ok": False, "error": str(e)}  # not cached: retried next build
    near = []
    for r in cone.itertuples():
        near.append({"main_id": r.main_id, "otype": r.otype, "sp_type": r.sp_type,
                     "sep_arcsec": round(float(r.sep_arcsec), 2)})  # fmt: skip
    out = {
        "ok": True,
        "by_tic_id": ({"main_id": ids.main_id.iloc[0], "otype": ids.otype.iloc[0]}
                      if len(ids) else None),
        "nearby": near,
        "provenance": {"service": "SIMBAD TAP", "urls": sorted({url, url2}),
                       "queries": [idq, cq], "utc": _now()},
    }
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, indent=1))
    return out


def simbad_otypedef(sess: requests.Session | None = None) -> dict[str, str]:
    """otype code -> description, cached as ``simbad_otypedef.csv`` (empty dict on failure)."""
    f = xmatch_dir() / "simbad_otypedef.csv"
    if not f.exists():
        try:
            df, url = _tap_csv(sess or _session(), "SELECT otype, label, description FROM otypedef")
        except RuntimeError:
            return {}
        f.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(f, index=False)
        _prov("simbad_otypedef", {"service": "SIMBAD TAP", "url": url, "table": "otypedef"})
    df = pd.read_csv(f, dtype=str, keep_default_na=False)
    return dict(zip(df.otype, df.description, strict=True))


# ---------------------------------------------------------------------------------------------
# VSX / Gaia (known-class cache, live fallback)


class CatalogCache:
    """VSX and Gaia vclassre rows for a set of TICs, from the known-class cache."""

    def __init__(self, tics):
        tics = {int(t) for t in tics}
        lab = pd.read_parquet(data_root() / "knownclass" / "labels.parquet",
                              columns=["tic", "label", "label_source", "label_detail", "excluded",
                                       "exclude_reason", "covered_vsx", "covered_gaia_vclass"])  # fmt: skip
        self.labels = lab[lab.tic.isin(tics)].set_index("tic")
        self.vsx = self._load("vsx", tics)
        self.gaia = self._load("gaia_vclass", tics)
        self.prov = {}
        for s in ("vsx", "gaia_vclass"):
            p = catalogs.raw_dir(s) / "provenance.json"
            if p.exists():
                pr = json.loads(p.read_text())
                self.prov[s] = {k: pr.get(k) for k in ("service", "catalog", "radius_arcsec",
                                                      "first_query_utc", "last_query_utc")}  # fmt: skip

    @staticmethod
    def _load(source: str, tics: set[int]) -> pd.DataFrame:
        try:
            df = catalogs.load_raw(source)
        except FileNotFoundError:
            return pd.DataFrame({"tic": pd.Series(dtype=np.int64)})
        return df[df.tic.isin(tics)].copy()

    def covered(self, tic: int, source: str) -> bool:
        col = {"vsx": "covered_vsx", "gaia_vclass": "covered_gaia_vclass"}[source]
        return bool(tic in self.labels.index and self.labels.at[tic, col])


def vsx_live(tic: int, ra: float, dec: float, refresh: bool = False) -> dict:
    """Live VSX cone for a TIC the known-class cache never covered; cached as JSON."""
    f = xmatch_dir() / "vsx_live" / f"{int(tic)}.json"
    if f.exists() and not refresh:
        return json.loads(f.read_text())
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier

    c = SkyCoord(ra * u.deg, dec * u.deg)
    for server in VIZIER_MIRRORS:
        try:
            v = Vizier(columns=["OID", "Name", "Type", "max", "min", "Period", "RAJ2000",
                                "DEJ2000"], row_limit=-1, vizier_server=server, timeout=60)  # fmt: skip
            res = v.query_region(c, radius=catalogs.VSX_RADIUS * u.arcsec, catalog=catalogs.VSX_CAT)
        except Exception:  # noqa: BLE001, S112 (network service: try the next mirror)
            continue
        rows = []
        if len(res):
            df = res[0].to_pandas()
            df["sep_arcsec"] = catalogs._sep(ra, dec, df.RAJ2000.to_numpy(), df.DEJ2000.to_numpy())
            rows = [{"Name": str(r.Name).strip(), "Type": str(r.Type).strip(),
                     "Period": None if pd.isna(r.Period) else float(r.Period),
                     "sep_arcsec": float(r.sep_arcsec)} for r in df.itertuples()]  # fmt: skip
        out = {"ok": True, "rows": rows,
               "provenance": {"service": f"VizieR ({server})", "catalog": catalogs.VSX_CAT,
                              "radius_arcsec": catalogs.VSX_RADIUS, "utc": _now()}}  # fmt: skip
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(out, indent=1))
        return out
    return {"ok": False, "error": "VizieR mirrors unreachable"}


# ---------------------------------------------------------------------------------------------
# TOI / CTOI


def _prov(name: str, info: dict) -> None:
    p = xmatch_dir() / "provenance.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old[name] = {**info, "utc": _now()}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(old, indent=1))


def _exofop(name: str, url: str, refresh: bool) -> pd.DataFrame | None:
    f = xmatch_dir() / f"exofop_{name}.csv"
    stale = (not f.exists()) or refresh or (
        dt.datetime.now().timestamp() - f.stat().st_mtime > EXOFOP_MAX_AGE_DAYS * 86400)
    if stale:
        try:
            r = _session().get(url, timeout=120)
            r.raise_for_status()
            if not r.text.startswith("TIC ID"):
                raise ValueError("unexpected ExoFOP CSV header")
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(".tmp")
            tmp.write_text(r.text)
            tmp.replace(f)
            _prov(f"exofop_{name}", {"service": "ExoFOP-TESS", "url": url,
                                     "bytes": len(r.content)})  # fmt: skip
        except (requests.RequestException, ValueError):
            if not f.exists():
                return None  # offline and never cached: the card says "TOI list unavailable"
    return pd.read_csv(f, dtype=str, keep_default_na=False)


class ToiLists:
    def __init__(self, refresh: bool = False):
        toi = _exofop("toi", TOI_URL, refresh)
        ctoi = _exofop("ctoi", CTOI_URL, refresh)
        self.ok = toi is not None and ctoi is not None
        self.toi = self._index(toi, ["TOI", "TFOPWG Disposition", "TESS Disposition",
                                     "Period (days)", "Depth (ppm)", "Comments"])  # fmt: skip
        self.ctoi = self._index(ctoi, ["CTOI", "Promoted to TOI", "User Disposition",
                                       "Period (days)", "Depth ppm", "Notes"])  # fmt: skip

    @staticmethod
    def _index(df: pd.DataFrame | None, cols: list[str]) -> dict[int, list[dict]]:
        out: dict[int, list[dict]] = {}
        if df is None or "TIC ID" not in df:
            return out
        cols = [c for c in cols if c in df]
        for r in df[["TIC ID", *cols]].itertuples(index=False):
            try:
                t = int(r[0])
            except ValueError:
                continue
            out.setdefault(t, []).append({c: str(v)[:200] for c, v in zip(cols, r[1:],
                                                                          strict=True)})  # fmt: skip
        return out

    def lookup(self, tic: int) -> dict:
        return {"toi": self.toi.get(int(tic), []), "ctoi": self.ctoi.get(int(tic), [])}


# ---------------------------------------------------------------------------------------------
# Per-star summary


def star_xmatch(tic: int, ra: float, dec: float, cache: CatalogCache, tois: ToiLists,
                refresh: bool = False, sess: requests.Session | None = None) -> dict:  # fmt: skip
    tic = int(tic)
    out: dict = {"simbad": simbad(tic, ra, dec, refresh, sess)}
    if cache.covered(tic, "vsx"):
        v = cache.vsx[cache.vsx.tic == tic].sort_values("sep_arcsec")
        out["vsx"] = {"ok": True, "source": "knownclass cache",
                      "rows": [{"Name": r.Name, "Type": r.Type,
                                "Period": None if pd.isna(r.Period) else float(r.Period),
                                "sep_arcsec": round(float(r.sep_arcsec), 2)}
                               for r in v.itertuples()]}  # fmt: skip
    else:
        out["vsx"] = {**vsx_live(tic, ra, dec, refresh), "source": "live"}
    if cache.covered(tic, "gaia_vclass"):
        g = cache.gaia[cache.gaia.tic == tic].sort_values("sep_arcsec")
        out["gaia"] = {"ok": True, "rows": [
            {"source_id": int(r.Source), "class": r.Class, "score": round(float(r.ClassSc), 3),
             "sep_arcsec": round(float(r.sep_arcsec), 2)} for r in g.itertuples()]}  # fmt: skip
    else:
        out["gaia"] = {"ok": False, "error": "not in the known-class cache"}
    if tic in cache.labels.index:
        lab = cache.labels.loc[tic]
        out["kc_label"] = {k: (None if pd.isna(lab[k]) else
                               (bool(lab[k]) if k == "excluded" else str(lab[k])))
                           for k in ("label", "label_source", "label_detail", "excluded",
                                     "exclude_reason")}  # fmt: skip
    out["toi"] = tois.lookup(tic) if tois.ok else None
    out["verdict"] = verdict(out)
    return out


def verdict(x: dict) -> dict:
    """Plain-language catalogue summary: ``{"kind": ..., "text": ...}``.

    kinds: ``toi`` (a TESS planet candidate), ``young`` (a catalogued young star or irregular
    dimmer: the class we hunt, look closely), ``common`` (a well-known common variable: probably
    ``known-boring-variable``), ``variable`` (catalogued variable, type unclear), ``none``.
    """
    notes, kind = [], "none"
    sb = x.get("simbad") or {}
    me = sb.get("by_tic_id")
    if me is None and sb.get("ok"):
        close = [r for r in sb.get("nearby", []) if r["sep_arcsec"] <= MATCH_RADIUS]
        me = close[0] if close else None
    vsx = [r for r in (x.get("vsx") or {}).get("rows", []) if r["sep_arcsec"] <= VSX_LABEL_RADIUS]
    young_vsx = [r for r in (x.get("vsx") or {}).get("rows", []) if vsx_excluded(r["Type"])]
    young_sb = [r for r in sb.get("nearby", []) if r["otype"] in SIMBAD_YOUNG]
    young_gaia = [r for r in (x.get("gaia") or {}).get("rows", []) if r["class"] in GAIA_EXCLUDE]

    t = x.get("toi") or {}
    if t.get("toi") or t.get("ctoi"):
        kind = "toi"
        for r in t.get("toi", []):
            notes.append(f"TOI {r.get('TOI')} ({r.get('TFOPWG Disposition') or 'no TFOPWG disp.'})")
        for r in t.get("ctoi", []):
            notes.append(f"CTOI {r.get('CTOI')}")
    if young_vsx or young_sb or young_gaia:
        kind = "young" if kind == "none" else kind
        names = [f"{r['Name']} ({r['Type']}, {r['sep_arcsec']:.0f}\")" for r in young_vsx]
        names += [f"{r['main_id']} ({r['otype']}, {r['sep_arcsec']:.0f}\")" for r in young_sb]
        names += [f"Gaia DR3 {r['source_id']} ({r['class']})" for r in young_gaia]
        notes.append("catalogued young star / irregular dimmer within 1 px: " + "; ".join(names))
    common = []
    if vsx and vsx_class(vsx[0]["Type"]) and not vsx_excluded(vsx[0]["Type"]):
        common.append(f"VSX {vsx[0]['Type']} ({vsx[0]['Name']})")
    if me and me["otype"] in SIMBAD_COMMON:
        common.append(f"SIMBAD {me['otype']} ({me['main_id']})")
    kl = x.get("kc_label") or {}
    if kl.get("label_source") == "prsa2022":
        common.append("Prša+2022 TESS eclipsing binary")
    if common:
        if kind == "none":
            kind = "common"
        notes.append("known common variable: " + "; ".join(common))
    elif kind == "none" and (vsx or (me and me["otype"] in {"V*", "V*?", "var"})):
        kind = "variable"
        notes.append("catalogued variable, type unspecified: "
                     + (f"VSX {vsx[0]['Type']} ({vsx[0]['Name']})" if vsx else me["main_id"]))
    if not notes:
        notes.append("no variable-star or planet-candidate entry found")
    return {"kind": kind, "text": "; ".join(notes)}
