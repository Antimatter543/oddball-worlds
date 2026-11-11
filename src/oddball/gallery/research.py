"""Catalogue research for vetting gallery cards (docs/VETTING.md F10, E8-E10).

Batch, anonymous VizieR cone searches (CfA mirror first, then IDIA; CDS Strasbourg times out
from here) for a list of stars, cached per catalogue under ``<data>/gallery/research/``:

* **Gaia DR3** (``I/355/gaiadr3``) within 150" (G < 19.5): the target match, RUWE, parallax, the
  variability flag, a CMD height above a rough main sequence, and every neighbour that could
  blend into a TESS aperture (with the largest depth it could fake, 10^(-0.4 dG)).
* **AllWISE** (``II/328/allwise``) within 6": W1-W2, W2-W3, W1-W4 and an infrared-excess flag.
* **VSX** (``B/vsx/vsx``) within 120": a known variable on the target or near it.
* **Gaia DR3 variability**: EB candidates (``I/358/veb``) and classifications (``I/358/vclassre``).
* **Kostov+2025** TESS EB tables (``J/ApJS/279/50``) within 21".
* **Young-star lists**: Marton+2019 Gaia/WISE YSO probabilities (``J/MNRAS/487/2522``; it lists
  every source, so only ``LY``/``SY`` (YSO probability) > 0.5 means "YSO candidate") and
  Kounkel+2020 young strings (``J/AJ/160/279``), within 3".

Only coordinates and catalogue ids leave this machine; the User-Agent is astroquery's own.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from oddball.common.paths import data_root

# IDIA first: on 2026-11-11 the CfA mirror returned Gmag = 999999999 for every row of a
# multi-target Gaia DR3 cone (IDIA gave the real values). Responses with that sentinel in a
# magnitude column are rejected and re-asked on the next mirror.
MIRRORS = ("vizier.idia.ac.za", "vizier.cfa.harvard.edu")
SENTINEL = 1e8
MAG_COLS = ("Gmag", "W1mag", "Tmag", "Vmag")
TESS_PX = 21.0

# id -> (vizier catalogue, radius arcsec, columns, column filters)
CATALOGUES = {
    "gaia": ("I/355/gaiadr3", 105.0,
             ["Source", "RA_ICRS", "DE_ICRS", "Gmag", "BP-RP", "Plx", "e_Plx", "RUWE", "VarFlag"],
             {"Gmag": 19.5}),  # max G, applied here (VizieR column_filters fail silently)
    "allwise": ("II/328/allwise", 6.0, ["**"], {}),
    "vsx": ("B/vsx/vsx", 120.0, ["Name", "Type", "Period", "max", "min", "RAJ2000", "DEJ2000"], {}),
    "gaia_veb": ("I/358/veb", 5.0, ["**"], {}),
    "gaia_vclass": ("I/358/vclassre", 5.0, ["**"], {}),
    "kostov": ("J/ApJS/279/50", 21.0, ["**"], {}),
    "marton_yso": ("J/MNRAS/487/2522", 3.0, ["**"], {}),
    "kounkel": ("J/AJ/160/279", 3.0, ["**"], {}),
}  # fmt: skip

# Rough main sequence, M_G against BP-RP (Pecaut & Mamajek-style dwarf sequence, rounded).
# Used only to say "on / above / below the MS"; no reddening correction.
MS_BPRP = np.array([-0.1, 0.0, 0.2, 0.38, 0.55, 0.73, 0.82, 0.98, 1.25, 1.43, 1.84, 2.2, 2.5,
                    2.8, 3.2, 4.0])  # fmt: skip
MS_MG = np.array([0.5, 1.0, 1.9, 2.6, 3.4, 4.3, 4.65, 5.4, 6.4, 6.9, 8.1, 9.2, 10.2, 11.3, 12.6,
                  15.0])  # fmt: skip


def research_dir() -> Path:
    return data_root() / "gallery" / "research"


def _vizier(cat: str, cols, filters, timeout: float = 120.0):
    from astroquery.vizier import Vizier

    return Vizier(columns=list(cols) + ["_r"], row_limit=-1,
                  timeout=timeout)  # fmt: skip


def query_catalogue(key: str, stars: pd.DataFrame, chunk: int = 40, sleep: float = 1.0,
                    refresh: bool = False) -> pd.DataFrame:  # fmt: skip
    """Rows of catalogue ``key`` within its radius of each star (columns + tic + sep_arcsec).

    Cached as ``<research>/<key>.parquet`` plus the list of TICs already queried, so a re-run
    only queries new stars."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    cat, radius, cols, filters = CATALOGUES[key]
    d = research_dir()
    d.mkdir(parents=True, exist_ok=True)
    fp, fq = d / f"{key}.parquet", d / f"{key}_queried.json"
    have = pd.read_parquet(fp) if fp.exists() and not refresh else pd.DataFrame()
    done = set(json.loads(fq.read_text())["tics"]) if fq.exists() and not refresh else set()
    todo = stars[~stars.tic.isin(done)].drop_duplicates("tic")
    new = []
    for i in range(0, len(todo), chunk):
        part = todo.iloc[i : i + chunk]
        c = SkyCoord(part.ra.to_numpy(), part.dec.to_numpy(), unit="deg")
        res, err = None, None
        # A mirror sometimes answers an empty list within a second (seen on the CfA mirror for
        # a Gaia cone that IDIA filled with 5,066 rows). Empty is indistinguishable from "no
        # source", so an empty answer is re-asked on the next mirror before it is believed.
        for server in MIRRORS:
            v = _vizier(cat, cols, filters)
            v.VIZIER_SERVER = server
            try:
                r_ = v.query_region(c, radius=radius * u.arcsec, catalog=cat)
            except Exception as e:  # noqa: BLE001
                err = repr(e)[:200]
                continue
            bad = any(c_ in tab.colnames and np.nanmax(np.abs(np.ma.filled(
                tab[c_].astype(float), np.nan))) >= SENTINEL for tab in r_ for c_ in MAG_COLS
                if len(tab))  # fmt: skip
            if bad:
                err = f"{server}: sentinel magnitudes in the response"
                continue
            res = r_
            if len(r_):
                break
        if res is None:
            raise RuntimeError(f"{key}: every VizieR mirror failed ({err})")
        for name, tab in zip(res.keys(), res, strict=True):
            df = tab.to_pandas().drop(columns=["_q", "_r"], errors="ignore")
            # assign each row to its nearest query star (VizieR's _q is not reliable for
            # multi-table catalogues)
            pair = next(((a, b) for a, b in (("RA_ICRS", "DE_ICRS"), ("RAJ2000", "DEJ2000"),
                                             ("RAdeg", "DEdeg"), ("RAICRS", "DEICRS"),
                                             ("_RA", "_DE"))
                         if a in df and b in df), None)  # fmt: skip
            if pair is None:
                raise RuntimeError(f"{key}/{name}: no coordinate columns in {list(df)[:20]}")
            ra_, de_ = df[pair[0]], df[pair[1]]
            if ra_.dtype == object:  # sexagesimal strings
                rc = SkyCoord(ra_.astype(str).to_numpy(), de_.astype(str).to_numpy(),
                              unit=("hourangle", "deg"))  # fmt: skip
            else:
                rc = SkyCoord(ra_.to_numpy(float), de_.to_numpy(float), unit="deg")
            idx, sep, _ = rc.match_to_catalog_sky(c)
            df.insert(0, "tic", part.tic.to_numpy()[idx].astype("int64"))
            df.insert(1, "table", name)
            df["sep_arcsec"] = sep.arcsec
            df = df[df.sep_arcsec <= 1.01 * radius]
            if "Gmag" in filters and "Gmag" in df:
                df = df[pd.to_numeric(df.Gmag, errors="coerce") <= filters["Gmag"]]
            new.append(df)
        done |= {int(t) for t in part.tic}
        time.sleep(sleep)
    if new:
        nd = pd.concat(new, ignore_index=True)
        for c_ in nd.columns:
            if nd[c_].dtype == object:
                nd[c_] = nd[c_].astype(str)
        have = pd.concat([have, nd], ignore_index=True) if len(have) else nd
        have.to_parquet(fp, index=False)
    fq.write_text(json.dumps({"catalogue": cat, "radius_arcsec": radius, "columns": cols,
                              "filters": filters, "mirrors": MIRRORS, "tics": sorted(done),
                              "utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}))  # fmt: skip
    return have[have.tic.isin(set(stars.tic))] if len(have) else have


def ms_height(bp_rp: float, abs_g: float) -> float:
    """Magnitudes above (+) or below (-) the rough main sequence at this colour."""
    if not (np.isfinite(bp_rp) and np.isfinite(abs_g)):
        return np.nan
    return float(np.interp(bp_rp, MS_BPRP, MS_MG) - abs_g)


def gaia_summary(g: pd.DataFrame, tmag: float) -> dict:
    """Pick the Gaia counterpart (close, G consistent with Tmag) and summarise neighbours."""
    if g is None or not len(g):
        return {"gaia_match": False}
    g = g.copy()
    g["Gmag"] = pd.to_numeric(g.Gmag, errors="coerce")
    near = g[g.sep_arcsec <= 5.0]
    if len(near):
        cost = near.sep_arcsec / 2.0 + (near.Gmag - (tmag + 0.5)).abs()
        t = near.loc[cost.idxmin()]
    else:
        return {"gaia_match": False}
    out = {"gaia_match": True, "gaia_source": str(t.Source), "gaia_sep": float(t.sep_arcsec),
           "G": float(t.Gmag), "bp_rp": float(pd.to_numeric(t["BP-RP"], errors="coerce")),
           "plx": float(t.Plx), "plx_snr": float(t.Plx / t.e_Plx) if t.e_Plx else np.nan,
           "ruwe": float(t.RUWE), "gaia_varflag": str(t.VarFlag)}  # fmt: skip
    if np.isfinite(out["plx"]) and out["plx"] > 0 and out["plx_snr"] > 5:
        out["dist_pc"] = 1000.0 / out["plx"]
        out["abs_G"] = out["G"] + 5 * np.log10(out["plx"] / 100.0)
        out["ms_height"] = ms_height(out["bp_rp"], out["abs_G"])
    else:
        out["dist_pc"] = out["abs_G"] = out["ms_height"] = np.nan
    nb = g[(g.Source.astype(str) != str(t.Source))].copy()
    nb["dG"] = nb.Gmag - out["G"]
    nb["max_fake_depth"] = 10 ** (-0.4 * nb.dG.clip(lower=0))
    for r_px in (1, 2, 3):
        m = nb[(nb.sep_arcsec <= r_px * TESS_PX) & (nb.dG <= 5.0)]
        out[f"n_nb_{r_px}px"] = len(m)
    m1 = nb[nb.sep_arcsec <= 2 * TESS_PX].sort_values("dG")
    if len(m1):
        b = m1.iloc[0]
        out.update({"brightest_nb_2px_dG": float(b.dG), "brightest_nb_2px_sep": float(b.sep_arcsec),
                    "brightest_nb_2px_source": str(b.Source),
                    "brightest_nb_2px_varflag": str(b.VarFlag),
                    "flux_ratio_nb_2px": float((10 ** (-0.4 * m1.dG)).sum())})  # fmt: skip
    return out


def wise_summary(w: pd.DataFrame) -> dict:
    """AllWISE colours and a simple infrared-excess flag for the nearest source within 6"."""
    if w is None or not len(w):
        return {"wise_match": False}
    w = w.sort_values("sep_arcsec").iloc[0]

    def num(c):
        return float(pd.to_numeric(w[c], errors="coerce"))

    w1, w2, w3, w4 = num("W1mag"), num("W2mag"), num("W3mag"), num("W4mag")
    e3, e4 = num("e_W3mag"), num("e_W4mag")
    out = {"wise_match": True, "wise_sep": float(w.sep_arcsec), "W1": w1,
           "W1_W2": w1 - w2, "W2_W3": w2 - w3 if np.isfinite(e3) else np.nan,
           "W1_W4": w1 - w4 if np.isfinite(e4) else np.nan, "wise_ccf": str(w.ccf),
           "wise_ex": str(w.ex), "wise_qph": str(w.qph)}  # fmt: skip
    # photospheres sit near 0 in all three colours (M dwarfs up to ~0.3 in W1-W2). Upper limits
    # (no error) never count as an excess.
    ex = []
    if np.isfinite(out["W1_W2"]) and out["W1_W2"] > 0.35:
        ex.append("W1-W2")
    if np.isfinite(out["W2_W3"]) and out["W2_W3"] > 0.6 and e3 < 0.2:
        ex.append("W2-W3")
    if np.isfinite(out["W1_W4"]) and out["W1_W4"] > 1.2 and e4 < 0.3:
        ex.append("W1-W4")
    out["ir_excess"] = ",".join(ex)
    out["wise_confused"] = str(w.ccf).strip("0 ") not in ("", "nan") or str(w.ex) not in ("0", "nan")
    return out


def vsx_summary(v: pd.DataFrame) -> dict:
    if v is None or not len(v):
        return {"vsx_on_target": "", "vsx_within_2px": "", "vsx_within_120": 0}
    v = v.sort_values("sep_arcsec")

    def fmt(r):
        p = pd.to_numeric(r.Period, errors="coerce")
        ps = f" P={p:.4g}d" if np.isfinite(p) else ""
        return f"{r.Name} [{r.Type}{ps}] {r.sep_arcsec:.0f}\""

    on = v[v.sep_arcsec <= 5.0]
    px = v[(v.sep_arcsec > 5.0) & (v.sep_arcsec <= 2 * TESS_PX)]
    far = v[v.sep_arcsec > 2 * TESS_PX].head(3)
    return {"vsx_on_target": "; ".join(fmt(r) for r in on.itertuples()),
            "vsx_within_2px": "; ".join(fmt(r) for r in px.itertuples()),
            "vsx_nearest_beyond_2px": "; ".join(fmt(r) for r in far.itertuples()),
            "vsx_within_120": len(v)}  # fmt: skip


def simple_summary(df: pd.DataFrame, cols, max_sep: float) -> str:
    if df is None or not len(df):
        return ""
    df = df[df.sep_arcsec <= max_sep].sort_values("sep_arcsec")
    parts = []
    for r in df.itertuples(index=False):
        d = r._asdict()
        bits = [f"{c}={d[c]}" for c in cols if c in d and str(d[c]) not in ("nan", "", "None")]
        parts.append(f"{d['table']}: " + " ".join(bits) + f" ({d['sep_arcsec']:.1f}\")")
    return "; ".join(parts[:3])


def research(stars: pd.DataFrame, refresh: bool = False, log=print) -> pd.DataFrame:
    """One row per star (columns tic, ra, dec, tmag in ``stars``) with the catalogue evidence."""
    raw = {}
    for key in CATALOGUES:
        t0 = time.time()
        try:
            raw[key] = query_catalogue(key, stars, refresh=refresh)
            log(f"  {key}: {len(raw[key])} rows ({time.time() - t0:.0f} s)")
        except RuntimeError as e:
            raw[key] = pd.DataFrame()
            log(f"  {key}: FAILED {e}")
    rows = []
    for s in stars.itertuples(index=False):
        tic = int(s.tic)

        def sub(k, tic=tic):
            d = raw.get(k)
            return d[d.tic == tic] if d is not None and len(d) else None

        r = {"tic": tic}
        r.update(gaia_summary(sub("gaia"), float(s.tmag)))
        r.update(wise_summary(sub("allwise")))
        r.update(vsx_summary(sub("vsx")))
        r["gaia_veb"] = simple_summary(sub("gaia_veb"), ["Freq", "ModelType", "Depth1"], 3.0)
        r["gaia_vclass"] = simple_summary(sub("gaia_vclass"), ["Class", "ClassSc"], 3.0)
        r["kostov"] = simple_summary(sub("kostov"), ["TIC", "Per", "Period", "P"], 21.0)
        r["yso_lists"] = "; ".join(x for x in (
            simple_summary(sub("marton_yso"), ["LY", "SY", "LMS"], 3.0),
            simple_summary(sub("kounkel"), ["Theia", "Age", "logAge"], 3.0)) if x)  # fmt: skip
        rows.append(r)
    return pd.DataFrame(rows)
