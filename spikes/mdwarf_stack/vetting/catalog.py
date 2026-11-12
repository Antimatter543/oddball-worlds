"""Star-level vetting information: Gaia neighbours (contamination) and catalogue cross-matches.

Fail-closed: every source records "ok" or the error; a failed query is never read as "not
known" (vet._catalogue_status turns it into status `error`, which blocks tier A).

Sources
  Gaia DR3 cone 105" (oddball.gallery.research, VizieR IDIA then CfA, sentinel-checked)
  EBs: Kostov+2025 J/ApJS/279/50 (21"), Gaia DR3 I/358/veb (5"), VSX B/vsx (eclipsing types, 21")
  TOIs: local ExoFOP table copy (data/planets_spike/catalogs/toi.csv, 2026-11-11), any
        disposition; FP/FA TOIs are kept as "toi_fp" (a known false positive at P -> fail)
  CTOIs: live ExoFOP download (pipe format), cached per run in <OUT>/vetting/cache/ctoi.txt
  known planets: local exoplanet.eu copy (data/planets_spike/catalogs/exoplanet_eu.csv), 30"

Usage: python -I vetting/catalog.py dev test hosts   (writes <OUT>/vetting/star_info.json)
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "src"))
from common import CAT_DIR, OUT  # noqa: E402

V = OUT / "vetting"
CACHE = V / "cache"
os.environ.setdefault("ODDBALL_DATA", str(CACHE))  # research.py caches under here, not data/
INFO = V / "star_info.json"
CTOI_URL = "https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=pipe"
PX = 21.0


def stars_for(which: str) -> pd.DataFrame:
    if which in ("dev", "test"):
        c = pd.read_csv(CAT_DIR / "controls_g2.csv")
        c = c[c["split"] == which.upper()]
        return pd.DataFrame({"tic": c.tic.astype(int), "ra": c.ra, "dec": c.dec, "tmag": c.Tmag})
    if which == "hosts":
        h = pd.read_csv(CAT_DIR / "holdout_g2.csv")
        h = h[h["in_range"] & h["tic"].notna()].drop_duplicates("tic")
        out = pd.DataFrame({"tic": h.tic.astype(int), "ra": h.ra, "dec": h.dec, "tmag": h.tmag})
        t = pd.read_csv(CAT_DIR / "tic_mdwarfs_50pc.csv") if (CAT_DIR / "tic_mdwarfs_50pc.csv"
                                                                ).exists() else None
        from make_jobs import HOSTS

        extra = []
        for tic in HOSTS:
            if tic in set(out.tic):
                continue
            row = t[t.iloc[:, 0] == tic] if t is not None else []
            if len(row):
                r = row.iloc[0]
                extra.append({"tic": tic, "ra": float(r.get("ra")), "dec": float(r.get("dec")),
                              "tmag": float(r.get("Tmag"))})
                continue
            toi = pd.read_csv(CAT_DIR / "toi.csv", low_memory=False)
            row = toi[toi["TIC ID"] == tic]
            if len(row):
                r = row.iloc[0]
                extra.append({"tic": tic, "ra": float(r["RA"]) if not isinstance(r["RA"], str)
                              else _sexa(r["RA"], r["Dec"])[0],
                              "dec": float(r["Dec"]) if not isinstance(r["Dec"], str)
                              else _sexa(r["RA"], r["Dec"])[1], "tmag": float(r["TESS Mag"])})
        return pd.concat([out, pd.DataFrame(extra)], ignore_index=True) if extra else out
    raise ValueError(which)


def _sexa(ra, dec):
    from astropy.coordinates import SkyCoord

    c = SkyCoord(str(ra), str(dec), unit=("hourangle", "deg"))
    return float(c.ra.deg), float(c.dec.deg)


def _ctoi():
    import requests

    from oddball.common.http import ANON_USER_AGENT

    CACHE.mkdir(parents=True, exist_ok=True)
    fp = CACHE / "ctoi.txt"
    if not fp.exists() or time.time() - fp.stat().st_mtime > 86400:
        r = requests.get(CTOI_URL, headers={"User-Agent": ANON_USER_AGENT}, timeout=120)
        r.raise_for_status()
        if not r.text.startswith("TIC ID|"):
            raise RuntimeError("unexpected CTOI response")
        fp.write_text(r.text)
    from io import StringIO

    return pd.read_csv(StringIO(fp.read_text()), sep="|", low_memory=False)


def _sep(ra1, de1, ra2, de2):
    r1, d1, r2, d2 = (np.radians(x) for x in (ra1, de1, ra2, de2))
    c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    return np.degrees(np.arccos(np.clip(c, -1, 1))) * 3600


def _gaia_block(g: pd.DataFrame, tmag: float) -> dict:
    if g is None or not len(g):
        return {"error": "no Gaia DR3 rows within 105\" (query empty)"}
    g = g.copy()
    for c in ("Gmag", "Plx", "sep_arcsec"):
        g[c] = pd.to_numeric(g[c], errors="coerce")
    # nearby M dwarfs move up to tens of arcsec between the TIC epoch (2000) and Gaia (2016), so
    # the counterpart is searched within 60" among parallax > 5 mas sources (or a 2-parameter
    # solution within 5"), and must have G within 1.5 mag of Tmag + 1.6 (M-dwarf G - T)
    cand = g[((g.sep_arcsec <= 60) & (g.Plx > 5)) | ((g.sep_arcsec <= 5) & g.Plx.isna())]
    cand = cand[(cand.Gmag - (tmag + 1.6)).abs() <= 1.5]
    if not len(cand):  # faint stars whose Gaia parallax disagrees with TIC: the source on top
        cand = g[(g.sep_arcsec <= 3) & ((g.Gmag - (tmag + 1.0)).abs() <= 2.0)]
    if not len(cand):
        return {"error": "no Gaia counterpart (60\", parallax > 5 mas, G ~ Tmag + 1.6)"}
    cost = (cand.Gmag - (tmag + 1.6)).abs() + cand.sep_arcsec / 30
    t = cand.loc[cost.idxmin()]
    nb = g[g.Source.astype(str) != str(t.Source)].copy()
    # separations from the Gaia position of the target (proper motion: TIC is epoch 2000)
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    ct = SkyCoord(float(t.RA_ICRS), float(t.DE_ICRS), unit="deg")
    cn = SkyCoord(nb.RA_ICRS.astype(float).to_numpy(), nb.DE_ICRS.astype(float).to_numpy(),
                  unit="deg")
    nb["sep_t"] = ct.separation(cn).to(u.arcsec).value
    nb["dG"] = nb.Gmag - float(t.Gmag)
    near = nb[nb.sep_t <= 3 * PX].sort_values("dG")
    return {"target": {"source": str(t.Source), "G": float(t.Gmag), "sep": float(t.sep_arcsec),
                       "plx": float(t.Plx), "ruwe": float(pd.to_numeric(t.RUWE, errors="coerce")),
                       "ra": float(t.RA_ICRS), "dec": float(t.DE_ICRS)},
            "neighbours": [{"source": str(r.Source), "G": float(r.Gmag), "dG": float(r.dG),
                            "sep": float(r.sep_t), "ra": float(r.RA_ICRS),
                            "dec": float(r.DE_ICRS)} for r in near.itertuples()]}


def build(which: list[str]) -> dict:
    from oddball.gallery.research import query_catalogue

    stars = pd.concat([stars_for(w) for w in which], ignore_index=True).drop_duplicates("tic")
    info = json.loads(INFO.read_text()) if INFO.exists() else {}
    src = {}
    res = {}
    for key in ("gaia", "kostov", "gaia_veb", "vsx"):
        try:
            res[key] = query_catalogue(key, stars)
            src[key] = "ok"
        except Exception as e:  # noqa: BLE001
            res[key] = None
            src[key] = f"error: {e!r}"[:200]
    try:
        ctoi = _ctoi()
        src["ctoi"] = "ok"
    except Exception as e:  # noqa: BLE001
        ctoi = None
        src["ctoi"] = f"error: {e!r}"[:200]
    toi = pd.read_csv(CAT_DIR / "toi.csv", low_memory=False)
    eu = pd.read_csv(CAT_DIR / "exoplanet_eu.csv", low_memory=False)
    src["toi"] = "ok"
    src["exoplanet_eu"] = "ok"
    now = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    for s in stars.itertuples():
        tic = int(s.tic)
        rec = {"tic": tic, "ra": float(s.ra), "dec": float(s.dec), "tmag": float(s.tmag),
               "utc": now, "sources": dict(src)}
        g = res["gaia"]
        if g is None:
            rec["gaia"] = {"error": src["gaia"]}
        else:
            rec["gaia"] = _gaia_block(g[g.tic == tic] if len(g) and "tic" in g else None,
                                      float(s.tmag))
        objs = []
        def _p(x):
            v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
            return float(v) if np.isfinite(v) and v > 0 else None

        for r in toi[toi["TIC ID"] == tic].to_dict("records"):
            disp = str(r.get("TFOPWG Disposition", ""))
            tdisp = str(r.get("TESS Disposition", ""))
            kind = "toi_fp" if disp in ("FP", "FA") or tdisp in ("FP", "FA") else "toi"
            objs.append({"kind": kind, "name": f"TOI-{r['TOI']}", "P": _p(r.get("Period (days)")),
                         "disp": disp})
        if ctoi is not None:
            for r in ctoi[ctoi["TIC ID"] == tic].to_dict("records"):
                objs.append({"kind": "ctoi", "name": f"CTOI {r['CTOI']}",
                             "P": _p(r.get("Period (days)"))})
        sep = _sep(float(s.ra), float(s.dec), eu.ra.to_numpy(float), eu.dec.to_numpy(float))
        for j in np.flatnonzero(sep < 30):
            p = eu.orbital_period.iloc[j]
            objs.append({"kind": "planet", "name": str(eu.name.iloc[j]),
                         "P": float(p) if np.isfinite(p) else None,
                         "status": str(eu.planet_status.iloc[j])})
        for key, rad in (("kostov", 21.0), ("gaia_veb", 5.0), ("vsx", 21.0)):
            t = res[key]
            if t is None or not len(t) or "tic" not in t:
                continue
            rows = t[(t.tic == tic) & (pd.to_numeric(t.sep_arcsec, errors="coerce") <= rad)]
            if key == "kostov":  # table2 is the unvetted NN-flagged list, not EBs
                rows = rows[rows.table.astype(str).str.contains("table3|table4")]
            if key == "vsx" and len(rows):
                rows = rows[rows.Type.astype(str).str.match(r"^E(A|B|W|$|\|)") |
                            rows.Type.astype(str).isin(["E", "EA", "EB", "EW", "E:"])]
            for r in rows.to_dict("records"):
                pcol = next((c for c in r if c.lower() in ("per", "period", "p", "porb")), None)
                p = pd.to_numeric(pd.Series([r.get(pcol)]), errors="coerce").iloc[0] if pcol \
                    else np.nan
                objs.append({"kind": "eb", "name": f"{key}:{r.get('Name', r.get('TIC', tic))}",
                             "P": float(p) if np.isfinite(p) else None,
                             "sep": float(r["sep_arcsec"])})
        rec["catalogue"] = {"objects": objs, "sources": {k: ("ok" if v == "ok" else v) for k, v
                                                         in src.items() if k != "gaia"}}
        info[str(tic)] = rec
    V.mkdir(parents=True, exist_ok=True)
    INFO.write_text(json.dumps(info, indent=1, default=str))
    return info


def load_info() -> dict:
    return json.loads(INFO.read_text()) if INFO.exists() else {}


if __name__ == "__main__":
    inf = build(sys.argv[1:])
    n_err = sum(any(v != "ok" for v in r["sources"].values()) for r in inf.values())
    print(len(inf), "stars;", n_err, "with a failed source")
    print({k: v for k, v in next(iter(inf.values()))["sources"].items()})
