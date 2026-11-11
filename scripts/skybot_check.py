"""Known-asteroid check (SkyBoT cone search from TESS, observer code C57) for top candidates.

    uv run python -I scripts/skybot_check.py --scorer ens_v2 --top 200 \
        --extra labels/triage_s0032_v1.csv
    uv run python -I scripts/skybot_check.py --sector 43 --scorer ens_v3 --top 0 \
        --candidates outputs/candidates/v1/skybot_targets_s0043.csv --tag cand_v1

docs/VETTING.md F3. For each of the top ``--top`` real light curves of ``--scorer`` that survive
its veto (score >= 0; known positives excluded), and for the TICs in ``--extra`` (evaluation
only), query SkyBoT at the epoch of the star's strongest ``cmv3`` event: the strongest residual
event for a surviving star, the strongest event of either kind for an ``--extra`` star. Cone
radius ``--radius`` (default 5 arcmin), asteroids only (``-objFilter=100``), observer
``-loc=C57`` was meant to give TESS's viewpoint, but **SkyBoT returns geocentric positions for
C57** (2026-11-11: (535) Montague at BJD 2459482.9 is at the same place, to 0.02", for C57 and for
the geocentre ``500``, while JPL Horizons with TESS as observer, ``500@-95``, puts it 190" away,
where the TESS pixels saw it cross). TESS orbits up to ~80 Earth radii out, so its parallax for a
main-belt object is several arcmin. The default ``--observer tess`` therefore queries the
geocentre with a wider cone (``--radius`` 12') and moves every body to TESS's viewpoint with its
geocentric distance and TESS's geocentric position (Horizons vectors, cached per sector).
``--observer c57`` reproduces the old, uncorrected check.

A star is *explained* when a body with V <= ``--vmax`` lies within ``--match`` arcsec
(6 px x 21 arcsec, the stamp's background pixels) at the epoch. Each body's offset is also
propagated over +-1 h with its SkyBoT motion to report the closest approach in that window.

Anonymous HTTP, serial, ``--sleep`` seconds between queries. Every response is cached with its
URL, time and HTTP status under ``<data>/skybot/sNNNN/`` (re-runs read the cache). Writes
``outputs/veto/cmv3/sNNNN/skybot_<scorer>.csv``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oddball.common.http import ANON_USER_AGENT
from oddball.common.paths import data_root, outputs_root, sector_dir
from oddball.common.splits import split_array
from oddball.tess.arrays import dir_metadata

URL = "https://ssp.imcce.fr/webservices/skybot/api/conesearch.php"
UA = ANON_USER_AGENT
COLS = ["num", "name", "ra_h", "dec_deg", "cls", "vmag", "err_arcsec", "d_arcsec",
        "dra_arcsec_h", "ddec_arcsec_h", "dg_au", "dh_au"]  # fmt: skip


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def query(ra: float, dec: float, jd: float, radius_deg: float, cache: Path, sleep: float,
          loc: str = "C57"):  # fmt: skip
    params = {"-ep": f"{jd:.5f}", "-ra": f"{ra:.6f}", "-dec": f"{dec:.6f}",
              "-rd": f"{radius_deg:.5f}", "-loc": loc, "-mime": "text", "-output": "basic",
              "-objFilter": "100"}  # fmt: skip
    url = URL + "?" + urllib.parse.urlencode(params)
    key = f"ra{ra:.5f}_dec{dec:.5f}_jd{jd:.5f}_r{radius_deg:.4f}".replace("-", "m")
    if loc != "C57":  # old cache files (C57) keep their names
        key += f"_loc{loc}"
    f = cache / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text()), False
    status, text, err = None, "", None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as r:
                status = r.status
                text = r.read().decode("utf-8", "replace")
            err = None
            break
        except Exception as e:  # noqa: BLE001
            err = repr(e)
            time.sleep(sleep * (attempt + 2))
    rec = {"url": url, "utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
           "http_status": status, "error": err, "text": text}  # fmt: skip
    if err is None:
        f.write_text(json.dumps(rec))
    time.sleep(sleep)
    return rec, True


def parse(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= len(COLS):
            rows.append(parts[: len(COLS)])
    df = pd.DataFrame(rows, columns=COLS)
    for c in ("vmag", "err_arcsec", "d_arcsec", "dra_arcsec_h", "ddec_arcsec_h"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


HORIZONS = "https://ssd.jpl.nasa.gov/api/horizons.api"
AU_KM = 149597870.7


def tess_vectors(jd0: float, jd1: float, cache: Path) -> pd.DataFrame:
    """TESS's geocentric ICRF position (au) every 30 min over [jd0, jd1] (TDB), from JPL Horizons
    (target -95, centre 500@399), cached as one CSV per span."""
    f = cache / f"tess_vec_{jd0:.1f}_{jd1:.1f}.csv"
    if f.exists():
        return pd.read_csv(f)
    params = {"format": "text", "COMMAND": "'-95'", "CENTER": "'500@399'",
              "EPHEM_TYPE": "'VECTORS'", "REF_PLANE": "'FRAME'", "START_TIME": f"'JD {jd0:.4f}'",
              "STOP_TIME": f"'JD {jd1:.4f}'", "STEP_SIZE": "'30m'", "OUT_UNITS": "'AU-D'",
              "VEC_TABLE": "'1'", "CSV_FORMAT": "'YES'"}  # fmt: skip
    req = urllib.request.Request(HORIZONS + "?" + urllib.parse.urlencode(params),
                                 headers={"User-Agent": UA})  # fmt: skip
    with urllib.request.urlopen(req, timeout=120) as r:
        text = r.read().decode("utf-8", "replace")
    body = text.split("$$SOE")[1].split("$$EOE")[0]
    rows = [[float(p[0]), float(p[2]), float(p[3]), float(p[4])]
            for p in (ln.split(",") for ln in body.strip().splitlines())]  # fmt: skip
    df = pd.DataFrame(rows, columns=["jd", "x", "y", "z"])
    if len(df) < 2 or df[["x", "y", "z"]].abs().max().max() > 0.01:  # TESS stays < ~0.003 au
        raise RuntimeError("Horizons TESS vectors look wrong")
    df.to_csv(f, index=False)
    return df


def to_tess_view(ra_deg, dec_deg, dg_au, r_tess) -> tuple[np.ndarray, np.ndarray]:
    """Shift geocentric astrometric positions to TESS's viewpoint (parallax), given each body's
    geocentric distance (au) and TESS's geocentric position (au, ICRF)."""
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    u = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)
    v = u * np.asarray(dg_au, float)[:, None] - np.asarray(r_tess, float)[None, :]
    ra2 = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0
    dec2 = np.degrees(np.arcsin(v[:, 2] / np.linalg.norm(v, axis=1)))
    return ra2, dec2


def sep_arcsec(ra1, dec1, ra0, dec0):
    r1, d1, r0, d0 = (np.radians(np.asarray(x, float)) for x in (ra1, dec1, ra0, dec0))
    c = np.sin(d1) * np.sin(d0) + np.cos(d1) * np.cos(d0) * np.cos(r1 - r0)
    return np.degrees(np.arccos(np.clip(c, -1, 1))) * 3600.0


def closest_in_window(df: pd.DataFrame, ra0: float, dec0: float, hours: float = 1.0):
    """Closest approach (arcsec) of each body to (ra0, dec0) over +-hours (linear motion).
    Uses ``ra_deg``/``dec_deg`` when present (parallax-corrected), else the SkyBoT columns."""
    if df.empty:
        return np.array([])
    if "ra_deg" in df:
        ra, dec = df.ra_deg.to_numpy(), df.dec_deg_num.to_numpy()
    else:
        ra = np.array([_hms(x) for x in df.ra_h]) * 15.0
        dec = np.array([_dms(x) for x in df.dec_deg])
    x = (ra - ra0) * np.cos(np.radians(dec0)) * 3600.0
    y = (dec - dec0) * 3600.0
    vx, vy = df.dra_arcsec_h.to_numpy(), df.ddec_arcsec_h.to_numpy()
    tt = np.linspace(-hours, hours, 49)[None, :]
    d = np.hypot(x[:, None] + vx[:, None] * tt, y[:, None] + vy[:, None] * tt)
    return d.min(1)


def _hms(s: str) -> float:
    h, m, sec = (float(v) for v in s.split())
    return h + m / 60 + sec / 3600


def _dms(s: str) -> float:
    p = s.split()
    sign = -1.0 if p[0].startswith("-") else 1.0
    d, m, sec = abs(float(p[0])), float(p[1]), float(p[2])
    return sign * (d + m / 60 + sec / 3600)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sector", type=int, default=32)
    ap.add_argument("--scorer", default="ens_v2")
    ap.add_argument("--veto", default="cmv3")
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--extra", type=Path, default=None)
    ap.add_argument("--candidates", type=Path, default=None,
                    help="CSV with a tic column: queried like the top list (strongest residual "
                         "event, or an optional epoch_btjd column), group 'cand' "
                         "(e.g. outputs/candidates/v1/skybot_targets_*.csv)")
    ap.add_argument("--tag", default=None, help="output name skybot_<tag>.csv (default: scorer)")
    ap.add_argument("--observer", choices=("tess", "c57"), default="tess",
                    help="tess: geocentric query + parallax shift to TESS (default); c57: old")
    ap.add_argument("--radius", type=float, default=None,
                    help="arcmin (default 12 for --observer tess, 5 for c57)")
    ap.add_argument("--match", type=float, default=126.0, help="arcsec")
    ap.add_argument("--vmax", type=float, default=19.0)
    ap.add_argument("--sleep", type=float, default=2.0)
    a = ap.parse_args()
    if a.radius is None:
        a.radius = 12.0 if a.observer == "tess" else 5.0
    loc = "500" if a.observer == "tess" else "C57"
    sec = a.sector
    cache = data_root() / "skybot" / f"s{sec:04d}"
    cache.mkdir(parents=True, exist_ok=True)
    meta = dir_metadata(sector_dir(sec))[["tic", "ra", "dec", "tmag", "camera", "ccd"]]
    vd = outputs_root() / "veto" / a.veto / f"s{sec:04d}"
    v = pd.read_parquet(vd / "real.parquet")
    s = pd.read_parquet(outputs_root() / "scores" / a.scorer / f"s{sec:04d}" / "real.parquet",
                        columns=["tic", "score"])  # fmt: skip
    s = s[(s.score >= 0) & (split_array(s.tic.to_numpy()) != "known")]
    top = s.sort_values("score", ascending=False).head(a.top).assign(group="top")
    top["rank"] = np.arange(1, len(top) + 1)
    rows = [top]
    if a.candidates:
        cd = pd.read_csv(a.candidates)
        cr = pd.DataFrame({"tic": cd.tic.astype("int64"), "group": "cand",
                           "rank": np.arange(1, len(cd) + 1)})  # fmt: skip
        if "epoch_btjd" in cd:  # optional per-target epoch (e.g. a crossing found by eye)
            cr["epoch_given"] = cd.epoch_btjd.astype(float).to_numpy()
        rows.append(cr)
    if a.extra:
        ex = pd.read_csv(a.extra)
        rows.append(pd.DataFrame({"tic": ex.tic, "group": "extra:" + ex.label.astype(str)}))
    q = pd.concat(rows, ignore_index=True).merge(meta, on="tic", how="left")
    vc = f"{a.veto}_"
    q = q.merge(v[["tic", f"{vc}res_t", f"{vc}res_snr", f"{vc}sys_t", f"{vc}sys_snr",
                   f"{vc}res_ev", f"{vc}sys_ev", f"{vc}flag"]], on="tic", how="left")  # fmt: skip
    use_sys = (~q.group.isin(["top", "cand"])) & (q[f"{vc}sys_snr"] > q[f"{vc}res_snr"])
    q["epoch_btjd"] = np.where(use_sys, q[f"{vc}sys_t"], q[f"{vc}res_t"])
    q["epoch_event"] = np.where(use_sys, q[f"{vc}sys_ev"], q[f"{vc}res_ev"])
    if "epoch_given" in q:
        g = np.isfinite(q.epoch_given.to_numpy(float))
        q.loc[g, "epoch_btjd"] = q.loc[g, "epoch_given"]
        q.loc[g, "epoch_event"] = "given"
        q = q.drop(columns="epoch_given")
    out = []
    n_new = 0
    tv = None
    if a.observer == "tess":
        ep = q.epoch_btjd.to_numpy(float)
        ep = ep[np.isfinite(ep)] + 2457000.0
        tv = tess_vectors(np.floor(ep.min()) - 1.0, np.ceil(ep.max()) + 1.0, cache)
    for i, r in enumerate(q.itertuples(index=False)):
        if not np.isfinite(r.epoch_btjd):
            out.append({"tic": r.tic, "status": "no event"})
            continue
        jd = float(r.epoch_btjd) + 2457000.0  # BJD_TDB; SkyBoT epoch (minutes off: < 1 px)
        rec, new = query(float(r.ra), float(r.dec), jd, a.radius / 60.0, cache, a.sleep, loc)
        n_new += new
        if rec.get("error"):
            out.append({"tic": r.tic, "status": f"error {rec['error']}"})
            continue
        df = parse(rec["text"])
        if tv is not None and len(df):
            rt = [np.interp(jd, tv.jd, tv[c]) for c in ("x", "y", "z")]
            ra_g = np.array([_hms(x) for x in df.ra_h]) * 15.0
            de_g = np.array([_dms(x) for x in df.dec_deg])
            df["ra_deg"], df["dec_deg_num"] = to_tess_view(ra_g, de_g, pd.to_numeric(
                df.dg_au, errors="coerce").to_numpy(), rt)  # fmt: skip
            df["d_arcsec"] = sep_arcsec(df.ra_deg, df.dec_deg_num, float(r.ra), float(r.dec))
        dmin = closest_in_window(df, float(r.ra), float(r.dec))
        bright = df.vmag.to_numpy() <= a.vmax if len(df) else np.array([], bool)
        at_ep = df.d_arcsec.to_numpy() if len(df) else np.array([])
        m_ep = bright & (at_ep <= a.match)
        m_win = bright & (dmin <= a.match)
        best = int(np.argmin(dmin)) if len(df) else None
        out.append({
            "tic": r.tic, "status": "ok", "n_bodies": len(df),
            "explained_at_epoch": bool(m_ep.any()), "explained_pm1h": bool(m_win.any()),
            "nearest_name": df.name.iloc[best] if best is not None else "",
            "nearest_v": float(df.vmag.iloc[best]) if best is not None else np.nan,
            "nearest_d_epoch": float(at_ep[best]) if best is not None else np.nan,
            "nearest_d_pm1h": float(dmin[best]) if best is not None else np.nan,
            "nearest_speed": (float(np.hypot(df.dra_arcsec_h.iloc[best],
                                             df.ddec_arcsec_h.iloc[best]))
                              if best is not None else np.nan)})  # fmt: skip
        if (i + 1) % 20 == 0:
            log(f"  {i + 1}/{len(q)} queried ({n_new} new)")
    # one result per query row (a star can be in the top list and in --extra)
    res = pd.concat([q.reset_index(drop=True),
                     pd.DataFrame(out).drop(columns="tic").reset_index(drop=True)], axis=1)
    tag = a.tag or a.scorer
    p = vd / f"skybot_{tag}.csv"
    res.to_csv(p, index=False)
    prov = {"service": URL, "observer": a.observer, "skybot_loc": loc,
            "parallax": "Horizons TESS (-95) geocentric vectors" if tv is not None else None,
            "objFilter": 100, "radius_arcmin": a.radius,
            "match_arcsec": a.match, "vmax": a.vmax, "sleep_s": a.sleep, "cache": str(cache),
            "n_queries": len(q), "n_new": n_new, "scorer": a.scorer, "veto": a.veto,
            "top": a.top, "candidates": str(a.candidates) if a.candidates else None,
            "tag": tag, "run_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}  # fmt: skip
    (cache / f"provenance_{tag}.json").write_text(json.dumps(prov, indent=1))
    g = res.groupby("group").agg(n=("tic", "size"), explained=("explained_pm1h", "mean"))
    log(f"wrote {p}\n{g}")


if __name__ == "__main__":
    main()
