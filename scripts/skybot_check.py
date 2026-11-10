"""Known-asteroid check (SkyBoT cone search from TESS, observer code C57) for top candidates.

    uv run python -I scripts/skybot_check.py --scorer ens_v2 --top 200 \
        --extra labels/triage_s0032_v1.csv

docs/VETTING.md F3. For each of the top ``--top`` real light curves of ``--scorer`` that survive
its veto (score >= 0; known positives excluded), and for the TICs in ``--extra`` (evaluation
only), query SkyBoT at the epoch of the star's strongest ``cmv3`` event: the strongest residual
event for a surviving star, the strongest event of either kind for an ``--extra`` star. Cone
radius ``--radius`` (default 5 arcmin), asteroids only (``-objFilter=100``), observer
``-loc=C57`` (TESS; a geocentric query would be off by arcminutes of parallax).

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

from oddball.common.paths import data_root, outputs_root, sector_dir
from oddball.common.splits import split_array
from oddball.tess.arrays import dir_metadata

URL = "https://ssp.imcce.fr/webservices/skybot/api/conesearch.php"
UA = "oddball-worlds research script (anonymous; contact via github antimatter543)"
COLS = ["num", "name", "ra_h", "dec_deg", "cls", "vmag", "err_arcsec", "d_arcsec",
        "dra_arcsec_h", "ddec_arcsec_h", "dg_au", "dh_au"]  # fmt: skip


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def query(ra: float, dec: float, jd: float, radius_deg: float, cache: Path, sleep: float):
    params = {"-ep": f"{jd:.5f}", "-ra": f"{ra:.6f}", "-dec": f"{dec:.6f}",
              "-rd": f"{radius_deg:.5f}", "-loc": "C57", "-mime": "text", "-output": "basic",
              "-objFilter": "100"}  # fmt: skip
    url = URL + "?" + urllib.parse.urlencode(params)
    key = f"ra{ra:.5f}_dec{dec:.5f}_jd{jd:.5f}_r{radius_deg:.4f}".replace("-", "m")
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


def closest_in_window(df: pd.DataFrame, ra0: float, dec0: float, hours: float = 1.0):
    """Closest approach (arcsec) of each body to (ra0, dec0) over +-hours (linear motion)."""
    if df.empty:
        return np.array([])
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
    ap.add_argument("--radius", type=float, default=5.0, help="arcmin")
    ap.add_argument("--match", type=float, default=126.0, help="arcsec")
    ap.add_argument("--vmax", type=float, default=19.0)
    ap.add_argument("--sleep", type=float, default=2.0)
    a = ap.parse_args()
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
    if a.extra:
        ex = pd.read_csv(a.extra)
        rows.append(pd.DataFrame({"tic": ex.tic, "group": "extra:" + ex.label.astype(str)}))
    q = pd.concat(rows, ignore_index=True).merge(meta, on="tic", how="left")
    vc = f"{a.veto}_"
    q = q.merge(v[["tic", f"{vc}res_t", f"{vc}res_snr", f"{vc}sys_t", f"{vc}sys_snr",
                   f"{vc}res_ev", f"{vc}sys_ev", f"{vc}flag"]], on="tic", how="left")  # fmt: skip
    use_sys = (q.group != "top") & (q[f"{vc}sys_snr"] > q[f"{vc}res_snr"])
    q["epoch_btjd"] = np.where(use_sys, q[f"{vc}sys_t"], q[f"{vc}res_t"])
    q["epoch_event"] = np.where(use_sys, q[f"{vc}sys_ev"], q[f"{vc}res_ev"])
    out = []
    n_new = 0
    for i, r in enumerate(q.itertuples(index=False)):
        if not np.isfinite(r.epoch_btjd):
            out.append({"tic": r.tic, "status": "no event"})
            continue
        jd = float(r.epoch_btjd) + 2457000.0  # BJD_TDB; SkyBoT epoch (minutes off: < 1 px)
        rec, new = query(float(r.ra), float(r.dec), jd, a.radius / 60.0, cache, a.sleep)
        n_new += new
        if rec.get("error"):
            out.append({"tic": r.tic, "status": f"error {rec['error']}"})
            continue
        df = parse(rec["text"])
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
    res = q.merge(pd.DataFrame(out), on="tic", how="left")
    p = vd / f"skybot_{a.scorer}.csv"
    res.to_csv(p, index=False)
    prov = {"service": URL, "observer": "C57", "objFilter": 100, "radius_arcmin": a.radius,
            "match_arcsec": a.match, "vmax": a.vmax, "sleep_s": a.sleep, "cache": str(cache),
            "n_queries": len(q), "n_new": n_new, "scorer": a.scorer, "veto": a.veto,
            "top": a.top, "run_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}  # fmt: skip
    (cache / f"provenance_{a.scorer}.json").write_text(json.dumps(prov, indent=1))
    g = res.groupby("group").agg(n=("tic", "size"), explained=("explained_pm1h", "mean"))
    log(f"wrote {p}\n{g}")


if __name__ == "__main__":
    main()
