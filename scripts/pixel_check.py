"""TESScut pixel check for every card of a gallery (docs/VETTING.md F11; README TODO).

    uv run python -I scripts/pixel_check.py --gallery cand_v1_n150
    uv run python -I scripts/pixel_check.py --gallery cand_v1_n150 --events extra.csv

For each card: the FFI cutout of its main sector, the difference image around its strongest
``cmv3`` residual event, and the signal-locus map (``oddball.gallery.pixels``). ``--events`` adds
or overrides rows (columns tic, sector, t_ev, w_hours, kind), e.g. another sector's event.

Writes ``<gallery>/cards/<tic>_s<NNNN>_pix.png`` (or ``..._t<t_ev>_pix.png`` for ``--events``
rows), ``<gallery>/pixels.csv``, and keeps each binned cube under
``<data>/gallery/pixels/<tic>_s<NNNN>.npz`` (no FITS is kept). Anonymous HTTP, at most
``--workers`` concurrent requests (TESScut allows 10 per user).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from oddball.common.paths import data_root, outputs_root
from oddball.gallery import pixels as P
from oddball.gallery import research as R
from oddball.tess.store import load_lightcurves

EV_HOURS = {"2h": 2.0, "8h": 8.0, "32h": 32.0}


def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def parse_event(ev: str) -> tuple[float, str]:
    parts = str(ev).split("/")
    kind = parts[1] if len(parts) > 1 else "dip"
    w_h = EV_HOURS.get(parts[2], 8.0) if len(parts) > 2 else 8.0
    return w_h, kind


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--events", type=Path, default=None)
    ap.add_argument("--only-events", action="store_true", help="skip the gallery's own events")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--size", type=int, default=11)
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    gdir = outputs_root() / "gallery" / a.gallery
    man = json.loads((gdir / "manifest.json").read_text())
    jobs = []
    if not a.only_events:
        for c in man["cards"]:
            ps = {p["sector"]: p for p in c["candidate"]["per_sector"]}[c["sector"]]
            w_h, kind = parse_event(ps["res_event"])
            jobs.append({"tic": c["tic"], "sector": c["sector"], "ra": c["ra"], "dec": c["dec"],
                         "tmag": c["tmag"], "t_ev": ps["res_t"], "w_hours": w_h, "kind": kind,
                         "tag": f"{c['tic']}_s{c['sector']:04d}"})  # fmt: skip
    if a.events:
        pos = {c["tic"]: c for c in man["cards"]}
        for r in pd.read_csv(a.events).itertuples():
            c = pos[int(r.tic)]
            jobs.append({"tic": int(r.tic), "sector": int(r.sector), "ra": c["ra"], "dec": c["dec"],
                         "tmag": c["tmag"], "t_ev": float(r.t_ev), "w_hours": float(r.w_hours),
                         "kind": str(r.kind),
                         "tag": f"{int(r.tic)}_s{int(r.sector):04d}_t{float(r.t_ev):.2f}"})  # fmt: skip
    jobs = pd.DataFrame(jobs)
    log(f"{len(jobs)} pixel jobs for {jobs.tic.nunique()} stars")
    # Gaia neighbours for the overlay (cached research queries)
    gaia = R.query_catalogue("gaia", jobs[["tic", "ra", "dec", "tmag"]].drop_duplicates("tic"))
    # template light curves (PDCSAP) per sector
    tmpl = {}
    for s, g in jobs.groupby("sector"):
        lcs = load_lightcurves(int(s), g.tic.unique())
        for tic, lc in lcs.items():
            f = np.asarray(lc["flux"], float)
            tmpl[(int(tic), int(s))] = (np.asarray(lc["time"], float), f / np.nanmedian(f) - 1.0)
    cube_dir = data_root() / "gallery" / "pixels"
    rows = []

    def one(j):
        cp = cube_dir / f"{j.tic}_s{j.sector:04d}.npz"
        if cp.exists() and not a.redo:
            cube = P.load_cube(cp)
        else:
            cube = P.fetch_cube(j.ra, j.dec, j.sector, a.size)
            P.save_cube(cube, cp)
        tt, tf = tmpl.get((j.tic, j.sector), (None, None))
        res = P.analyse(cube, j.ra, j.dec, j.t_ev, j.w_hours / 24.0, j.kind, tt, tf)
        g = gaia[gaia.tic == j.tic]
        nbs = []
        if len(g):
            from astropy.coordinates import SkyCoord

            gm = pd.to_numeric(g.Gmag, errors="coerce").to_numpy()
            tgt = R.gaia_summary(g, j.tmag)
            g0 = tgt.get("G", np.nan)
            sel = np.isfinite(gm) & (gm - g0 < 5.0) & (g.sep_arcsec.to_numpy() > 1.0)
            if sel.sum() > 25:  # crowded fields (LMC, 30 Dor): only the 25 brightest
                cut = np.sort(gm[sel])[24]
                sel &= gm <= cut
            if sel.any():
                sc = SkyCoord(g.RA_ICRS.to_numpy()[sel], g.DE_ICRS.to_numpy()[sel], unit="deg")
                x, y = P.wcs_of(cube).world_to_pixel(sc)
                nbs = list(zip(x, y, gm[sel] - g0, strict=True))
        title = (f"TIC {j.tic}  S{j.sector}  pixel check of the event at BTJD {j.t_ev:.2f} "
                 f"({j.kind}, {j.w_hours:g} h)")  # fmt: skip
        png = f"cards/{j.tag}_pix.png"
        P.render(res, cube, gdir / png, title, j.t_ev, j.w_hours / 24.0, nbs)
        s = P.summary(res)
        s.update({"tic": j.tic, "sector": j.sector, "t_ev": j.t_ev, "w_hours": j.w_hours,
                  "kind": j.kind, "png": png, "verdict": P.verdict(res)})  # fmt: skip
        return s

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(one, j): j for j in jobs.itertuples(index=False)}
        for i, fu in enumerate(as_completed(futs)):
            j = futs[fu]
            try:
                rows.append(fu.result())
            except Exception as e:  # noqa: BLE001
                rows.append({"tic": j.tic, "sector": j.sector, "t_ev": j.t_ev,
                             "verdict": f"ERROR {type(e).__name__}: {str(e)[:120]}"})  # fmt: skip
            if (i + 1) % 10 == 0:
                log(f"  {i + 1}/{len(jobs)} done ({time.time() - t0:.0f} s)")
    out = pd.DataFrame(rows)
    p = gdir / ("pixels.csv" if not a.only_events else "pixels_events.csv")
    if p.exists() and (a.events is not None):
        old = pd.read_csv(p)
        out = pd.concat([old, out]).drop_duplicates(["tic", "sector", "t_ev"], keep="last")
    out.to_csv(p, index=False)
    (gdir / "pixels_provenance.json").write_text(json.dumps({
        "service": P.TESSCUT, "size_px": a.size, "workers": a.workers, "n_jobs": len(jobs),
        "cube_dir": str(cube_dir), "utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "product": "TESScut SPOC FFI cutouts, 30-min bins, local background subtracted"}, indent=1))
    log(f"wrote {p} ({len(out)} rows, {out.verdict.str.startswith('ERROR').sum()} errors)")


if __name__ == "__main__":
    main()
