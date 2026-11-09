"""Pull Rubin/LSST objects (n_det >= 2) around young-star catalogue entries and cross-match.

Step 1 queries ALeRCE once per sky tile that holds catalogue stars (cached on disk, so
re-runs are free). Step 2 cross-matches locally with a KD-tree.

    uv run python -I scripts/rubin_crossmatch.py [--cell-deg 2] [--workers 2] [--max-tiles N]

Outputs (data/rubin/):
    tiles.csv            per-tile query log (n objects, pages, truncated, seconds)
    rubin_objects.parquet  every LSST object returned (deduplicated on oid)
    matches.parquet      catalogue <-> Rubin pairs within --radius arcsec
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from oddball.rubin import catalog, crossmatch
from oddball.rubin.alerce_api import BASE_URL, AlerceLSST, data_root

_local = threading.local()
OBJ_COLS = ["oid", "meanra", "meandec", "firstmjd", "lastmjd", "n_det", "n_forced", "class_name",
            "classifier_name", "probability"]


def client() -> AlerceLSST:
    if not hasattr(_local, "api"):
        _local.api = AlerceLSST(min_interval=0.5)
    return _local.api


def query_tile(row, stars: np.ndarray, min_ndet: int, max_pages: int,
               star_switch: int) -> tuple[dict, list[dict]]:
    """Page through a tile's cone. If the tile is dense (> 1 page) but holds at most
    ``star_switch`` catalogue stars, switch to one 2.5" cone per star instead (n_cat requests
    instead of up to ``max_pages``)."""
    api = client()
    t = time.time()
    items: list[dict] = []
    pages = 0
    truncated = False
    has_lsst = False
    mode = "tile"
    err = ""
    try:
        for page in range(1, max_pages + 1):
            d = api.list_objects(page=page, ra=round(row.ra_c, 5), dec=round(row.dec_c, 5),
                                 radius=round(row.radius_deg * 3600, 1),
                                 n_det=[min_ndet, 1_000_000])
            pages += 1
            got = (d or {}).get("items") or []
            has_lsst = has_lsst or bool(got)
            items.extend(got)
            if not (d or {}).get("has_next"):
                break
            if page == 1 and len(stars) <= star_switch:
                mode = "stars"
                items = []
                for ra, dec in stars:
                    items.extend(star_cone(ra, dec, min_ndet))
                break
        else:
            truncated = True
    except Exception as e:  # noqa: BLE001 - log and move on; cache makes a re-run resume
        err = f"{type(e).__name__}: {str(e)[:120]}"
    log = {"tile_id": row.tile_id, "ra_c": row.ra_c, "dec_c": row.dec_c,
           "radius_deg": row.radius_deg, "n_cat": row.n_cat, "n_obj": len(items),
           "has_lsst": has_lsst, "mode": mode, "pages": pages, "truncated": truncated,
           "error": err, "seconds": round(time.time() - t, 2)}
    return log, [{k: it.get(k) for k in OBJ_COLS} for it in items]


def star_cone(ra: float, dec: float, min_ndet: int, radius_arcsec: float = 2.5) -> list[dict]:
    try:
        d = client().list_objects(ra=round(ra, 6), dec=round(dec, 6), radius=radius_arcsec,
                                  n_det=[min_ndet, 1_000_000])
    except Exception:  # noqa: BLE001
        return []
    return [{k: it.get(k) for k in OBJ_COLS} for it in (d or {}).get("items") or []]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-deg", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--min-ndet", type=int, default=2)
    ap.add_argument("--max-pages", type=int, default=100)
    ap.add_argument("--max-tiles", type=int, default=0)
    ap.add_argument("--radius", type=float, default=1.0, help="match radius, arcsec")
    ap.add_argument("--star-switch", type=int, default=50,
                    help="dense tiles with <= this many catalogue stars use per-star cones")
    args = ap.parse_args()

    out_dir = data_root() / "rubin"
    cat = catalog.load_union()
    tiles = crossmatch.make_tiles(cat["ra"], cat["dec"], cell_deg=args.cell_deg)
    tiles = tiles.sort_values("n_cat", ascending=False).reset_index(drop=True)
    if args.max_tiles:
        tiles = tiles.head(args.max_tiles)
    print(f"{len(cat)} catalogue rows, {len(tiles)} tiles", flush=True)
    tid_all = crossmatch.tile_ids(cat["ra"], cat["dec"], cell_deg=args.cell_deg)
    stars_by_tile = {k: v[["ra", "dec"]].to_numpy()
                     for k, v in cat.assign(tile_id=tid_all).groupby("tile_id")}

    logs, objs = [], []
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(query_tile, r, stars_by_tile[r.tile_id], args.min_ndet,
                          args.max_pages, args.star_switch) for r in tiles.itertuples()]
        for k, f in enumerate(as_completed(futs), 1):
            log, items = f.result()
            logs.append(log)
            objs.extend(items)
            if log["has_lsst"] or log["error"] or k % 100 == 0:
                print(f"[{k}/{len(tiles)} {time.time()-t0:.0f}s] {log['tile_id']} "
                      f"ra={log['ra_c']:.1f} dec={log['dec_c']:.1f} n_cat={log['n_cat']} "
                      f"n_obj={log['n_obj']} mode={log['mode']} {log['error']}", flush=True)
            if k % 200 == 0:
                pd.DataFrame(logs).to_csv(out_dir / "tiles.csv", index=False)
    logs_df = pd.DataFrame(logs)
    logs_df.to_csv(out_dir / "tiles.csv", index=False)

    # Truncated (very dense) tiles: fall back to one small cone per catalogue star.
    trunc = set(logs_df.loc[logs_df["truncated"].astype(bool), "tile_id"])
    n_star_cones = 0
    if trunc:
        stars = cat[pd.Series(tid_all).isin(trunc).to_numpy()]
        print(f"{len(trunc)} truncated tiles -> {len(stars)} per-star cones", flush=True)
        with ThreadPoolExecutor(args.workers) as ex:
            futs = [ex.submit(star_cone, r.ra, r.dec, args.min_ndet) for r in stars.itertuples()]
            for k, f in enumerate(as_completed(futs), 1):
                objs.extend(f.result())
                if k % 500 == 0:
                    print(f"  star cones {k}/{len(futs)}", flush=True)
        n_star_cones = len(stars)
    objs_df = pd.DataFrame(objs, columns=OBJ_COLS).drop_duplicates("oid").reset_index(drop=True)
    objs_df["oid"] = objs_df["oid"].astype("int64")
    objs_df.to_parquet(out_dir / "rubin_objects.parquet", index=False)

    m = crossmatch.match(cat, objs_df, radius_arcsec=args.radius)
    m.to_parquet(out_dir / "matches.parquet", index=False)
    prov = {"run_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "api": BASE_URL,
            "cell_deg": args.cell_deg, "min_ndet": args.min_ndet, "radius_arcsec": args.radius,
            "tiles": len(tiles), "tiles_with_lsst": int(logs_df["has_lsst"].sum()),
            "tiles_star_mode": int((logs_df["mode"] == "stars").sum()),
            "tiles_error": int((logs_df["error"] != "").sum()),
            "tiles_truncated": int(logs_df["truncated"].sum()),
            "star_cones_for_truncated_tiles": n_star_cones,
            "rubin_objects": len(objs_df), "match_pairs": len(m),
            "matched_oids": int(m["oid"].nunique()) if len(m) else 0,
            "matched_cat_rows": int(m.groupby(["cat", "cat_id"]).ngroups) if len(m) else 0,
            "by_catalog": m.groupby("cat")["cat_id"].nunique().to_dict() if len(m) else {}}
    (out_dir / "crossmatch_provenance.json").write_text(json.dumps(prov, indent=1))
    print(json.dumps(prov, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
