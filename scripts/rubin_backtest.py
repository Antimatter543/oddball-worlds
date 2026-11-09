"""Backtest the outburst detector over every young star matched to a Rubin object.

Needs data/rubin/matches.parquet + rubin_objects.parquet (scripts/rubin_crossmatch.py).

    uv run python -I scripts/rubin_backtest.py [--workers 2] [--limit N] [--sep 1.0]

Outputs:
    data/rubin/backtest_results.parquet  one row per matched Rubin object (all verdicts)
    outputs/rubin/candidates.csv         flagged objects, ranked
    outputs/rubin/plots/<oid>.png        light curves of candidates (+ near misses)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time

from oddball.rubin.alerce_api import BASE_URL, AlerceLSST, data_root
from oddball.rubin.detect import DetectConfig, detect_object
from oddball.rubin.lightcurve import assemble
from oddball.rubin.plots import plot_candidate

_local = threading.local()


def client() -> AlerceLSST:
    if not hasattr(_local, "api"):
        _local.api = AlerceLSST(min_interval=0.5)
    return _local.api


def load_lc(oid: int) -> pd.DataFrame:
    api = client()
    return assemble(api.detections(oid), api.forced_photometry(oid))


def run_one(oid: int, cfg: DetectConfig):
    try:
        lc = load_lc(oid)
        res = detect_object(lc, cfg)
        s = res.summary()
        s.update({"oid": oid, "n_lc": len(lc), "n_bad": int(lc["bad"].sum()) if len(lc) else 0,
                  "bands_seen": "".join(sorted(set(lc["band"]))) if len(lc) else "",
                  "lc_mjd_min": float(lc["mjd"].min()) if len(lc) else np.nan,
                  "lc_mjd_max": float(lc["mjd"].max()) if len(lc) else np.nan,
                  "error": ""})
        return s, lc, res
    except Exception as e:  # noqa: BLE001 - log and continue
        return {"oid": oid, "flagged": False, "error": f"{type(e).__name__}: {str(e)[:150]}"}, \
            None, None


def mjd_to_date(m):
    return Time(m, format="mjd").to_value("iso", subfmt="date") if np.isfinite(m) else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sep", type=float, default=1.0, help="max match separation, arcsec")
    ap.add_argument("--min-points", type=int, default=2,
                    help="min n_det+n_forced (2 suffices with a template baseline)")
    ap.add_argument("--near-miss-plots", type=int, default=10)
    args = ap.parse_args()
    root = data_root() / "rubin"
    out = Path("outputs/rubin")
    (out / "plots").mkdir(parents=True, exist_ok=True)

    m = pd.read_parquet(root / "matches.parquet")
    objs = pd.read_parquet(root / "rubin_objects.parquet").set_index("oid")
    m = m[m["sep_arcsec"] <= args.sep]
    # one row per Rubin object, carrying every catalogue identity that matched it
    agg = m.sort_values("sep_arcsec").groupby("oid").agg(
        cat_ids=("cat_id", lambda s: "|".join(dict.fromkeys(s))),
        cats=("cat", lambda s: ",".join(sorted(set(s)))),
        sep_arcsec=("sep_arcsec", "min"), n_rubin_near=("n_rubin_near", "max"),
        n_cat_rows=("cat_id", "size"))
    agg = agg.join(objs[["meanra", "meandec", "n_det", "n_forced", "firstmjd", "lastmjd",
                         "class_name", "classifier_name"]])
    agg["name"] = agg["cat_ids"].str.split("|").str[0]
    eligible = agg[(agg["n_det"].fillna(0) + agg["n_forced"].fillna(0)) >= args.min_points]
    if args.limit:
        eligible = eligible.head(args.limit)
    print(f"matched Rubin objects: {len(agg)}; eligible (n_det+n_forced>={args.min_points}): {len(eligible)}",
          flush=True)

    cfg = DetectConfig()
    rows, keep = [], {}
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run_one, int(oid), cfg): int(oid) for oid in eligible.index}
        for k, f in enumerate(as_completed(futs), 1):
            s, lc, res = f.result()
            rows.append(s)
            pd_ = s.get("p_peak_dmag", np.nan)
            if s.get("flagged") or (lc is not None and np.isfinite(pd_) and pd_ >= 0.7):
                keep[s["oid"]] = (lc, res)
            if k % 50 == 0 or s.get("flagged"):
                print(f"[{k}/{len(futs)} {time.time()-t0:.0f}s] oid={s['oid']} "
                      f"flagged={s.get('flagged')} dmag={pd_} {s.get('reasons', '')} "
                      f"{s.get('error', '')}", flush=True)
    res_df = pd.DataFrame(rows).set_index("oid").join(agg, how="left").reset_index()
    c = SkyCoord(ra=res_df["meanra"].to_numpy() * u.deg, dec=res_df["meandec"].to_numpy() * u.deg)
    res_df["l"], res_df["b"] = c.galactic.l.deg, c.galactic.b.deg
    res_df.to_parquet(root / "backtest_results.parquet", index=False)

    cand = res_df[res_df["flagged"].astype(bool)].copy()
    cand = cand.sort_values(["p_peak_dmag"], ascending=False)
    near = res_df[~res_df["flagged"].astype(bool) & (res_df["p_peak_dmag"] >= 0.7)].sort_values(
        "p_peak_dmag", ascending=False).head(args.near_miss_plots)
    for df, tag in ((cand, "cand"), (near, "near")):
        paths = []
        for r in df.itertuples():
            lc, res = keep.get(r.oid, (None, None))
            if lc is None:
                paths.append("")
                continue
            title = (f"{tag} {r.name}  LSST {r.oid}  ra={r.meanra:.5f} dec={r.meandec:.5f}  "
                     f"l={r.l:.2f} b={r.b:.2f}\npeak d={r.p_peak_dmag:.2f} mag ({r.p_band}) "
                     f"rise {r.p_rise_time_lo:.0f}-{r.p_rise_time:.0f} d  {r.flags} {r.reasons}")
            paths.append(str(plot_candidate(lc, res, title, out / "plots" / f"{tag}_{r.oid}.png")))
        df["plot"] = paths
    cand["t_peak_date"] = cand["p_t_peak"].map(mjd_to_date)
    cand["t_run_start_date"] = cand["p_t_run_start"].map(mjd_to_date)
    cols = ["name", "oid", "meanra", "meandec", "l", "b", "cats", "cat_ids", "sep_arcsec",
            "n_rubin_near", "p_band", "p_baseline_mag", "p_peak_mag", "p_peak_dmag",
            "t_run_start_date", "t_peak_date", "p_rise_time_lo", "p_rise_time", "p_rise_resolved",
            "p_run_len", "xband", "flags", "diff_ratio", "n_det", "n_forced", "bands_seen",
            "class_name", "plot"]
    cand[cols].to_csv(out / "candidates.csv", index=False)
    near.to_csv(out / "near_misses.csv", index=False)
    reasons = res_df["reasons"].fillna("").str.split(":").str[0].value_counts().to_dict()
    prov = {"run_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "api": BASE_URL,
            "config": cfg.__dict__, "matched_objects": len(agg), "eligible": len(eligible),
            "processed": len(res_df), "errors": int((res_df["error"].fillna("") != "").sum()),
            "candidates": len(cand), "reasons": reasons,
            "lc_mjd_range": [float(res_df["lc_mjd_min"].min()), float(res_df["lc_mjd_max"].max())]}
    (out / "backtest_provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    print(json.dumps(prov, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
