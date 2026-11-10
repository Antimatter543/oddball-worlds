"""Second null set ``n2``: instrument and Solar-System false positives for the ``cmv3`` veto
(docs/VETTING.md §2, "Also add to the null-injection set"; docs/ML_DESIGN.md §13).

Same layout as ``oddball.inject.build`` (``<data>/injections/n2/sNNNN/{val,test}/``), every row
``is_null``. Three kinds:

* ``null_asteroid``: a body moving on a straight line across a CCD at 15-80 arcsec/h for 3-10 d.
  For every star it passes within 6 px, at distance ``r(t)``, the 30-min-binned light curve gets
  ``SAP += A (g_ap(r) - K_BKG g_bkg(r))``, ``BKG += A K_BKG g_bkg(r)``, ``PDCSAP += SAP change /
  CROWDSAP``, with ``g_ap = exp(-r^2 / 2)`` (1-px PSF in the aperture) and ``g_bkg`` = 1 for
  2.5 <= r <= 6 px (the background pixels), smooth edges. Raw aperture flux (SAP + BKG) moves
  only when the body is in the aperture. So a star the body passes at 3 px dips with a rising
  background, and one it crosses brightens with a dip on either side (dip-spike-dip).
  ``A = 10^(-0.4 (m - T))`` with one magnitude ``m`` per track, chosen so that the first
  evaluated star's peak PDCSAP change is the drawn depth.
* ``null_step``: a flux step at a momentum dump with an exponential recovery (tau 0.5-5 d), in
  20 stars of one CCD, random sign per star (pointing moves flux into or out of the aperture).
* ``null_column``: the same event (Gaussian, box or ramp, 2 h - 2 d) in every star within
  +-1.5 columns of one CCD column (a bleed or smear column), and nowhere else.

**Carriers.** A track or column passes stars of every split; all of them get the signal (so the
cross-star tests see what the real sky would show), but only stars of the set's own split are
evaluated. The others are written with ``split = "carrier"`` (never ranked; known positives are
never used). Depths of evaluated stars: log-uniform over 0.1-30% at the first evaluated star of
a track; physically scaled for the others.

Positions are the ``tess-point`` (col, row) pixel positions cached by ``cmv2``.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from oddball.common.paths import sector_dir
from oddball.common.splits import known_positive_tics, split_array
from oddball.common.strata import ccd_index, stratum_id
from oddball.inject.apply import apply_model, sap_dilution
from oddball.inject.build import (
    INJ_COLUMNS,
    INJ_SCHEMA,
    SPLIT_CODE,
    _atomic,
    _git_sha,
    _read_hosts,
    _times,
    injection_dir,
)
from oddball.inject.sampling import DEPTH_RANGE, draw_null_event, loguniform
from oddball.tess.arrays import dir_metadata

KINDS2 = ("null_asteroid", "null_step", "null_column")
K_BKG = 0.5
R_AP, R_IN, R_OUT = 1.0, 2.5, 6.0  # px: PSF sigma, background annulus inner / outer radius
SPEED = (15.0, 80.0)  # arcsec/h
DUR_TRACK = (3.0, 10.0)  # d
PX_ARCSEC = 21.0
NSUB = 15


def _asteroid_response(t_bins: np.ndarray, bin_days: float, sx, sy, x0, y0, vx, vy, t0):
    """Per-unit-amplitude (ap, bkg) responses averaged over each bin (``NSUB`` sub-samples).
    ``vx, vy`` in px/day, ``t0`` the time at (x0, y0)."""
    off = (np.arange(NSUB) + 0.5) / NSUB - 0.5
    tt = t_bins[:, None] + off[None, :] * bin_days
    r = np.hypot(x0 + vx * (tt - t0) - sx, y0 + vy * (tt - t0) - sy)
    g_ap = np.exp(-0.5 * (r / R_AP) ** 2)
    edge = 0.3
    g_bkg = ((1 / (1 + np.exp(np.clip(-(r - R_IN) / edge, -50, 50))))
             * (1 / (1 + np.exp(np.clip((r - R_OUT) / edge, -50, 50)))))
    return g_ap.mean(1), g_bkg.mean(1)


def _star_table(meta: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    m = meta.merge(cells[["tic", "col", "row", "own_ccd"]], on="tic", how="left")
    m = m[m.own_ccd.fillna(False).astype(bool)].copy()
    m["split"] = split_array(m.tic.to_numpy())
    m = m[m.split != "known"]
    assert not m.tic.isin(known_positive_tics()).any()
    m["ccd_idx"] = ccd_index(m.camera, m.ccd)
    m["stratum"] = stratum_id(m.tmag, m.camera, m.ccd)
    return m.reset_index(drop=True)


def plan_asteroids(stars, split, n_tracks, rng, t_range, bin_days, used):
    """List of (track params, member rows: tic, d_min, t_c, evaluated)."""
    out = []
    tries = 0
    while len(out) < n_tracks and tries < 50 * n_tracks:
        tries += 1
        c = int(rng.integers(16))
        st = stars[(stars.ccd_idx == c) & ~stars.tic.isin(used)]
        if len(st) < 50:
            continue
        v_px_d = loguniform(rng, *SPEED) / PX_ARCSEC * 24.0
        ang = rng.uniform(0, 2 * np.pi)
        vx, vy = v_px_d * np.cos(ang), v_px_d * np.sin(ang)
        dur = loguniform(rng, *DUR_TRACK)
        t_a = rng.uniform(t_range[0] - dur / 2, t_range[1] - dur / 2)
        x0, y0 = rng.uniform(45, 2092), rng.uniform(1, 2048)
        ex, ey = st.col.to_numpy() - x0, st.row.to_numpy() - y0
        tc = t_a + (ex * vx + ey * vy) / v_px_d**2
        dperp = np.abs(ex * vy - ey * vx) / v_px_d
        ok = (dperp <= R_OUT) & (tc >= t_a) & (tc <= t_a + dur)
        mem = st[ok].assign(d_min=dperp[ok], t_c=tc[ok])
        ev = mem[mem.split == split]
        if len(mem) < 3 or len(ev) == 0:
            continue
        used.update(mem.tic.tolist())
        p = {"x0": x0, "y0": y0, "vx": vx, "vy": vy, "t0": t_a, "t_end": t_a + dur,
             "speed_arcsec_h": v_px_d * PX_ARCSEC / 24.0, "ccd_idx": c}  # fmt: skip
        out.append((p, mem.assign(evaluated=mem.split == split)))
    return out


def plan_columns(stars, split, n_events, rng, used, half=1.5):
    out = []
    tries = 0
    while len(out) < n_events and tries < 100 * n_events:
        tries += 1
        c = int(rng.integers(16))
        c0 = rng.uniform(100, 2000)
        st = stars[(stars.ccd_idx == c) & ~stars.tic.isin(used)]
        mem = st[np.abs(st.col.to_numpy() - c0) <= half]
        if len(mem) < 4 or not (mem.split == split).any():
            continue
        used.update(mem.tic.tolist())
        out.append(({"column": c0, "ccd_idx": c}, mem.assign(evaluated=mem.split == split)))
    return out


def plan_steps(stars, split, n_events, hosts_per_event, rng, used):
    out = []
    ccds = rng.choice(16, size=n_events, replace=n_events > 16)
    for c in ccds:
        st = stars[(stars.ccd_idx == c) & (stars.split == split) & ~stars.tic.isin(used)]
        g = st.iloc[rng.choice(len(st), size=min(hosts_per_event, len(st)), replace=False)]
        used.update(g.tic.tolist())
        out.append(({"ccd_idx": int(c)}, g.assign(evaluated=True)))
    return out


def build_null2_set(sector: int, split: str, cells: pd.DataFrame, name: str = "n2",
                    seed: int = 20261110, n_tracks: int = 40, n_columns: int = 40,
                    n_steps: int = 10, hosts_per_step: int = 20, log=print) -> dict:  # fmt: skip
    if split not in ("val", "test"):
        raise ValueError("null sets are built for val or test only")
    out = injection_dir(name, sector, split)
    if (out / "injections.parquet").exists():
        raise SystemExit(f"{out} already exists; delete it to rebuild")
    (out / "shards").mkdir(parents=True, exist_ok=True)
    meta = dir_metadata(sector_dir(sector))
    bin_days = float(meta.bin_minutes.iloc[0]) / 1440.0
    stars = _star_table(meta, cells)
    rng = np.random.default_rng(np.random.SeedSequence([seed, sector, SPLIT_CODE[split]]))
    used: set[int] = set()
    t_range = (float(meta.t_start.median()), float(meta.t_stop.median()))

    plan = []  # one row per light curve: tic, kind, event_id, evaluated, params
    eid = 0
    for p, mem in plan_asteroids(stars, split, n_tracks, rng, t_range, bin_days, used):
        p["depth_target"] = loguniform(rng, *DEPTH_RANGE)
        for r in mem.itertuples(index=False):
            plan.append({"tic": int(r.tic), "kind": "null_asteroid", "event_id": eid,
                         "evaluated": bool(r.evaluated), "p": {**p, "d_min": float(r.d_min),
                                                               "t_c": float(r.t_c)}})  # fmt: skip
        eid += 1
    for p, mem in plan_columns(stars, split, n_columns, rng, used):
        p["depth"] = loguniform(rng, *DEPTH_RANGE)
        p["seed"] = int(rng.integers(2**31))
        for r in mem.itertuples(index=False):
            plan.append({"tic": int(r.tic), "kind": "null_column", "event_id": eid,
                         "evaluated": bool(r.evaluated), "p": dict(p)})  # fmt: skip
        eid += 1
    for p, mem in plan_steps(stars, split, n_steps, hosts_per_step, rng, used):
        p["depth"] = loguniform(rng, *DEPTH_RANGE)
        p["tau_d"] = loguniform(rng, 0.5, 5.0)
        for r in mem.itertuples(index=False):
            plan.append({"tic": int(r.tic), "kind": "null_step", "event_id": eid,
                         "evaluated": True, "p": {**p, "sign": int(rng.choice([-1, 1]))}})
        eid += 1
    plan = pd.DataFrame(plan).merge(stars, on="tic", how="left", validate="one_to_one")
    plan["inj_id"] = [f"{name}-s{sector:04d}-{split}-{i:05d}" for i in range(len(plan))]

    # per-event shared quantities that need the light curves (column event shape/time, dump
    # times, the track magnitude) are fixed on first use below
    dump_times = _dump_times(sector, plan[plan.kind == "null_step"], bin_days)
    col_models: dict[int, dict] = {}
    track_mag: dict[int, float] = {}
    step_dump: dict[int, float] = {}
    # the magnitude of each track is set by its first evaluated star: process those first
    plan["_order"] = np.where(plan.evaluated, 0, 1)
    rows_by_shard = {}
    hosts_all = {}
    for shard, grp in plan.groupby("shard"):
        rows_by_shard[int(shard)] = grp
        hosts_all.update(_read_hosts(sector, int(shard), grp.tic.tolist()))
    meta_cols = [c for c in meta.columns if c != "shard"]
    records, lc_out = {}, {}
    for r in plan.sort_values(["_order", "event_id", "tic"]).itertuples(index=False):
        h = hosts_all[int(r.tic)]
        flux = np.asarray(h["flux"], np.float64)
        sap = np.asarray(h["sap_flux"], np.float64)
        bkg = np.asarray(h["bkg"], np.float64)
        t = _times(int(h["bin0"]), len(flux), bin_days)
        valid = np.isfinite(flux)
        c = sap_dilution(r.crowdsap)
        p = dict(r.p)
        if r.kind == "null_asteroid":
            pr = p
            gap, gbk = _asteroid_response(t, bin_days, r.col, r.row, pr["x0"], pr["y0"],
                                          pr["vx"], pr["vy"], pr["t0"])  # fmt: skip
            d_sap = gap - K_BKG * gbk
            if r.event_id not in track_mag:
                peak = float(np.max(np.abs(d_sap[valid]) / c)) if valid.any() else 0.0
                if peak <= 0:
                    peak = 1.0
                a0 = pr["depth_target"] / peak
                track_mag[r.event_id] = float(r.tmag) - 2.5 * np.log10(a0)
            a = 10 ** (-0.4 * (track_mag[r.event_id] - float(r.tmag)))
            s_add, b_add = a * d_sap, a * K_BKG * gbk
            f_add = s_add / c
            mx = np.max(np.abs(f_add)) if f_add.size else 0.0
            if mx > 0.5:  # keep any single star's change physical (<= 50%)
                k = 0.5 / mx
                s_add, b_add, f_add = s_add * k, b_add * k, f_add * k
            f_new, s_new, b_new = flux + f_add, sap + s_add, bkg + b_add
            dev = f_add
            p.update(mag=track_mag[r.event_id], amp=float(a))
            t0 = (float(t[valid][np.argmax(np.abs(dev[valid]))]) if valid.any()
                  else float(pr["t_c"]))  # fmt: skip
            depth = float(np.max(np.abs(dev[valid]))) if valid.any() else 0.0
            dur = 2 * R_OUT / max(np.hypot(pr["vx"], pr["vy"]), 1e-6)
        elif r.kind == "null_step":
            if r.event_id not in step_dump:
                step_dump[r.event_id] = (float(rng.choice(dump_times)) if len(dump_times)
                                         else float(t[len(t) // 2]))  # fmt: skip
            td = step_dump[r.event_id]
            p["dump_btjd"] = td
            m = _step_model(t, bin_days, td, p["sign"] * p["depth"], p["tau_d"])
            f_new, s_new, _ = _apply_m(flux, sap, m, c)
            b_new = bkg
            dev = m - 1.0
            t0, depth, dur = td, float(p["depth"]), float(p["tau_d"])
        else:  # null_column
            if r.event_id not in col_models:
                erng = np.random.default_rng(p["seed"])
                q = draw_null_event("null_ccd", erng, t, valid.astype(float), p["depth"])
                col_models[r.event_id] = q
            q = col_models[r.event_id]
            f_new, s_new, mm = apply_model(flux, sap, t, q, bin_days, crowdsap=r.crowdsap)
            b_new = bkg
            dev = mm - 1.0
            p["model"] = q
            t0, depth, dur = float(q["t0_btjd"]), float(q["depth"]), float(q["duration_d"])
        split_out = split if r.evaluated else "carrier"
        k = int(np.argmin(np.abs(t - t0)))
        lc_out[r.inj_id] = {
            "tic": int(r.tic), "sector": int(sector), "bin0": int(h["bin0"]),
            "flux": np.asarray(f_new, np.float32), "flux_err": np.asarray(h["flux_err"], np.float32),
            "npts": np.asarray(h["npts"], np.uint16), "qflags": np.asarray(h["qflags"], np.int32),
            "sap_flux": np.asarray(s_new, np.float32), "bkg": np.asarray(b_new, np.float32),
            "inj_id": r.inj_id, "host_tic": int(r.tic)}  # fmt: skip
        records[r.inj_id] = {
            "inj_id": r.inj_id, "host_tic": int(r.tic), "sector": int(sector),
            "split": split_out, "kind": r.kind, "depth": depth, "duration_d": dur,
            "t0_btjd": t0, "params_json": json.dumps(p, sort_keys=True, default=float),
            "is_null": True, "event_id": int(r.event_id), "tmag": float(r.tmag),
            "camera": int(r.camera), "ccd": int(r.ccd), "stratum": int(r.stratum),
            "crowdsap": float(r.crowdsap) if r.crowdsap is not None else np.nan,
            "sap_depth": depth * c,
            "binned_depth": float(np.max(np.abs(dev[valid]))) if valid.any() else 0.0,
            "peak_bin_valid": bool(valid[k])}  # fmt: skip
        records[r.inj_id]["_meta"] = {cc: getattr(r, cc) for cc in meta_cols}

    # write shard by shard (source-shard order)
    for out_idx, (shard, grp) in enumerate(sorted(rows_by_shard.items())):
        ids = grp.inj_id.tolist()
        lc_tbl = pa.Table.from_pylist([lc_out[i] for i in ids], schema=INJ_SCHEMA)
        _atomic(lc_tbl, out / "shards" / f"lc_{out_idx:05d}.parquet", row_group_size=1000)
        mrows = []
        for i in ids:
            mr = dict(records[i]["_meta"])
            mr.update(inj_id=i, host_tic=records[i]["host_tic"], kind=records[i]["kind"],
                      is_null=True, shard=out_idx)  # fmt: skip
            mrows.append(mr)
        _atomic(pa.Table.from_pylist(mrows), out / "shards" / f"meta_{out_idx:05d}.parquet")
    inj = pd.DataFrame([{k: v for k, v in rec.items() if k != "_meta"}
                        for rec in records.values()], columns=INJ_COLUMNS)  # fmt: skip
    inj = inj.sort_values("inj_id").reset_index(drop=True)
    _atomic(pa.Table.from_pandas(inj, preserve_index=False), out / "injections.parquet")
    metas = [pq.read_table(f) for f in sorted((out / "shards").glob("meta_*.parquet"))]
    _atomic(pa.concat_tables(metas, promote_options="default"), out / "metadata.parquet")
    src_prov = sector_dir(sector) / "provenance.json"
    src = json.loads(src_prov.read_text()) if src_prov.exists() else {}
    ev = inj[inj.split == split]
    prov = {
        "name": name, "sector": sector, "split": split, "seed": seed,
        "created_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _git_sha(), "n_lightcurves": len(inj), "n_evaluated": len(ev),
        "n_carrier": int((inj.split == "carrier").sum()),
        "evaluated_by_kind": ev.kind.value_counts().to_dict(),
        "events_by_kind": inj.groupby("kind").event_id.nunique().to_dict(),
        "k_bkg": K_BKG, "radii_px": [R_AP, R_IN, R_OUT], "speed_arcsec_h": SPEED,
        "track_days": DUR_TRACK, "dump_times": [float(x) for x in dump_times],
        "host_rule": "evaluated: split_of(tic)==split, known excluded; carriers: any split "
                     "except known (split='carrier', never ranked); positions from tess-point",
        "source": {k: src.get(k) for k in ("product", "hlsp_version", "doi",
                                           "download_dates_utc", "spoc_data_release",
                                           "bin_minutes", "quality_bitmask")},
    }  # fmt: skip
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    log(f"wrote {len(inj)} light curves ({len(ev)} evaluated) to {out}")
    return prov


def _dump_times(sector, hosts: pd.DataFrame, bin_days: float) -> np.ndarray:
    if hosts.empty:
        return np.array([])
    dump = {}
    for shard, grp in hosts.groupby("shard"):
        for r in _read_hosts(sector, int(shard), grp.tic.tolist()).values():
            q = np.asarray(r["qflags"], np.int64)
            for j in np.flatnonzero((q & 32) > 0):
                g = int(r["bin0"]) + int(j)
                dump[g] = dump.get(g, 0) + 1
    if not dump:
        return np.array([])
    g = np.array(sorted(k for k, v in dump.items() if v >= 0.5 * len(hosts)))
    if g.size == 0:
        return np.array([])
    runs = np.split(g, np.flatnonzero(np.diff(g) > 1) + 1)
    return np.array([(r_[0] + 0.5) * bin_days for r_ in runs])


def _step_model(t: np.ndarray, bin_days: float, td: float, amp: float, tau: float):
    off = (np.arange(NSUB) + 0.5) / NSUB - 0.5
    tt = t[:, None] + off[None, :] * bin_days
    m = np.where(tt >= td, amp * np.exp(-np.clip(tt - td, 0, None) / tau), 0.0)
    return 1.0 + m.mean(1)


def _apply_m(flux, sap, m, c):
    m_sap = 1.0 - c * (1.0 - m)
    return flux * m, sap * m_sap, m

