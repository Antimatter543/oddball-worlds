"""Pre-render a labelling gallery for any scorer (docs/ML_DESIGN.md §7).

    uv run python -I scripts/gallery_build.py --scorer ssl_v2_iforest --sector 32 --n 60

Reads ``outputs/scores/<scorer>/s<NNNN>/real.parquet`` (the §9 score-file contract), picks
~70% top-ranked stars plus ~30% exploration (``oddball.gallery.select``), and writes
``outputs/gallery/<name>/{manifest.json,cards/*.png}``. Serve it with
``scripts/gallery_serve.py``. Already-labelled stars (``labels/labels_v1.csv``) are skipped
unless ``--relabel``; known positives are never shown.

Optional inputs, used when present: a time-resolved anomaly map for the scorer (the ML
``timeresolved_*`` files, found via ``outputs/ml/*/score_meta.json``, or ``--timeresolved``),
the known-class predictions (``outputs/knownclass/<kc>/``), the classical scorer's file (for
its score and the ``sys_flag`` systematics columns). Catalogue answers are cached under
``data/gallery/xmatch/`` (``--offline`` uses only the cache).

``--candidates v1`` instead builds one card per star of a cross-sector candidate list
(``scripts/build_candidates.py``), in its priority order, skipping known positives and
labelled stars: the star's best-ranked sector is the main panel (zoom on its strongest
``cmv3`` residual event, CCD neighbours), and every other ingested sector of the star is shown
below with its ranks there.

TODO: TESScut pixel thumbnails (§7) are not drawn yet.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from oddball.common.paths import REPO_ROOT, data_root, outputs_root
from oddball.gallery import cards as C
from oddball.gallery import xmatch as X
from oddball.gallery.labels import DEFAULT_LABELS_CSV, LabelStore, validate_name
from oddball.gallery.select import select, subscore_columns
from oddball.knownclass.catalogs import _sep
from oddball.ml.data import sector_origin
from oddball.tess.plots import surprise_time
from oddball.tess.store import load_lightcurves, metadata

META_COLS = ["tic", "camera", "ccd", "tmag", "ra", "dec", "crowdsap", "rms_bin", "teff",
             "radius", "n_bins_filled", "n_bins"]  # fmt: skip
SYS_COLS = ["cm_frac", "dump_frac", "sys_frac", "sys_flag", "n_anom"]
NEIGHBOUR_TMAG_MAX = 14.0


def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def jsonable(v):
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [jsonable(x) for x in v]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def read_rows(path: Path, tics, columns=None) -> pd.DataFrame:
    return ds.dataset(path).to_table(filter=ds.field("tic").isin([int(t) for t in tics]),
                                     columns=columns).to_pandas()  # fmt: skip


def find_timeresolved(scorer: str, sector: int) -> tuple[Path | None, str]:
    sdir = f"s{sector:04d}"
    direct = [outputs_root() / "scores" / scorer / sdir / "timeresolved_real.parquet",
              outputs_root() / "ml" / scorer / f"timeresolved_{sdir}_real.parquet"]  # fmt: skip
    for p in direct:
        if p.exists():
            return p, f"{scorer} time-resolved anomaly"
    for meta in sorted((outputs_root() / "ml").glob("*/score_meta.json")):
        try:
            sc = json.loads(meta.read_text()).get("scorers", {})
        except json.JSONDecodeError:
            continue
        p = meta.parent / f"timeresolved_{sdir}_real.parquet"
        if scorer in sc and p.exists():
            return p, f"{meta.parent.name} reconstruction surprise (same model)"
    return None, ""


def neighbours_for(row, meta: pd.DataFrame, k: int) -> pd.DataFrame:
    m = meta[(meta.camera == row.camera) & (meta.ccd == row.ccd) & (meta.tic != row.tic)]
    bright = m[m.tmag <= NEIGHBOUR_TMAG_MAX]
    m = bright if len(bright) >= k else m
    # cheap box pre-cut (0.5 deg), then exact separations
    cosd = max(np.cos(np.radians(row.dec)), 0.05)
    for r in (0.5, 2.0, 12.0):
        c = m[(np.abs(m.dec - row.dec) < r) & (np.abs(((m.ra - row.ra + 180) % 360) - 180) * cosd < r)]
        if len(c) >= k:
            break
    c = c.assign(sep_arcsec=_sep(row.ra, row.dec, c.ra.to_numpy(), c.dec.to_numpy()))
    return c.nsmallest(k, "sep_arcsec")


def rel(p: Path) -> str:
    """Path relative to the repo when inside it (symlinks resolved), else absolute."""
    p = Path(p)
    for base in (REPO_ROOT, REPO_ROOT.resolve()):
        for q in (p, p.resolve()):
            if q.is_relative_to(base):
                return str(q.relative_to(base))
    return str(p)


def git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10, check=False).stdout.strip()  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scorer", default=None, help="score file to draw from (one sector)")
    ap.add_argument("--candidates", default=None,
                    help="candidate list name (outputs/candidates/<name>/candidates.parquet, "
                         "scripts/build_candidates.py): one card per star across sectors, in "
                         "priority order; --n cards")
    ap.add_argument("--sector", type=int, default=32)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--explore", type=float, default=0.3, help="exploration share (§7: ~0.3)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--neighbours", type=int, default=5)
    ap.add_argument("--knownclass", default="kc_v1", help="known-class model dir (if present)")
    ap.add_argument("--timeresolved", type=Path, default=None,
                    help="time-resolved map parquet (tic, surprise) to use instead of auto")
    ap.add_argument("--name", default=None, help="gallery id (default <scorer>_s<NNNN>_n<N>_seed<S>)")
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS_CSV)
    ap.add_argument("--relabel", action="store_true", help="include already-labelled stars")
    ap.add_argument("--order", choices=("shuffled", "rank"), default=None,
                    help="card order: shuffled (default for --scorer, so the head does not "
                         "anchor the labels) or by rank (default for --candidates: priority)")
    ap.add_argument("--offline", action="store_true", help="catalogue cache only, no network")
    ap.add_argument("--refresh-xmatch", action="store_true")
    ap.add_argument("--dpi", type=int, default=80)
    a = ap.parse_args()
    a.order = a.order or ("rank" if a.candidates else "shuffled")
    if a.candidates:
        return build_candidate_gallery(a)
    if not a.scorer:
        ap.error("give --scorer or --candidates")

    scorer = validate_name(a.scorer, "scorer")
    sdir = f"s{a.sector:04d}"
    name = validate_name(a.name or f"{scorer}_{sdir}_n{a.n}_seed{a.seed}", "gallery name")
    score_path = outputs_root() / "scores" / scorer / sdir / "real.parquet"
    if not score_path.exists():
        raise SystemExit(f"no score file {score_path}")
    gdir = outputs_root() / "gallery" / name
    t_start = time.time()

    scores = pd.read_parquet(score_path)
    missing = {"tic", "sector", "score"} - set(scores.columns)
    if missing:
        raise SystemExit(f"{score_path} lacks columns {sorted(missing)} (ML_DESIGN §9)")
    scores = scores[scores.sector == a.sector]
    meta = metadata(a.sector)[META_COLS]
    scores = scores.merge(meta, on="tic", how="left")
    store = LabelStore(a.labels)
    labelled = set() if a.relabel else set(store.latest())
    chosen = select(scores, a.n, a.explore, a.seed, exclude=labelled)
    log(f"{scorer} {sdir}: {len(scores)} scored, {len(labelled)} already labelled; chose "
        f"{len(chosen)} ({chosen.pick.value_counts().to_dict()})")

    # sub-score percentiles over the whole sector
    subs = subscore_columns(scores)
    pct = {c: scores.set_index("tic")[c].rank(pct=True) for c in subs}
    sys_src = scorer if set(SYS_COLS) <= set(scores.columns) else None
    cls_path = outputs_root() / "scores" / "classical" / sdir / "real.parquet"
    cls = None
    if cls_path.exists() and (scorer != "classical" or sys_src is None):
        cls = pd.read_parquet(cls_path, columns=["tic", "score", *SYS_COLS]).set_index("tic")
        cls["rank"] = cls.score.rank(ascending=False, method="min").astype(int)
        sys_src = sys_src or "classical"
    kc_path = outputs_root() / "knownclass" / a.knownclass / sdir / "real.parquet"
    kc = read_rows(kc_path, chosen.tic).set_index("tic") if kc_path.exists() else None

    tr_path, tr_label = (a.timeresolved, f"time-resolved anomaly ({a.timeresolved.name})") \
        if a.timeresolved else find_timeresolved(scorer, a.sector)
    tr = {}
    origin = sector_origin(a.sector)
    if tr_path is not None:
        trd = read_rows(tr_path, chosen.tic, ["tic", "surprise"])
        tr = {int(r.tic): np.asarray(r.surprise, np.float32) for r in trd.itertuples()}
    log(f"time-resolved map: {tr_path} ({len(tr)} of {len(chosen)} chosen stars covered)")

    # neighbours and other sectors
    nbs = {int(r.tic): neighbours_for(r, meta, a.neighbours) for r in chosen.itertuples()}
    others: dict[int, list[tuple[int, int, int]]] = {}
    other_secs = sorted(int(p.parent.name[1:]) for p in
                        (data_root() / "tess-spoc").glob("s*/metadata.parquet"))  # fmt: skip
    other_secs = [s for s in other_secs if s != a.sector]
    for s in other_secs:
        m = read_rows(data_root() / "tess-spoc" / f"s{s:04d}" / "metadata.parquet", chosen.tic,
                      ["tic", "camera", "ccd"])  # fmt: skip
        for r in m.itertuples():
            others.setdefault(int(r.tic), []).append((s, int(r.camera), int(r.ccd)))
    need = set(chosen.tic.astype(int)) | {int(t) for d in nbs.values() for t in d.tic}
    log(f"loading {len(need)} light curves from {sdir}")
    lcs = load_lightcurves(a.sector, need)
    dumps = C.sector_dump_times(lcs.values())
    olcs = {}
    for s in other_secs:
        want = [t for t, v in others.items() if any(x[0] == s for x in v)]
        if want:
            olcs[s] = load_lightcurves(s, want)
    log(f"other sectors: {sum(len(v) for v in olcs.values())} light curves; "
        f"{len(dumps)} momentum-dump bins in {sdir}")

    # catalogues
    cat = X.CatalogCache(chosen.tic)
    tois = X.ToiLists(refresh=a.refresh_xmatch) if not a.offline else None
    otypes = X.simbad_otypedef() if not a.offline else {}
    sess = X._session()

    if a.order == "shuffled":
        chosen = chosen.sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    else:
        chosen = chosen.sort_values("rank").reset_index(drop=True)
    card_list = []
    for i, r in enumerate(chosen.itertuples()):
        tic = int(r.tic)
        lc = lcs.get(tic)
        if lc is None:
            log(f"  TIC {tic}: no light curve in {sdir}, skipped")
            continue
        if tic in tr:
            trace = C.Trace(surprise_time(origin, len(tr[tic])), tr[tic], tr_label)
            win = C.window_from_trace(trace)
        else:
            win, trace = C.classical_window(lc)
        nb_rows = nbs[tic]
        nb_list = [C.Neighbour(int(n.tic), float(n.tmag), float(n.sep_arcsec), lcs.get(int(n.tic)))
                   for n in nb_rows.itertuples()]  # fmt: skip
        oth = [C.OtherSector(s, cam, ccd, olcs[s][tic]) for s, cam, ccd in others.get(tic, [])
               if tic in olcs.get(s, {})]  # fmt: skip
        title = (f"TIC {tic}   S{a.sector} cam{int(r.camera)}/ccd{int(r.ccd)}   Tmag {r.tmag:.2f}"
                 f"   {scorer} rank {int(r.rank)} (top {r.rank_pct:.2f}%)   [{r.pick}]")  # fmt: skip
        png = f"cards/{tic}_{sdir}.png"
        C.render(C.CardData(tic, a.sector, title, lc, dumps, win, trace, nb_list, oth),
                 gdir / png, dpi=a.dpi)  # fmt: skip

        xm = None
        if not a.offline or (X.xmatch_dir() / "simbad" / f"{tic}.json").exists():
            xm = X.star_xmatch(tic, r.ra, r.dec, cat, tois or _NoToi(), a.refresh_xmatch, sess)
        card = {
            "tic": tic, "sector": a.sector, "png": png, "rank": int(r.rank),
            "rank_pct": float(r.rank_pct), "score": float(r.score), "pick": r.pick,
            "reason": r.reason, "split": r.split, "disagreement": float(r.disagreement),
            "tmag": float(r.tmag), "camera": int(r.camera), "ccd": int(r.ccd),
            "ra": float(r.ra), "dec": float(r.dec), "crowdsap": float(r.crowdsap),
            "rms_ppt": float(r.rms_bin) * 1e3, "teff": float(r.teff), "radius": float(r.radius),
            "subscores": {c: {"value": float(getattr(r, c)), "pct": float(pct[c].get(tic))}
                          for c in subs},
            "window": {"t0": win.t0, "t1": win.t1, "source": win.source},
            "timeresolved": trace.label if trace is not None else None,
            "neighbours": [{"tic": n.tic, "tmag": n.tmag, "sep_arcsec": round(n.sep_arcsec, 1),
                            "has_lc": n.lc is not None} for n in nb_list],
            "other_sectors": [o.sector for o in oth],
            "xmatch": xm,
        }  # fmt: skip
        if sys_src == scorer:
            card["sys"] = {c: getattr(r, c) for c in SYS_COLS}
        elif cls is not None and tic in cls.index:
            card["sys"] = {c: cls.at[tic, c] for c in SYS_COLS}
        if cls is not None and scorer != "classical" and tic in cls.index:
            card["classical"] = {"score": float(cls.at[tic, "score"]),
                                 "rank": int(cls.at[tic, "rank"])}  # fmt: skip
        if kc is not None and tic in kc.index:
            k = kc.loc[tic]
            probs = {c[5:]: float(k[c]) for c in kc.columns if c.startswith("kc_p_")
                     and c != "kc_pmax"}  # fmt: skip
            card["knownclass"] = {
                "model": a.knownclass, "pred": str(k.kc_pred), "pmax": float(k.kc_pmax),
                "explained": float(k.kc_explained), "cat_label": str(k.get("cat_label", "")),
                "top": sorted(probs.items(), key=lambda kv: -kv[1])[:3]}  # fmt: skip
        card_list.append(card)
        if (i + 1) % 10 == 0:
            log(f"  rendered {i + 1}/{len(chosen)}")

    manifest = {
        "gallery": name, "scorer": scorer, "sector": a.sector, "n": len(card_list),
        "built_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "git": git_sha(),
        "selection": {"n_requested": a.n, "explore_frac": a.explore, "seed": a.seed,
                      "order": a.order, "excluded_labelled": len(labelled),
                      "counts": chosen.pick.value_counts().to_dict(),
                      "rule": "top by score; explore-top5 = uniform in the rest of the top 5%; "
                              "explore-mid = ranks 5-25% weighted by sub-score disagreement^2; "
                              "split 'known' never shown"},
        "inputs": {
            "scores": {"path": rel(score_path), "rows": len(scores),
                "mtime_utc": dt.datetime.fromtimestamp(score_path.stat().st_mtime, dt.UTC)
                .isoformat(timespec="seconds")},
            "timeresolved": rel(tr_path) if tr_path else None,
            "knownclass": rel(kc_path) if kc is not None else None,
            "classical": rel(cls_path) if cls is not None else None, "sys_source": sys_src,
            "lightcurves": {"sector": a.sector, "product": "TESS-SPOC FFI (HLSP v1), 30-min bins",
                            "data_root": rel(data_root())},
            "xmatch": {"dir": rel(X.xmatch_dir()), "catalog_cache": cat.prov,
                       "simbad_servers": list(X.SIMBAD_TAP), "offline": a.offline,
                       "exofop": {"toi": X.TOI_URL, "ctoi": X.CTOI_URL}},
        },
        "otypes": {k: otypes[k] for k in sorted(used_otypes(card_list)) if k in otypes},
        "cards": card_list,
    }  # fmt: skip
    gdir.mkdir(parents=True, exist_ok=True)
    tmp = gdir / "manifest.json.tmp"
    tmp.write_text(json.dumps(jsonable(manifest), indent=1))
    tmp.replace(gdir / "manifest.json")
    log(f"wrote {len(card_list)} cards -> {gdir} in {time.time() - t_start:.0f} s")
    print(f"serve with: uv run python -I scripts/gallery_serve.py --gallery {name}")


# --------------------------------------------------------------------------------------------
# Cross-sector candidate galleries (--candidates)

EV_HOURS = {"2h": 2.0, "8h": 8.0, "32h": 32.0}


def event_window(t_ev: float, ev: str, snr: float) -> C.Window | None:
    """Zoom window centred on a cmv3 residual event (``ev`` = ``pdc/dip/8h/none``)."""
    if t_ev is None or not np.isfinite(t_ev):
        return None
    parts = str(ev).split("/")
    ch = parts[0] if parts and parts[0] in ("pdc", "sap") else "pdc"
    kind = parts[1] if len(parts) > 1 else "event"
    w_h = EV_HOURS.get(parts[2], 8.0) if len(parts) > 2 else 8.0
    hw = C._clip_halfwidth(1.5 * w_h / 24.0 + 0.5)
    name = {"pdc": "PDCSAP", "sap": "SAP"}[ch]
    return C.Window(t_ev - hw, t_ev + hw, ch, f"strongest cmv3 residual event: {name} {kind}, "
                    f"{w_h:g} h box, SNR {snr:.0f}, at BTJD {t_ev:.2f}")  # fmt: skip


def recon_traces(sector: int, tics) -> tuple[dict[int, np.ndarray], str | None]:
    """Per-bin ssl_v2 reconstruction surprise for ``tics`` from the run's memmap (all real stars
    of the sector, written by ``ml_score.py``), keyed by TIC; ({}, None) if absent."""
    from oddball.common.paths import sector_dir
    from oddball.ml.data import cache_dir_for

    mp = outputs_root() / "ml" / "ssl_v2" / f"score_meta_s{sector:04d}.json"
    if not mp.exists():
        return {}, None
    m = json.loads(mp.read_text())
    sfx = "_pt" if m.get("mask_seeding") == "tic" else ""
    f = Path(m["run"]) / "score" / f"surprise_real_s{sector:04d}_b{m['mask_block']}{sfx}.npy"
    if not f.exists():
        f = REPO_ROOT / "runs" / Path(m["run"]).name / "score" / f.name
    if not f.exists():
        return {}, None
    idx = pd.read_parquet(cache_dir_for(sector_dir(sector)) / "index.parquet", columns=["tic"])
    pos = pd.Series(np.arange(len(idx)), index=idx.tic.astype(int).to_numpy())
    sur = np.load(f, mmap_mode="r")
    if len(sur) != len(idx):
        log(f"  {f}: {len(sur)} rows but the cache has {len(idx)}; no trace")
        return {}, None
    out = {int(t): np.asarray(sur[pos[int(t)]], np.float32) for t in tics if int(t) in pos.index}
    return out, rel(f)


def build_candidate_gallery(a) -> None:
    lst = validate_name(a.candidates, "candidate list")
    cdir = outputs_root() / "candidates" / lst
    cand = pd.read_parquet(cdir / "candidates.parquet")
    ranks = pd.read_parquet(cdir / "candidates_allsectors.parquet")
    name = validate_name(a.name or f"cand_{lst}_n{a.n}", "gallery name")
    gdir = outputs_root() / "gallery" / name
    t_start = time.time()
    store = LabelStore(a.labels)
    labelled = set() if a.relabel else set(store.latest())
    pool = cand[cand.gallery_ok & ~cand.tic.isin(labelled)].sort_values("priority")
    chosen = pool.head(a.n).copy()
    log(f"candidate list {lst}: {len(cand)} stars, {int(cand.gallery_ok.sum())} gallery-eligible, "
        f"{len(labelled)} already labelled; chose {len(chosen)} "
        f"(multi-sector {int(chosen.multi.sum())}; best sector "
        f"{chosen.best_sector.value_counts().sort_index().to_dict()})")
    by_sec = {s: g.set_index("tic") for s, g in ranks.groupby("sector")}
    sectors = sorted(by_sec)

    # light curves: each star in every ingested sector; neighbours in its best sector
    metas = {s: metadata(s)[META_COLS] for s in sectors}
    nbs = {}
    for r in chosen.itertuples():
        m = metas[int(r.best_sector)]
        row = m[m.tic == r.tic].iloc[0]
        nbs[int(r.tic)] = neighbours_for(row, m, a.neighbours)
    lcs, dumps, traces, tr_files = {}, {}, {}, {}
    for s in sectors:
        want = {int(t) for t in chosen.tic if int(t) in by_sec[s].index}
        want |= {int(t) for r in chosen.itertuples() if int(r.best_sector) == s
                 for t in nbs[int(r.tic)].tic}  # fmt: skip
        if not want:
            continue
        lcs[s] = load_lightcurves(s, want)
        dumps[s] = C.sector_dump_times(lcs[s].values())
        primary = [int(r.tic) for r in chosen.itertuples() if int(r.best_sector) == s]
        traces[s], tr_files[s] = recon_traces(s, primary)
        log(f"  S{s}: {len(lcs[s])} light curves, {len(dumps[s])} dump bins, recon trace for "
            f"{len(traces[s])} of {len(primary)} primary stars")

    cat = X.CatalogCache(chosen.tic)
    tois = X.ToiLists(refresh=a.refresh_xmatch) if not a.offline else None
    otypes = X.simbad_otypedef() if not a.offline else {}
    sess = X._session()
    if a.order == "shuffled":
        chosen = chosen.sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    card_list = []
    for i, r in enumerate(chosen.itertuples()):
        tic, s0 = int(r.tic), int(r.best_sector)
        lc = lcs.get(s0, {}).get(tic)
        if lc is None:
            log(f"  TIC {tic}: no light curve in S{s0}, skipped")
            continue
        rk = by_sec[s0].loc[tic]
        m0 = metas[s0].set_index("tic").loc[tic]
        trace = None
        if tic in traces.get(s0, {}):
            v = traces[s0][tic]
            trace = C.Trace(surprise_time(sector_origin(s0), len(v)), v,
                            "ssl_v2 reconstruction surprise")  # fmt: skip
        win = event_window(rk.cmv3_res_t, rk.cmv3_res_ev, rk.cmv3_res_snr)
        if win is None:
            win, tr2 = (C.window_from_trace(trace), trace) if trace is not None else \
                C.classical_window(lc)
            trace = trace or tr2
        nb_list = [C.Neighbour(int(n.tic), float(n.tmag), float(n.sep_arcsec),
                               lcs[s0].get(int(n.tic))) for n in nbs[tic].itertuples()]  # fmt: skip

        def note(s, tic=tic):
            q = by_sec[s].loc[tic]
            tag = "CANDIDATE " if (q.in_cls or q.in_ens) else ""
            fl = " cmv3-flagged" if q.cmv3_flag else ""
            return (f"{tag}cls #{int(q.cls_rank)} / ens #{int(q.ens_rank)} of "
                    f"{int(q.n_stars)}{fl}")  # fmt: skip

        oth = []
        for s in sectors:
            if s == s0 or tic not in by_sec[s].index or tic not in lcs.get(s, {}):
                continue
            q = by_sec[s].loc[tic]
            oth.append(C.OtherSector(s, int(q.camera), int(q.ccd), lcs[s][tic], note=note(s),
                                     mark_t=float(q.cmv3_res_t)))  # fmt: skip
        secs_txt = f"cand in S{r.sectors_cand.replace(' ', ',S')}; obs S{r.sectors_obs.replace(' ', ',S')}"
        title = (f"TIC {tic}   S{s0} cam{int(rk.camera)}/ccd{int(rk.ccd)}   Tmag {m0.tmag:.2f}   "
                 f"{note(s0)}   [{secs_txt}]")  # fmt: skip
        png = f"cards/{tic}_s{s0:04d}.png"
        C.render(C.CardData(tic, s0, title, lc, dumps[s0], win, trace, nb_list, oth),
                 gdir / png, dpi=a.dpi)  # fmt: skip
        xm = None
        if not a.offline or (X.xmatch_dir() / "simbad" / f"{tic}.json").exists():
            xm = X.star_xmatch(tic, float(r.ra), float(r.dec), cat, tois or _NoToi(),
                               a.refresh_xmatch, sess)  # fmt: skip
        best_src = "classical" if rk.cls_rank <= rk.ens_rank else "ens_v3"
        per_sector = []
        for s in sectors:
            if tic in by_sec[s].index:
                q = by_sec[s].loc[tic]
                per_sector.append({"sector": s, "cls_rank": int(q.cls_rank),
                                   "ens_rank": int(q.ens_rank), "n_stars": int(q.n_stars),
                                   "candidate": bool(q.in_cls or q.in_ens),
                                   "cmv3_flag": bool(q.cmv3_flag), "eb_like": float(q.eb_like),
                                   "res_event": str(q.cmv3_res_ev),
                                   "res_t": float(q.cmv3_res_t)})  # fmt: skip
        card = {
            "tic": tic, "sector": s0, "png": png, "rank": int(r.best_rank),
            "rank_pct": 100.0 * float(r.best_rank) / float(rk.n_stars),
            "score": float(rk.cls_score if best_src == "classical" else rk.ens_score),
            "pick": "multi-sector" if r.multi else "single-sector",
            "reason": f"candidates {lst}: best {best_src} rank {int(r.best_rank)} in S{s0}; "
                      f"candidate in S{r.sectors_cand}",
            "split": r.split, "disagreement": 0.0, "priority": int(r.priority),
            "tmag": float(m0.tmag), "camera": int(rk.camera), "ccd": int(rk.ccd),
            "ra": float(r.ra), "dec": float(r.dec), "crowdsap": float(m0.crowdsap),
            "rms_ppt": float(m0.rms_bin) * 1e3, "teff": float(m0.teff),
            "radius": float(m0.radius),
            "subscores": {f"score_{k}": {"value": float(getattr(r, f"pct_{k}")),
                                         "pct": float(getattr(r, f"pct_{k}"))}
                          for k in ("classical", "iforest", "recon")},
            "window": {"t0": win.t0, "t1": win.t1, "source": win.source},
            "timeresolved": trace.label if trace is not None else None,
            "neighbours": [{"tic": n.tic, "tmag": n.tmag, "sep_arcsec": round(n.sep_arcsec, 1),
                            "has_lc": n.lc is not None} for n in nb_list],
            "other_sectors": [o.sector for o in oth],
            "candidate": {"list": lst, "multi": bool(r.multi), "sectors_cand": r.sectors_cand,
                          "sectors_obs": r.sectors_obs, "source": r.source,
                          "per_sector": per_sector, "cat_class": r.cat_class,
                          "kostov_nn": bool(r.kostov_nn), "eb_like_max": float(r.eb_like_max),
                          "skybot_checked": bool(r.skybot_checked),
                          "triage_s32": r.triage_s32},
            "xmatch": xm,
        }  # fmt: skip
        card_list.append(card)
        if (i + 1) % 10 == 0:
            log(f"  rendered {i + 1}/{len(chosen)}")

    manifest = {
        "gallery": name, "scorer": f"candidates_{lst}",
        "sector": "/".join(str(s) for s in sectors), "n": len(card_list),
        "built_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "git": git_sha(),
        "selection": {"n_requested": a.n, "order": a.order, "seed": a.seed,
                      "excluded_labelled": len(labelled),
                      "counts": chosen.multi.map({True: "multi-sector",
                                                  False: "single-sector"}).value_counts().to_dict(),
                      "rule": "gallery_ok (uncatalogued, not EB-like, not SkyBoT-explained, not a "
                              "known positive) in candidates.parquet priority order: multi-sector "
                              "first, S32-triaged stars last, then best per-sector rank"},
        "inputs": {
            "candidates": {"path": rel(cdir / "candidates.parquet"),
                           "provenance": json.loads((cdir / "provenance.json").read_text())},
            "timeresolved": {str(s): f for s, f in tr_files.items()},
            "lightcurves": {"sectors": sectors,
                            "product": "TESS-SPOC FFI (HLSP v1), 30-min bins",
                            "data_root": rel(data_root())},
            "xmatch": {"dir": rel(X.xmatch_dir()), "catalog_cache": cat.prov,
                       "simbad_servers": list(X.SIMBAD_TAP), "offline": a.offline,
                       "exofop": {"toi": X.TOI_URL, "ctoi": X.CTOI_URL}},
        },
        "otypes": {k: otypes[k] for k in sorted(used_otypes(card_list)) if k in otypes},
        "cards": card_list,
    }  # fmt: skip
    gdir.mkdir(parents=True, exist_ok=True)
    tmp = gdir / "manifest.json.tmp"
    tmp.write_text(json.dumps(jsonable(manifest), indent=1))
    tmp.replace(gdir / "manifest.json")
    log(f"wrote {len(card_list)} cards -> {gdir} in {time.time() - t_start:.0f} s")
    print(f"serve with: uv run python -I scripts/gallery_serve.py --gallery {name}")


def used_otypes(card_list: list[dict]) -> set[str]:
    """SIMBAD otype codes that appear on any card (their descriptions go in the manifest)."""
    out = set()
    for c in card_list:
        sb = (c.get("xmatch") or {}).get("simbad") or {}
        out |= {r["otype"] for r in sb.get("nearby", [])}
        if sb.get("by_tic_id"):
            out.add(sb["by_tic_id"]["otype"])
    return out


class _NoToi:
    ok = False

    def lookup(self, tic):  # pragma: no cover - offline only
        return None


if __name__ == "__main__":
    main()
