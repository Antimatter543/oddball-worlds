"""Candidate list v1 across the ingested sectors (frozen S32 rules, nothing tuned here).

    uv run python -I scripts/build_candidates.py                       # list + SkyBoT targets
    uv run python -I scripts/skybot_check.py --sector 43 --scorer ens_v3 --top 0 \
        --candidates outputs/candidates/v1/skybot_targets_s0043.csv --tag cand_v1   # and 44
    uv run python -I scripts/build_candidates.py                       # again: merges SkyBoT

Per sector (``oddball.score.candidates``): the union of the classical top ``--n`` and the
``ens_v3`` top ``--n``, both after the cmv3 veto and the EB down-weight. Merged by TIC with every
sector's ranks; ``multi`` marks stars that are candidates in 2+ sectors. Cross-matched against
``data/known_positives.csv`` (by TIC) and the catalogue labels in
``data/knownclass/labels_eb.parquet`` (VSX, Gaia DR3 vclassre / veb, Prsa+2022, Kostov+2025),
plus the eyeballed S32 triage (``labels/triage_s0032_v1.csv``).

Writes ``outputs/candidates/<name>/``: ``candidates.parquet`` (one row per TIC),
``candidates_sectors.parquet`` (one row per TIC x candidate sector),
``candidates_allsectors.parquet`` (the candidates' ranks in every sector they were observed in,
candidate there or not; used by the gallery), ``sector_stats.csv``,
``ccd.csv`` (camera/CCD distribution against the sector's stars), ``skybot_targets_sNNNN.csv``
(ecliptic sectors, best-ranked first, within ``--skybot-budget`` queries), ``summary.md`` and
``provenance.json``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oddball.common.paths import REPO_ROOT, data_root, outputs_root, sector_dir
from oddball.common.splits import split_array
from oddball.score import candidates as K
from oddball.tess.arrays import dir_metadata

SECTORS = (32, 33, 38, 43, 44, 55)
ECLIPTIC = (43, 44)
ENS = "ens_v3"
V3COLS = ["tic", "cmv3_flag", "cmv3_res_snr", "cmv3_res_t", "cmv3_res_ev", "cmv3_sys_snr",
          "cmv3_sys_t", "cmv3_sys_ev"]  # fmt: skip
ENSCOLS = ["tic", "stratum", "score", "score_raw", "eb_like", "score_classical",
           "score_ssl_v2_iforest", "score_ssl_v2_recon"]  # fmt: skip


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_sector(sec: int, alpha: float) -> tuple[pd.DataFrame, dict]:
    sd = outputs_root() / "scores"
    meta = dir_metadata(sector_dir(sec))[["tic", "ra", "dec", "tmag", "camera", "ccd"]]
    e = pd.read_parquet(sd / ENS / f"s{sec:04d}" / "real.parquet", columns=ENSCOLS)
    e = e.rename(columns={"score": "ens_score", "score_raw": "ens_raw",
                          "score_classical": "pct_classical", "score_ssl_v2_iforest":
                          "pct_iforest", "score_ssl_v2_recon": "pct_recon"})  # fmt: skip
    v = pd.read_parquet(outputs_root() / "veto" / "cmv3" / f"s{sec:04d}" / "real.parquet",
                        columns=V3COLS)  # fmt: skip
    t = meta.merge(e, on="tic", how="inner", validate="one_to_one")
    t = t.merge(v, on="tic", how="left", validate="one_to_one")
    t["cmv3_flag"] = t.cmv3_flag.fillna(False).astype(bool)
    t["cls_score"], t["cls_raw"] = K.classical_eb(t.pct_classical.to_numpy(), t.eb_like.to_numpy(),
                                                  t.stratum.to_numpy(), t.cmv3_flag.to_numpy(),
                                                  alpha)  # fmt: skip
    t["split"] = split_array(t.tic.to_numpy())
    t["sector"] = sec
    stats = {"sector": sec, "n_metadata": len(meta), "n_scored": len(t),
             "cmv3_flag_rate": float(t.cmv3_flag.mean()),
             "eb_like_gt_0.5_rate": float((t.eb_like > K.EB_LIKE_FLAG).mean())}  # fmt: skip
    c2 = outputs_root() / "veto" / "cmv2" / f"s{sec:04d}" / "real.parquet"
    if c2.exists():
        stats["cmv2_flag_rate"] = float(pd.read_parquet(c2, columns=["cmv2_flag"]).cmv2_flag.mean())
    return t, stats


def catalog_table(tics: np.ndarray) -> pd.DataFrame:
    lab = pd.read_parquet(data_root() / "knownclass" / "labels_eb.parquet",
                          columns=["tic", "label_any", "label_source", "label_detail",
                                   "exclude_reason", "kostov_nn"])  # fmt: skip
    lab = lab[lab.tic.isin(tics)].drop_duplicates("tic")
    typ = lab.label_detail.fillna("").map(lambda s: (re.findall(r"\[([^\]]+)\]", s) or [""])[-1])
    excl = lab.exclude_reason.isin(["vsx_type", "gaia_class"])
    lab["cat_class"] = np.where(lab.label_any != "", lab.label_any,
                                np.where(excl, "OTHER:" + typ.where(typ != "", "?"), ""))  # fmt: skip
    lab["catalogued"] = lab.cat_class != ""
    return lab.rename(columns={"label_source": "cat_source", "label_detail": "cat_detail"})[
        ["tic", "cat_class", "cat_source", "cat_detail", "catalogued", "kostov_nn"]]


def known_table() -> pd.DataFrame:
    k = pd.read_csv(data_root() / "known_positives.csv")
    k = k.groupby("tic").agg(kp_name=("name", "first"), kp_class=("class", "first")).reset_index()
    return k


def triage_table() -> pd.DataFrame:
    p = REPO_ROOT / "labels" / "triage_s0032_v1.csv"
    t = pd.read_csv(p)
    t["triage_s32"] = t.label + np.where(t.subtype.fillna("") != "", "/" + t.subtype.fillna(""), "")
    return t.drop_duplicates("tic")[["tic", "triage_s32"]]


def skybot_results(name: str) -> pd.DataFrame:
    parts = []
    for sec in ECLIPTIC:
        p = outputs_root() / "veto" / "cmv3" / f"s{sec:04d}" / f"skybot_{name}.csv"
        if p.exists():
            r = pd.read_csv(p)
            r["sector"] = sec
            parts.append(r)
    if not parts:
        return pd.DataFrame(columns=["tic", "sector"])
    r = pd.concat(parts, ignore_index=True)
    keep = ["tic", "sector", "status", "explained_pm1h", "nearest_name", "nearest_v",
            "nearest_d_pm1h", "epoch_btjd"]  # fmt: skip
    r = r[[c for c in keep if c in r]].drop_duplicates(["tic", "sector"])
    return r.rename(columns={c: f"skybot_{c}" for c in keep if c not in ("tic", "sector")})


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in r)
                   + " |")  # fmt: skip
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="v1")
    ap.add_argument("--n", type=int, default=K.TOP_N)
    ap.add_argument("--sectors", type=int, nargs="+", default=list(SECTORS))
    ap.add_argument("--skybot-budget", type=int, default=600)
    ap.add_argument("--skybot-tag", default="cand_v1")
    a = ap.parse_args()
    t0 = time.time()
    od = outputs_root() / "candidates" / a.name
    od.mkdir(parents=True, exist_ok=True)
    prov_e = json.loads((outputs_root() / "scores" / ENS / "s0032" / "provenance.json").read_text())
    alpha = float(prov_e["alpha"])
    log(f"{ENS} rule from S32: {prov_e['rule']} {prov_e['weights']} alpha {alpha}; the same "
        f"alpha is used for the classical EB down-weight")

    longs, stats, ccds, observed, per_sector_prov, allrank, tcs = [], [], [], {}, {}, [], []
    for sec in a.sectors:
        t, st = load_sector(sec, alpha)
        observed[sec] = set(t.tic.astype(int))
        t = K.add_ranks(t, a.n)
        allrank.append(t[["tic", "sector", "camera", "ccd", "cls_rank", "ens_rank", "cls_score",
                          "ens_score", "cmv3_flag", "eb_like", "cmv3_res_t", "cmv3_res_ev",
                          "cmv3_res_snr", "in_cls", "in_ens"]].assign(n_stars=len(t)))  # fmt: skip
        c = K.sector_candidates(t, a.n)
        st.update(n_cand=len(c), n_cls=int(c.in_cls.sum()), n_ens=int(c.in_ens.sum()),
                  n_both=int((c.in_cls & c.in_ens).sum()),
                  n_known_split=int((c.split == "known").sum()),
                  cand_eb_like_gt_0_5=int((c.eb_like > K.EB_LIKE_FLAG).sum()))  # fmt: skip
        stats.append(st)
        cc = K.ccd_excess(c, t)
        cc.insert(0, "sector", sec)
        ccds.append(cc)
        tc = K.time_clusters(c)
        tc.insert(0, "sector", sec)
        tcs.append(tc)
        st["max_time_cluster"] = int(tc.n.max()) if len(tc) else 0
        # chance level: the same count with camera/CCD labels shuffled among the candidates
        rng = np.random.default_rng(sec)
        sh = []
        for _ in range(50):
            cs = c.copy()
            perm = rng.permutation(len(cs))
            cs["camera"], cs["ccd"] = cs.camera.to_numpy()[perm], cs.ccd.to_numpy()[perm]
            t2 = K.time_clusters(cs)
            sh.append(int(t2.n.max()) if len(t2) else 0)
        st["max_time_cluster_shuffled_p95"] = float(np.percentile(sh, 95))
        st["n_in_time_clusters"] = int(tc.n.sum()) if len(tc) else 0
        st["camera_counts"] = " ".join(f"{k}:{v}" for k, v in
                                       c.camera.value_counts().sort_index().items())  # fmt: skip
        longs.append(c)
        ep = outputs_root() / "scores" / ENS / f"s{sec:04d}" / "provenance.json"
        per_sector_prov[sec] = json.loads(ep.read_text()).get("built_utc") if ep.exists() else None
        log(f"S{sec}: {len(t)} stars, cmv3 flags {st['cmv3_flag_rate']:.4f}, eb_like > 0.5 "
            f"{st['eb_like_gt_0.5_rate']:.4f}; candidates {len(c)} (classical {st['n_cls']}, "
            f"ens_v3 {st['n_ens']}, both {st['n_both']}); worst CCD ratio "
            f"{cc.ratio.max():.2f} (cam{int(cc.loc[cc.ratio.idxmax(), 'camera'])}/"
            f"ccd{int(cc.loc[cc.ratio.idxmax(), 'ccd'])})")
    long = pd.concat(longs, ignore_index=True)
    wide = K.merge_sectors(long, observed)

    # star-level context from the best sector, worst-case EB and veto across candidate sectors
    best = long.assign(best_rank=long[["cls_rank", "ens_rank"]].min(axis=1)).sort_values(
        ["tic", "best_rank", "sector"]).drop_duplicates("tic")  # fmt: skip
    ctx = best[["tic", "ra", "dec", "tmag", "camera", "ccd", "split", "cmv3_res_t", "cmv3_res_ev",
                "cmv3_res_snr", "pct_classical", "pct_iforest", "pct_recon"]].rename(
        columns={"camera": "best_camera", "ccd": "best_ccd"})  # fmt: skip
    agg = long.groupby("tic").agg(eb_like_max=("eb_like", "max"), in_cls_any=("in_cls", "any"),
                                  in_ens_any=("in_ens", "any")).reset_index()  # fmt: skip
    wide = wide.merge(ctx, on="tic").merge(agg, on="tic")
    # veto flags of the star in sectors where it is *not* a candidate (information only)
    vet = []
    for sec in a.sectors:
        p = outputs_root() / "veto" / "cmv3" / f"s{sec:04d}" / "real.parquet"
        f = pd.read_parquet(p, columns=["tic", "cmv3_flag"])
        vet.append(f[f.tic.isin(wide.tic) & f.cmv3_flag].assign(sector=sec))
    vet = pd.concat(vet, ignore_index=True)
    vs = vet.groupby("tic").sector.apply(lambda s: " ".join(str(x) for x in sorted(s)))
    wide["cmv3_flag_sectors"] = wide.tic.map(vs).fillna("")
    # source of the candidacy
    wide["source"] = np.select([wide.in_cls_any & wide.in_ens_any, wide.in_cls_any],
                               ["both", "classical"], "ens_v3")  # fmt: skip

    # catalogues
    cat = catalog_table(wide.tic.to_numpy())
    wide = wide.merge(cat, on="tic", how="left")
    wide["catalogued"] = wide.catalogued.fillna(False).astype(bool)
    wide["cat_class"] = wide.cat_class.fillna("")
    wide["kostov_nn"] = wide.kostov_nn.fillna(False).astype(bool)
    kp = known_table()
    wide = wide.merge(kp, on="tic", how="left")
    wide["known_positive"] = wide.kp_name.notna()
    wide = wide.merge(triage_table(), on="tic", how="left")
    wide["triage_s32"] = wide.triage_s32.fillna("")
    sk = skybot_results(a.skybot_tag)
    if len(sk):
        skw = sk.sort_values("skybot_explained_pm1h", ascending=False).drop_duplicates("tic")
        wide = wide.merge(skw.drop(columns=["sector"]).rename(
            columns={"skybot_epoch_btjd": "skybot_epoch"}), on="tic", how="left")  # fmt: skip
        wide["skybot_checked"] = wide.skybot_status.eq("ok")
        wide["skybot_explained"] = wide.skybot_explained_pm1h.fillna(False).astype(bool)
    else:
        wide["skybot_checked"] = False
        wide["skybot_explained"] = False

    # gallery priority: uncatalogued, not EB-like, not asteroid-explained, multi-sector first
    wide["eb_like_flag"] = wide.eb_like_max > K.EB_LIKE_FLAG
    wide["gallery_ok"] = (~wide.catalogued & ~wide.known_positive & ~wide.eb_like_flag
                          & ~wide.skybot_explained & (wide.split != "known"))  # fmt: skip
    wide = wide.sort_values(["gallery_ok", "multi", "triage_s32", "best_rank", "tic"],
                            ascending=[False, False, True, True, True]).reset_index(drop=True)
    wide["priority"] = np.arange(1, len(wide) + 1)
    wide.to_parquet(od / "candidates.parquet", index=False)
    long.to_parquet(od / "candidates_sectors.parquet", index=False)
    ar = pd.concat(allrank, ignore_index=True)
    ar[ar.tic.isin(wide.tic)].to_parquet(od / "candidates_allsectors.parquet", index=False)
    st = pd.DataFrame(stats)
    st.to_csv(od / "sector_stats.csv", index=False)
    cc = pd.concat(ccds, ignore_index=True)
    cc.to_csv(od / "ccd.csv", index=False)
    tca = pd.concat(tcs, ignore_index=True)
    tca.to_csv(od / "time_clusters.csv", index=False)

    # SkyBoT targets: ecliptic-sector candidates, best rank first, within the query budget
    per = a.skybot_budget // len(ECLIPTIC)
    sky_info = {}
    for sec in ECLIPTIC:
        if sec not in a.sectors:
            continue
        c = long[(long.sector == sec) & (long.split != "known")].copy()
        c["best_rank"] = c[["cls_rank", "ens_rank"]].min(axis=1)
        c = c[np.isfinite(c.cmv3_res_t)].sort_values(["best_rank", "tic"]).head(per)
        c[["tic"]].assign(label="cand").to_csv(od / f"skybot_targets_s{sec:04d}.csv", index=False)
        sky_info[sec] = {"n_targets": len(c), "max_best_rank": int(c.best_rank.max())}

    # summary
    n_multi = int(wide.multi.sum())
    lines = [f"# Candidate list {a.name} ({dt.date.today().isoformat()})", "",
             (f"Union per sector of the classical top {a.n} and the {ENS} top {a.n}, after the "
             f"cmv3 veto and the EB down-weight (alpha {alpha}, frozen from S32). Built by "
             "`scripts/build_candidates.py`. Nothing was tuned on these sectors. ssl_v2 was "
             "trained on S33/38/43/44/55 (self-supervised, no labels), so its iforest/recon "
             "scores there are in-sample; S32 is held out."), "",
             (f"- **{len(wide)} unique stars** from {len(long)} star-sector candidates; "
             f"**{n_multi} are candidates in 2+ sectors**."),
             f"- source: {wide.source.value_counts().to_dict()}",
             (f"- catalogued (VSX / Gaia / Prsa / Kostov label): {int(wide.catalogued.sum())}; "
             f"by class {wide[wide.catalogued].cat_class.value_counts().head(12).to_dict()}"),
             (f"- known positives in the list: {int(wide.known_positive.sum())} "
             f"({', '.join(wide[wide.known_positive].kp_name.astype(str).tolist()[:20])})"),
             (f"- eb_like > 0.5 in a candidate sector: {int(wide.eb_like_flag.sum())}; Kostov NN "
             f"flag (soft): {int(wide.kostov_nn.sum())}"),
             (f"- SkyBoT checked: {int(wide.skybot_checked.sum())}, explained by a known "
             f"asteroid (V <= 19 within 126\" +-1 h): {int(wide.skybot_explained.sum())}"),
             (f"- gallery-eligible (uncatalogued, not EB-like, not asteroid, not known): "
             f"{int(wide.gallery_ok.sum())}, of which multi-sector "
             f"{int((wide.gallery_ok & wide.multi).sum())}"), "",
             "## Per sector", "", md_table(st.round(4)), "",
             "## Camera/CCD distribution (worst three cells per sector by ratio)", "",
             md_table(cc.sort_values(["sector", "ratio"], ascending=[True, False])
                      .groupby("sector").head(3).round(3)), "",
             ("## Time-coincident candidates (same camera/CCD, residual event in the same "
             "0.5-d bin, >= 3 stars; largest 15)"), "",
             ("Stratum percentiles spread the list evenly over CCDs by construction, so a "
             "CCD-wide artefact would show here rather than as a CCD excess."), "",
             md_table(tca.head(15)) if len(tca) else "none", "",
             "## Multi-sector candidates (first 40 by priority)", "",
             md_table(wide[wide.multi].head(40)[["tic", "sectors_cand", "sectors_obs",
                                                 "best_rank", "source", "cat_class", "tmag",
                                                 "eb_like_max", "triage_s32"]]), ""]
    (od / "summary.md").write_text("\n".join(lines))
    prov = {"name": a.name, "sectors": a.sectors, "top_n": a.n, "ens": ENS,
            "ens_rule_from_s0032": {k: prov_e[k] for k in ("rule", "weights", "alpha")},
            "classical_eb_alpha": alpha, "ens_built_utc": per_sector_prov,
            "catalogues": str(data_root() / "knownclass" / "labels_eb.parquet"),
            "known_positives": str(data_root() / "known_positives.csv"),
            "skybot_tag": a.skybot_tag, "skybot_targets": sky_info,
            "n_unique": len(wide), "n_multi": n_multi, "seconds": round(time.time() - t0),
            "built_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}  # fmt: skip
    (od / "provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    log(f"wrote {od}: {len(wide)} stars ({n_multi} multi-sector), {int(wide.gallery_ok.sum())} "
        f"gallery-eligible; SkyBoT targets {sky_info}")


if __name__ == "__main__":
    main()
