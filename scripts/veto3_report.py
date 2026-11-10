"""Before / after report for the ``cmv3`` veto (docs/ML_DESIGN.md §13).

    uv run python -I scripts/veto3_report.py --scorers classical ssl_v2_iforest ens_v1 ens_v2

Per scorer and split (val, then test), with no veto, ``cmv2`` and ``cmv3`` (the evaluator's
rule: flagged real and injected light curves to the bottom, re-rank within split x stratum):
recall in the top 1% at ~0.3 / 1 / 3% depth (v1 astrophysical injections), the share of them
flagged, and the null false-positive rate per null type (v1 ``null_ccd``, ``null_dump``; n2
``null_asteroid``, ``null_step``, ``null_column``; n2 carriers are never ranked).

* ``cmv2`` on n2 is the cmv2 rule (PDCSAP only, t_sys 10, t_res 10) on ``cmv3``'s ``cmv2``
  configuration with the real masks only, i.e. without the set's own binomial CCD map (n2's
  nulls are local; see ``build_veto3.py``). On v1 and the real stars it is the cmv2 build.
* Ensembles are re-scored from their components with their recorded rule (the pre-veto
  ``score_raw``) so every veto is applied the same way.

Evaluation-only sections (nothing here is tuned): the known positives' flags, and the
2026-11-10 triage labels (``labels/triage_s0032_v1.csv``): how many artefacts each veto flags
and how many eclipsing binaries and variables it wrongly flags. Also the ablation table on val
from the tune step. Writes ``outputs/veto/cmv3/sNNNN/report.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ensemble import combine
from tune_veto3 import NULL_TYPES, load_all_truth, metrics

from oddball.common.paths import REPO_ROOT, sector_dir
from oddball.common.splits import split_array
from oddball.common.strata import stratum_id
from oddball.eval.harness import score_dir, veto_dir
from oddball.eval.report import md_table
from oddball.knownclass.explain import stratum_pct
from oddball.score import veto3 as V3
from oddball.tess.arrays import dir_metadata


def scores_for(scorer: str, sec: int, meta: pd.DataFrame, truth: pd.DataFrame):
    """(real: tic, score; inj: inj_id, score) with the pre-veto score for ensembles."""
    sd = score_dir(scorer, sec)
    pp = sd / "provenance.json"
    prov = json.loads(pp.read_text()) if pp.exists() else {}
    if "components" not in prov:
        r = pd.read_parquet(sd / "real.parquet", columns=["tic", "score"])
        i = pd.concat([pd.read_parquet(sd / f"inj_{n}.parquet", columns=["inj_id", "score"])
                       for n in ("v1", "n2")], ignore_index=True)  # fmt: skip
        return r, i
    if prov.get("ka"):
        raise SystemExit(f"{scorer}: explain-away ensembles are not supported here")
    comps, rule = prov["components"], prov["rule"]
    w = tuple(prov["weights"]) if prov["weights"] else None
    real = meta[["tic", "stratum"]].copy()
    inj = truth[["inj_id", "stratum"]].copy()
    for c in comps:
        cd = score_dir(c, sec)
        real = real.merge(pd.read_parquet(cd / "real.parquet", columns=["tic", "score"])
                          .rename(columns={"score": c}), on="tic", how="inner")  # fmt: skip
        ii = pd.concat([pd.read_parquet(cd / f"inj_{n}.parquet", columns=["inj_id", "score"])
                        for n in ("v1", "n2")], ignore_index=True).rename(columns={"score": c})
        inj = inj.merge(ii, on="inj_id", how="left")
    Pr = np.column_stack([stratum_pct(real[c].fillna(-np.inf), real.stratum,
                                      real[c].fillna(-np.inf), real.stratum) for c in comps])
    Pi = np.column_stack([stratum_pct(real[c].fillna(-np.inf), real.stratum,
                                      inj[c].fillna(-np.inf), inj.stratum) for c in comps])
    sr, si = combine(Pr, rule, w), combine(Pi, rule, w)
    er = stratum_pct(sr, real.stratum, sr, real.stratum)
    ei = stratum_pct(sr, real.stratum, si, inj.stratum)
    # consistency with the published file (pre-veto score)
    pub = pd.read_parquet(sd / "real.parquet", columns=["tic", "score_raw"])
    j = real[["tic"]].assign(e=er).merge(pub, on="tic")
    assert np.allclose(j.e, j.score_raw), f"{scorer}: re-scored ensemble differs from its file"
    return real[["tic"]].assign(score=er), inj[["inj_id"]].assign(score=ei)


def flags(sec: int):
    v2 = veto_dir("cmv2", sec)
    v3 = veto_dir("cmv3", sec)
    p2 = json.loads((v2 / "provenance.json").read_text())["flag"]
    r2 = pd.read_parquet(v2 / "real.parquet", columns=["tic", "cmv2_flag"])
    i2 = pd.read_parquet(v2 / "inj_v1.parquet", columns=["inj_id", "cmv2_flag"])
    n2 = pd.read_parquet(v3 / "inj_n2.parquet")
    n2["cmv2_flag"] = V3.veto_flag(n2, "cmv2", p2["t_sys"], p2["t_res"], p2["pdc_only"])
    r3 = pd.read_parquet(v3 / "real.parquet")
    i3 = pd.concat([pd.read_parquet(v3 / "inj_v1.parquet"), n2], ignore_index=True)
    fr = r2.merge(r3[["tic", "cmv3_flag"]], on="tic", how="outer", validate="one_to_one")
    fi = pd.concat([i2, n2[["inj_id", "cmv2_flag"]]], ignore_index=True).merge(
        i3[["inj_id", "cmv3_flag"]], on="inj_id", how="outer", validate="one_to_one")
    return fr, fi, r3, p2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sector", type=int, default=32)
    ap.add_argument("--scorers", nargs="+", default=["classical", "ssl_v2_iforest", "ens_v1",
                                                     "ens_v2"])  # fmt: skip
    ap.add_argument("--triage", type=Path, default=REPO_ROOT / "labels" / "triage_s0032_v1.csv")
    a = ap.parse_args()
    sec = a.sector
    vd = veto_dir("cmv3", sec)
    meta = dir_metadata(sector_dir(sec))[["tic", "tmag", "camera", "ccd"]]
    meta["stratum"] = stratum_id(meta.tmag, meta.camera, meta.ccd)
    meta["split"] = split_array(meta.tic.to_numpy())
    truth = load_all_truth(sec)
    fr, fi, r3, p2 = flags(sec)
    prov3 = json.loads((vd / "provenance.json").read_text())
    fl3 = prov3["flag"]
    lines = [f"# cmv3 veto: S{sec} before / after", "",
             (f"Chosen on val: configuration `{fl3['config']}` (labels "
              f"{', '.join(fl3['labels']) or 'none'}), pdc_only={fl3['pdc_only']}, "
              f"t_sys={fl3['t_sys']}, t_res={fl3['t_res']}; flags "
              f"{fl3['flag_frac_real_all']:.3f} of all real stars (cmv2: "
              f"{fr.cmv2_flag.mean():.3f}). cmv2 rule: pdc_only={p2['pdc_only']}, "
              f"t_sys={p2['t_sys']}, t_res={p2['t_res']}."), ""]

    rows = []
    for s in a.scorers:
        r, i = scores_for(s, sec, meta, truth)
        r = r.merge(meta, on="tic", how="left").merge(fr, on="tic", how="left")
        i = truth.merge(i, on="inj_id", how="left").merge(fi, on="inj_id", how="left")
        for df in (r, i):
            df["score"] = df.score.astype(float).fillna(-np.inf)
            for c in ("cmv2_flag", "cmv3_flag"):
                df[c] = df[c].fillna(False).astype(bool)
        for split in ("val", "test"):
            rr, ii = r[r.split == split].reset_index(drop=True), i[i.split == split].reset_index(
                drop=True)
            for v in ("none", "cmv2", "cmv3"):
                f_r = np.zeros(len(rr), bool) if v == "none" else rr[f"{v}_flag"].to_numpy()
                f_i = np.zeros(len(ii), bool) if v == "none" else ii[f"{v}_flag"].to_numpy()
                m = metrics(rr, ii, f_r, f_i)
                rows.append({"scorer": s, "split": split, "veto": v,
                             "flagged real": round(m["flag_real"], 3),
                             "recall ~0.3/1/3%": f"{m['~0.3%']:.3f} / {m['~1%']:.3f} / "
                                                 f"{m['~3%']:.3f}",
                             "all depths": round(m["all"], 3),
                             "astro flagged": round(m["astro_flagged"], 3),
                             **{k[5:]: round(m[f"fpr_{k}"], 3) for k in NULL_TYPES},
                             "null mean": round(m["null_fpr_mean"], 3)})  # fmt: skip
    tab = pd.DataFrame(rows)
    lines += ["## Recall and null FPR by type (top 1%, evaluator veto rule)", "",
              ("Null columns: false-positive rate in the top 1% per null type. n2 has "
               f"{truth[(truth.set == 'n2') & (truth.split == 'test')].kind.value_counts().to_dict()}"
               " evaluated nulls on test."), "", md_table(tab), ""]

    # known positives
    kp = pd.read_csv(REPO_ROOT / "docs" / "known_positives.csv")[["tic", "name"]]
    k = meta[meta.split == "known"][["tic"]].merge(kp, on="tic", how="left").merge(
        fr, on="tic", how="left").merge(
        r3[["tic", "cmv3_sys_snr", "cmv3_res_snr", "cmv3_sys_ev", "cmv3_res_ev",
            "cmv3_sys_snr_pdc", "cmv3_res_snr_pdc"]], on="tic", how="left")  # fmt: skip
    lines += ["## Known positives (evaluation only)", "",
              (f"Flagged: cmv2 {int(k.cmv2_flag.sum())} / {len(k)}, cmv3 "
               f"{int(k.cmv3_flag.sum())} / {len(k)}."), "",
              md_table(k.sort_values(["cmv3_flag", "cmv2_flag"], ascending=False).round(2)), ""]

    # triage
    if a.triage.exists():
        tr = pd.read_csv(a.triage).merge(fr, on="tic", how="left").merge(
            r3[["tic", "cmv3_sys_snr_pdc", "cmv3_res_snr_pdc", "cmv3_sys_ev", "cmv3_res_ev",
                *[f"n_{lb}" for lb in V3.LABELS]]], on="tic", how="left")  # fmt: skip
        g = tr.groupby("label").agg(n=("tic", "size"), cmv2_flagged=("cmv2_flag", "sum"),
                                    cmv3_flagged=("cmv3_flag", "sum")).reset_index()
        tr["sub"] = tr.subtype.fillna("").str.split(" ").str[0]
        g2 = tr[tr.label == "artefact"].groupby("sub").agg(
            n=("tic", "size"), cmv2_flagged=("cmv2_flag", "sum"),
            cmv3_flagged=("cmv3_flag", "sum")).reset_index()  # fmt: skip
        lines += ["## Eyeballed triage stars (evaluation only)", "",
                  ("Labels: `labels/triage_s0032_v1.csv` (96 gallery stars, 2026-11-10; "
                   "'triage-2' rows were re-labelled from the cards for this record because the "
                   "first pass stored only counts)."), "", md_table(g), "", md_table(g2), "",
                  md_table(tr.drop(columns=["source", "sub"]).sort_values(
                      ["label", "cmv3_flag"]).round(2)), ""]  # fmt: skip

    # ablation (val)
    bpc = fl3["best_per_config"]
    ab = pd.DataFrame([{"config": c, "J3 (val)": round(x["J3"], 4), "pdc_only": x["pdc_only"],
                        "t_sys": x["t_sys"], "t_res": x["t_res"],
                        "flagged real val": round(x["flag_frac_real_val"], 3),
                        **{f"{s}: {k[5:]}": round(x["per_scorer"][s][f"fpr_{k}"], 3)
                           for s in fl3["scorers"][:1] for k in NULL_TYPES},
                        "recall_mean": round(np.mean([x["per_scorer"][s]["recall_mean"]
                                                      for s in fl3["scorers"]]), 3)}
                       for c, x in bpc.items()]).sort_values("J3 (val)", ascending=False)
    lines += ["## Label ablation on val (best grid point per configuration)", "", md_table(ab),
              ""]
    rc = prov3["real_counts"]
    lines += ["## Asteroid tracks and columns on the real sector", "",
              (f"- {rc['points']} track points; {rc['track_pairs']} track pairs; "
               f"{rc['track_lcs']} light curves with a track-labelled event; "
               f"{rc['column_lcs']} with a column-labelled event."),
              (f"- Chance-track null (point times shuffled within each CCD): "
               f"{prov3['chance_track_null']}."), ""]
    sk = vd / "skybot_ens_v2.csv"
    if sk.exists():
        s = pd.read_csv(sk)
        gs = s.groupby("group").agg(n=("tic", "size"), ok=("status", lambda x: (x == "ok").sum()),
                                    explained=("explained_pm1h", "sum")).reset_index()
        lines += ["## SkyBoT (C57) known-asteroid check", "", md_table(gs), ""]
    (vd / "report.md").write_text("\n".join(lines))
    print(f"report: {vd / 'report.md'}")
    print(md_table(tab))


if __name__ == "__main__":
    main()
