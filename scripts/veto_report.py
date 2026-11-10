"""Before / after table for a systematics veto (docs/ML_DESIGN.md §12).

    uv run python -I scripts/veto_report.py --scorers classical ssl_v2_iforest

For each scorer and split (val first, then test): the flagged fraction of real stars, the null
false-positive rate and recall in the top 1% at ~0.3 / 1 / 3% depth with no veto, the v0
``sys_flag`` veto and the per-event ``cmv2`` veto (evaluator rule). Also the known positives'
flags and the strongest systematic / residual event of each (evaluation only). Writes
``outputs/veto/cmv2/sNNNN/report.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oddball.eval.harness import load, score_dir, veto_dir
from oddball.eval.report import md_table

DEPTHS = ("~0.3%", "~1%", "~3%")


def rows_for(scorer: str, sector: int, inj: str) -> list[dict]:
    out = []
    eds = {v: load(scorer, sector, inj, v) for v in ("cmv2", "sys_flag")}
    for split in ("val", "test"):
        for lab, ed, col in (("none", eds["cmv2"], "top1%"), ("sys_flag (v0)", eds["sys_flag"],
                             "top1%_veto"), ("cmv2", eds["cmv2"], "top1%_veto")):  # fmt: skip
            s = ed.inj[ed.inj.split == split]
            astro, nul = s[~s.is_null], s[s.is_null]
            real = ed.real[ed.real.split == split]
            row = {"scorer": scorer, "split": split, "veto": lab,
                   "flagged real": 0.0 if lab == "none" else float(real.veto_flag.mean()),
                   "null FPR": nul[col].mean(), "null_ccd": nul[nul.kind == "null_ccd"][col].mean(),
                   "null_dump": nul[nul.kind == "null_dump"][col].mean()}  # fmt: skip
            for d in DEPTHS:
                row[f"recall {d}"] = astro[astro.depth_bin == d][col].mean()
            row["recall all"] = astro[col].mean()
            row["astro flagged"] = 0.0 if lab == "none" else float(astro.veto_flag.mean())
            out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scorers", nargs="+", default=["classical", "ssl_v2_iforest"])
    ap.add_argument("--sector", type=int, default=32)
    ap.add_argument("--inj", default="v1")
    a = ap.parse_args()
    rows = [r for s in a.scorers for r in rows_for(s, a.sector, a.inj)]
    tab = pd.DataFrame(rows)
    vd = veto_dir("cmv2", a.sector)
    vr = pd.read_parquet(vd / "real.parquet")
    kp = pd.read_csv(Path(__file__).resolve().parents[1] / "docs" / "known_positives.csv")
    kp = kp[["tic", "name"]].drop_duplicates("tic")
    old = pd.read_parquet(score_dir("classical", a.sector) / "real.parquet",
                          columns=["tic", "sys_flag", "sys_frac"])
    kv = kp.merge(vr, on="tic").merge(old, on="tic")
    kcols = ["name", "tic", "sys_flag", "sys_frac", "cmv2_flag", "cmv2_res_snr_pdc",
             "cmv2_sys_snr_pdc", "cmv2_res_snr", "cmv2_sys_snr", "cmv2_res_t", "cmv2_res_ev",
             "cmv2_sys_t", "cmv2_sys_ev"]  # fmt: skip
    L = ["# Systematics veto `cmv2` vs v0 `sys_flag`", "",
         f"Command: `python scripts/veto_report.py --scorers {' '.join(a.scorers)}`", "",
         ("Evaluator rule: flagged real and injected light curves go to the bottom, re-rank "
          "within split x stratum; a flagged injection counts as not recovered."), "",
         md_table(tab, "{:.3f}"), "",
         "## Known positives (evaluation only; never tuned on)", "",
         md_table(kv[kcols], "{:.4g}"), ""]  # fmt: skip
    (vd / "report.md").write_text("\n".join(L))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(tab.round(3).to_string())
    print(kv[kcols].round(2).to_string())
    print(f"report: {vd / 'report.md'}")


if __name__ == "__main__":
    main()
