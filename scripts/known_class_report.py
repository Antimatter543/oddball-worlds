"""Diagnostics for an explain-away scorer: what leaves and what fills the real top 1%.

    uv run python -I scripts/known_class_report.py --base classical --ka classical_ka --kc kc_v1

Writes ``outputs/knownclass/<kc>/sNNNN/report_<ka>.md``:

* the base scorer's real top 1% (within Tmag x CCD stratum, known positives left out): how many
  stars the explain-away scorer drops out of it, by predicted known class and catalogue label;
* the composition of the explain-away scorer's own top 1%;
* injection recall in the top 1% by the host's catalogue class (the host keeps its catalogue
  class even though hosts are never training labels), for the requested split;
* the known positives (``docs/known_positives.csv``, ``split == "known"``): stratum percentile
  under both scorers, the class the model gives them and the explained share ``E``. "Wrongly
  explained away" = in the base top 1% but not in the explain-away top 1%. Evaluation only.

It reads score files and the known-class outputs only; nothing is tuned here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oddball.common.paths import data_root, outputs_root
from oddball.eval.harness import known_table, load, rank_within
from oddball.eval.report import md_table


def top1(ed, veto: bool = False) -> pd.Series:
    """Real top 1% within stratum; ``veto`` sends sys_flag light curves to the bottom first
    (the evaluator's symmetric veto), so the top 1% is then drawn from unflagged stars only."""
    r = ed.real
    flag = r.sys_flag.astype(bool).to_numpy() if veto else np.zeros(len(r), bool)
    r = r.assign(score=np.where(flag, -np.inf, r.score))
    ref = r[r.split != "known"]
    pct = rank_within(ref, r, ["stratum"])
    return pd.Series((pct >= 0.99) & ~flag, index=r.tic.to_numpy())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="classical")
    ap.add_argument("--ka", default="classical_ka")
    ap.add_argument("--kc", default="kc_v1")
    ap.add_argument("--sector", type=int, default=32)
    ap.add_argument("--inj", default="v1")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    sec = a.sector
    eb, ek = load(a.base, sec, a.inj, "sys_flag"), load(a.ka, sec, a.inj, "sys_flag")
    kc = pd.read_parquet(outputs_root() / "knownclass" / a.kc / f"s{sec:04d}" / "real.parquet")
    labels = pd.read_parquet(data_root() / "knownclass" / "labels.parquet")
    lab = labels.set_index("tic")
    kc["cat"] = lab.label_any.reindex(kc.tic).fillna("").replace("", "(none)").to_numpy()
    kc = kc.set_index("tic")

    tb, tk = top1(eb), top1(ek)
    real = ek.real.set_index("tic")
    known = real.split == "known"
    tb, tk = tb[~known.reindex(tb.index).to_numpy()], tk[~known.reindex(tk.index).to_numpy()]
    in_b = tb[tb].index
    in_k = tk[tk].index
    gone = in_b.difference(in_k)
    new = in_k.difference(in_b)
    sysf = real.sys_flag.astype(bool) if "sys_flag" in real else pd.Series(False, real.index)

    def comp(idx, title):
        d = pd.DataFrame({"kc_pred": kc.kc_pred.reindex(idx).to_numpy(),
                          "cat": kc.cat.reindex(idx).to_numpy()})  # fmt: skip
        by_pred = d.kc_pred.value_counts().rename_axis("predicted class").reset_index(name="n")
        by_cat = d.cat.value_counts().rename_axis("catalogue class").reset_index(name="n")
        pmax = kc.kc_pmax.reindex(idx)
        s = (f"### {title}: {len(idx)} stars\n\nsys_flag rate {sysf.reindex(idx).mean():.2f}; "
             f"median kc_pmax {pmax.median():.2f}\n\n" + md_table(by_pred) + "\n\n"
             + md_table(by_cat))  # fmt: skip
        return s, d

    parts = [f"# Explain-away diagnostics: {a.base} -> {a.ka} (S{sec})\n",
             (f"Real stars, top 1% within Tmag x CCD stratum (known positives excluded). "
              f"Known-class model `{a.kc}`.\n")]  # fmt: skip
    s1, _ = comp(in_b, f"{a.base} top 1%")
    s2, _ = comp(gone, f"explained away (in {a.base} top 1%, not in {a.ka} top 1%)")
    s3, _ = comp(in_k, f"{a.ka} top 1%")
    s4, _ = comp(new, f"new in {a.ka} top 1%")
    parts += [s1, s2, s3, s4]

    # the same with the sys_flag veto applied to both scorers
    if "sys_flag" in real and "sys_flag" in eb.real:
        vb, vk = top1(eb, veto=True), top1(ek, veto=True)
        vb, vk = vb[~known.reindex(vb.index).to_numpy()], vk[~known.reindex(vk.index).to_numpy()]
        vin_b, vin_k = vb[vb].index, vk[vk].index
        parts.append("## With the sys_flag veto (flagged light curves ranked last)")
        parts += [comp(vin_b, f"{a.base} top 1%, veto")[0],
                  comp(vin_b.difference(vin_k), "explained away, veto")[0],
                  comp(vin_k, f"{a.ka} top 1%, veto")[0],
                  comp(vin_k.difference(vin_b), f"new in {a.ka} top 1%, veto")[0]]  # fmt: skip
        rows = []
        for name, ed in ((a.base, eb), (a.ka, ek)):
            inj = ed.inj[ed.inj.split == a.split]
            astro = inj[~inj.is_null]
            r = {"scorer": name, "null FPR": inj[inj.is_null]["top1%"].mean(),
                 "null FPR (veto)": inj[inj.is_null]["top1%_veto"].mean()}  # fmt: skip
            for dbin in ("~0.3%", "~1%", "~3%"):
                g = astro[astro.depth_bin == dbin]
                r[f"recall {dbin}"] = g["top1%"].mean()
                r[f"recall {dbin} (veto)"] = g["top1%_veto"].mean()
            rows.append(r)
        parts.append(f"### Injection recall in the top 1% with and without the veto "
                     f"({a.split} split)\n\n" + md_table(pd.DataFrame(rows).round(3)))

    # injection recall by host catalogue class
    rows = []
    for name, ed in ((a.base, eb), (a.ka, ek)):
        inj = ed.inj[(ed.inj.split == a.split) & ~ed.inj.is_null].copy()
        inj["host_cat"] = lab.label_any.reindex(inj.host_tic).fillna("").replace(
            "", "(none)").to_numpy()  # fmt: skip
        inj["host_pred"] = kc.kc_pred.reindex(inj.host_tic).to_numpy()
        for (hc,), g in inj.groupby(["host_cat"]):
            rows.append({"scorer": name, "host catalogue class": hc, "n": len(g),
                         "recall top1% (all depths)": g["top1%"].mean(),
                         "recall top1% (~1-3%)": g[g.depth_bin.isin(["~1%", "~3%"])]["top1%"].mean(),
                         "AUROC": g.pct.mean()})  # fmt: skip
    t = pd.DataFrame(rows).pivot_table(index=["host catalogue class"], columns="scorer",
                                       values=["n", "recall top1% (all depths)",
                                               "recall top1% (~1-3%)"])  # fmt: skip
    t.columns = [f"{v} [{s}]" for v, s in t.columns]
    for c in t.columns:
        if c.startswith("n ["):
            t[c] = t[c].astype("Int64").astype(str)
    t = t.reset_index()
    parts.append(f"### Injection recall by host catalogue class ({a.split} split, non-null)\n\n"
                 + md_table(t))  # fmt: skip

    # known positives: evaluation only (never used to tune alpha / tau)
    pb, pk = f"pct [{a.base}]", f"pct [{a.ka}]"
    kb, kk = known_table(eb.real, sec), known_table(ek.real, sec)
    kt = kb[["name", "tic", "class", "tmag", "pct_stratum"]].rename(columns={"pct_stratum": pb})
    kt = kt.merge(kk[["tic", "pct_stratum"]].rename(columns={"pct_stratum": pk}), on="tic")
    kt["kc_pred"] = kc.kc_pred.reindex(kt.tic).to_numpy()
    kt["kc_pmax"] = kc.kc_pmax.reindex(kt.tic).round(2).to_numpy()
    if "kc_explained" in real:
        kt["E"] = real.kc_explained.reindex(kt.tic).round(2).to_numpy()
    kt["catalogue"] = kc.cat.reindex(kt.tic).to_numpy()
    if "sys_flag" in real:
        kt["sys_flag"] = real.sys_flag.astype(bool).reindex(kt.tic).to_numpy()
    kin_b, kin_k = kt[pb] >= 0.99, kt[pk] >= 0.99
    kt["lost"] = kin_b & ~kin_k
    kt[pb], kt[pk] = kt[pb].round(3), kt[pk].round(3)
    lost = kt[kt.lost]
    why = ", ".join(f"{n} as {p}" for n, p in zip(lost.name, lost.kc_pred))
    parts.append(f"### Known positives ({len(kt)} in S{sec}; evaluation only)\n\n"
                 f"In the top 1%: {int(kin_b.sum())} under {a.base}, {int(kin_k.sum())} under "
                 f"{a.ka}. Wrongly explained away (in {a.base} top 1%, not in {a.ka}): "
                 f"{len(lost)}{f' ({why})' if why else ''}.\n\n"
                 + md_table(kt.sort_values(pb, ascending=False)))  # fmt: skip

    out = outputs_root() / "knownclass" / a.kc / f"s{sec:04d}" / f"report_{a.ka}.md"
    out.write_text("\n\n".join(parts) + "\n")
    print("\n\n".join(parts))
    print(f"\nreport: {out}")
    print(f"in {a.base} top1%: {len(in_b)}; explained away: {len(gone)} "
          f"({len(gone) / max(len(in_b), 1):.0%}); {a.ka} top1%: {len(in_k)}")
    _ = np  # keep numpy import for interactive use


if __name__ == "__main__":
    main()
