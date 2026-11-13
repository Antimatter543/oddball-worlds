"""Part 2.1: the sector gap of the Track C survey and what could fill it.

Inputs: the merged survey records (predicted sectors = tess-point on-silicon through S107, the
survey's last data sector; sectors searched; product per sector), FU_OUT/gap_products.csv
(every MAST timeseries product per TIC and sector, gap_query.py), FU_OUT/gap_tic.csv (TIC
contamination ratio), FU_OUT/tglc_check.json (bucket check of TGLC S1-S55, tglc_check.py).

Writes FU_OUT/gap_tables.txt and FU_OUT/gap_stars.csv (per star: Tmag, contratio, predicted,
searched, and the sectors each filler could add).
Usage: python -I followup/gap_analysis.py
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fu

LAST = 107
QLP_BAD = set(range(99, 105))  # QLP S99-S104: timestamp errors, not yet redelivered (MAST page)
TARS_MAX = 96
TGLC_MAX = 55


def tm_bin(t):
    for lo, hi in ((0, 11), (11, 12), (12, 13), (13, 13.5), (13.5, 14), (14, 15), (15, 99)):
        if lo <= t < hi:
            return f"{lo}-{hi}" if hi < 99 else "15+"
    return "?"


def sec_bin(s):
    for lo, hi in ((1, 13), (14, 26), (27, 39), (40, 55), (56, 69), (70, 83), (84, 96),
                   (97, 107)):
        if lo <= s <= hi:
            return f"S{lo}-{hi}"
    return "other"


def cr_bin(c):
    if not np.isfinite(c):
        return "n/a"
    for lo, hi in ((0, 0.01), (0.01, 0.1), (0.1, 0.5), (0.5, 1e9)):
        if lo <= c < hi:
            return f"{lo}-{hi if hi < 1e9 else 'inf'}"
    return "n/a"


def main():
    recs = []
    with gzip.open(fu.OUT / "survey" / "merged" / "results.jsonl.gz", "rt") as fh:
        for line in fh:
            r = json.loads(line)
            recs.append(r)
    tg = pd.read_csv(fu.OUT / "gate2" / "survey_targets.csv").set_index("tic")
    tic_tab = pd.read_csv(fu.FU_OUT / "gap_tic.csv").drop_duplicates("tic").set_index("tic")
    prod = pd.read_csv(fu.FU_OUT / "gap_products.csv")
    prod = prod[prod["sector"] > 0]
    avail = defaultdict(set)  # (tic, sector) -> kinds
    for t, s, k in zip(prod["tic"], prod["sector"], prod["kind"]):
        avail[(int(t), int(s))].add(str(k))
    tg_ok = {}
    tcf = fu.FU_OUT / "tglc_check.json"
    if tcf.exists():
        tg_ok = json.loads(tcf.read_text())
    L = []
    rows = []
    cause = Counter()
    by = {"tmag": defaultdict(Counter), "sector": defaultdict(Counter),
          "contratio": defaultdict(Counter)}
    fill_kinds = Counter()
    extra = Counter()
    for r in recs:
        tic = r["tic"]
        tmag = float(tg.loc[tic, "Tmag"])
        cr = float(tic_tab.loc[tic, "contratio"]) if tic in tic_tab.index else float("nan")
        pred = {s for s in r["sectors_pred"] if s <= LAST}
        found = {int(s) for s in (r.get("sectors_found") or []) if int(s) <= LAST}
        fails = {int(f["sector"]) for f in (r.get("sector_failures") or [])}
        miss = pred - found
        add = {"qlp": set(), "tars": set(), "tglc_mast": set(), "t16": set(),
               "eleanor": set(), "other_hlsp": set(), "none": set()}
        for s in sorted(miss):
            k = avail.get((tic, s), set())
            if "spoc120" in k or "tess-spoc" in k:
                c = "SPOC/TESS-SPOC listed but not used" + (" (conversion failed)" if s in fails
                                                            or r["status"] == "ok" else "")
            else:
                c = "no SPOC/TESS-SPOC product"
            cause[c] += 1
            for key, val in (("tmag", tm_bin(tmag)), ("sector", sec_bin(s)),
                             ("contratio", cr_bin(cr))):
                by[key][val]["missing"] += 1
            alts = []
            if "qlp" in k and s not in QLP_BAD:
                add["qlp"].add(s)
                alts.append("qlp")
            if "tars" in k:
                add["tars"].add(s)
                alts.append("tars")
            if "tglc" in k:
                add["tglc_mast"].add(s)
                alts.append("tglc")
            if "t16" in k:
                add["t16"].add(s)
                alts.append("t16")
            if "gsfc-eleanor-lite" in k:
                add["eleanor"].add(s)
                alts.append("eleanor")
            other = k - {"spoc120", "spoc20", "dvt", "tess_other", "tess-spoc", "qlp", "tars",
                         "tglc", "t16", "gsfc-eleanor-lite"}
            if other:
                add["other_hlsp"].add(s)
                alts += sorted(other)
            if not alts:
                add["none"].add(s)
            fill_kinds[tuple(sorted(set(alts))) or ("nothing listed",)] += 1
        for s in pred:
            for key, val in (("tmag", tm_bin(tmag)), ("sector", sec_bin(s)),
                             ("contratio", cr_bin(cr))):
                by[key][val]["pred"] += 1
                if s in found:
                    by[key][val]["found"] += 1
        # near-edge / unpredicted sectors with any light-curve product
        ex = {s for s in range(1, LAST + 1)
              if (tic, s) in avail and s not in pred and s not in found
              and avail[(tic, s)] - {"dvt", "spoc20", "tess_other"}}
        extra[len(ex)] += 1
        rows.append({"tic": tic, "status": r["status"], "tmag": tmag, "contratio": cr,
                     "n_pred": len(pred), "n_found": len(found & pred),
                     "n_found_all": len(found), "missing": " ".join(map(str, sorted(miss))),
                     **{f"add_{k}": " ".join(map(str, sorted(v))) for k, v in add.items()},
                     "extra_unpredicted": " ".join(map(str, sorted(ex)))})
    df = pd.DataFrame(rows)
    df.to_csv(fu.FU_OUT / "gap_stars.csv", index=False)
    n_pred = int(df["n_pred"].sum())
    n_found = int(df["n_found"].sum())
    L.append(f"targets {len(df)}; star-sectors on silicon through S{LAST} (tess-point) {n_pred}; "
             f"searched {n_found} ({n_found / n_pred:.3f}); missing {n_pred - n_found}")
    L.append(f"searched star-sectors outside the tess-point list: "
             f"{int((df['n_found_all'] - df['n_found']).sum())}")
    L.append("\n## missing star-sectors by cause")
    for k, v in cause.most_common():
        L.append(f"- {k}: {v}")
    for key in ("tmag", "sector", "contratio"):
        L.append(f"\n## by {key}: predicted / searched / missing / searched fraction")
        L.append("| bin | predicted | searched | missing | searched/pred |")
        L.append("|---|---|---|---|---|")
        for b in sorted(by[key]):
            c = by[key][b]
            L.append(f"| {b} | {c['pred']} | {c['found']} | {c['missing']} | "
                     f"{c['found'] / max(1, c['pred']):.2f} |")
        if key == "tmag":
            pd.DataFrame([{"bin_lo": float(b.split("-")[0]) if b != "15+" else 15.0,
                           "frac": by[key][b]["found"] / max(1, by[key][b]["pred"])}
                          for b in by[key]]).sort_values("bin_lo").to_csv(
                fu.FU_OUT / "gap_tables_frac.csv", index=False)
    L.append("\n## what MAST lists for the missing star-sectors (TGLC S12-S55 is NOT in MAST's "
             "API; see the bucket check)")
    for k, v in fill_kinds.most_common(25):
        L.append(f"- {'+'.join(k)}: {v}")
    L.append("\n## no-data stars")
    nd = df[df["status"] == "no_data"]
    L.append(f"- {len(nd)} stars; Tmag median {nd['tmag'].median():.2f}; predicted sectors "
             f"{int(nd['n_pred'].sum())}")
    for k in ("qlp", "tars", "tglc_mast", "t16", "eleanor", "other_hlsp", "none"):
        c = nd[f"add_{k}"].fillna("").astype(str).str.split().str.len()
        L.append(f"- {k}: stars with >= 1 sector {int((c > 0).sum())}, star-sectors {int(c.sum())}")
    L.append(f"\n## unpredicted sectors with a light-curve product (near-edge): "
             f"{dict(sorted(extra.items()))}")
    if tg_ok:
        L.append(f"\n## TGLC bucket check: {tg_ok.get('summary')}")
    txt = "\n".join(L) + "\n"
    (fu.FU_OUT / "gap_tables.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
