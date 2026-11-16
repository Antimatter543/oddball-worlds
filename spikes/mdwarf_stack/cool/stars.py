"""Star-level table of everything the two Track C surveys flagged (no new search, no re-vetting).

Inputs (read only):
  <OUT>/survey/vet/graded.jsonl, <OUT>/survey100/vet/graded.jsonl   long-range signals >= tau_long, v2-graded
  <OUT>/survey*/merged/results.jsonl.gz                              short-range (1-10 d) signals
  <OUT>/gate2/survey_targets.csv, <OUT>/gate3/targets_100pc.csv      coordinates, Tmag, TIC radius / mass

Output: <OUT>/cool/signals_long.csv, signals_short.csv, stars.csv

Usage: python -I cool/stars.py
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import OUT

COOL = OUT / "cool"
TAU = json.loads((HERE.parent / "gate2" / "frozen.json").read_text())


def targets() -> pd.DataFrame:
    a = pd.read_csv(OUT / "gate2" / "survey_targets.csv")
    a["survey"] = "50pc"
    b = pd.read_csv(OUT / "gate3" / "targets_100pc.csv")
    b["survey"] = "100pc"
    cols = ["tic", "ra", "dec", "Tmag", "Teff", "rad", "mass", "d_pc", "survey"]
    for c in cols:
        if c not in b:
            b[c] = np.nan
    return pd.concat([a[cols], b[cols]], ignore_index=True).drop_duplicates("tic")


def long_signals() -> pd.DataFrame:
    out = []
    for tag, surv in (("survey", "50pc"), ("survey100", "100pc")):
        for line in open(OUT / tag / "vet" / "graded.jsonl"):
            r = json.loads(line)
            s, g = r["stats"], r["grade"]
            st = g["status"]
            sec = s.get("secondary") or {}
            oe = s.get("odd_even") or {}
            cat = ((s.get("star") or {}).get("catalogue") or {}).get("objects") or []
            out.append({
                "key": r["key"], "tic": r["tic"], "survey": surv, "sig": r["sig_index"],
                "P": s["P"], "t0": s["t0"], "dur_h": s["dur_h"], "depth": s["depth"],
                "mes": s["mes"], "n_tr": s.get("n_transits"), "tier": g["tier"],
                "sec_status": st.get("secondary"), "sec_z": sec.get("z_sec"),
                "sec_excess": sec.get("excess"), "sec_phase_d": sec.get("phase_sec"),
                "sec_mes": sec.get("mes_sec"),
                "oe_status": st.get("odd_even"), "oe_z": oe.get("z"), "d_odd": oe.get("d_odd"),
                "d_even": oe.get("d_even"),
                "sap_status": st.get("sap"), "gap_status": st.get("gap_edge"),
                "dur_ratio": (s.get("duration_density") or {}).get("ratio"),
                "dur_status": st.get("duration_density"), "dom": (s.get("per_transit") or {}).get("dom"),
                "uniq_status": st.get("uniqueness"), "sys_status": st.get("systematics_window"),
                "model_status": st.get("model_compare"), "pixel_status": st.get("pixel"),
                "cat_status": st.get("catalogue"),
                "cat_objects": "; ".join(f"{o.get('kind')}:{o.get('name')}@{o.get('P')}" for o in cat),
                "reasons": " | ".join(g.get("reasons") or []),
                "n_sectors": r.get("n_sectors"),
            })  # fmt: skip
    return pd.DataFrame(out)


def short_signals() -> pd.DataFrame:
    out = []
    for tag, surv in (("survey", "50pc"), ("survey100", "100pc")):
        for line in gzip.open(OUT / tag / "merged" / "results.jsonl.gz", "rt"):
            r = json.loads(line)
            if r.get("status") != "ok":
                continue
            for i, s in enumerate(r["search"]["short"]["signals"]):
                out.append({"tic": r["tic"], "survey": surv, "sig": i, "P": s["P"], "t0": s["t0"],
                            "dur_h": s["dur_h"], "depth": s["depth"], "mes": s["mes"],
                            "n_ep": s["n_ep"], "dom": s["dom"], "n_sectors": r["n_sectors_found"],
                            "above_tau": s["mes"] >= TAU["tau_short"]})  # fmt: skip
    return pd.DataFrame(out)


def main():
    COOL.mkdir(parents=True, exist_ok=True)
    t = targets()
    L = long_signals().merge(t.drop(columns="survey"), on="tic", how="left")
    S = short_signals().merge(t.drop(columns="survey"), on="tic", how="left")
    L.to_csv(COOL / "signals_long.csv", index=False)
    S.to_csv(COOL / "signals_short.csv", index=False)
    g = L.groupby("tic")
    stars = pd.DataFrame({
        "n_long": g.size(), "max_mes": g.mes.max(), "max_depth": g.depth.max(),
        "tiers": g.tier.agg(lambda x: "".join(sorted(set(x)))),
        "n_sec_fail": g.sec_status.agg(lambda x: int((x == "fail").sum())),
        "n_oe_fail": g.oe_status.agg(lambda x: int((x == "fail").sum())),
        "n_sap_fail": g.sap_status.agg(lambda x: int((x == "fail").sum())),
        "n_gap_fail": g.gap_status.agg(lambda x: int((x == "fail").sum())),
        "cat": g.cat_objects.agg(lambda x: " ".join(sorted({c for v in x for c in str(v).split("; ") if c and c != "nan"}))),
    }).reset_index().merge(t, on="tic", how="left")  # fmt: skip
    stars.to_csv(COOL / "stars.csv", index=False)
    print("long signals", len(L), "stars", L.tic.nunique(), L.groupby("survey").tic.nunique().to_dict())
    print("short signals", len(S), "stars", S.tic.nunique(), "| >= tau_short", int(S.above_tau.sum()),
          "on", S[S.above_tau].tic.nunique(), "stars; tau_short", TAU["tau_short"])  # fmt: skip


if __name__ == "__main__":
    main()
