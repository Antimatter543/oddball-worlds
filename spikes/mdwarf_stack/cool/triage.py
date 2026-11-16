"""First-pass triage of the flagged stars from the survey's own numbers (no light curves).

Per star:
  harmonic   a long signal sits at an integer multiple (n <= 60) of a short-search signal (>= 6 MES)
             or of a catalogued period (graded catalogue objects) within 0.1%: a short-period object
  eb_like    any long signal with a secondary-eclipse fail, odd/even fail, or depth >= 1% at MES >= 15
  artefact   every long signal fails SAP or gap_edge or systematics_window (hard)
  known      TOI / CTOI / planet / EB catalogued in the graded rows
Writes <OUT>/cool/triage.csv; prints counts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import OUT

COOL = OUT / "cool"


def harmonic_of(P, cands, tol=1e-3, nmax=60):
    best = None
    for pc in cands:
        if not np.isfinite(pc) or pc <= 0 or pc >= P * 1.001:
            continue
        n = round(P / pc)
        if 1 <= n <= nmax and abs(P - n * pc) <= tol * P:
            if best is None or n < best[1]:
                best = (pc, n)
    return best


def main():
    L = pd.read_csv(COOL / "signals_long.csv")
    S = pd.read_csv(COOL / "signals_short.csv")
    out = []
    for tic, g in L.groupby("tic"):
        sh = S[S.tic == tic]
        catP = []
        for v in g.cat_objects.dropna():
            for o in str(v).split("; "):
                try:
                    catP.append(float(o.rsplit("@", 1)[1]))
                except (IndexError, ValueError):
                    pass
        cands = list(sh.P) + catP
        harm = [harmonic_of(p, cands) for p in g.P]
        n_harm = sum(h is not None for h in harm)
        top = g.sort_values("mes", ascending=False).iloc[0]
        hard_art = ((g.sap_status == "fail") | (g.gap_status == "fail") | (g.sys_status == "fail"))
        out.append({
            "tic": tic, "survey": top.survey, "Tmag": top.Tmag, "rad": top.rad, "d_pc": top.d_pc,
            "n_long": len(g), "n_harm": n_harm,
            "harm_base": min((h[0] for h in harm if h), default=np.nan),
            "top_P": top.P, "top_depth": top.depth, "top_mes": top.mes, "top_dur_h": top.dur_h,
            "top_ntr": top.n_tr, "top_tier": top.tier,
            "n_sec_fail": int((g.sec_status == "fail").sum()), "n_oe_fail": int((g.oe_status == "fail").sum()),
            "n_art": int(hard_art.sum()),
            "short_best_P": sh.sort_values("mes").P.iloc[-1] if len(sh) else np.nan,
            "short_best_mes": sh.mes.max() if len(sh) else np.nan,
            "short_best_depth": sh.sort_values("mes").depth.iloc[-1] if len(sh) else np.nan,
            "known": "; ".join(sorted({o for v in g.cat_objects.dropna() for o in str(v).split("; ")})),
            "tiers": "".join(sorted(set(g.tier))),
        })  # fmt: skip
    T = pd.DataFrame(out)
    T["cls0"] = np.select(
        [T.known.str.contains("toi:|ctoi:|planet:"), T.n_harm == T.n_long, T.n_art == T.n_long,
         (T.n_sec_fail + T.n_oe_fail > 0) | ((T.top_depth >= 0.01) & (T.top_mes >= 15))],
        ["known_planet_or_toi", "harmonic_of_short", "artefact_like", "eb_like"], "other")  # fmt: skip
    T.to_csv(COOL / "triage.csv", index=False)
    print(T.cls0.value_counts())
    print(pd.crosstab(T.cls0, T.known.str.contains("eb:")))


if __name__ == "__main__":
    main()
