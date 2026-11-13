"""Pre-registration tables: ephemeris of each leftover signal, every predicted epoch while the star
was on silicon (tess-point + TESScut sector lists), its 1-sigma timing error, and whether the
survey's own light curve (frozen preprocessing) covered it. Uses only the survey's data.

For the 2-transit TIC 57830249 it also lists the period aliases P_n = (t2 - t1) / n.

Writes FU_OUT/ephem.json and prints markdown. Usage: python -I followup/ephem.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fu  # noqa: E402
from get_lightcurve import get_lightcurve  # noqa: E402

NOW = 4357.5  # 2026-11-13 BTJD


def main():
    sigs = fu.load_signals()
    dossier = json.loads((fu.OUT / "survey" / "vet" / "dossier.json").read_text())
    res = {}
    for key in fu.KEYS:
        r = sigs[key]
        s = r["stats"]
        dz = dossier[key]
        e = fu.fit_ephem(s)
        lc, meta = get_lightcurve(r["tic"], max_bytes=200_000_000)
        pp = fu.prep(lc)
        D = e["D"]
        met = fu.Meter(pp, D, exclude=list(s["epochs"]))
        tlo, thi = float(pp["idx"].min() * fu.BIN_D), float(pp["idx"].max() * fu.BIN_D)
        secs = fu.sector_windows(dz["ra"], dz["dec"])
        used = sorted({int(x["sector"]) for x in meta["sectors"]})
        eps = []
        for sw in secs:
            for n in fu.epochs_between(e, sw["lo"], sw["hi"]):
                tp, sp = fu.predict(e, n)
                half = fu.NSIG_WIN * sp
                ms = met.window(tp, half)
                cov = float(np.mean([m is not None for m in ms]))
                c0 = met.at(tp)
                status = ("future" if tp > NOW else
                          "searched" if (c0 is not None and sw["sector"] in used) else
                          "gap_in_searched_sector" if sw["sector"] in used else
                          "sector_not_searched")
                eps.append({"n": n, "t": tp, "sig_h": 24 * sp, "half_h": 24 * half,
                            "sector": sw["sector"], "on_tesspoint": sw["tesspoint"],
                            "on_tesscut": sw["tesscut"], "survey_window_cover": cov,
                            "status": status,
                            "detected": any(abs(tp - x) < 0.5 for x in s["epochs"])})
        out = {"key": key, "tic": r["tic"], "ephem": e, "sectors_used": used,
               "sectors_on_silicon": [x["sector"] for x in secs],
               "tesscut_status": secs[0]["tesscut_status"] if secs else "none",
               "epochs": eps, "baseline": [tlo, thi]}
        if len(s["epochs"]) == 2:
            dt = s["epochs"][1] - s["epochs"][0]
            al = []
            for n in range(1, 181):
                P = dt / n
                k = np.arange(-int((s["epochs"][0] - tlo) / P) - 1, int((thi - s["epochs"][0]) / P) + 2)
                tt = s["epochs"][0] + k * P
                other = [float(x) for x in tt if met.at(float(x)) is not None
                         and min(abs(x - y) for y in s["epochs"]) > 0.5]
                al.append({"n": n, "P": P, "n_other_searched": len(other)})
            out["aliases"] = al
        res[key] = out
        print(f"### {key}: P = {e['P']:.5f} +- {24 * np.sqrt(e['cov'][1][1]):.3f} h "
              f"(search {s['P']:.5f}), T0 = {e['T0']:.4f} +- {24 * np.sqrt(e['cov'][0][0]):.2f} h, "
              f"D = {24 * D:.1f} h, depth = {1e3 * e['depth']:.2f} ppt, chi2/dof {e['chi2']:.2f}/"
              f"{e['dof']}, epoch sigma (h) {np.round(e['sig_epoch_h'], 2).tolist()}, residuals "
              f"(h) {np.round(e['res_h'], 2).tolist()}")
        print(f"sectors used {used}; on silicon {[x['sector'] for x in secs]}; TESScut list "
              f"{secs[0]['tesscut_status'] if secs else ''}")
        print("| n | BTJD | 1 sigma (h) | window +- (h) | sector | status | survey cover of window |")
        print("|---|---|---|---|---|---|---|")
        for x in eps:
            tag = " (detected)" if x["detected"] else ""
            print(f"| {x['n']} | {x['t']:.3f} | {x['sig_h']:.2f} | {x['half_h']:.2f} | "
                  f"{x['sector']}{'' if x['on_tesspoint'] else ' (TESScut only)'} | "
                  f"{x['status']}{tag} | {x['survey_window_cover']:.2f} |")
        if "aliases" in out:
            free = [a for a in out["aliases"] if a["n_other_searched"] == 0]
            print(f"aliases with no other epoch in searched data: "
                  f"{[(a['n'], round(a['P'], 3)) for a in free]}")
        print()
    (fu.FU_OUT / "ephem.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
