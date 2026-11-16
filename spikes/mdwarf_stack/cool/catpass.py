"""Catalogue pass for every star the two Track C surveys flagged (long >= tau_long, short >= tau_short).

Anonymous, cached, fail-closed: each source per star is "ok" or the error text; a failed query is
"not checked", never "not found".

Sources
  Gaia DR3 counterpart  VizieR I/355/gaiadr3 105" cone (oddball.gallery.research), the vetting rule
                        (vetting/catalog._gaia_block): 60", parallax > 5 mas, G ~ Tmag + 1.6
  Gaia DR3 by source_id ESA TAP: gaia_source (RUWE, RV stats, NSS flag, variability flag),
                        vari_eclipsing_binary, vari_classifier_result, nss_two_body_orbit
  VizieR cones          VSX B/vsx/vsx (120" from TIC pos, judged at 30"), Prsa+2022 J/ApJS/258/16
                        (42"), Kostov+2025 J/ApJS/279/50 tables 3/4 (21"), IJspeert+2024 J/A+A/691/A242
                        (21"; an OBAF-star catalogue, so M dwarfs are not expected), ASAS-SN II/366 (30"),
                        ZTF periodic variables Chen+2020 J/ApJS/249/18 (10"), ATLAS variables Heinze+2018
                        J/AJ/156/241 (10"), AllWISE II/328/allwise (10"), Marton+2019 YSO J/MNRAS/487/2522
                        (3"), Kounkel+2020 strings J/AJ/160/279 (3"). Gaia-position cones use the Gaia
                        DR3 (2016) position of the counterpart (high proper motion).
  SIMBAD                Harvard TAP mirror, 30" cone from the Gaia position: main_id, otype, nbref
  ExoFOP                local TOI table (2026-11-16 copy) by TIC; live CTOI download
  exoplanet.eu          local copy, 30"

Writes <OUT>/cool/cat/: <key>.parquet caches (research.py layout), gaia_match.csv, gaia_tap_*.csv,
simbad.jsonl, status.json.
Usage: python -I cool/catpass.py [step ...]   steps: gaia vizier tap simbad (default all)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "src"))
from common import OUT

COOL = OUT / "cool"
CATD = COOL / "cat"
os.environ["ODDBALL_DATA"] = str(CATD)  # research.py caches under <ODDBALL_DATA>/gallery/research

from oddball.common.http import ANON_USER_AGENT
from oddball.gallery import research as R

R.CATALOGUES.update({
    "prsa": ("J/ApJS/258/16", 42.0, ["**"], {}),
    "ijspeert": ("J/A+A/691/A242", 21.0, ["**"], {}),
    "asassn_g": ("II/366", 30.0, ["**"], {}),
    "ztf_g": ("J/ApJS/249/18", 10.0, ["**"], {}),
    "atlas_g": ("J/AJ/156/241", 10.0, ["**"], {}),
    "allwise_g": ("II/328/allwise", 10.0, ["**"], {}),
    "marton_g": ("J/MNRAS/487/2522", 3.0, ["**"], {}),
    "kounkel_g": ("J/AJ/160/279", 3.0, ["**"], {}),
})  # fmt: skip
TIC_POS = ["vsx", "prsa", "kostov", "ijspeert"]
GAIA_POS = ["asassn_g", "ztf_g", "atlas_g", "allwise_g", "marton_g", "kounkel_g"]
SIMBAD_TAP = "https://simbad.harvard.edu/simbad/sim-tap/sync"
GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"


def star_list() -> pd.DataFrame:
    L = pd.read_csv(COOL / "signals_long.csv")
    S = pd.read_csv(COOL / "signals_short.csv")
    tics = set(L.tic) | set(S[S.above_tau].tic)
    a = pd.read_csv(OUT / "gate2" / "survey_targets.csv")
    b = pd.read_csv(OUT / "gate3" / "targets_100pc.csv")
    t = pd.concat([a[["tic", "ra", "dec", "Tmag"]], b[["tic", "ra", "dec", "Tmag"]]]).drop_duplicates("tic")
    t = t[t.tic.isin(tics)].rename(columns={"Tmag": "tmag"})
    return t.reset_index(drop=True)


def status_update(key, val):
    fp = CATD / "status.json"
    s = json.loads(fp.read_text()) if fp.exists() else {}
    s[key] = val
    fp.write_text(json.dumps(s, indent=1))


def step_gaia(stars):
    from vetting.catalog import _gaia_block

    g = R.query_catalogue("gaia", stars)
    rows = []
    for s in stars.itertuples():
        b = _gaia_block(g[g.tic == s.tic], float(s.tmag))
        if "error" in b:
            rows.append({"tic": s.tic, "gaia_error": b["error"]})
            continue
        t = b["target"]
        nb = b["neighbours"]
        n1 = [x for x in nb if x["sep"] <= 21]
        rows.append({"tic": s.tic, "source_id": t["source"], "G": t["G"], "plx": t["plx"], "ruwe": t["ruwe"],
                     "ra_g": t["ra"], "dec_g": t["dec"], "sep_tic": t["sep"],
                     "n_nb_21": len(n1), "nb_min_dG_21": min((x["dG"] for x in n1), default=None),
                     "nb_list_63": json.dumps(nb[:12])})  # fmt: skip
    m = pd.DataFrame(rows)
    m.to_csv(CATD / "gaia_match.csv", index=False)
    status_update("gaia_cone", {"ok": True, "n": len(m), "no_counterpart": int(m.source_id.isna().sum())})
    print("gaia counterparts", int(m.source_id.notna().sum()), "of", len(m))


def gpos(stars):
    m = pd.read_csv(CATD / "gaia_match.csv")
    x = stars.merge(m[["tic", "ra_g", "dec_g"]], on="tic", how="left")
    x["ra"] = x.ra_g.fillna(x.ra)
    x["dec"] = x.dec_g.fillna(x.dec)
    return x[["tic", "ra", "dec", "tmag"]]


def step_vizier(stars):
    sg = gpos(stars)
    for key in TIC_POS + GAIA_POS:
        try:
            df = R.query_catalogue(key, stars if key in TIC_POS else sg, chunk=40, sleep=1.0)
            status_update(key, {"ok": True, "rows": len(df), "catalogue": R.CATALOGUES[key][0]})
            print(key, "ok", len(df))
        except Exception as e:  # noqa: BLE001
            status_update(key, {"ok": False, "error": repr(e)[:300]})
            print(key, "ERROR", repr(e)[:200])


def _tap(url, q, fmt="json", timeout=180):
    import requests

    r = requests.post(url, data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": fmt, "QUERY": q},
                      headers={"User-Agent": ANON_USER_AGENT}, timeout=timeout)  # fmt: skip
    r.raise_for_status()
    j = r.json()
    cols = [c["name"] for c in j["metadata"]]
    return pd.DataFrame(j["data"], columns=cols)


def step_tap():
    m = pd.read_csv(CATD / "gaia_match.csv", dtype={"source_id": "Int64"})
    ids = [int(x) for x in m.source_id.dropna()]
    qs = {
        "gaia_source": "SELECT source_id, ruwe, radial_velocity, radial_velocity_error, rv_nb_transits, "
                       "rv_chisq_pvalue, rv_amplitude_robust, ipd_frac_multi_peak, non_single_star, "
                       "phot_variable_flag, has_epoch_photometry, bp_rp, phot_g_mean_mag, parallax "
                       "FROM gaiadr3.gaia_source WHERE source_id IN ({})",
        "veb": "SELECT * FROM gaiadr3.vari_eclipsing_binary WHERE source_id IN ({})",
        "vclass": "SELECT source_id, classifier_name, best_class_name, best_class_score "
                  "FROM gaiadr3.vari_classifier_result WHERE source_id IN ({})",
        "nss": "SELECT source_id, nss_solution_type, period, period_error, eccentricity, semi_amplitude_primary, "
               "semi_amplitude_secondary, significance, goodness_of_fit FROM gaiadr3.nss_two_body_orbit "
               "WHERE source_id IN ({})",
        "vrot": "SELECT source_id, best_rotation_period, max_activity_index "
                "FROM gaiadr3.vari_rotation_modulation WHERE source_id IN ({})",
    }  # fmt: skip
    for name, q in qs.items():
        parts = []
        try:
            for i in range(0, len(ids), 400):
                parts.append(_tap(GAIA_TAP, q.format(",".join(map(str, ids[i:i + 400])))))
                time.sleep(1.0)
            df = pd.concat(parts, ignore_index=True)
            df.to_csv(CATD / f"gaia_tap_{name}.csv", index=False)
            status_update(f"gaia_tap_{name}", {"ok": True, "rows": len(df), "n_ids": len(ids)})
            print(name, "ok", len(df))
        except Exception as e:  # noqa: BLE001
            status_update(f"gaia_tap_{name}", {"ok": False, "error": repr(e)[:300]})
            print(name, "ERROR", repr(e)[:200])


def step_simbad(stars):
    sg = gpos(stars)
    fp = CATD / "simbad.jsonl"
    done = set()
    if fp.exists():
        for line in open(fp):
            r = json.loads(line)
            if r.get("ok"):
                done.add(r["tic"])
    n_err = 0
    with open(fp, "a") as fh:
        for s in sg.itertuples():
            if s.tic in done:
                continue
            q = ("select main_id, otype, ra, dec, nbref, "
                 f"distance(point('icrs',ra,dec),point('icrs',{s.ra},{s.dec}))*3600 as sep from basic "
                 f"where contains(point('icrs',ra,dec),circle('icrs',{s.ra},{s.dec},{30 / 3600}))=1 order by sep")  # fmt: skip
            rec = {"tic": int(s.tic)}
            for attempt in range(3):
                try:
                    df = _tap(SIMBAD_TAP, q, timeout=60)
                    rec.update(ok=True, rows=df.to_dict("records"))
                    break
                except Exception as e:  # noqa: BLE001
                    rec.update(ok=False, error=repr(e)[:200])
                    time.sleep(5 * (attempt + 1))
            n_err += not rec["ok"]
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            time.sleep(0.4)
    status_update("simbad", {"ok": n_err == 0, "n_err_this_run": n_err})
    print("simbad errors", n_err)


def main():
    CATD.mkdir(parents=True, exist_ok=True)
    steps = sys.argv[1:] or ["gaia", "vizier", "tap", "simbad"]
    stars = star_list()
    print("stars", len(stars), flush=True)
    for st in steps:
        t = time.time()
        if st == "gaia":
            step_gaia(stars)
        elif st == "vizier":
            step_vizier(stars)
        elif st == "tap":
            step_tap()
        elif st == "simbad":
            step_simbad(stars)
        print(st, "done", round(time.time() - t), "s", flush=True)


if __name__ == "__main__":
    main()
