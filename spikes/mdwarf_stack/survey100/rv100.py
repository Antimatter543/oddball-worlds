"""Spectroscopic RV / binarity look-up for 100 pc survivors (anonymous VizieR + ESA Gaia TAP).

Per TIC: 10" VizieR cones (IDIA mirror, then CfA; an empty answer is re-asked on the other mirror)
in LAMOST DR11 (V/162; LRS + MRS tables), LAMOST DR7 (V/156), APOGEE DR17 (III/286), GALAH DR4
(II/385) and RAVE DR6 (III/283); plus the Gaia DR3 RV statistics of the target and of every Gaia
source within 63" (rv_chisq_pvalue, rv_amplitude_robust, ruwe, ipd_frac_multi_peak,
non_single_star), taken from the cached research cone (VizieR I/355; ESA TAP only as a fallback).
Gaia gives RV statistics only for grvs_mag <= 12, so most of these faint stars have none. The
Gaia mean RV counts as one more epoch. A failed query is recorded as "not checked", never as
"nothing there".
Pre-registered binarity rule (as in followup/archive/rv_mrs_clean.py): >= 2 epochs with constancy
chi2 p < 0.0027 and peak-to-peak > 2 km/s; Gaia: rv_chisq_pvalue < 0.01 with rv_amplitude_robust
> 10 km/s, or RUWE > 1.4, or non_single_star > 0.
Writes <OUT>/survey100/vet/rv100.json. Usage: python -I survey100/rv100.py TIC [TIC ...]
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common100 import VET

UA = {"User-Agent": "astro-research-client/1.0"}
SERVERS = ("vizier.idia.ac.za", "vizier.cfa.harvard.edu")
C_KMS = 299792.458
CATS = {"LAMOST DR11": "V/162", "LAMOST DR7": "V/156", "APOGEE DR17": "III/286",
        "GALAH DR4": "II/385", "RAVE DR6": "III/283"}
# barycentric columns first: APOGEE allvis "RV" is VREL (no barycentric correction), VHelio is the
# barycentric one; the APOGEE "catalog" row is the visit average and is skipped when visits exist.
RVCOLS = ("VHelio", "HRV", "RVbr1", "Vhelio", "HRVcor", "rv", "RV", "rv_comp_1", "RVel")


def cell(v):
    if np.ma.is_masked(v):
        return None
    a = np.asarray(v)
    if a.ndim:
        return str(v)
    x = a.item()
    if isinstance(x, bytes):
        x = x.decode()
    if isinstance(x, float) and (math.isnan(x) or abs(x) >= 9.9e8):
        return None
    return x


def vizier(ra, dec, cat):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier

    log = []
    for server in SERVERS:
        try:
            v = Vizier(columns=["**", "_r"], row_limit=200, vizier_server=server, timeout=90)
            tl = v.query_region(SkyCoord(ra * u.deg, dec * u.deg), radius=10 * u.arcsec, catalog=cat)
            got = {t.meta.get("name", cat): [{k: cell(r[k]) for k in t.colnames} for r in t] for t in tl}
            n = sum(len(x) for x in got.values())
            log.append(f"ok {server} {n} rows")
            if n:
                return {"status": "ok", "log": log, "tables": got}
        except Exception as ex:  # noqa: BLE001
            log.append(f"failed {server} {ex!r}"[:200])
        time.sleep(1.0)
    ok = any(x.startswith("ok") for x in log)
    return {"status": "ok" if ok else "not checked", "log": log, "tables": {}}


def epochs(tables):
    """(table, rv, err, mjd-or-None) for rows that carry an RV and an error."""
    pts = []
    for tn, rows in tables.items():
        if tn.endswith("/catalog") and tn.replace("/catalog", "/allvis") in tables:
            continue
        for r in rows:
            col = next((c for c in RVCOLS if r.get(c) is not None and isinstance(r.get(c), (int, float))), None)
            mjd = next((r.get(c) for c in ("MJD", "mjd", "ObsDate", "Date") if r.get(c) is not None), None)
            meta = {"table": tn, "mjd": mjd, "_r": r.get("_r"), "ncomp": r.get("Ncomp"),
                    "id": r.get("APOGEE") or r.get("ObsID")}
            if col is None:
                # LAMOST low-resolution tables give the redshift z (+- e_z) instead of an RV;
                # S/N in i >= 10 required (the M-dwarf LRS RVs are poor below that)
                z, ez = r.get("z"), r.get("e_z")
                if (isinstance(z, (int, float)) and isinstance(ez, (int, float)) and ez > 0
                        and (r.get("snri") or 0) >= 10 and abs(z) < 0.01):
                    pts.append(dict(meta, col="z (c z)", rv=C_KMS * float(z), err=C_KMS * float(ez),
                                    snr=r.get("snri")))
                continue
            ecol = next((c for c in (f"e_{col}", "e_RV", "e_HRV", "e_rv", "e_RVbr1", "e_Vhelio")
                         if isinstance(r.get(c), (int, float)) and r.get(c)), None)
            pts.append(dict(meta, col=col, rv=float(r[col]), err=float(r[ecol]) if ecol else None,
                            snr=r.get("snrr") or r.get("snr")))
    return dedupe(pts)


def dedupe(pts):
    """The same LAMOST exposure appears in several tables and in DR7 and DR11: keep one per
    (MJD, RV). Applied per catalogue and again over all catalogues together."""
    seen, out = set(), []
    for p in pts:
        k = (float(p["mjd"]), round(p["rv"], 1)) if isinstance(p["mjd"], (int, float)) else id(p)
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def rule(pts):
    from scipy import stats

    p = [x for x in pts if x["err"]]
    if len(p) < 2:
        return {"n": len(p), "verdict": "fewer than 2 RV epochs with errors"}
    rv = np.array([x["rv"] for x in p])
    e = np.array([x["err"] for x in p])
    w = 1 / e**2
    mu = np.sum(w * rv) / w.sum()
    chi2 = float(np.sum(((rv - mu) / e) ** 2))
    pv = float(stats.chi2.sf(chi2, len(p) - 1))
    ptp = float(rv.max() - rv.min())
    return {"n": len(p), "chi2": chi2, "p": pv, "ptp_kms": ptp,
            "verdict": "binary (rule)" if pv < 0.0027 and ptp > 2 else "no RV variation detected"}


def gaia_tap(ids):
    import requests

    q = ("SELECT source_id, radial_velocity, radial_velocity_error, rv_nb_transits, rv_chisq_pvalue, "
         "rv_renormalised_gof, rv_amplitude_robust, ruwe, ipd_frac_multi_peak, non_single_star, "
         "phot_variable_flag, phot_g_mean_mag FROM gaiadr3.gaia_source WHERE source_id IN ("
         + ",".join(str(int(i)) for i in ids) + ")")
    try:
        r = requests.post("https://gea.esac.esa.int/tap-server/tap/sync",
                          data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "QUERY": q},
                          headers=UA, timeout=120)
        r.raise_for_status()
        j = r.json()
        cols = [m["name"] for m in j["metadata"]]
        return {"status": "ok", "rows": {str(row[0]): dict(zip(cols, row)) for row in j["data"]}}
    except Exception as ex:  # noqa: BLE001
        return {"status": f"not checked: {ex!r}"[:250], "rows": {}}


def from_vizier(x):
    """Gaia DR3 (VizieR I/355 row, cached by research_candidates) -> the TAP column names."""
    return {"source_id": x["Source"], "radial_velocity": x.get("RV"), "radial_velocity_error": x.get("e_RV"),
            "rv_nb_transits": x.get("RVNper"), "rv_chisq_pvalue": x.get("RVchi2"),
            "rv_renormalised_gof": x.get("RVgof"), "rv_amplitude_robust": x.get("RVamp"),
            "ruwe": x.get("RUWE"), "ipd_frac_multi_peak": x.get("IPDfmp"), "non_single_star": x.get("NSS"),
            "phot_variable_flag": x.get("VarFlag"), "phot_g_mean_mag": x.get("Gmag"), "grvs_mag": x.get("GRVSmag")}


def gaia_flags(g):
    if not g:
        return []
    f = []
    if g.get("ruwe") and g["ruwe"] > 1.4:
        f.append(f"RUWE {g['ruwe']:.2f}")
    if (g.get("rv_chisq_pvalue") is not None and g["rv_chisq_pvalue"] < 0.01
            and (g.get("rv_amplitude_robust") or 0) > 10):
        f.append(f"RV p {g['rv_chisq_pvalue']:.3g}, amp {g['rv_amplitude_robust']:.1f} km/s")
    if g.get("non_single_star"):
        f.append(f"non_single_star {g['non_single_star']}")
    if (g.get("ipd_frac_multi_peak") or 0) > 10:
        f.append(f"ipd_frac_multi_peak {g['ipd_frac_multi_peak']}")
    return f


def main():
    info = json.loads((VET / "star_info.json").read_text())
    fp = VET / "rv100.json"
    out = json.loads(fp.read_text()) if fp.exists() else {}
    for tic in sys.argv[1:]:
        st = info[str(int(tic))]
        g = (st.get("gaia") or {}).get("target") or {}
        ra, dec = g.get("ra", st["ra"]), g.get("dec", st["dec"])
        res = {"ra": ra, "dec": dec, "cats": {}}
        for name, cat in CATS.items():
            q = vizier(ra, dec, cat)
            pts = epochs(q["tables"])
            res["cats"][name] = {"status": q["status"], "log": q["log"],
                                 "n_rows": sum(len(x) for x in q["tables"].values()),
                                 "tables": {k: len(v) for k, v in q["tables"].items()}, "rv_points": pts}
            time.sleep(0.5)
        allpts = dedupe([p for c in res["cats"].values() for p in c["rv_points"]])
        # Gaia DR3 statistics from the cached research cone (VizieR I/355, 63"); the ESA TAP is
        # asked only when no cached cone exists (it timed out from here on 2026-11-15).
        rp = VET / "research" / f"{int(tic)}.json"
        d = json.loads(rp.read_text()) if rp.exists() else {}
        if d.get("gaia", {}).get("status") == "ok":
            rows = d["gaia"]["data"]["rows"]
            res["gaia"] = {"status": "ok (VizieR I/355 research cone)",
                           "rows": {str(x["Source"]): from_vizier(x) for x in rows}}
            tgt = min(rows, key=lambda x: x["_r"])
            g = dict(g, source=tgt["Source"])
        else:
            ids = [int(g["source"])] if g.get("source") else []
            res["gaia"] = gaia_tap(ids) if ids else {"status": "not checked: no Gaia ids", "rows": {}}
        tg = res["gaia"]["rows"].get(str(g.get("source")))
        if tg and tg.get("radial_velocity") is not None and tg.get("radial_velocity_error"):
            # Gaia's mean RV (2014-2017) counts as one more epoch against the survey spectra
            allpts.append({"table": "Gaia DR3 mean RV", "col": "radial_velocity",
                           "rv": float(tg["radial_velocity"]), "err": float(tg["radial_velocity_error"]),
                           "mjd": "2014-2017 mean", "snr": None})
        res["rule"] = rule(allpts)
        res["gaia_target_flags"] = gaia_flags(tg)
        res["gaia_neighbour_flags"] = {s: gaia_flags(r) for s, r in res["gaia"]["rows"].items()
                                       if s != str(g.get("source")) and gaia_flags(r)}
        out[str(int(tic))] = res
        print(tic, {k: (v["status"], v["n_rows"], len(v["rv_points"])) for k, v in res["cats"].items()},
              res["rule"], "gaia", res["gaia"]["status"], "target", res["gaia_target_flags"],
              {k: v for k, v in (tg or {}).items() if k.startswith("rv") or k == "ruwe"},
              "neighbours", res["gaia_neighbour_flags"], flush=True)
        fp.write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
