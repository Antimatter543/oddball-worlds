"""Per-star catalogue summary from the catalogue pass (cool/catpass.py caches). Fail-closed.

For each star: what every catalogue says ON the target and what it says about NEIGHBOURS (a catalogued
variable within 1-2 TESS pixels is a blend suspect), and whether each source answered.
  on-target radii: VSX 30" (TIC position; VSX is mostly J2000), Prsa / Kostov / IJspeert by TIC id
  (or 10"), ASAS-SN 30", ZTF / ATLAS 5" (Gaia 2016 position), Gaia tables by source_id,
  SIMBAD 5" from the Gaia position (otherwise the nearest object is listed as a neighbour).
Writes <OUT>/cool/cat_summary.csv and <OUT>/cool/cat_periods.csv (tic, P, src: catalogued periods
for the light-curve characterisation).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import OUT

COOL = OUT / "cool"
CATD = COOL / "cat"
RES = CATD / "gallery" / "research"
CAT_DIR = Path("/home/antiraedus/Projects/oddball-worlds/data/planets_spike/catalogs")
YOUNG_OTYPES = {"Y*O", "Y*?", "TT*", "TT?", "pr*", "pr?", "Ae*", "Ae?", "BD*", "BD?", "out", "HH"}
# SIMBAD otype codes (2026 nomenclature short forms); variable / binary codes of interest
EB_OTYPES = {"EB*", "EB?", "SB*", "**", "El*", "El?", "Al*", "WU*", "bL*"}


def _num(x):
    v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    return float(v) if np.isfinite(v) else np.nan


def load(key):
    fp = RES / f"{key}.parquet"
    return pd.read_parquet(fp) if fp.exists() else None


def main():
    status = json.loads((CATD / "status.json").read_text())
    gm = pd.read_csv(CATD / "gaia_match.csv", dtype={"source_id": "Int64"})
    stars = gm[["tic", "source_id", "G", "plx", "ruwe", "n_nb_21", "nb_min_dG_21"]].copy()
    t = {k: load(k) for k in ("vsx", "prsa", "kostov", "ijspeert", "asassn_g", "ztf_g", "atlas_g",
                              "allwise_g", "marton_g", "kounkel_g", "capi_dip")}  # fmt: skip
    gt = {}
    for k in ("gaia_source", "veb", "vclass", "nss", "vrot"):
        fp = CATD / f"gaia_tap_{k}.csv"
        gt[k] = pd.read_csv(fp, dtype={"source_id": "Int64"}) if fp.exists() else None
    simb = {}
    if (CATD / "simbad.jsonl").exists():
        for line in open(CATD / "simbad.jsonl"):
            r = json.loads(line)
            if r.get("ok") or r["tic"] not in simb:
                simb[r["tic"]] = r
    toi = pd.read_csv(CAT_DIR / "toi_2026-11-16.csv", low_memory=False)
    ctoi_fp = OUT / "vetting" / "cache" / "ctoi.txt"
    ctoi = pd.read_csv(ctoi_fp, sep="|", low_memory=False) if ctoi_fp.exists() else None

    rows, periods = [], []
    for s in stars.itertuples():
        tic, sid = int(s.tic), s.source_id
        on, nb, var_on, young, src_bad = [], [], [], [], []

        def chk(key):
            st = status.get(key)
            if not st or not st.get("ok"):
                src_bad.append(key)
                return None
            return t.get(key)

        v = chk("vsx")
        if v is not None and len(v):
            x = v[v.tic == tic]
            for r in x.itertuples():
                sep = float(r.sep_arcsec)
                typ = str(r.Type)
                p = _num(r.Period)
                lab = f"VSX {r.Name} [{typ}] P={p:g}"
                if sep <= 30:
                    (on if typ.startswith("E") else var_on).append(lab)
                    if np.isfinite(p) and p > 0:
                        periods.append((tic, p, "vsx"))
                elif sep <= 63:
                    nb.append(f"{lab} at {sep:.0f}\"")
        for key, lab0 in (("prsa", "Prsa+2022"), ("kostov", "Kostov+2025"), ("ijspeert", "IJspeert+2024")):
            v = chk(key)
            if v is None or not len(v):
                continue
            x = v[v.tic == tic]
            if key == "kostov":
                x = x[x.table.astype(str).str.contains("table3|table4")]
            if key == "ijspeert":
                x = x[x.table.astype(str).str.contains("eb")]
            for r in x.to_dict("records"):
                ttic = _num(r.get("TIC"))
                p = _num(r.get("Per", r.get("Per-TESS")))
                lab = f"{lab0} TIC {int(ttic) if np.isfinite(ttic) else '?'} P={p:g}"
                if (np.isfinite(ttic) and int(ttic) == tic) or r["sep_arcsec"] <= 10:
                    on.append(lab)
                    if np.isfinite(p) and p > 0:
                        periods.append((tic, p, key))
                else:
                    nb.append(f"{lab} at {r['sep_arcsec']:.0f}\"")
        for key, lab0, rad, pcol, tcol, idcol in (
                ("asassn_g", "ASAS-SN", 30, "Per", "Type", "ASASSN-V"),
                ("ztf_g", "ZTF", 5, "Per", "Type", "ID"), ("atlas_g", "ATLAS", 5, "fp-lngfitper", "Class", "ATOID")):
            v = chk(key)
            if v is None or not len(v):
                continue
            x = v[(v.tic == tic) & (v.sep_arcsec <= 63)]
            if key == "ztf_g":
                x = x[x.table.astype(str).str.contains("table2")]
            for r in x.to_dict("records"):
                p = _num(r.get(pcol))
                typ = str(r.get(tcol))
                lab = f"{lab0} {r.get(idcol)} [{typ}] P={p:g}"
                if r["sep_arcsec"] <= rad:
                    (on if (typ.upper().startswith("E") or typ in ("EW", "EA", "EB")) else var_on).append(lab)
                    if np.isfinite(p) and p > 0:
                        periods.append((tic, p, key))
                else:
                    nb.append(f"{lab} at {r['sep_arcsec']:.0f}\"")
        # Gaia by source id
        gsrc = {}
        for k, df in gt.items():
            st = status.get(f"gaia_tap_{k}")
            if not st or not st.get("ok") or df is None:
                src_bad.append(f"gaia_{k}")
                continue
            if sid is pd.NA or sid is None:
                continue
            gsrc[k] = df[df.source_id == sid]
        if "veb" in gsrc and len(gsrc["veb"]):
            r = gsrc["veb"].iloc[0]
            fq = _num(r.get("frequency", r.get("Freq")))  # ESA name, or the VizieR I/358/veb name
            p = 1 / fq if fq > 0 else np.nan
            on.append(f"Gaia DR3 EB P={p:g}")
            if np.isfinite(p):
                periods.append((tic, p, "gaia_veb"))
        if "vclass" in gsrc and len(gsrc["vclass"]):
            for r in gsrc["vclass"].itertuples():
                var_on.append(f"Gaia class {r.best_class_name} ({r.best_class_score:.2f})")
                if str(r.best_class_name).upper().startswith("YSO"):
                    young.append("Gaia class YSO")
        if "nss" in gsrc and len(gsrc["nss"]):
            for r in gsrc["nss"].itertuples():
                lab = f"Gaia NSS {r.nss_solution_type} P={_num(r.period):g}"
                (on if "Eclipsing" in str(r.nss_solution_type) else var_on).append(lab)
                if np.isfinite(_num(r.period)):
                    periods.append((tic, _num(r.period), "gaia_nss"))
        gs = gsrc.get("gaia_source")
        rv = {}
        if gs is not None and len(gs):
            r = gs.iloc[0]
            rv = {"rv_amp": _num(r.rv_amplitude_robust), "rv_p": _num(r.rv_chisq_pvalue),
                  "rv_n": _num(r.rv_nb_transits), "nss_flag": _num(r.non_single_star),
                  "ipd_multi": _num(r.ipd_frac_multi_peak), "gaia_var": str(r.phot_variable_flag)}  # fmt: skip
        if "vrot" in gsrc and len(gsrc["vrot"]):
            var_on.append(f"Gaia rotation P={_num(gsrc['vrot'].iloc[0].best_rotation_period):.3g}")
        # young-star lists
        for key, lab0 in (("marton_g", "Marton+2019"), ("kounkel_g", "Kounkel+2020")):
            v = chk(key)
            if v is None or not len(v):
                continue
            x = v[v.tic == tic]
            if key == "marton_g" and len(x):
                pr = max(_num(x.iloc[0].get("LY", np.nan)), _num(x.iloc[0].get("SY", np.nan)))
                if pr > 0.5:
                    young.append(f"Marton YSO prob {pr:.2f}")
            elif len(x):
                young.append(f"Kounkel string {x.iloc[0].get('Theia', '')}")
        # dipper / complex-rotator lists (only Capistrant+2022 has a VizieR table; see cool/catextra.py)
        cpv = []
        v = chk("capi_dip")
        if v is not None and len(v):
            x = v[(v.tic == tic) & (v.sep_arcsec <= 10)]
            if len(x):
                cpv.append("Capistrant+2022 dipper")
                young.append("Capistrant+2022 dipper")
        w = chk("allwise_g")
        wise = {}
        if w is not None and len(w):
            x = w[w.tic == tic].sort_values("sep_arcsec")
            if len(x):
                r = x.iloc[0]
                w1, w2, w3, w4 = (_num(r.get(c)) for c in ("W1mag", "W2mag", "W3mag", "W4mag"))
                wise = {"W1-W2": w1 - w2, "W2-W3": w2 - w3, "W1-W4": w1 - w4}
                # M dwarf photospheres have W1-W4 ~ 0.3-0.9; > 1.5 with a real W4 detection is excess
                qph = str(r.get("qph", ""))
                if np.isfinite(w1 - w4) and w1 - w4 > 1.5 and len(qph) == 4 and qph[3] == "A":
                    young.append(f"WISE W1-W4 = {w1 - w4:.2f} (excess)")
        # SIMBAD
        sb = simb.get(tic)
        sim_main, sim_type, sim_ref, sim_nb = "", "", np.nan, []
        if sb is None or not sb.get("ok"):
            src_bad.append("simbad")
        else:
            for o in sb["rows"]:
                if str(o.get("otype", "")).startswith("Pl"):
                    continue
                if o["sep"] <= 5 and not sim_main:
                    sim_main, sim_type, sim_ref = o["main_id"], o["otype"], o.get("nbref")
                elif o["sep"] <= 63 and (o["otype"] in EB_OTYPES or "V*" in str(o["otype"])):
                    sim_nb.append(f"{o['main_id']} [{o['otype']}] at {o['sep']:.0f}\"")
            if sim_type in YOUNG_OTYPES:
                young.append(f"SIMBAD type {sim_type}")
            if sim_type in EB_OTYPES:
                on.append(f"SIMBAD {sim_main} [{sim_type}]")
        # ExoFOP
        tois = [f"TOI-{r['TOI']} ({r.get('TFOPWG Disposition', '')}/{r.get('TESS Disposition', '')}) P={_num(r.get('Period (days)')):g}"
                for r in toi[toi["TIC ID"] == tic].to_dict("records")]  # fmt: skip
        ctois = []
        if ctoi is not None:
            ctois = [f"CTOI {r['CTOI']} P={_num(r.get('Period (days)')):g}" for r in ctoi[ctoi["TIC ID"] == tic].to_dict("records")]
        else:
            src_bad.append("ctoi")
        for x in tois + ctois:
            try:
                periods.append((tic, float(x.rsplit("P=", 1)[1]), "toi"))
            except ValueError:
                pass
        rows.append({"tic": tic, "source_id": sid, "G": s.G, "ruwe": s.ruwe, "n_nb_21": s.n_nb_21,
                     "nb_min_dG_21": s.nb_min_dG_21, "eb_on": " | ".join(on), "var_on": " | ".join(var_on),
                     "nb_var": " | ".join(nb + sim_nb), "simbad": sim_main, "simbad_type": sim_type,
                     "simbad_nbref": sim_ref, "toi": " | ".join(tois), "ctoi": " | ".join(ctois),
                     "young": " | ".join(young), "cpv": " | ".join(cpv), "not_checked": ",".join(sorted(set(src_bad))), **rv,
                     **{k: round(v, 2) for k, v in wise.items()}})  # fmt: skip
    out = pd.DataFrame(rows)
    out.to_csv(COOL / "cat_summary.csv", index=False)
    pd.DataFrame(periods, columns=["tic", "P", "src"]).drop_duplicates().to_csv(COOL / "cat_periods.csv", index=False)
    print(len(out), "stars; with EB on target", (out.eb_on != "").sum(), "; with TOI/CTOI",
          ((out.toi != "") | (out.ctoi != "")).sum(), "; young flags", (out.young != "").sum(),
          "; any source not checked", (out.not_checked != "").sum())  # fmt: skip


if __name__ == "__main__":
    main()
