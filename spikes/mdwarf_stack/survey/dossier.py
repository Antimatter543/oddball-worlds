"""Per-signal dossier for the shortlist: implied planet, every transit with its data-quality flags,
next predicted transits, TESS sectors, and the catalogue research (research_candidates.py).

Reads OUT/survey/vet/graded.jsonl (survey_tables.py), gate2/survey_targets.csv (TIC Teff, distance),
vetting/star_info.json (Gaia neighbours), the SAP files' CROWDSAP, and OUT/survey/vet/research/.
Writes OUT/survey/vet/dossier.json and prints a text block per key.

Usage: python -I survey/dossier.py KEY [KEY ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "vetting"))
sys.path.insert(0, str(HERE))

from common import OUT
from physics import implied, next_transits, tess_sectors, timing_sigma

NOW_BTJD = 4356.5  # 2026-11-12
LAST_DATA_SECTOR = 107  # latest sector in the survey's downloads


def research_summary(tic: int) -> dict:
    fp = OUT / "survey" / "vet" / "research" / f"{tic}.json"
    if not fp.exists():
        return {"status": "not run"}
    d = json.loads(fp.read_text())
    out = {k: d[k]["status"] for k in ("exofop", "nasa", "simbad", "gaia", "gaia_nss", "prsa_eb")
           if k in d}
    if d.get("exofop", {}).get("status") == "ok":
        e = d["exofop"]["data"]
        out["exofop_tois"] = [(t.get("toi"), t.get("period"), t.get("disposition") or
                               t.get("tfopwg_disp")) for t in (e.get("tois") or [])]
        out["exofop_ctois"] = [(t.get("ctoi"), t.get("period"), t.get("user_disposition"))
                               for t in (e.get("ctois") or [])]
        out["exofop_n"] = {k: len(e.get(k) or []) for k in ("imaging", "spectroscopy",
                                                            "time_series", "stellar_companions")}
        out["exofop_companions"] = e.get("stellar_companions")
    if d.get("nasa", {}).get("status") == "ok":
        out["nasa_planets"] = [(r["pl_name"], r["pl_orbper"]) for r in d["nasa"]["data"]]
    if d.get("simbad", {}).get("status") == "ok":
        s = d["simbad"]["data"]
        out["simbad"] = [(o["main_id"], o["otype"], o["nbref"], round(o["sep"], 1))
                         for o in s["objects"]]
        out["simbad_ids"] = s["target_ids"][:25]
    if d.get("gaia", {}).get("status") == "ok":
        rows = d["gaia"]["data"]["rows"]
        out["gaia_rows"] = [{k: r.get(k) for k in ("Source", "_r", "Gmag", "BP-RP", "Plx", "RUWE",
                                                   "NSS", "RV", "e_RV", "RVamp", "VarFlag",
                                                   "IPDfmp", "IPDgofha")} for r in rows]
    if d.get("gaia_nss", {}).get("status") == "ok":
        out["gaia_nss_rows"] = d["gaia_nss"]["data"]["rows"]
    if d.get("prsa_eb", {}).get("status") == "ok":
        out["prsa_eb_rows"] = d["prsa_eb"]["data"]["rows"]
    return out


def text(o: dict) -> str:
    im = o["implied"]
    px = o.get("pixel") or {}
    rs = o["research"]
    L = [(f"{o['key']} tier {o['tier']}: P {o['P']:.4f} d, t0 {o['t0']:.3f}, MES {o['mes']:.2f}, depth "
         f"{o['depth_ppt']:.2f} ppt, dur {o['dur_h']:.1f} h (circular central {o['dc_h']:.2f} h, ratio "
         f"{o['dur_ratio']:.2f}), n_tr {o['n_tr']}, {o['n_sectors']} sectors, dom {o['dom']:.2f}, "
         f"odd/even z {o['odd_even_z']}"),
         (f"  star: Teff {o['teff']:.0f} K, R {o['rstar']:.3f}, M {o['mstar']:.3f}, d {o['d_pc']:.1f} pc, "
         f"Tmag {o['tmag']:.2f}; Gaia target {o['gaia_target']}"),
         (f"  implied: Rp {im['rp_re']:.2f} Re, a {im['a_au']:.4f} AU, S {im['s_earth']:.2f} S_e, Teq "
         f"{im['teq_a03_k']:.0f} K (A=0.3) / {im['teq_a0_k']:.0f} K (A=0); {im['zone']}; HZ S runaway "
         f"{im['hz']['runaway_greenhouse']:.2f} .. max-GH {im['hz']['maximum_greenhouse']:.2f}"),
         "  epochs: " + "; ".join(f"{e['t']} S{e['sector']} SES {e['ses']} flags {e['flags']} gap "
                                  f"{e['gap_dist_h']} h crowd {e['crowdsap']}" for e in o["epochs"]),
         f"  next (BTJD, +-1sig h): {o['next']}",
         f"  future sectors after S{LAST_DATA_SECTOR}: {o['future_sectors']}",
         (f"  pixel: {px.get('status')} {px.get('why')} n_strong {px.get('n_strong')} S_t "
         f"{px.get('S_t')} S_nb {px.get('S_nb')} nb_sep {px.get('nb_sep')} nb_dG {px.get('nb_dG')}"),
         f"  reasons: {o['reasons']}; info: {o['info']}",
         "  neighbours <= 63\": " + "; ".join(f"G{n['G']:.2f} dG {n['dG']:.2f} at {n['sep']:.1f}\""
                                             for n in o["neighbours_63"][:8]),
         f"  research status: { {k: rs.get(k) for k in ('exofop', 'nasa', 'simbad', 'gaia', 'gaia_nss', 'prsa_eb')} }",
         (f"  exofop: TOIs {rs.get('exofop_tois')} CTOIs {rs.get('exofop_ctois')} counts "
         f"{rs.get('exofop_n')}"),
         f"  nasa: {rs.get('nasa_planets')}",
         f"  simbad: {rs.get('simbad')}",
         f"  simbad ids: {rs.get('simbad_ids')}"]
    for g in sorted(rs.get("gaia_rows") or [], key=lambda r: r["_r"])[:5]:
        L.append(f"  gaia {g['Source']} r {g['_r']:.2f}\" G {g['Gmag']} BP-RP {g['BP-RP']} plx "
                 f"{g['Plx']} RUWE {g['RUWE']} NSS {g['NSS']} RV {g['RV']} eRV {g['e_RV']} RVamp "
                 f"{g['RVamp']} Var {g['VarFlag']} IPDfmp {g['IPDfmp']}")
    L.append(f"  gaia NSS orbits: {len(rs.get('gaia_nss_rows') or [])}; Prsa EB rows: "
             f"{len(rs.get('prsa_eb_rows') or [])}")
    return "\n".join(L)


def main():
    keys = set(sys.argv[1:])
    rows = [json.loads(x) for x in (OUT / "survey" / "vet" / "graded.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["key"] in keys]
    tg = pd.read_csv(OUT / "gate2" / "survey_targets.csv").set_index("tic")
    info = json.loads((OUT / "vetting" / "star_info.json").read_text())
    res = {}
    for r in rows:
        s, tic = r["stats"], r["tic"]
        t = tg.loc[tic]
        teff = float(t["Teff"]) if np.isfinite(t["Teff"]) else 3100.0
        st = info.get(str(tic), {})
        g = (st.get("gaia") or {})
        ra, dec = (g["target"]["ra"], g["target"]["dec"]) if "target" in g else (t["ra"], t["dec"])
        z = np.load(OUT / "vetting" / "sap" / f"tic{tic}.npz", allow_pickle=False)
        crowd = {int(m["sector"]): m.get("crowdsap") for m in json.loads(str(z["meta"]))["sectors"]}
        imp = implied(s["P"], s["depth"], r["mass"], r["radius"], teff)
        nxt = next_transits(s["t0"], s["P"], NOW_BTJD, 3)
        sig = [timing_sigma(s["P"], s["dur_h"] / 24, s["epochs"], x) for x in nxt]
        future = tess_sectors(float(ra), float(dec), after_sector=LAST_DATA_SECTOR)
        fut_tr = []
        for f in future:
            if f["start"] is None:
                continue
            k0 = int(np.ceil((f["start"] - s["t0"]) / s["P"]))
            k1 = int(np.floor((f["end"] - s["t0"]) / s["P"]))
            fut_tr.append((f["sector"], round(f["start"], 1), round(f["end"], 1),
                           [round(s["t0"] + k * s["P"], 2) for k in range(k0, k1 + 1)]))
        flags = (s.get("systematics_window") or {}).get("flags") or [[]] * len(s["epochs"])
        gapd = ((s.get("v2") or {}).get("gap_edge") or {}).get("dist_h") or [None] * len(s["epochs"])
        ep = [{"t": round(e, 3), "sector": int(sec), "ses": round(float(v), 2),
               "flags": fl, "gap_dist_h": None if gd is None else round(gd, 1),
               "crowdsap": crowd.get(int(sec))}
              for e, sec, v, fl, gd in zip(s["epochs"], s["epoch_sector"], s["epoch_ses"], flags,
                                           gapd)]
        nb = [n for n in g.get("neighbours", []) if n["sep"] <= 63]
        out = {"key": r["key"], "tic": tic, "tier": r["grade"]["tier"], "P": s["P"],
               "t0": s["t0"], "mes": s["mes"], "depth_ppt": 1e3 * s["depth"], "dur_h": s["dur_h"],
               "n_tr": s["n_transits"], "n_sectors": r["n_sectors"], "teff": teff,
               "rstar": r["radius"], "mstar": r["mass"], "d_pc": float(t["d_pc"]),
               "tmag": float(t["Tmag"]), "ra": float(ra), "dec": float(dec), "implied": imp,
               "epochs": ep, "next": [(round(x, 3), round(24 * y, 1)) for x, y in zip(nxt, sig)],
               "future_sectors": fut_tr, "neighbours_63": nb,
               "gaia_target": g.get("target"), "reasons": r["grade"]["reasons"],
               "status": r["grade"]["status"], "info": r["grade"].get("info"),
               "dc_h": (s.get("duration_density") or {}).get("dc_h"),
               "dur_ratio": (s.get("duration_density") or {}).get("ratio"),
               "odd_even_z": (s.get("odd_even") or {}).get("z"),
               "dom": (s.get("per_transit") or {}).get("dom"),
               "pixel": s.get("pixel"), "research": research_summary(tic)}
        res[r["key"]] = out
        print(text(out))
        print("-" * 80)
    fp = OUT / "survey" / "vet" / "dossier.json"
    old = json.loads(fp.read_text()) if fp.exists() else {}
    old.update(res)
    fp.write_text(json.dumps(old, indent=1, default=float))


if __name__ == "__main__":
    main()
