"""Per-star vetting dossier for a candidate gallery (text next to each card image).

    uv run python -I scripts/vet_dossier.py --gallery cand_v1_n150

For every card: per-sector ranks, cmv3 events and EB-test features (``ebf_v1``), a Lomb-Scargle
period and amplitude of PDCSAP per sector, robust PDCSAP/SAP amplitudes, SkyBoT, the catalogue
research (``oddball.gallery.research``: Gaia DR3 incl. RUWE and CMD height, AllWISE excess, VSX
within 120", Gaia variability, Kostov+2025, young-star lists), the card's SIMBAD/TOI cross-match
and the pixel check (``pixels.csv``).

Writes ``<gallery>/research.csv`` (one row per star), ``<gallery>/dossier.csv`` (everything,
flat) and ``<gallery>/dossier.md`` (one block per card, in card order).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

import numpy as np
import pandas as pd

from oddball.common.paths import outputs_root
from oddball.gallery import research as R
from oddball.tess.store import load_lightcurves, metadata

EB_COLS = ["eb_P", "eb_n_events", "eb_mean_depth", "eb_is_periodic", "eb_sec_ok", "eb_sec_depth",
           "eb_asym", "eb_fwhm", "eb_depth_fex", "eb_n_extra"]  # fmt: skip


def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def ls_period(t: np.ndarray, f: np.ndarray) -> dict:
    from astropy.timeseries import LombScargle

    m = np.isfinite(t) & np.isfinite(f)
    if m.sum() < 100:
        return {}
    t, f = t[m], f[m] / np.nanmedian(f[m]) - 1.0
    freq = np.linspace(1 / 13.0, 1 / 0.05, 20000)
    p = LombScargle(t, f).power(freq)
    k = int(np.argmax(p))
    per = 1 / freq[k]
    ph = (t / per) % 1.0
    # semi-amplitude of the best sinusoid
    X = np.column_stack([np.ones_like(t), np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)])
    b, *_ = np.linalg.lstsq(X, f, rcond=None)
    return {"ls_P": float(per), "ls_power": float(p[k]), "ls_amp_ppt": 1e3 * float(np.hypot(*b[1:]))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gallery", required=True)
    a = ap.parse_args()
    gdir = outputs_root() / "gallery" / a.gallery
    man = json.loads((gdir / "manifest.json").read_text())
    cards = man["cards"]
    cand = pd.read_parquet(outputs_root() / "candidates" / "v1" / "candidates.parquet")
    cand = cand.set_index("tic")
    stars = pd.DataFrame([{"tic": c["tic"], "ra": c["ra"], "dec": c["dec"], "tmag": c["tmag"]}
                          for c in cards])  # fmt: skip
    log(f"research for {len(stars)} stars")
    res = R.research(stars, log=log).set_index("tic")
    res.to_csv(gdir / "research.csv")
    pix = pd.read_csv(gdir / "pixels.csv") if (gdir / "pixels.csv").exists() else pd.DataFrame()
    # per-sector light-curve stats and EB features
    per = {}
    sectors = sorted({p["sector"] for c in cards for p in c["candidate"]["per_sector"]})
    for s in sectors:
        tics = [c["tic"] for c in cards if any(p["sector"] == s for p in c["candidate"]["per_sector"])]
        lcs = load_lightcurves(s, tics)
        meta = metadata(s).set_index("tic")
        ebf = pd.read_parquet(outputs_root() / "knownclass" / "features" / "ebf_v1" /
                              f"s{s:04d}" / "real.parquet", columns=["tic"] + EB_COLS)  # fmt: skip
        ebf = ebf[ebf.tic.isin(tics)].set_index("tic")
        for tic in tics:
            d = {}
            lc = lcs.get(tic)
            if lc is not None:
                t = np.asarray(lc["time"], float)
                f = np.asarray(lc["flux"], float)
                sp = np.asarray(lc["sap_flux"], float)
                fn = f / np.nanmedian(f)
                sn = sp / np.nanmedian(sp)
                d.update(ls_period(t, f))
                d["ptp_pdc_pct"] = 100 * float(np.nanpercentile(fn, 99) - np.nanpercentile(fn, 1))
                d["ptp_sap_pct"] = 100 * float(np.nanpercentile(sn, 99) - np.nanpercentile(sn, 1))
                d["min_pdc_pct"] = 100 * float(1 - np.nanpercentile(fn, 0.5))
            if tic in meta.index:
                d["crowdsap"] = float(meta.loc[tic, "crowdsap"])
            if tic in ebf.index:
                d.update({k: ebf.loc[tic, k] for k in EB_COLS})
            per[(tic, s)] = d
    rows, blocks = [], [(f"# Dossier: {a.gallery} ({len(cards)} cards), built "
                        f"{dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M} UTC\n")]  # fmt: skip
    for i, c in enumerate(cards):
        tic = c["tic"]
        cr = cand.loc[tic]
        rr = res.loc[tic].to_dict() if tic in res.index else {}
        xm = c.get("xmatch") or {}
        sb = (xm.get("simbad") or {})
        sb_id = (sb.get("by_tic_id") or {})
        sb_near = "; ".join(f"{n['main_id']} [{n['otype']}] {n['sep_arcsec']:.0f}\""
                            for n in (sb.get("nearby") or [])[:4])  # fmt: skip
        toi = xm.get("toi") or {}
        pr = pix[(pix.tic == tic) & (pix.sector == c["sector"])] if len(pix) else pix
        pv = pr.iloc[0].to_dict() if len(pr) else {}
        row = {"order": i + 1, "tic": tic, "card": c["png"], "pix_card": pv.get("png"),
               "best_sector": c["sector"], "pick": c["pick"], "tmag": c["tmag"], "ra": c["ra"],
               "dec": c["dec"], "sectors_obs": cr.sectors_obs, "sectors_cand": cr.sectors_cand,
               "skybot_checked": bool(cr.skybot_checked), "skybot_explained": bool(cr.skybot_explained),
               "simbad_id": sb_id.get("main_id", ""), "simbad_otype": sb_id.get("otype", ""),
               "simbad_near": sb_near, "toi": json.dumps(toi.get("toi") or []),
               "ctoi": json.dumps(toi.get("ctoi") or []),
               "xmatch_verdict": (xm.get("verdict") or {}).get("text", ""),
               "pix_verdict": pv.get("verdict", "")}  # fmt: skip
        row.update({f"r_{k}": v for k, v in rr.items()})
        row.update({k: pv.get(k) for k in ("diff_peak_snr", "diff_offset_px", "diff_neg_peak_snr",
                                            "diff_neg_offset_px", "locus_peak_snr",
                                            "locus_offset_px")})  # fmt: skip
        lines = [(f"## {i + 1}. TIC {tic}  ({c['pick']}; best S{c['sector']} rank {c['rank']}; "
                 f"Tmag {c['tmag']:.2f}; RA {c['ra']:.4f} Dec {c['dec']:.4f})"),
                 f"- cards: `{c['png']}`, pixels `{pv.get('png')}`",
                 (f"- sectors observed {cr.sectors_obs}; candidate in {cr.sectors_cand}; "
                 f"SkyBoT checked {bool(cr.skybot_checked)}, explained {bool(cr.skybot_explained)}")]  # fmt: skip
        for p in c["candidate"]["per_sector"]:
            s = p["sector"]
            d = per.get((tic, s), {})
            row.update({f"s{s}_{k}": v for k, v in d.items()})
            row[f"s{s}_ranks"] = f"cls {p['cls_rank']} / ens {p['ens_rank']}"
            rt = p["res_t"] if p["res_t"] is not None else np.nan
            row[f"s{s}_event"] = f"{p['res_event']} @ {rt:.2f}"
            eb = (f"EB-test: P={d.get('eb_P', np.nan):.4g} n_ev={d.get('eb_n_events')} "
                  f"depth={d.get('eb_mean_depth', np.nan):.4g} periodic={d.get('eb_is_periodic')} "
                  f"sec={d.get('eb_sec_ok')} asym={d.get('eb_asym', np.nan):.2g}")  # fmt: skip
            lines.append(
                f"- S{s}: ranks cls {p['cls_rank']} / ens {p['ens_rank']} of {p['n_stars']}"
                f"{' CANDIDATE' if p['candidate'] else ''}{' cmv3-flagged' if p['cmv3_flag'] else ''}; "
                f"event {p['res_event']} @ BTJD {rt:.2f}; eb_like {p['eb_like'] or np.nan:.3f}; "
                f"LS P={d.get('ls_P', np.nan):.4g} d amp={d.get('ls_amp_ppt', np.nan):.3g} ppt "
                f"pow={d.get('ls_power', np.nan):.2f}; PDC 1-99% range {d.get('ptp_pdc_pct', np.nan):.2f}%, "
                f"SAP {d.get('ptp_sap_pct', np.nan):.2f}%, deepest(0.5%) {d.get('min_pdc_pct', np.nan):.2f}%; "
                f"CROWDSAP {d.get('crowdsap', np.nan):.2f}; {eb}")  # fmt: skip
        g = rr
        lines.append(
            f"- Gaia: match {g.get('gaia_match')} G={g.get('G', np.nan):.2f} BP-RP={g.get('bp_rp', np.nan):.2f} "
            f"plx={g.get('plx', np.nan):.3f} (S/N {g.get('plx_snr', np.nan):.0f}) d={g.get('dist_pc', np.nan):.0f} pc "
            f"M_G={g.get('abs_G', np.nan):.2f} above-MS={g.get('ms_height', np.nan):+.2f} mag "
            f"RUWE={g.get('ruwe', np.nan):.2f} VarFlag={g.get('gaia_varflag')}")  # fmt: skip
        lines.append(
            f"- Neighbours (Gaia, dG<5): {g.get('n_nb_1px')} within 1 px, {g.get('n_nb_2px')} within 2 px, "
            f"{g.get('n_nb_3px')} within 3 px; brightest within 2 px dG={g.get('brightest_nb_2px_dG', np.nan):+.2f} "
            f"at {g.get('brightest_nb_2px_sep', np.nan):.0f}\" (VarFlag {g.get('brightest_nb_2px_varflag')}); "
            f"summed neighbour/target flux within 2 px {g.get('flux_ratio_nb_2px', np.nan):.2f}")  # fmt: skip
        lines.append(
            f"- AllWISE: match {g.get('wise_match')} W1-W2={g.get('W1_W2', np.nan):.2f} "
            f"W2-W3={g.get('W2_W3', np.nan):.2f} W1-W4={g.get('W1_W4', np.nan):.2f} "
            f"excess=[{g.get('ir_excess', '')}] ccf={g.get('wise_ccf')} ex={g.get('wise_ex')} "
            f"confused={g.get('wise_confused')}")  # fmt: skip
        lines.append(f"- VSX: on target [{g.get('vsx_on_target', '')}]; within 2 px "
                     f"[{g.get('vsx_within_2px', '')}]; next beyond [{g.get('vsx_nearest_beyond_2px', '')}] "
                     f"(total within 120\": {g.get('vsx_within_120')})")  # fmt: skip
        lines.append(f"- Gaia var: veb [{g.get('gaia_veb', '')}] class [{g.get('gaia_vclass', '')}]; "
                     f"Kostov+25 [{g.get('kostov', '')}]; young lists [{g.get('yso_lists', '')}]")  # fmt: skip
        lines.append(f"- SIMBAD: by TIC [{sb_id.get('main_id', '')} {sb_id.get('otype', '')}]; "
                     f"within 21\" [{sb_near}]; TOI {toi.get('toi') or []} CTOI {toi.get('ctoi') or []}; "
                     f"card verdict: {row['xmatch_verdict']}")  # fmt: skip
        lines.append(f"- Pixels: {pv.get('verdict', 'not run')}")
        blocks.append("\n".join(lines) + "\n")
        rows.append(row)
    pd.DataFrame(rows).to_csv(gdir / "dossier.csv", index=False)
    (gdir / "dossier.md").write_text("\n".join(blocks))
    log(f"wrote {gdir / 'dossier.md'} and dossier.csv ({len(rows)} stars)")


if __name__ == "__main__":
    main()
