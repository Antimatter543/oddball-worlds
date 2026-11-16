"""Per-star labels for the cool-objects pass (labels/cool_objects_v1.csv): every one of the 1,256 stars with a
long-range signal (any grade) or a short-range signal at or above tau_short.

  depth = "full"      eyeballed light curve + period work, and (for kept objects) a pixel check and a catalogue look
          "lc_look"   ebscan light-curve fold exists but was not eyeballed in detail: automatic label only
          "catalogue" catalogue cross-match only
  source = "eyeball" (MANUAL below, from the contact sheets / per-star plots / pixel checks of 2026-11-16)
           or "auto" (rules on the catalogue summary and the first-pass triage class)

MANUAL entries: tic -> (label, detail). Labels:
  eb_new_candidate      eclipsing pair, on target, not found as an eclipsing binary in the catalogues checked
  eb_partly_known       eclipsing on target; a catalogue lists the star as variable/binary but not this eclipse
  eb_known              eclipsing binary already catalogued (VSX / Prsa+2022 / Kostov+2025 / Gaia DR3 / SIMBAD / TOI FP)
  eclipser_few_events   2-3 deep eclipses on target, period not pinned down
  single_event          one deep dip on target (pixel check passed)
  blend_neighbour       the dips come from a neighbouring star (pixel check failed or a catalogued neighbour EB at
                        the same period)
  off_target_or_unclear dip not localised on the target (pixel check soft / no signal on target pixels)
  young_dipper_cpv      young-star dips: complex periodic variable or dipper
  transit_like          shallow periodic dip, no secondary, not a TOI: needs follow-up before calling it anything
  odd                   real but not explained
  unclear               seen by eye, not resolved (no pixel check); not counted as a real object
  rotator_flare         spots / flares only (not an object of interest here)
  artefact              systematics, data edges, crowding
  known_planet_or_toi   TOI / CTOI / confirmed planet host
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import OUT

C = OUT / "cool"
REPO = HERE.parents[2]

MANUAL = {
    # --- new-looking eclipsers (pixel check passed) ---
    308186412: ("eclipser_few_events", "3 eclipses 8-15% deep, T14~5 h, BTJD 1579.03/2094.95/2352.91; P=257.96 d if all are the "
                "same eclipse (515.9/773.9 d if one is a secondary); pixel pass (S10/S29/S38); 6.5\" neighbour dG=0.96 "
                "unresolved and co-moving (same parallax 11.77 mas and proper motion), so the eclipses may belong to "
                "either star; SIMBAD Y*? (young-association candidate: BANYAN XIII, SPYGLASS, Laos+2022); "
                "no Gaia EB/orbit/class"),
    238891168: ("eb_partly_known", "SCR J0241-7719: 2 eclipses 27%/41% deep at the lowest point, ~4 h, BTJD 1673.12/3172.44; "
                "pixel pass. Gaia DR3 has an astrometric orbit (I/357/tbooc): P=152.36+-0.45 d, e=0.56, incl 87.4+-2.7 deg. "
                "The two eclipses fall at the orbit's two different conjunctions (predicted spacing 126.6 d, 68% "
                "120-133; needed 128.1+-4.1), so they are primary + secondary; TESS coverage then allows only "
                "P=152.735-152.823 d (cool/gaiaorbit.py). Eclipses not catalogued anywhere checked"),
    344769778: ("eb_partly_known", "P=18.58115 d eccentric: primary 25%, secondary 4.4% at phase 0.218 (T14 1.8/2.7 h); "
                "pixel pass; ATLAS (Heinze+2018) lists it as a 'dubious' variable with P=18.1888 d; VSX lists a "
                "different Gaia source nearby as type S (no period)"),
    435680606: ("eb_new_candidate", "P=3.00782 d, 40%/34% (near-twin M+M); pixel pass; LP 13-133"),
    261590581: ("eb_partly_known", "P=4.45725 d, 20%/14%; pixel pass; LEHPM 5710: Gaia DR3 variable (VSX VAR, P=0.0196 d "
                "is wrong) but no eclipse period catalogued"),
    468353260: ("eb_new_candidate", "P=5.87063 d, 27%/15%; pixel pass"),
    10747609: ("eb_new_candidate", "P=5.98383 d, 15%/13%, secondary at phase 0.482 (slightly eccentric); pixel pass but a "
               "co-moving star 2.4\" away (G 14.6, same parallax 12.9 mas) is unresolved, so the eclipses may belong to it"),
    316822151: ("eb_new_candidate", "P=5.02091 d eccentric: primary 20%, secondary 4.7% at phase 0.921; pixel pass; a "
                "fainter co-moving star 1.8\" away (G 17.3, same parallax) is unresolved"),
    94917984: ("eb_partly_known", "P=1.03034 d, 6%/5%; pixel pass; known double-lined spectroscopic binary (APOGEE, "
               "Kounkel+2021) but no eclipses catalogued"),
    162689162: ("eb_new_candidate", "P=21.02706 d eccentric: 21%/22%, secondary at phase 0.398, T14 2.4 h; pixel pass; "
                "no neighbour within 12\""),
    147304495: ("eb_new_candidate", "P=13.14687 d, 16%/9.5%, T14 5.0 h (long for a 0.24 R_sun star: the eclipsing star "
                "may be larger than the TIC radius says); pixel pass; 4\" neighbour G 20.7 (too faint to matter)"),
    7322727: ("eb_new_candidate", "P=1.90517 d, 4.5%/2.0%, T14 2.4 h (long for a 0.20 R_sun star); pixel pass; shallow, "
              "so possibly a diluted or grazing pair"),
    233466569: ("eb_new_candidate", "P=0.51392 d, 5.9%/2.6%; pixel pass, but the co-moving companion LSPM J1524+3942S "
                "6.6\" away (same parallax 14.2 mas and motion) is unresolved and also lit up in the pixel check: the "
                "eclipses may belong to either star"),
    446169146: ("eb_new_candidate", "LP 7-265: a ~3% V-shaped dip every 0.40488 d, phase-coherent over 13 sectors (BTJD 1683-3479), "
                "odd/even equal (2.93%/2.89%), no other dip: a grazing pair with equal eclipses and P=0.80976 d, or "
                "P=0.40488 d with an unseen second eclipse; on a 5-7% spot wave; pixel pass (8 dips in S41, SNR 12.8 "
                "vs -1.0); no Gaia star within 12\"; found late on sheet r"),
    # 308186412 / 238891168 extra notes (Gaia DR3 via VizieR mirrors): see the entries above
    # --- off target / unclear (pixel check soft) ---
    181110184: ("off_target_or_unclear", "P=10.7593 d, 2.3%/2.4%, T14 7.5 h; difference centroid > 1 px off in 5/5 epochs"),
    182659260: ("off_target_or_unclear", "P=4.91034 d, 6.1%; difference centroid > 1 px off in 9/9 epochs"),
    281928832: ("off_target_or_unclear", "P=10.18288 d, 2.0% transit-like with a 0.7% secondary; centroid off in 5/10 epochs"),
    290270132: ("off_target_or_unclear", "P=3.72381 d, 1.7%; centroid off in 11/11 epochs; co-moving twin 3.2\" away "
                "(same parallax 13.5 mas) also unresolved"),
    8990565: ("off_target_or_unclear", "one dip BTJD 2110.11; difference centroid 5 px off"),
    74998095: ("off_target_or_unclear", "one dip BTJD 2548.67; difference centroid 2.4 px off"),
    75728727: ("off_target_or_unclear", "two dips BTJD 2398.01, 2412.51; centroid 5-6 px off"),
    # --- more blends (pixel check fail) ---
    445589439: ("blend_neighbour", "P=13.746 d, 8%; dip stronger on a star 40\" away"),
    4607424: ("blend_neighbour", "one dip BTJD 2957.55, stronger on a star 56\" away"),
    366443401: ("blend_neighbour", "one dip BTJD 2974.92, much stronger on a star 56\" away"),
    195383541: ("blend_neighbour", "6.40 d variable-depth dips, stronger on a star 38\" away"),
    400211921: ("blend_neighbour", "P=1.34702 d, 1.7%; stronger on a star 38\" away"),
    388436472: ("blend_neighbour", "P=1.54913 d, 2.4%/0.9%; stronger on a star 38\" away"),
    # --- young stars: complex periodic variables (CPVs) ---
    2234692: ("young_dipper_cpv", "CPV, spin 0.5433 d; known (Bouma+2024, via SIMBAD references)"),
    425933644: ("young_dipper_cpv", "UPM J0014-6003, CPV; known (Zhan+2019, Bouma+2024)"),
    234295610: ("young_dipper_cpv", "CPV; known (Zhan+2019, Guenther+2022, Bouma+2024)"),
    402980664: ("young_dipper_cpv", "LP 12-502, CPV; known (Bouma+2024)"),
    280945693: ("young_dipper_cpv", "UCAC4 103-046417, CPV; known (Bouma+2024)"),
    22739419: ("young_dipper_cpv", "new-looking CPV: spin 0.351275 d (8.4 h), sharp dips that change between S18, S44 "
               "and S58; pixel pass (weak: SNR 19.6 vs 10.1 on a neighbour; 5.2\" background star G 15.8 unresolved); "
               "not in Capistrant+2022, no Bouma+2024/Zhan+2019 reference in SIMBAD; 94.9 pc"),
    220599904: ("young_dipper_cpv", "new-looking CPV: spin 0.62852 d (15.1 h); two sharp dips (2.5% and 5%) per spin in "
                "S31-33 (Oct-Dec 2020) only, smooth spot wave in S5/S6 (2018) and S87 (2024); pixel pass (S32 dips, "
                "SNR 27 vs 4.5); 53.2 pc; SIMBAD lists BANYAN young-group papers but no CPV paper"),
    394664599: ("young_dipper_cpv", "new-looking CPV: spin 1.36323 d; sharp dips to 15% at two phases in S38 (Apr 2021), "
                "one ~6% dip in S65 (May 2023); pixel pass (SNR 66 vs 5.6); 95.4 pc; SIMBAD Y*? with emission-line "
                "references but no CPV paper"),
    178155030: ("rotator_flare", "spin 0.48649 d, 5% spot wave whose shape changes a lot between sectors, a small sharp "
                "notch in S106: at most a weak CPV candidate; GALEX-young (SIMBAD)"),
    410421717: ("rotator_flare", "spin 0.9854 d, 4% wave with small sharp notches in S94/S95: at most a weak CPV candidate"),
    264599508: ("rotator_flare", "two spin periods (0.329 and 0.375 d) in different sectors: probably two stars"),
    # --- other seen, not followed further ---
    457801578: ("odd", "one sector: ~15% dips ~12 h long; VSX (Gaia) calls it DSCT, which an M dwarf cannot be; "
                "no pixel check"),
    233684011: ("blend_suspect_neighbour_eb", "3.64 d dips whose depth changes between sectors (CVZ star): "
                "suspected blend, not pixel-checked"),
    132529464: ("transit_like", "2.15 d, ~1% deep but long; not pixel-checked"),
    # --- deepest folds among the light-curve-only stars, eyeballed late (sheets p-r) ---
    350296414: ("artefact", "ramp at a sector start (fold depth > 100%)"),
    373431017: ("artefact", "+-2x swings: PDC crowding correction on a faint star"),
    59525277: ("artefact", "deep drops at sector edges with ramps; ZTF EW variable 30-60\" away"),
    198312554: ("artefact", "two 5% drops in one sector next to a gap"),
    458587958: ("artefact", "6% dip next to a 5% spike: glitch"),
    68265933: ("artefact", "messy drops in one sector"),
    392065283: ("artefact", "ramp + single low points"),
    360130454: ("rotator_flare", "spot wave"),
    458358301: ("rotator_flare", "3.8 d spot wave; the 'dips' are wave troughs and S44 glitches (pixel check on target)"),
    231712880: ("off_target_or_unclear", "7% sharp-ingress dips BTJD 1349.11 (S1) and 3162.09 (S68); centroid 1.6 px off"),
    172842767: ("off_target_or_unclear", "5.5% dips BTJD 2550.05, 2587.05; centroid 2.5-4.6 px off"),
    249744864: ("off_target_or_unclear", "4% dips BTJD 1755.04, 1774.52; centroid 1.3 px off"),
    166180049: ("off_target_or_unclear", "one 6.5%, ~8 h V-shaped dip BTJD 2635.56; pixel check inconclusive (6.1 on target vs "
                "5.8 on a neighbour)"),
    285903673: ("unclear", "7% V-shaped dips in S79 (BTJD 3457.13, 3474.73, 3479.59); pixel check not done (TESScut HTTP 502)"),
    455055150: ("unclear", "two ~3% dips, 20.26 d apart; noisy; not pixel-checked"),
    185247277: ("unclear", "several 1-3% dips in S72 with slow shapes; not pixel-checked"),
    302607052: ("unclear", "9.86 d ~2% dip, noisy; not pixel-checked"),
    # --- first-pass "eb_like" stars not looked at before (sheets s-t) ---
    98540149: ("off_target_or_unclear", "P=0.95685 d, 2.5%/1.5% V-shaped; centroid off in 3/5 epochs; a G 13.6 star 4.9\" "
               "away and a VSX variable nearby"),
    144325909: ("off_target_or_unclear", "P=3.85358 d, 1.4% transit-like, no secondary; difference centroid 2 px off in 4/4 "
                "epochs and the peak far brighter elsewhere: light from another star"),
    441804457: ("artefact", "no clear signal in the fold"),
    73308879: ("artefact", "no clear signal in the fold"),
    235558447: ("artefact", "drop at a data edge"),
    229479346: ("unclear", "4.92 d, 0.35% dip; not pixel-checked"),
    434171112: ("unclear", "4.49 d, 0.6% dip, 2 sectors, noisy; not pixel-checked"),
    # --- deeper uncatalogued short-range signals not folded before (sheet u) ---
    294328887: ("young_dipper_cpv", "UCAC4 154-009819: CPV, spin 0.35449 d, sharp evolving dips to 10% (S98), 42 sectors; "
                "known (Bouma+2024 via SIMBAD; also in an EB list, Justesen+2021)"),
    78537191: ("off_target_or_unclear", "P=4.58681 d, 2-8% dips; difference centroid 5 px off in 3/3 epochs"),
    166760940: ("unclear", "3.10 d, ~1% weak; not pixel-checked"),
    239366978: ("unclear", "0.516 d, one noisy sector; not pixel-checked"),
    389214311: ("artefact", "noise"),
    39970966: ("artefact", "flares and ramps"),
    # --- isolated planet-sized short-range signals (batch 2, sheets n-o): planet-vetting territory, not followed here ---
    152364094: ("transit_like", "2.152 d, 0.35% deep, no secondary; not a TOI; not pixel-checked"),
    137609439: ("transit_like", "4.294 d, 0.2% deep; not a TOI; not pixel-checked"),
    348905376: ("unclear", "1.004 d, 0.44% broad V-shaped dip: rotation or a blend; not pixel-checked"),
    355303117: ("artefact", "noise"),
    232623462: ("artefact", "noise"),
    71561595: ("rotator_flare", "0.710 d spot wave"),
    336741253: ("rotator_flare", "1.18 d spot wave that changes between sectors"),
    # --- known re-found ---
    142588954: ("eb_known", "LSPM J1112+7626, P=41.0326 d eccentric (VSX)"),
    199574208: ("eb_known", "CM Dra, P=1.2684 d"),
    427240420: ("eb_known", "CzeV390, P=1.3884 d"),
    358908226: ("eb_known", "MNIC V134, P=1.0006 d"),
    142979644: ("eb_known", "NEV364, P=4.1790 d"),
    950715112: ("blend_neighbour", "sees NEV364 (TIC 142979644) next door, P=4.1790 d"),
    392651143: ("eb_known", "NV Lyr, P=1.5414 d"),
    408651394: ("eb_known", "CCDM J04404+3127B, P=2.0481 d"),
    471012590: ("eb_known", "LP 133-373, P=1.6280 d"),
    22131853: ("eb_known", "HAT-196-0006238, P=1.7583 d"),
    156067195: ("eb_known", "HAT-148-0000574, VSX P=3.5502 d (TESS fold at half)"),
    172517058: ("eb_known", "Prsa+2022, P=2.4365 d twin"),
    166090445: ("eb_known", "Prsa+2022, P=5.3246 d"),
    91542943: ("eb_known", "Prsa+2022, P=5.3267 d"),
    160492103: ("eb_known", "Kostov+2025, P=6.2839 d"),
    230235519: ("eb_known", "Kostov+2025, P=9.6920 d eccentric (secondary at phase 0.389)"),
    272086159: ("eb_known", "TOI-176.01 = TFOP false positive (EB), P=16.1554 d eccentric, secondary 0.6% at phase 0.718"),
    461579405: ("blend_neighbour", "Prsa+2022 lists an EB (P=13.918 d) under this TIC, but the pixel check puts the "
                "dips on a star 36\" away"),
    # --- blends ---
    182469311: ("blend_neighbour", "3.015 d dips sit on a star 41\" away (pixel fail); Prsa+2022 lists the blend under this TIC"),
    334672710: ("blend_neighbour", "two dips (BTJD 2346.0, 3075.3) sit on a star 38\" away (pixel fail)"),
    159505078: ("blend_neighbour", "28.856 d dips sit on a star 37\" away (pixel fail)"),
    189084771: ("blend_neighbour", "15.03 d dips sit on a star 46\" away (pixel fail); Prsa+2022 lists it under this TIC"),
    294768511: ("blend_neighbour", "0.88482 d = VSX EB Gaia DR3 5809727395730297856 32\" away"),
    158329689: ("blend_neighbour", "1.5336 d = CF Ant (VSX EA) 60\" away"),
    284783117: ("blend_neighbour", "depth >100% in PDC; VSX Gaia EB 11.2459 d next door"),
    172166157: ("blend_neighbour", "HD 258728 B: light from a 9th-mag star 15\" away (earlier note)"),
    # --- singles / few events off target ---
    35665763: ("single_event", "one 23% dip, ~1.5 h, BTJD 3230.51 (2023-10-13); pixel pass; LP 588-14; no Gaia "
               "neighbour within 12\""),
    143997061: ("blend_neighbour", "dips stronger on a star 42\" away (pixel fail)"),
    98662599: ("off_target_or_unclear", "one 28% dip BTJD 3036.08; pixel soft (difference centroid 2.2 px off)"),
    436931412: ("off_target_or_unclear", "one 28% dip BTJD 2906.92; pixel soft (centroid 2.1 px off, peak not on target)"),
    365334255: ("off_target_or_unclear", "one 8% dip BTJD 3245.35; difference centroid 6 px off"),
    171428220: ("off_target_or_unclear", "one 10% dip BTJD 2585.26; difference centroid 6 px off"),
    344926802: ("off_target_or_unclear", "two dips BTJD 1521.54, 2560.45; centroid 6-7 px off"),
    219779728: ("off_target_or_unclear", "two dips BTJD 2898.51, 3485.97; no dip on the target pixels"),
    # --- young stars ---
    57830249: ("young_dipper_cpv", "TWA 33: deep (to 68%) irregular dips in S10, shallower later, smooth 2.5% rotation by "
               "S90-101; pixel pass; known dipper (Capistrant+2022) and in Bouma+2024"),
    193136669: ("young_dipper_cpv", "TWA 34: 1.567 d complex periodic variable with sharp dips; known (Bouma+2024, Capistrant+2022)"),
    146725602: ("young_dipper_cpv", "known dipper (Capistrant+2022): 1.77 d quasi-periodic +-5-9% with sharp 15-20% dips; its co-moving companion TIC 146725601 (31\") "
                "shows the same 1.768 d V-shaped dips in some sectors only: one of the pair is a complex rotator, "
                "light shared between apertures"),
    146725601: ("odd", "companion of TIC 146725602, same 1.768 d dips (see that entry)"),
    # --- rotators / artefacts seen ---
    220523369: ("rotator_flare", "rotation + flares"),
    299486246: ("rotator_flare", "10% rotator"),
    322009100: ("rotator_flare", "rotator"),
    300866996: ("rotator_flare", "0.61 d sinusoid"),
    270465315: ("rotator_flare", "rotator"),
    68982826: ("artefact", "event at a data edge"),
    198160317: ("artefact", "SAP/gap artefact"),
    284191177: ("artefact", "artefact-like"),
    392569631: ("artefact", "artefact-like"),
    236817870: ("artefact", "artefact-like"),
    420901501: ("artefact", "artefact-like"),
    171522073: ("artefact", "noise"),
    184093368: ("artefact", "noise"),
    395510283: ("artefact", "+-100% swing in one sector: PDC crowding correction"),
    345135733: ("artefact", "slow 40% ramps: systematics"),
    341795046: ("artefact", "dip at data edge"),
    351339273: ("artefact", "ramps at sector edges"),
    75160399: ("artefact", "ramps at sector edges"),
    251845156: ("artefact", "scattered-light structure"),
    387541118: ("artefact", "momentum-dump-like feature"),
    436644116: ("artefact", "messy single feature"),
    458613226: ("artefact", "noise"),
    231815218: ("artefact", "drop at the end of a sector"),
    452522881: ("artefact", "drop at the end of a sector"),
    358232435: ("rotator_flare", "rotation/noise"),
    202468443: ("known_planet_or_toi", "TOI-5268.01 (FP/PC) 2.0686 d"),
}


def auto_label(r, looked: bool) -> tuple[str, str]:
    if isinstance(r.toi, str) and r.toi or isinstance(r.ctoi, str) and r.ctoi:
        return "known_planet_or_toi", str(r.toi or r.ctoi)
    if r.cls0 == "known_planet_or_toi":
        return "known_planet_or_toi", str(r.known)
    if isinstance(r.eb_on, str) and r.eb_on:
        return "eb_known", r.eb_on[:160]
    if isinstance(r.cpv, str) and r.cpv:
        return "young_dipper_cpv", r.cpv
    if isinstance(r.nb_var, str) and "[E" in r.nb_var:
        return "blend_suspect_neighbour_eb", r.nb_var[:160]
    if isinstance(r.var_on, str) and r.var_on:
        return "catalogued_variable", r.var_on[:160]
    return "unreviewed_" + str(r.cls0 if isinstance(r.cls0, str) else "short_only"), ""


def main():
    K = pd.read_csv(C / "cat_summary.csv")
    T = pd.read_csv(C / "triage.csv")[["tic", "survey", "Tmag", "rad", "d_pc", "cls0", "known", "top_P", "top_mes"]]
    S = pd.read_csv(C / "signals_short.csv")
    S = S[S.above_tau]
    sb = S.sort_values("mes", ascending=False).drop_duplicates("tic")[["tic", "survey", "P", "mes", "Tmag", "rad", "d_pc"]]
    sb = sb.rename(columns={"P": "short_P", "mes": "short_mes"})
    D = K.merge(T, on="tic", how="left").merge(sb, on="tic", how="left", suffixes=("", "_s"))
    for c in ("survey", "Tmag", "rad", "d_pc"):
        D[c] = D[c].fillna(D[c + "_s"])
    E = pd.read_csv(C / "ebscan_table.csv")[["tic", "P"]].rename(columns={"P": "lc_P"})
    D = D.merge(E, on="tic", how="left")
    if "cpv" not in D.columns:
        D["cpv"] = ""
    rows = []
    for r in D.itertuples():
        looked = pd.notna(r.lc_P)
        if r.tic in MANUAL:
            lab, det = MANUAL[r.tic]
            src, depth = "eyeball", "full"
        else:
            lab, det = auto_label(r, looked)
            src, depth = "auto", ("lc_look" if looked else "catalogue")
        rows.append({"tic": r.tic, "survey": r.survey, "d_pc": round(r.d_pc, 1) if pd.notna(r.d_pc) else None,
                     "Tmag": r.Tmag, "label": lab, "detail": det, "source": src, "depth": depth,
                     "lc_P": round(r.lc_P, 5) if looked else None, "cls0": r.cls0,
                     "eb_on": r.eb_on, "var_on": r.var_on, "nb_var": r.nb_var, "toi": r.toi,
                     "simbad": r.simbad, "not_checked": r.not_checked})  # fmt: skip
    out = pd.DataFrame(rows)
    fp = REPO / "labels" / "cool_objects_v1.csv"
    out.to_csv(fp, index=False)
    print(fp, len(out))
    print(out.groupby(["source", "label"]).size().to_string())


if __name__ == "__main__":
    main()
