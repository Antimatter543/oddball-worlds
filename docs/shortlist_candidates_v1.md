# Shortlist from the cross-sector gallery (candidates v1), 2026-11-11

## Bottom line

**Nothing new and unexplained survived.** We vetted all 150 gallery stars, one by one, and none of
them earned the label "unexplained". The closest thing to what this project hunts for is a young
star already known to have a dusty disk. It dims irregularly by 10–30% (TIC 435923212, below).
That is real and on the right star, but it is an expected kind of object. No published report of
its dimming turned up in its SIMBAD references.

The other useful results:

* Three bright, uncatalogued **high-amplitude γ Doradus pulsator candidates**: stars that pulse
  by up to 22%. That is a recognised but rare class.
* About **30 eclipsing binaries** with no period in any variable-star catalogue.
* Two pipeline faults that would otherwise have kept producing false candidates: the asteroid
  check was off by several arcminutes, and some catalogued stars slip through the
  "uncatalogued" filter.

## What was looked at

The gallery `outputs/gallery/cand_v1_n150` has 150 cards, built from `outputs/candidates/v1`:

* **69 stars that were candidates in two or more sectors.** This is all of the multi-sector stars
  that pass the gallery filter.
* **81 top-ranked single-sector stars.**

The gallery filter keeps stars that are not in a variable-star catalogue, do not look like an
eclipsing binary to the automatic test, were not explained by a known asteroid, are not in a
crowded time bin, and are not a known positive. The scores were frozen: nothing was retrained or
retuned.

Each star had three things checked:

1. **The card** (`cards/<tic>_s<NNNN>.png`). This shows the light curve in every sector we hold,
   a zoom on the strongest event, and the nearest stars on the same CCD.
2. **A new pixel check** (`cards/<tic>_s<NNNN>_pix.png`). This takes a TESScut image cut-out and
   shows a difference image of the event, a map of where the star's variability lives, a 7×7 grid
   of per-pixel light curves, and the Gaia neighbours.
3. **A catalogue dossier** (`dossier.md`). This holds:
   * Gaia DR3 parallax, RUWE (high values hint at a binary) and position on the
     colour–magnitude diagram;
   * AllWISE infrared colours;
   * VSX variables within 120″;
   * Gaia variability classes;
   * Kostov+2025 TESS EBs;
   * young-star lists;
   * SIMBAD;
   * ExoFOP TOIs and CTOIs.

Labels and their evidence are in `labels/triage_allsectors_v1.csv`, one row per star. Each row
gives the class, confidence, reason, catalogue evidence, whether the star is already known, and
what would confirm or kill it. The vetting rules were written down before labelling
(`docs/VETTING.md` §5). Five parallel vetters each took 30 cards. I then personally re-reviewed:

* every low-confidence row;
* every pulsator and young-star row;
* the stars a catalogue cross-match contradicted;
* a random sample (4 of 4 agreed).

All 150 stars got the full depth: card, pixel card and dossier. None was vetted on the light curve
alone.

## Class counts (all 150)

| class | n | multi-sector | single-sector |
|---|---|---|---|
| eclipsing binary | 56 | 21 | 35 |
| scattered light (orbit-edge ramps) | 17 | 2 | 15 |
| rotation / starspots | 16 | 12 | 4 |
| pulsator | 13 | 10 | 3 |
| asteroid crossing | 11 | 0 | 11 |
| blend (signal from a neighbour) | 10 | 7 | 3 |
| stochastic (supergiant, red giant) | 7 | 6 | 1 |
| flare star | 5 | 5 | 0 |
| other systematic (PDC / crowding / background) | 5 | 1 | 4 |
| quiet (no real signal) | 3 | 0 | 3 |
| momentum-dump step | 2 | 0 | 2 |
| cataclysmic variable (dwarf nova) | 2 | 2 | 0 |
| known young star | 2 | 2 | 0 |
| Be-star outburst | 1 | 1 | 0 |
| **unexplained** | **0** | 0 | 0 |

* **Confidence:** high 86, med 58, low 6.
* **Already known:** 32 yes, 55 partly, 63 no. "Partly" means, for example, a known spectroscopic
  binary whose eclipses are not catalogued, or an unvetted machine-learning flag.

**Reading.** Multi-sector picks are mostly real stars doing ordinary things: binaries, spots,
pulsations and luminous LMC stars. Single-sector picks are dominated by one-off events. Of the 81:

* 35 are lone or periodic eclipses;
* 15 are scattered-light ramps;
* 11 are asteroids.

## Worth a second look (best first)

None of these is unexplained. They are the most interesting real signals in the batch, and each
has a named observation that would settle it.

### 1. TIC 435923212 = IRAS 04193+1518: a young star dimmed by its own dust (dipper)

* Card: `outputs/gallery/cand_v1_n150/cards/435923212_s0043.png`
* Pixels: `..._s0043_pix.png`
* Sectors: S32, S43, S44 (candidate in S32 and S43)

This star in the Taurus–Auriga region fades by 10–30% for about a day at a time, at irregular
intervals. Both the depth and the timing change from sector to sector. These are exactly the
features of a "dipper": a young star seen through clumps of dust in the inner edge of its own
planet-forming disk.

**Why the boring explanations fail:**

* **Not an instrument effect.** The fades appear in both kinds of photometry (SAP and PDCSAP).
  They do not line up with orbit edges.
* **Not a neighbour.** The pixel difference image puts the lost light on this star (0.1 pixel
  off-centre, signal-to-noise 270). The only neighbour within two pixels is 4.6 magnitudes
  fainter.
* **Not an eclipsing binary.** The dips are neither periodic nor of constant depth.

**Why a disk is the right explanation:**

* **Strong infrared excess.** WISE W1−W4 = 5.0 mag, a class-II disk signature.
* **Very red and above the main sequence.** Gaia puts it 3.4 mag above the main sequence (pre-main
  sequence and/or reddened) at 289 pc.
* **High YSO probability.** Marton+2019 give 96%.

**How sure:** high that the dimming is real and on target. It is a known *type* of object, not a
new one.

**Already known?** Partly. SIMBAD lists it as a young-star candidate with a disk (Rebull+2011,
Esplin+2014 WISE disk surveys). None of its 10 SIMBAD references reports photometric variability,
and it is not in VSX.

**What would confirm it:** colour photometry during a dip (dust makes the star redder as it fades)
and a spectrum showing accretion (Hα emission). ASAS-SN or ZTF long-term light curves would show
whether deeper, longer fades also happen.

### 2. TIC 411991552: an uncatalogued high-amplitude pulsator (up to 22%)

* Card: `outputs/gallery/cand_v1_n150/cards/411991552_s0044.png`
* Sectors: S43, S44

This late-F star (G = 11.8, 456 pc) brightens in sharp peaks of 5–22%, separated by 1.3–1.9 days,
from a flat baseline. The peak heights rise and fall in a beating pattern. The same few
frequencies (0.54, 0.73, 0.76 and 1.03 cycles per day) are present in both sectors.

**Why the boring explanations fail:**

* **Not an instrument effect.** It is identical in SAP and PDCSAP.
* **Not a blend.** It is uncrowded (98% of the aperture light is the star's own). The pixels put
  the signal on target (0.2 px), and no Gaia star within 6 magnitudes lies within two pixels.
* **Not flares or bursts.** The peaks are too symmetric and too regular for flares, and there is
  no infrared excess to suggest an accretion burster.
* **Not an RR Lyrae star.** Those are about 3 mag more luminous.
* **Not a heartbeat binary.** Those repeat at a single period.

**What it is:** the pattern fits a γ Doradus star. Such stars pulsate in several slow modes at
once, and when the modes combine non-linearly the light curve shows sharp peaks. γ Dor stars are
normally a few percent in amplitude. Paunzen et al. 2020 (MNRAS 499, 3976) describe a
"high-amplitude γ Doradus" group whose combined variations exceed 0.3 mag, so 22% (≈0.2 mag) fits
that rare group. RUWE is 2.5, so it is probably a binary as well.

**How sure:** medium that it is a high-amplitude γ Dor star. High that it is real and on target.

**Already known?** No. There is no VSX, SIMBAD, Gaia variability or Kostov entry.

**What would confirm it:** a spectrum (an F-type star, no emission) and a frequency analysis of
more TESS sectors. Coherent g-mode frequencies plus their sum and difference frequencies would
settle it.

### 3. TIC 19701483 and TIC 307568749: two more of the same kind

**TIC 19701483** (TYC 1861-1525-1)
* Card: `cards/19701483_s0043.png`
* Sectors: S43, S44

It shows peaks of 3–12% about 1.4 days apart, with beating heights. It is a known spectroscopic
binary (Gaia DR3 orbit, 13.97 d). Folding the light curve on that orbit does not line the peaks
up, so they are pulsation, not tidal "heartbeats".

**TIC 307568749** (TYC 8937-1903-1)
* Card: `cards/307568749_s0038.png`
* Sectors: S32, S38

It shows peaks of 5–10% every 0.75–0.9 days on an F star sitting on the main sequence.

**For both:** uncrowded, on target in pixels, no catalogue entry for variability, medium
confidence. A smaller relative of the same family is in VSX: TIC 262389393, classed γ Dor by
ASAS-SN, which shows 2–13% peaks here. That supports this reading. Lower-amplitude members of the
same family in this gallery: 229969755, 238028090, 293476405, 367176360.

## Uncatalogued eclipsing binaries that look solid

* None of these is in VSX, the Gaia DR3 EB list, Prša+2022, Kostov+2025's vetted tables or
  IJspeert+2024.
* Most carry only the unvetted "eclipse-like" neural-network flag from Kostov+2025 (table 2),
  which gives no period.
* Every period comes from our fold across all the sectors we hold.
* Depths are PDCSAP (crowding-corrected).

| TIC | sectors | period (d) | depth(s) % | note |
|---|---|---|---|---|
| 100527598 | 32 33 | 42.60 | 40 | |
| 220481009 | 32 33 38 | 28.92 | 27 | 57.8 if the two eclipses are equal |
| 127191878 | 43 44 | 25.80 | 39 | no catalogue entry at all |
| 262309535 | 55 | 24.36 | 5.8 | flat-bottomed |
| 283629409 | 32 43 44 | 23.15 | 40 / 18 | eccentric |
| 294785844 | 32 33 38 | 21.48 | 46 / 43 | eccentric (secondary at phase 0.32) |
| 309657897 | 32 33 38 | 20.66 | 38 / 23 | eccentric (secondary at phase 0.59) |
| 348961777 | 32 33 38 | 20.33 | 18 / 9 | eccentric |
| 95697669 | 44 | 19.375 | 37 / 23 | our EB test found half this period |
| 82505578 | 44 | 17.67 (or 8.835) | 43 / 43 | K-dwarf pair at 147 pc |
| 366628994 | 44 | 12.895 (or ~25.8) | 28 / 30 | |
| 399931058 | 44 | 11.729 | 28 / 11 | secondary at phase 0.525 |
| 269314859 | 55 | 11.75 | 7 | medium confidence |
| 374015350 | 38 | 8.417 | 40.5 / 32.5 | our EB test found 21.23 d (wrong) |
| 293773129 | 32 43 | 8.089 | 2.6 / 1.4 | spotted, RS CVn-like |
| 1250902 | 32 | 5.729 (or 11.46) | 28 | |
| 389007178 | 43 44 | 3.8995 | 45 | no secondary; K dwarf |
| 143277066 | 32 33 | 3.8135 (or 7.627) | 25 | |
| 231934082 | 32 33 | 3.4995 | 29 | |
| 260239131 | 32 33 38 | 0.32373 | 3.1 / 2.2 | near-contact; medium confidence |

**Known binaries whose eclipses are not catalogued:**

* TIC 55662895 (TYC 8881-240-1): a red giant in a Gaia SB1 orbit (P = 36.98 d, e = 0.36). It
  shows 0.4–0.8%, ~1-day eclipses at the predicted conjunctions in S32, S33 and S38.
* TIC 240026725: a 44% single eclipse.

**Lower bound on the period only** (two eclipses seen):

| TIC | period (d) | depths % |
|---|---|---|
| 407621961 | ≥ 19.3 | 31 / 38 |
| 12157630 | ≥ 20.7 | 21 / 36 |
| 127203223 | ≥ 17.4 | 32 / 31 |
| 34551996 | 36.4 or 18.2 | 25 / 7.5 |
| 317281056 | > 35 or ~22 | 20 / 2.8 |

**Single eclipses**, period unknown. Each one is symmetric, on target in the pixels, and alone in
its sector. These are the lone dips that a symmetric exocomet could in principle mimic, so they
need a second eclipse to confirm.

| TIC | depth % | TIC | depth % |
|---|---|---|---|
| 284003896 | 52 | 22530487 | 16 |
| 261061569 | 37 | 116067735 | 12 |
| 380035972 | 34 | 97362640 | 13.5 |
| 275843704 | 33 | 443549970 | 9.5 |
| 307487369 | 33 | 230396307 | 23 |
| 187758814 | 27 | 422897442 | 18 |
| 140968283 | 21 | 55237844 | 18 |

* 38467924 has two eclipses in different sectors (37% and 11.5%), so it is probably eccentric.
* 167810221 shows 5% in S33 and 13% in S38, probably a primary and a secondary.

**Not on this list:**

* The nine "uncatalogued" EBs found in the IJspeert+2024 TESS OBAF EB catalogue (`J/A+A/691/A242`):
  80594139, 115639773, 207780183, 234226107, 371811873, 381133258, 389247591, 427818034 and
  443115497.
* TIC 96289372 and TIC 366811495. Each shows a pure 0.800-day sinusoid, which could be an
  ellipsoidal binary or rotation (low confidence). The shared period is a coincidence, not a
  shared instrument signal: refined periods are 0.79989 and 0.80012 d, phases differ by 127°,
  the stars are 4.7° apart, and each is on target.

## Already known in the literature (in the gallery although flagged "uncatalogued")

* **Dwarf novae:** VW Hyi and CZ Ori. Both are VSX dwarf novae but passed the gallery's
  catalogue filter.
* **LMC luminous stars:**
  * SK −69 167 and SK −69 149 (B supergiants);
  * HD 269440 (an α Cyg variable);
  * HD 269062 (an emission-line star);
  * HD 269927 A+B (next to a Wolf–Rayet star);
  * CPD−69 457 in 30 Doradus (75 Gaia stars within one TESS pixel: a blend).
* **RS CVn star:** TIC 171260676, in Gaia and VSX.
* **ASAS-SN γ Dor:** TIC 262389393.
* **Spectroscopic binaries:** TYC 643-682-1 (period 4.542 d; listed as a SuperWASP false
  positive) and others in the EB table above.
* **Blends with a catalogued variable neighbour:**
  * ASAS J054549-2709.6;
  * LINEAR 20443973;
  * ZTF J053722.05+333449.7;
  * OGLE LMC-ECL-32026;
  * OGLE LMC550.06.000012;
  * WISE J040528.8-650412.
* **Asteroids.** All 11 asteroid labels are confirmed by the corrected SkyBoT check (below). The
  bodies include:
  * (22) Kalliope;
  * (42) Isis;
  * (690) Wratislavia;
  * (535) Montague;
  * (50) Virginia (the vetter's identification; the check finds a V ≤ 19 body there);
  * Sigelinde, Okuda, Daishi, Lomnický Štít, Murgas;
  * 2000 XP4 (V 18.8, 5.5″ off), which explains TIC 405266889. The vetter had called that star
    "quiet" with a possible faint mover, and it is now relabelled asteroid.

## Pipeline problems found during vetting

1. **The SkyBoT asteroid check used Earth's viewpoint, not TESS's.**
   * SkyBoT returns the same positions for observer code C57 (TESS) as for the geocentre: for
     (535) Montague the two agree to 0.02″.
   * JPL Horizons with TESS as observer puts the body 189″ away. That is where the pixels saw it
     cross.
   * Every earlier SkyBoT result in this project was therefore off by up to several arcminutes.
     This includes the S32 `ens_v2` check in PLAN.md.
   * The vetters had found asteroid crossings by eye that the check missed, including bright
     named ones such as (22) Kalliope, (42) Isis and (535) Montague.
   * **Fixed:** `scripts/skybot_check.py` now queries the geocentre with a 12′ cone and shifts each
     body to TESS's viewpoint using Horizons vectors. It matches Horizons to 0.2″ (unit test).
   * **Rerun with the fix** on the same S43/S44 targets (309 + 310 queries, 304 + 302 answered):
     * known-asteroid explanations rose from 12 to 54 in S43 and from 12 to 29 in S44;
     * in the candidate table, 83 of 572 checked stars are now explained (was 24);
     * gallery-eligible stars fell from 2,091 to 2,035 (still 69 multi-sector).
   * **Against the vetted gallery**, the rerun explains 8 of the 150 stars. All 8 were labelled
     asteroid by eye, and the check explains no star with any other label. The other 3 asteroid
     labels are confirmed when SkyBoT is asked at the crossing time the vetter saw: 2 S44 stars
     whose strongest-event epoch was a different event, and 1 S33 star (S33 had no bulk check).
   * The vetted gallery was not rebuilt. Eight of its stars would now be dropped as asteroids.
2. **The "uncatalogued" filter misses some catalogued stars.**
   * Ten gallery stars have a VSX entry within 5″, including two dwarf novae and an RS CVn.
   * Nine "uncatalogued" EBs are in IJspeert+2024.
   * The catalogue label set should add VSX CV and RS CVn types and IJspeert+2024.
3. **The EB test (`ebf_v1`) is weak on real long-period EBs.**
   * `eb_like` is about 0 for obvious EBs, for example a VSX EA (0.000) and TIC 389007178 (0.001).
   * It often locks onto half the period, or onto the spacing of momentum dumps (13.76 d for
     TIC 22933977).
   * That is why 56 of 150 "not EB-like" stars are eclipsing binaries.
4. **Pixel-check limits.**
   * The "variability OFF target" verdict is unreliable when the signal is weak, for saturated
     stars (bleed columns), and when a background gradient dominates.
   * "No FFI data in the event window" is also printed when a sector simply has no residual event.
   * The fixed 2–32 h event windows under-sample day-long events.
   * The verdict is a pointer for a human, not a veto.

## What is unverified or skipped

* **Periods.** Periods are from our own folds over 1–3 sectors. Several are ambiguous by a factor
  of 2, as marked. No radial velocities exist for any of them.
* **Kostov table 2.** "Kostov+2025 table 2 only" means flagged by an unvetted network. It is not
  a confirmed EB.
* **Literature checks** were SIMBAD references, VSX, Gaia DR3, Kostov+2025, IJspeert+2024/2021,
  Green+2023 ellipsoidal candidates, Marton+2019, Kounkel+2020 and ExoFOP TOI/CTOI. There was no
  full-text literature search beyond the high-amplitude γ Dor question.
* **Coverage.** Only the 150 gallery stars were vetted. The remaining 1,941 gallery-eligible
  candidates were not looked at.
* **Pixel checks** cover only each star's best sector.
* **Asteroid check coverage.** The corrected SkyBoT check covers the S43/S44 candidate targets
  plus the one S33 asteroid label. The S32 `ens_v2` result in PLAN.md is still the old
  Earth-view one. The other near-ecliptic cameras of S32/S33/S38/S55 were not rechecked.
* **Gallery selection.** The gallery was selected before the fix, using the Earth-view check. It
  was not rebuilt, so 8 of its 150 stars are asteroids that the corrected filter would now drop.
