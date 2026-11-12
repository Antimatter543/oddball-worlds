# Track C shortlist v1: the all-sector M-dwarf transit survey, vetted with the frozen v2 procedure (2026-11-12)

## Bottom line

**No planets were found. The survey is consistent with noise.** After the frozen v2 vetting, 13 signals on
12 stars are not rejected (3 tier A, 10 tier B). Noise alone is expected to leave about **26 [2, 83]** such
signals in a survey this size, or about 57 [30, 94] in the pessimistic case. The count matches or falls below
noise in every period and transit-count bin. Every one of the 13 is more likely a false alarm than a planet.
None should go to ExoFOP as a community TOI yet.

The pipeline works. It found every known transiting planet in its 10-60 d range that it was expected to find
(TOI-6002 b, TOI-1452 b, TOI-406 c, TOI-1224 c, TOI-715 b and the brown dwarf TOI-6508 b), without being told
where they were. It also refined the period of one community candidate (CTOI 278661431.01).

If one signal gets a follow-up, make it **TIC 2102868 (G 86-37)**. It is a 39.4-day, 0.14% dip seen three
times, and if real it would be about 2 Earth radii with roughly Earth's sunlight. Gaia flags this star as a
possible binary, though, and the pixel data show a hint that the dip comes from a neighbouring star. The
next predicted dips are 2026-11-22 18:27 UTC, 2027-01-01 04:01 and 2027-02-09 13:35 (each ± ~0.6 h, 1 sigma).

Phone images (`outputs/planets_spike/visuals/`): `survey_candidate1_tic2102868.png`,
`survey_candidate2_tic234295610.png` and `survey_candidate3_tic146636926.png`.

## What was searched (survey health)

The gate-2 search was frozen and ran unchanged on all 4,681 target M dwarfs: 2,700-3,400 K, closer than 50 pc,
T < 15, predicted 6+ TESS sectors. The run was 10 Kaggle shards at git 88a2a31 with frozen.json sha256
4194793b. Every record shows the same git head, frozen hash and code hashes.

| | value |
|---|---|
| searched (`ok`) | 4,350 |
| no SPOC / TESS-SPOC product (`no_data`) | 331 |
| real sectors per searched star, median (10/25/75/90%) | 6 (3 / 4 / 8 / 18) |
| searched stars with 1 / 2 / 3 / 4-5 / 6-12 / 13-25 / 26+ real sectors | 127 / 283 / 510 / 1,050 / 1,832 / 197 / 351 |
| searched stars with fewer than 6 real sectors | 1,970 (45%) |
| real / predicted sectors | 0.78 |
| shards | 10 × ~468 stars; ok 421-443, no_data 25-47, mean sectors 8.31-8.68, CPU 8.1-10.2 h each, 89.9 CPU-h total; no outlier shard |
| stars that lost a sector | 10, one sector each, because PDCSAP had no usable flux (`LightCurveError no usable PDCSAP flux`): TIC 142002844 S90, 193635891 S79, 198036199 S4, 232083054 S29, 232564158 S16, 307440881 S10, 313425320 S25, 340638718 S90, 356336967 S41, 407300140 S17 |
| transient errors (all recovered) | 713 S3 misses (MAST fallback), 80 retried conversions, 12 S3 connection errors |
| mass/radius fallback (0.3 M_sun) | 109 targets |
| products | 27,666 SPOC 2-min and 9,516 TESS-SPOC sector light curves |

**Sensitivity.** The fresh v2 validation set gives the share of injected planets that are detected, by real
sector count: 1-3 sectors 0.03, 4-5 sectors 0.10, 6-12 sectors 0.20, 13+ sectors 0.58. Weighted by this
survey's sector counts, that is about **0.19 per searched star**. These are injected 1-4 Earth-radius planets
at 10-60 d whose orbits happen to transit. Almost half the list has fewer than 6 real sectors, so the survey
is much shallower than the "6+ sectors" plan assumed (that rate was 0.54 on gate-2 TEST controls).

**Reproduction.** I rebuilt the light curves of the 252 vetted stars, and every star got exactly the survey's
sectors. I re-ran the frozen search on three of them (TIC 2102868, 234295610 and 146636926). All 11 long-range
signals came back identical to the survey records (P and MES equal to < 1e-9).

## Survey output vs the noise expectation

510 long-range signals at or above tau_long (8.054) fell on 252 stars. All were vetted with v2 (`run_vet2.py`,
`frozen_vet2.json` sha256 d5076c08, unchanged), with 0 measurement errors. Every A/B signal also got the pixel
check.

| tier (signals) | K (known) | A+ | A | B | C (rejected) |
|---|---|---|---|---|---|
| signals | 48 | 0 | 3 | 10 | 449 |
| stars (best tier) | 20 | 0 | 3 | 9 | 220 |

The noise expectation is N = 4,350 × r × s. r is the real-star false-alarm rate (1/118 from gate-2 real
controls; the pessimistic "fat tail" case uses 16/600 from fresh nulls). s is the share of null false alarms
(MES ≥ 7) that v2 leaves in each bin. Both were fixed before the survey (`docs/spikes/mdwarf_vetting.md`,
"Projection with v2"). Brackets are 95% intervals.

| bin | not rejected (A+/A/B), observed | noise expectation, r = 1/118 | fat-tail case |
|---|---|---|---|
| **all** | **13** | **26.3 [1.8, 83.2]** | 57.2 [30.3, 93.9] |
| tier A+ / A / B | 0 / 3 / 10 | 0.6 / 9.5 [0.6, 32] / 17.4 [1.2, 56] | 1.2 / 20.7 / 37.8 |
| 2 transits | 3 | 2.8 [0.1, 11.4] | 6.1 |
| 3 transits | 3 | 5.0 [0.3, 18.6] | 11.0 |
| 4 transits | 2 | 7.3 [0.4, 25.5] | 15.8 |
| 5+ transits | 5 | 12.9 [0.8, 42.3] | 28.0 |
| period 10-30 d | 5 | 9.5 [0.6, 32.2] | 20.7 |
| period 30-40 d | 1 | 9.5 [0.6, 32.2] | 20.7 |
| period 40 d+ | 7 | 8.4 [0.5, 28.9] | 18.3 |
| 40 d+ with 2-4 transits (the known pile-up region) | 6 | 7.3 [0.4, 25.3] | 15.8 |
| before vetting: stars with any signal ≥ tau_long | 252 | 54.9 [4.0, 169] | 119 [69, 182] |

**Verdict: no excess after vetting.** Before vetting there is a large raw excess (252 stars against ~55). It
is what vetting exists for: eclipsing binaries, known planets seen at multiples of their period, and PDC
artefacts. 49 signals have MES > 100. For example, TIC 102734241 #0 (808 ppt) and TIC 198160317 (163 ppt) are
PDC artefacts that vanish in SAP. The most common reasons for rejection were: the signal rests on one or two
of its transits (178), a secondary eclipse (167), a dip of similar strength elsewhere in the fold (159), only
two transits (134), transits at the edge of a data gap (109), and the dip being absent or different in SAP
flux (106). Full tables: `outputs/planets_spike/survey/vet/tables.txt`.

## Known planets: did the survey find them?

`spikes/mdwarf_stack/survey/known_check.py` → `outputs/planets_spike/survey/vet/known.csv` (133 catalogued
objects on the 252 stars: exoplanet.eu, TOI table refreshed 2026-11-12, live CTOIs).

- **Recovered blind in the 10-60 d range, tier K:** TOI-6002 b (10.90 d, MES 10.2 vs 10.3 predicted),
  TOI-1452 b (11.06 d, MES 28.4, on both stars of the pair; TOI-1760.01 is the FP twin), TOI-406 c (13.18 d,
  28.0), TOI-1224 c (17.95 d, 23.1), TOI-6508.01 (18.99 d, 161.7, a brown dwarf) and TOI-715 b (19.29 d, 24.4).
- **Short-period planets found at an integer multiple of their period** (the long search starts at 10 d):
  TOI-910 / LHS 475 b, TOI-771 b, TOI-7867.01, TOI-406 b, TOI-1224 b, TOI-6708.01, TOI-1743 b, TOI-1680 b,
  TOI-237 b, TOI-2120 b, TOI-2096 c, TOI-2136 b, TOI-210 b, and TOI-1276.01 (a TFOP false alarm).
- **CTOI 278661431.01 (17.63 d):** our 17.6776 d fits 6 transits over S1/27/67/68. The CTOI ephemeris is
  already 1.9 d off by S27, so our period is a refinement.
- **CTOI 233738446.01 (14.36 d):** matched, MES 150.8, but odd/even z = 4.86 and duration ratio 3.15 both fail.
  It looks like an eclipsing binary; the K tier overrides.
- **Missed, and why:** CTOI 271971130.02 (TOI-715 c candidate, 25.6 d) had predicted MES 8.2, below the
  gate-2 "predicted detectable" cut of 10.1. TOI-203.01 (51.9 d) is a TFOP false alarm with an 8.1 h duration,
  longer than the longest search box (4 h). TOI-4341.01 is a TFOP false alarm. TOI-6981 b (60.78 d, just past
  the range) was found as a two-transit signal at 40.52 d = 2/3 × 60.78 d. Its epochs are 12 periods apart.
  The catalogue test checks only integer multiples, so it went to tier C instead of K (gap_edge + duo).
  LP 890-9 b, TOI-4360.01, TOI-6245.01 and TOI-5716.01 matched only below tau_long.

## The 13 surviving signals, best first

All 12 stars were researched (`spikes/mdwarf_stack/survey/research_candidates.py`, anonymous, cached in
`outputs/planets_spike/survey/vet/research/`). Sources: ExoFOP; the NASA Exoplanet Archive (TAP RA/Dec box,
because `contains()` returned 403); SIMBAD (Harvard mirror, because CDS returned 404); Gaia DR3 (63″, with
RUWE, RV scatter and neighbours) and Gaia NSS orbits; the Gaia eclipsing-binary list (I/357/tbooc); and the
Prsa et al. TESS EB catalogue (J/ApJS/258/16). **Every query succeeded for every star.** No ExoFOP TOI and no
NASA planet exists on any of the 12 stars, except the CTOI and EB entries named below.

How the numbers below were derived:

- Radius comes from the box depth and the star's TIC radius. There is no crowding correction beyond what
  PDCSAP already does.
- a comes from Kepler's law with the TIC mass. S is the sunlight the planet gets (Earth = 1).
- Teq assumes Bond albedo 0.3 (and 0 in brackets) and full heat redistribution.
- HZ is the Kopparapu et al. (2013) habitable zone. "Conservative" runs from runaway greenhouse to maximum
  greenhouse. "Optimistic" runs from recent Venus to early Mars.
- Next dips have 1-sigma timing errors from a linear-ephemeris fit to the TESS epochs. Real errors are larger
  if the period is an alias.
- "Odds of being false" is the v2 validation's count of false alarms like this one (same tier and transit
  count) per 4,700 survey stars. If that is comparable to the handful of real planets the survey can find,
  the signal is more likely false than real.
- Cards are in `outputs/planets_spike/vetting/v2/cards/survey/`, as `survey-<TIC>-long_<k>.png`, plus `cards.md`.

| # | TIC | P (d) | depth (ppt) | transits | tier | radius if real (R_e) | Teq A=0.3 (K) | verdict | false alarms like it per 4,700 stars |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2102868 #0 | 39.398 | 1.44 | 3 | A | 1.95 | 245 (cons. HZ) | weak; best of a weak set | 0.6 [0, 3.8] |
| 2 | 234295610 #2 | 49.162 | 4.96 | 3 | A | 2.90 | 198 (cons. HZ) | weak; shape not transit-like | 0.6 [0, 3.8] |
| 3 | 146636926 #1 | 54.649 | 1.30 | 4 | B | 1.46 | 200 (cons. HZ) | weak; at threshold, transits at data gaps | 4.2 [0.2, 16] |
| 4 | 141816993 #0 | 22.138 | 0.52 | 5 | A | 0.96 (diluted) | 279 (opt. HZ) | weak; at threshold, dumps/gaps | 6.6 [0.4, 24] |
| 5 | 231888176 #0 | 27.809 | 1.90 | 4 | B | 1.48 | 240 (cons. HZ) | likely false (neighbour, alias) | 4.2 [0.2, 16] |
| 6 | 57830249 #2 | 45.068 | 9.73 | 2 | B | 5.3 | 216 | unclear; young star, period unknown | 3.0 [0.1, 12] |
| 7 | 323354765 #0 | 24.831 | 9.69 | 7 | B | 1.5 | 170 (too cold) | likely false (one transit dominates, crowded) | 7.9 [0.5, 27] |
| 8 | 441811894 #1 | 11.769 | 2.46 | 51 | B | 1.5 | 289 (too hot) | likely false (noise pattern) | 7.9 [0.5, 27] |
| 9 | 89615401 #4 | 13.046 | 5.57 | 15 | B | 2.1 | 266 (opt. HZ) | likely false (off target) | 7.9 [0.5, 27] |
| 10 | 410421717 #0 | 46.196 | 12.7 | 2 | B | 5.5 | 219 | likely eclipsing binary | 3.0 [0.1, 12] |
| 11 | 234284556 #2, #3 | 59.505 / 54.245 | 7.3 / 3.2 | 2 / 5 | B, B | — | — | known eclipsing binary | 3.0 / 7.9 |
| 12 | 172166157 #0 | 57.456 | 74.2 | 3 | B | — | — | artefact / contamination | 5.4 [0.3, 20] |

### 1. TIC 2102868 (G 86-37, Karmn J05173+321): tier A, 39.40 d

A 1.4 ppt (0.14%), 1.5 h dip seen three times: 2021-09-18 (S43), 2021-10-27 (S44) and 2024-12-13 (S86).
Individual strengths are SES 5.3, 4.2 and 7.9. None of the three falls on a momentum dump. The S86 dip is
3.2 h from a data gap; the other two are 30+ h from one. It passes every v2 test with zero strikes. The fold
shows nothing else of comparable strength, and SAP agrees with PDCSAP (z_SAP = -0.5).

- **Star:** 3,393 K, 0.47 R_sun, 32.7 pc, T = 10.8.
- **If on this star:** 1.95 R_e, a = 0.18 AU, S = 0.85, Teq 245 K (268 K at A = 0), inside the conservative HZ.
- **Against it:**
  - Gaia RUWE is 2.94, and the Gaia RV error is 2.7 km/s with an RV amplitude of 18.7 km/s. Both are typical
    of an unresolved binary. A companion could be the source of the dip, and it would also make the radius
    wrong.
  - Two fainter Gaia stars sit 6.5″ (ΔG 2.5) and 7.4″ away. The brighter one would need only a 1.4% eclipse to make the dip.
  - The pixel check found the difference-image centroid more than 1 px off the target in the one epoch strong
    enough to test. The frozen v2 rule needs two such epochs to strike, so this is a warning, not a fail.
  - A 3-transit A signal is a cell where the validation saw 0 of 48 null false alarms. The survey has 2 such
    signals against 0.6 [0, 3.8] expected, which is inside the noise interval.
- **Catalogues:** no ExoFOP TOI or CTOI, no NASA entry, no Gaia NSS orbit, no EB catalogue entry.
- **Next dips:** 2026-11-22 18:27, 2027-01-01 04:01 and 2027-02-09 13:35 UTC (BTJD 4367.269, 4406.668,
  4446.066; ± 0.6-0.7 h). The star is at RA 05h17m, Dec +32°, well placed from northern sites in these months.
- **TESS:** no sector is planned through S134 (Aug 2028, the end of the tess-point 0.9.5 table).
- **Card:** `survey-2102868-long_0.png`.

### 2. TIC 234295610 (2MASS J23515597-6447345): tier A, 49.16 d

A dip seen three times: 2018-07-27 (S1), 2020-08 (S28) and 2026-03-29 (S102), with SES 6.7, 4.8 and 4.5. The
box depth is 5.0 ppt, and the pixel check puts it on the target (pixel SNR 25.2 against 6.9 for the neighbour).

The averaged dip is lopsided: a sharp drop to ~1.5%, then a slow recovery over ~2 h. A transiting planet makes
a symmetric dip, so this shape points to stellar activity or a systematic. The star is also noisy for its
brightness. Its five other search signals (11.4-15.2 d, MES 8.3-10.1) are of comparable strength, and one is
stronger. That is how a noise floor looks, not how a planet looks.

- **Star:** 3,161 K, 0.38 R_sun, 48 pc, T = 12.5; CROWDSAP 0.71-0.90; one Gaia neighbour at 6.9″ (ΔG 3.1);
  RUWE 1.13.
- **If real:** 2.9 R_e, S = 0.36, Teq 198 K (216 K), inside the conservative HZ.
- **Next dips:** 2026-11-30 05:34, 2027-01-18 09:27 and 2027-03-08 13:20 UTC (± 1.1 h). Dec -65°, southern.
  No TESS sector is planned through S134.
- **Catalogues:** nothing.
- **Card:** `survey-234295610-long_2.png`.

### 3. TIC 146636926 (2MASS J02492136-4416063): tier B, 54.65 d

A 1.3 ppt, 4 h dip seen four times: 2018-10-09 (S3), 2025-10-20 (S97), 2026-05-27 (S104) and 2026-07-20 (S106).
MES 8.29 is less than 1 above the threshold, which is the one strike (low_margin). The pixel check puts it on
target (pixel SNR 16.1).

The dip itself is weak:

- The S3 dip is the weakest (SES 2.0) and sits on a momentum dump.
- The other three are 0.0, 1.0 and 1.4 h from a data gap. The frozen gap test passes them, but gap edges are
  where systematics live.
- 54.65 d is within 0.3% of 4 × 13.7 d, TESS's orbital period.
- Three other weaker signals on this star (37.7, 39.9 and 48.3 d) put dips within a few days of the same S106
  stretch.

- **Star:** 3,334 K, 0.37 R_sun, 31.8 pc, T = 11.5.
- **If real:** 1.46 R_e, S = 0.38, Teq 200 K (219 K), inside the conservative HZ.
- **Next dips:** 2026-12-31 19:52, 2027-02-24 11:26 and 2027-04-20 03:01 UTC (± 0.6-0.7 h).
  No TESS sector is planned through S134. A 1.3 ppt dip on a T = 11.5 star is hard from the ground.
- **Catalogues:** nothing; Gaia RV amplitude 13.7 km/s.
- **Card:** `survey-146636926-long_1.png`.

### 4. TIC 141816993 (StM 187 A, WDS J13418+5815A): tier A, 22.14 d

A 0.5 ppt, 4 h dip with five transits (S15, S48, S49, S75, S76; SES 1.9-6.2) and MES 8.20, barely above the
threshold.

- Two of the five transits fall on momentum dumps. One sits at the very edge of a data gap (0.0 h), on a
  downward ramp.
- The pixel test was uninformative, and its on-target pixel SNR was negative (-4.6).
- The star has a companion 0.76″ away of nearly equal brightness (ΔG 0.93, same parallax). TESS cannot
  separate the two, so the depth is diluted and either star could host the dip. On the brighter star the
  radius would be ~1.15 R_e. On the companion it would be about twice that.
- RUWE is 1.81.

- **If on the brighter star:** S = 1.45, Teq 279 K (305 K), inside the optimistic HZ only.
- **TESS:** looks again in S121-122 (Sep-Oct 2027) and S127-128 (Feb-Mar 2028), with about four predicted transits, so this one tests itself.
- **Next dips:** 2026-11-19 04:48, 2026-12-11 08:06 and 2027-01-02 11:25 UTC (± 1.3 h).
- **Card:** `survey-141816993-long_0.png`.

### 5-12. The rest (likely false or already explained)

- **TIC 231888176 (SCR J1639-4652, 19.3 pc), B, 27.81 d:** 1 h dip (duration ratio 0.19, short for this star),
  four transits, MES 8.16. The pixel check leans off target: 2 of 3 epochs off, and the neighbour's pixel SNR
  (15.6) equals the target's (15.2). The field is crowded (CROWDSAP ~0.67). 27.81 d is 1.5% from 2 × 13.7 d.
- **TIC 57830249 (TWA 33, a young TW Hydrae member), B, 45.07 d, two transits (2021-03 S36, 2026-02 S100):** a 9.7
  ppt dip on target (pixel SNR 29). With only two transits the true period could be 45.07/n d. Young stars dip
  for many reasons, and the Gaia RV scatter (amplitude 18.9 km/s) hints at a companion. Another two-transit
  signal on the same star (50.11 d) was rejected. TESS looks again in S114 (Feb 2027).
- **TIC 323354765 (DENIS J084834.9-785352, 2,915 K, 0.14 R_sun), B, 24.83 d:** seven transits, 9.7 ppt, but
  one transit carries 70% of the signal (dom 0.70). Crowded (CROWDSAP 0.70-0.84). Colder than the HZ (S = 0.20).
- **TIC 441811894, B, 11.77 d, 51 transits:** the star has six near-equal signals at 10.06-11.77 d (MES 8.0-9.1,
  2.1-2.5 ppt, 1 h). A planet gives one period. Six equal ones are a noise pattern. Pixel soft (1 of 2 off).
  Too hot for the HZ (S = 1.65).
- **TIC 89615401, B, 13.05 d (5th signal on the star):** the pixel check puts the dip off target in 6 of 6
  strong epochs. Crowded (CROWDSAP 0.58-0.77, Gaia IPDfmp 23).
- **TIC 410421717, B, 46.20 d, two transits (S95, S103):** 12.7 ppt, on target (pixel SNR 20.4). Gaia RV error
  10.4 km/s with an RV amplitude of 52.8 km/s strongly suggests a spectroscopic or eclipsing binary. The two
  epochs are 277.2 d apart, so the true period could be 277.2/n d.
- **TIC 234284556 (UPM J2222-6303), B, 59.50 d (2 transits) and 54.24 d (5 transits, odd/even z 4.0):** a
  known eclipsing binary. It is in the Prsa et al. TESS EB catalogue, SIMBAD type EB*, CTOI .01 at 1.106 d,
  and Gaia RV amplitude 32.8 km/s.
- **TIC 172166157 (HD 258728 B), B, 57.46 d:** 74 ppt in one transit (S43). There is no dip at the S60
  epoch (SES 0.3), and the S73 epoch sits on an edge and a dump. CROWDSAP is 0.06-0.09: the 9th-magnitude
  HD 258728 is 15″ away and dominates the aperture. It is an artefact or contamination.

One B signal became C after the pixel check: TIC 102764828 #0 (8 of 8 epochs off target; the star also has
a Prsa EB entry). I read the striking C signals. None looks wrongly rejected, and the only real one among
them is the TOI-6981 b alias above.

## What to do next

1. **Submit nothing to ExoFOP now.** By `mdwarf_vetting.md` tier A is "community TOI-grade at most", and all
   three A signals fail a closer look (binarity and centroid warnings, a lopsided shape, a diluting
   companion). With the survey consistent with noise, a CTOI would most likely waste TFOP time.
2. **TIC 2102868 is the only one worth a cheap test.**
   - Gaia DR4 (expected December 2026) publishes epoch astrometry and RVs. If they show an orbit, the star is
     a binary and the signal is very likely not a planet. This costs nothing.
   - If the star looks single, a 0.6-1 m telescope with ~1 ppt precision could watch one of the predicted
     dips (2026-11-22, 2027-01-01, 2027-02-09). A flat light curve at the predicted time would kill it.
3. **TIC 141816993 needs no action.** TESS re-observes it in S121-122 and S127-128 (2027-28). Re-run the frozen
   search then. A real 22.1 d signal would gain about four transits.
4. **Re-estimate the survey's yield with real sector counts** (about 0.19 detected per searched star, not
   0.54) before any decision on extending Track C. The deeper half of the list (≥ 6 real sectors, 2,380 stars)
   is where the sensitivity is.

## Not checked or not possible

- No spectroscopy, high-resolution imaging or ground photometry exists for any of the 12 stars on ExoFOP (all
  ExoFOP queries succeeded and returned no observations). On-target status is unproven for every signal.
- The pixel test was uninformative or not applicable for TIC 141816993 and TIC 234284556. It is untested on
  injections (no pixel-level injections), so a "pass" there is weak evidence.
- Ground-based visibility was judged from coordinates only; no observability calculation was done.
- I re-ran the search on 3 of the 252 vetted stars, not all of them. Their sector lists all matched.
- TESS sectors after S134 are not planned yet, so "no future sector" means none through Aug 2028.
- The 10 stars that each lost one sector to unusable PDCSAP flux were not searched again with SAP.

## Ideas for vetting v3 (not applied here; v2 is frozen)

- **Match catalogue objects by epochs, not period ratios.** TOI-6981 b was found at 2/3 of its period and
  rejected instead of recognised. Checking whether the detected epochs fall on the known ephemeris would
  catch any p/q alias.
- **Flag periods near n × 13.7 d (TESS's orbit) and stars with several near-equal signals.** Several B signals
  (146636926, 231888176, 441811894) look like the noise floor of their star. A "signal not unique on its
  own star" test would remove them.
- **Transit-shape symmetry.** A test comparing ingress and egress would have caught TIC 234295610.
- **Count single-strong-epoch centroid offsets as a soft strike** when the epoch is strong, with a Gaia
  neighbour inside 10″ (TIC 2102868).
- **A Gaia binarity flag** (RUWE > 1.4 or RV amplitude > 10 km/s) as an info line on every card, with the
  radius caveat it implies.
- **Calibrate the pixel test with pixel-level injections.**

## Reproduce

From `spikes/mdwarf_stack/`, with `PY` = a repo venv with wotan/numba, and every heavy step under
`nice -n 19 ionice -c3`:

```
$PY -I survey/vet_rows.py                          # survey records -> rows_long / jobs_long / tics_long
$PY -I survey/prep_vet.py                          # rebuild LCs + SAP for the 252 stars, check sectors
$PY -I vetting/catalog.py tics=<OUT>/survey/vet/tics_long.txt
VET_LC_DIR=data/planets_spike/lc_cache/lc $PY -I vetting/run_vet2.py survey --rows <OUT>/survey/vet/rows_long.jsonl \
    --jobs <OUT>/survey/vet/jobs_long.jsonl --shard 0/2   (and 1/2)
$PY -I vetting/pixel_check.py <shard file> KEY ...  # every A/B
$PY -I survey/survey_tables.py                     # tiers, noise expectation -> tables.txt / graded.jsonl
$PY -I survey/known_check.py                       # known.csv
$PY -I survey/research_candidates.py TIC ...       # catalogue research, cached
$PY -I survey/dossier.py KEY ...                   # implied planet, epochs, next dips, sectors
$PY -I vetting/cards2.py survey --tiers A+,A,B,K
$PY -I survey/phone_plot.py TIC P T0 DUR_H OUT.png --title ... --sub "...|..."
```
