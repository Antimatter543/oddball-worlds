# Track C shortlist, 50-100 pc extension v1: blind TESS search of 15,452 M dwarfs at 50-100 pc, vetted with the frozen v2 procedure (2026-11-16)

## Bottom line

**No convincing new planet. The survey output is consistent with noise and blends.** After the frozen v2
vetting, 23 signals on 19 stars are not rejected (4 tier A+, 2 A, 17 B). Measured on the gate-3 far-star
nulls, the false alarms that survive vetting, scaled to the 756 non-catalogue signals this survey produced,
come to **22 [12, 35]**. The pre-registered planet expectation was about 4.6 (12.2 before the 2.7x model
correction). Counts alone cannot separate "all false" from "a few real". Signal by signal, 8 of the 19 stars
have a specific false explanation, proven or likely (a binary star, a neighbouring star, a young disk star,
crowding). 9 more look like noise or systematics, or are untested two-transit signals, and 2 stay open. Every one
of the 23 is more likely false than real.

**No signal clears the community-TOI bar, so there is no CTOI draft and nothing goes to ExoFOP.**

The two open signals:

- **TIC 230081659** (2MASS J16233268+6721310, 67 pc, T = 13.1): a 0.12% dip every 34.61 days, seen at 22 transits
  over 42 TESS sectors. If real and on this star it would be about 1.4-1.7 Earth radii with 72% of Earth's sunlight
  (inside the habitable zone). It is the most planet-like signal in shape and in where it was found. No single
  transit is strong, though, so the pixels cannot place it. Three of four other reductions of the same data show it
  at only 55-70% of the depth a real dip would give. TESS returns for ~9 more transits in 2027-28 (S118-S126), which will test it for free.
- **TIC 447890696** (2MASS J10315803-4954073, 76 pc, T = 13.5): a 2% dip every 16.84 days, seen three times in
  March-April 2021. It is clearly in the pixel data: four independent reductions show it at SES 8-12. But it lasts
  only 1 hour, less than half the 2.4 h a central transit across this star would take, and it is V-shaped. That
  points to a grazing eclipse, or to a faint background star 9.8″ away that the pixels cannot separate from the
  target. TESS is observing it **now** (S110-S113; predicted dips ± 22 h).

One survivor, TIC 353875121 (32.3 d, 23 transits, would be 2.0 R_e), is **settled as a blend**. ZTF shows the
background star 28″ away (G = 17.2) dimming by ~20% at 11 of the TESS-predicted times in 2018-2025. It is an
eclipsing binary.

The pipeline works. It re-found all 6 TOIs with 10-60 d periods on searched stars blind, and 16 more catalogued
short-period objects at integer multiples of their periods.

Phone images (`outputs/planets_spike/visuals/`): `survey100_candidate1_tic230081659.png`,
`survey100_candidate2_tic447890696.png` and `survey100_candidate3_tic353875121.png`.

## What was searched (survey health)

The frozen gate-2 search (`frozen.json` sha256 4194793b, tau_long 8.054) ran unchanged on all 19,827 stars of
the gate-3 target list: 2,700-3,400 K M dwarfs at 50-100 pc, T < 15, predicted 6+ TESS sectors
(`outputs/planets_spike/gate3/targets_100pc.csv`). It ran as 25 Kaggle shards at git 111f07a. Every record has
the same git head, frozen hash and code hashes. The 25 manifests agree, and their target counts add up to
19,827. Script: `spikes/mdwarf_stack/survey100/health100.py`, output `outputs/planets_spike/survey100/vet/health.txt`.

| | value |
|---|---|
| searched (`ok`) | 15,452 |
| no SPOC / TESS-SPOC light curve (`no_data`) | 4,375 (22.1%; the gate-3 sample had 21.7%) |
| no-data share by brightness | T < 12: 1.2%; 12-13: 9.0%; 13-14: 14.9%; 14-15: 27.8% |
| real sectors per searched star, median (10/25/75/90%) | 5 (2 / 3 / 8 / 19); mean 7.94 |
| searched stars with 1 / 2 / 3 / 4-5 / 6-12 / 13-25 / 26+ real sectors | 1,204 / 1,660 / 2,473 / 3,047 / 4,987 / 805 / 1,276 |
| searched stars with fewer than 6 real sectors | 8,384 (54%) |
| real / predicted sectors | 0.688 on searched stars, 0.564 on all stars (gate-3 sample 0.666 / 0.560) |
| star-sectors | 122,694 (27,268 SPOC 2-min, 95,426 TESS-SPOC FFI); 118,859 from the S3 bucket, 3,835 from MAST |
| sectors after S86 | 3,681 star-sectors, including 527 from "sector 1751", the January 2026 3I/ATLAS special pointing (TESS-SPOC, BTJD 4055.8-4063.0) |
| shards | 25. `ok` share 0.750-0.807 (largest z 1.97); mean real sectors 7.56-8.20; 3.97-9.75 CPU-h each; 213.9 CPU-h in all (221 predicted). No outlier shard. |
| transient errors (all recovered) | 3,835 S3 misses (MAST fallback), 35 S3 connection errors, 1 S3 timeout, 344 retried conversions |
| sectors lost | 43 sectors on 41 stars, because PDCSAP had no usable flux (`LightCurveError no usable PDCSAP flux`). Those stars were searched on their other sectors. |
| mass/radius fallback (0.3 M_sun) | 546 stars |

**Sensitivity.** The gate-3 far-star injections give the share of transiting planets that the frozen search plus
v2 vetting keeps, by real-sector count. Weighted by this survey's sector counts, a planet that transits is found
with these probabilities per searched star:

- 1.0-1.5 R_e at 15-30 d: **0.0065**
- 1.0-1.5 R_e at 30-50 d: **0**
- 1.5-2 R_e at 15-50 d: **0.040**
- 2-3 R_e at 15-50 d: **0.160**

Almost all of that sits on the 2,081 stars with 13+ real sectors. Earth-size planets were never within reach
here; gate 3 said so before the run.

## Survey output vs the noise expectation

809 long-range signals at or above tau_long fell on 404 stars. The pre-registered vetting load was ~195. All
809 were vetted with v2 (`frozen_vet2.json` sha256 d5076c08, unchanged; `gate3/vet100.py`), in three batches of
≤ 140 stars. The light curves were deleted after each batch, and there were 0 measurement errors. Every A/B
signal got the frozen pixel check (30 signals). Script: `survey100/tables100.py`, output
`outputs/planets_spike/survey100/vet/tables.txt`.

| tier (signals) | K (known) | A+ | A | B | C (rejected) |
|---|---|---|---|---|---|
| signals | 53 | 4 | 2 | 17 | 733 |
| stars (best tier) | 20 | 4 | 2 | 13 | 365 |

Three expectations for the surviving (A+/A/B) false alarms, all fixed before this survey:

- **Realised mix:** the 756 non-K signals × the share of v2-vetted null signals that survive at 50 pc (13/462).
  This is the pre-registered F, recomputed with the signal count that actually happened. The pre-registered
  F = 5.5 [0, 16] assumed ~195 signals.
- **Noise only, far nulls:** 15,452 stars × the gate-3 far null rate (9/940) × the share of far nulls that v2
  leaves (20/42).
- **50 pc method:** the 50 pc survey's r = 1/118 and its fresh-null survival.

Brackets are 95% intervals. "Planets" columns are the gate-3 kept injections weighted by occurrence, scaled to
12.2 (the model as pre-registered) and 4.6 (corrected by the 2.7x over-prediction measured on the 50 pc survey).

| bin | not rejected, observed | realised mix (all false) | noise only, far nulls | noise only, 50 pc method | planets, model / corrected |
|---|---|---|---|---|---|
| **all** | **23** | **22.1 [12.0, 34.9]** | 74.4 [32, 137] | 93.6 [6.6, 295] | 12.2 / 4.6 |
| 2 transits | 2 | 6.1 [1.5, 13.8] | 1.8 [0, 9.5] | 10.0 | 0 / 0 |
| 3 transits | 6 | 5.9 [1.5, 13.2] | 27.2 [8.9, 58] | 17.9 | 0.4 / 0.1 |
| 4 transits | 5 | 12.4 [2.3, 29.2] | 23.6 [7.3, 52] | 25.9 | 0.3 / 0.1 |
| 5+ transits | 10 | 6.9 [2.4, 13.6] | 27.2 [8.9, 58] | 45.8 | 11.5 / 4.3 |
| P 10-15 d | 3 | 4.0 [0.7, 10.2] | 5.4 | 10.0 | 0 / 0 |
| P 15-30 d | 8 | 9.1 [2.3, 20.2] | 45.3 | 25.9 | 9.0 / 3.4 |
| P 30-40 d | 8 | 2.5 [0.2, 7.7] | 12.7 [2.6, 33] | 33.9 | 2.2 / 0.8 |
| P 40 d+ | 4 | 9.3 [4.0, 16.7] | 16.3 | 29.9 | 1.0 / 0.4 |
| stars with 1-3 / 4-5 / 6-12 sectors | 4 / 1 / 14 | 1.9 / 3.1 / 15.0 | 5.4 / 16.3 / 34.4 | n/a | 0 / 0.2 / 1.1 (corrected) |
| **stars with 13+ sectors** | **4** | **2.4 [0.2, 7.3]** | 23.6 | n/a | 8.7 / **3.3** |
| 40 d+ with 2-4 transits (the 50 pc pile-up) | 4 | 8.6 [3.4, 16.0] | 5.4 | 25.9 | 0 / 0 |

**Verdict: no excess over noise.** The 23 survivors match the 22 expected from false alarms at the realised
signal count. They are well below both per-star noise estimates. The only bin above its false-alarm interval is
P 30-40 d (8 against 2.5 [0.2, 7.7]), one bin of thirteen. Of those 8:

- 2 are explained false: the 353875121 blend and the young disk star TWA 34;
- 4 look like noise: 233719663, 88762886, 365466012 and 423088347;
- 1 is an untested two-transit signal (181110184);
- 1 is TIC 230081659. The planet-rich corner (5+ transits on 13+ sector stars) holds 4 survivors:
TIC 230081659, 353875121, 294328887 and 233719663. False alarms alone predict 2.4 there and planets ~3.3, so
counts cannot tell those two apart. Individually, 353875121 is a proven blend and 294328887 and 233719663 look
like noise.

The survivors also look like false alarms rather than injected planets:

| median | survivors (23) | kept injected planets | far nulls not rejected |
|---|---|---|---|
| transits | 4 | 18 | 4 |
| real sectors | 7 | 31 | 6 |
| share on 13+ sector stars | 0.17 | 0.71 | 0.30 |
| depth (ppt) | 10.4 | 5.1 | 5.4 |

**Why 4x the pre-registered vetting load.** 733 of the 809 signals are tier C, rejected for exactly what vetting
exists to catch. The most common reasons:

| reason for rejection | signals |
|---|---|
| secondary eclipse | 330 |
| one or two transits carry the signal | 310 |
| a comparable dip elsewhere in the fold | 243 |
| only two transits | 239 |
| absent or different in SAP flux | 183 + 119 |
| transits at gap edges | 113 + 100 |

48 stars filled all six signal slots above threshold. Those stars are dominated by systematics or variability.
The far-null sample (300 stars) evidently under-sampled such stars. That is why the per-signal "realised mix"
expectation is the fair one.

## Known planets: did the survey find them?

Scripts: `spikes/mdwarf_stack/survey/known_check.py --vet-dir <OUT>/survey100/vet` → `known.csv` (66 catalogued
objects on the 48 vetted stars that host one: the TOI table refreshed 2026-11-16, live CTOIs and exoplanet.eu), and
`survey100/known_all.py` → `known_all.csv` (every TOI with P 10-60 d on any of the 15,452 searched stars).

- **Every TOI with a 10-60 d period on a searched star was found blind (6 of 6), all tier K:**
  - TOI-450.01 (10.71 d, MES 216)
  - TOI-7870.01 (11.92 d, 3.4 R_e, MES 27.4)
  - TOI-1433.01 (12.44 d, MES 165)
  - TOI-7694.01 (13.67 d, 2.1 R_e, MES 12.3)
  - TOI-176.01 (16.16 d, a TFOP false positive / EB)
  - TOI-5575.01 (32.07 d, 9.5 R_e, MES 189)

  CTOI 406276109.01 (7.55 d) was found at twice its period.
- **Short-period objects found at an integer multiple of their period** (the long search starts at 10 d):
  TOI-218.02 (×2), TOI-805.01 (×3), TOI-762.01 (×3), TIC 165227846 b / CTOI .01 (×5), TOI-7384 b (×2),
  TOI-7281.01 (×3), TOI-5268.01 (×5, TFOP false positive), TOI-122.01 (×2), TOI-1746.01 (×4) and .02 (×3),
  TOI-1448 b (×3), CTOI 345324572.01 (×4), TOI-4184.01 (×3) and TOI-6022.01 (×6). All went to tier K.
- **Not found, and why:** Kepler-1649 c (19.5 d, 1.06 R_e) has no usable TESS ephemeris, and its sibling
  Kepler-1649 b predicts MES 0.6, far below the threshold. No known 1-3 R_e planet at 15-50 d exists on any
  searched star (gate 3 also found none on its 22 TOI hosts), so the survey had no known small planet in its
  target range to re-find.
- **No survivor is a known object.** None of the 19 survivor stars has an ExoFOP TOI or CTOI, a NASA Exoplanet
  Archive entry, a Gaia NSS orbit or a Prsa et al. EB entry. TIC 193136669 is the known young star TWA 34, but it
  hosts no known planet.

## The 23 surviving signals, best first

All 19 stars were researched with `survey/research_candidates.py` (anonymous; cached in
`outputs/planets_spike/survey100/vet/research/`), and every source answered for every star:

- ExoFOP
- the NASA Exoplanet Archive
- SIMBAD
- Gaia DR3 (63″ cone, with RUWE, RV and neighbours)
- Gaia NSS orbits
- the Prsa et al. TESS EB catalogue

Further checks per star:

- **RVs** (`survey100/rv100.py`): LAMOST DR7/DR11, APOGEE DR17, GALAH DR4 and RAVE DR6 (10″ cones), with the Gaia
  mean RV counted as one more epoch.
- **Neighbour-resolved pixels** (`survey100/pixel_nb.py`): every Gaia source within 3 px.
- **Searched neighbours** (`survey100/blend_check.py`).
- **Dossier** (`survey/dossier.py`): implied planet, epochs, next dips and future sectors.

For the top signals there were also:

- **Shorter-period aliases** (`survey100/alias100.py`): P/2-P/5, injected at the raw depth.
- **Other reductions of the same TESS sectors** (`survey100/rereduce.py`): QLP, TGLC, TARS and our own TESScut
  photometry, each judged against a raw-depth injection in that product.
- **ZTF light curves of neighbours** (`followup/archive/ztf_neighbours.py`).

How to read the table:

- Radius uses the survey's box depth (after the detrend, crowding-corrected PDCSAP) and the TIC radius.
  The detrend shrinks long dips, so it is a lower bound.
- Teq assumes Bond albedo 0.3 and full redistribution. HZ is Kopparapu et al. (2013).
- "False alarms like it" is the number of null false alarms in the same (tier, transit count) cell, scaled to the
  15,452 searched stars. The first figure is from the gate-3 far nulls (A+ uses the A cell); the bracket is
  from the 50 pc v2 validation. The whole survey expects ~4.6 planets.
- Cards: `outputs/planets_spike/survey100/vet/cards/survey100/` (`survey-<TIC>-long_<k>.png`, `cards_batch_*.md`).

| # | TIC (signal) | T mag | d (pc) | P (d) | depth (ppt) | transits | tier | radius if real (R_e) / Teq (K) | binarity | other reductions | verdict | false alarms like it |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 230081659 #0 | 13.08 | 67 | 34.607 | 1.23 | 22 | A | 1.41 (1.67 at raw depth) / 235, cons. HZ | none: RUWE 1.19; APOGEE 2019 and Gaia mean RV agree to 0.4 km/s | QLP 3.8σ, TESScut 4.0σ present; TARS 3.0σ, TGLC 2.8σ at ~60% of the expected depth | open, weak; more likely noise | 23.6 [7, 52] (21.9) |
| 2 | 447890696 #0 | 13.54 | 76 | 16.837 | 19.0 | 3 | A+ | 5.5 (more if grazing) / 287, too hot | RUWE 1.395 (just under 1.4); 1 RV epoch | seen in all four: QLP 10.7σ, TARS 10.1σ, TGLC 12.2σ, TESScut 8.4σ | open: grazing eclipse or background star | 1.8 [0, 9.5] (2.0) |
| 3 | 353875121 #0 | 13.89 | 76 | 32.306 | 3.83 | 23 | B | 2.02 / 225 | none on the target | not run | **blend: ZTF shows a 20% eclipse on a background star 28″ away** | 5.4 (25.8) |
| 4 | 146725602 #0, #3 | 14.04 | 94 | 47.876 / 17.381 | 34.9 / 13.1 | 3 / 9 | A+, B | 7.6 / 199 | none: 1 RV epoch, RUWE 1.04 | not run | likely artefact or noise (5 signals on the star; dump) | 1.8 (2.0) / 5.4 |
| 5 | 294328887 #2 | 14.23 | 98 | 14.179 | 6.43 | 40 | A+ | 3.3 / 291, too hot | none known | not run | likely noise (six equal signals, too short) | 1.8 (21.9) |
| 6 | 171524608 #0 | 14.10 | 55 | 20.121 | 6.99 | 5 | A | 2.06 / 230, cons. HZ | LAMOST LRS + Gaia mean: no significant variation (errors large) | not run | likely false (centroid off, two transits dominate) | 23.6 (21.9) |
| 7 | 241626984 #1 | 14.58 | 97 | 17.163 | 81 | 4 | A+ | 11 / 265 | none known | not run | likely contamination (CROWDSAP 0.13-0.23, emission-line star) | 1.8 (13.9) |
| 8 | 233719663 #0 | 14.60 | 88 | 36.871 | 7.03 | 12 | B | 2.48 / 202 | none known | not run | likely false (weaker in SAP, one transit dominates) | 5.4 (25.8) |
| 9 | 181110184 #0 | 13.87 | 62 | 32.601 (or /n) | 23.7 | 2 | B | 4.4 / 210 | none known | not run | untested two-transit signal; TESS S111 re-tests it | 1.8 (9.9) |
| 10 | 423088347 #0 | 13.28 | 80 | 31.698 | 4.17 | 3 | B | 2.81 / 245 | none known | not run | likely false (one transit carries 77%) | 27.2 (17.9) |
| 11 | 88762886 #0 | 12.44 | 61 | 37.086 | 2.75 | 3 | B | 2.50 / 242 | Gaia RV constant; LAMOST 7 km/s off (p 0.016, below the rule) | not run | likely noise (at threshold) | 27.2 (17.9) |
| 12 | 193729289 #0 | 14.82 | 86 | 59.471 | 23.3 | 3 | B | 3.8 / 170, too cold | unresolved neighbour 4.0″, ΔG 0.5 | not run | likely noise or the neighbour | 27.2 (17.9) |
| 13 | 365466012 #1 | 12.66 | 84 | 39.805 | 10.4 | 4 | B | 5.9 / 249 | none known | not run | likely noise (red-noise strike) | 12.7 (13.9) |
| 14 | 122141722 #0 | 13.77 | 57 | 16.579 | 8.91 | 4 | B | 2.66 / 262 | none known | not run | likely neighbour (centroid off 4/4, CROWDSAP 0.4-0.5) | 12.7 (13.9) |
| 15 | 349307345 #4 | 13.76 | 95 | 24.760 | 14.7 | 9 | B | 5.3 / 254 | none known | not run | likely neighbour (13″ star 1.9 mag brighter has the larger pixel signal) | 5.4 (25.8) |
| 16 | 358232435 #3 | 13.69 | 85 | 10.034 (or /n) | 29.3 | 2 | B | 6.9 / 346 | none known | not run | likely neighbour (centroid off 2/2), two transits | 1.8 (9.9) |
| 17 | 193136669 #0, #1, #5 | 13.06 | 61 | 31.64 / 26.51 / 45.26 | 33-43 | 4-6 | B ×3 | — | young star (TWA 34, disk host) | not run | young-star variability, not transits | 12.7 / 5.4 |
| 18 | 280945693 #0, #1 | 12.27 | 98 | 10.807 / 15.920 | 5.0 / 6.2 | 14 / 8 | B ×2 | — | **binary: Gaia RV amplitude 83 km/s (p = 0)** | not run | binary star | 5.4 |
| 19 | 270465315 #2 | 12.06 | 59 | 47.630 | 1.88 | 3 | B | — | **binary: LAMOST MRS RV -121 → -19 km/s between 2018 and 2022; RUWE 1.64** (Wolf 129) | not run | binary star | 27.2 (17.9) |

### 1. TIC 230081659 (2MASS J16233268+6721310): tier A, 34.607 d

A 1.2 ppt (0.12%), 3 h dip at 22 transits from 2019-07-28 (S14) to 2024-11-16 (S85). The star has 42 real
sectors in the northern continuous viewing zone. MES is 8.71, just above the 8.05 threshold.

No transit stands out. Individual SES values run from 0.2 to 4.3, and the most dominant transit carries 24% of
the signal. That is how a real shallow planet looks, and also how a slightly lucky noise fluctuation looks. It
passes every v2 test with zero strikes. Odd/even z is 0.19, SAP agrees with PDCSAP (z_SAP -1.3), the duration
fits the star's density, and it is the only signal on the star above the threshold (the next is MES 6.1).

- **Star:** 3,361 K, 0.37 R_sun, 0.36 M_sun, 67.3 pc, T = 13.08. CROWDSAP 0.87-0.90.
  - APOGEE (one visit, 2019-06-15) confirms a single-lined M dwarf: Teff 3,358 K, logg 4.68, v sin i 1.5 km/s.
  - The APOGEE RV (-15.63 ± 0.08 km/s) and the Gaia DR3 mean RV (-15.2 ± 1.9) agree. Gaia RUWE is 1.19, with no
    NSS orbit and no EB entry.
- **If on this star:** 1.41 R_e from the detected depth. The detrend keeps only 72% of a 3 h dip of this depth
  (`s_det` 0.72 from injections), so the raw depth is 1.72 ppt and the radius is **1.67 R_e**. a = 0.147 AU,
  S = 0.72, Teq 235 K (257 K at A = 0), inside the conservative habitable zone.
- **Aliases:** P/2 (17.30 d), P/3, P/4 and P/5 are all excluded. The extra epochs P/2 adds (23 of them) show
  joint SES -1.8, against 5.8 expected for a real dip injected at the raw depth. 34.61 d is 2.53 × 13.7 d, not a
  TESS-orbit alias.
- **Other reductions of the same sectors** (joint over all evaluated transits, at the predicted times; the
  expected depth comes from raw-depth injections into each product):

  | product | measured depth | expected depth | SES | rule result |
  |---|---|---|---|---|
  | QLP | 0.69 ± 0.18 ppt | 0.98 | 3.8 | present |
  | our TESScut photometry (4 sectors) | 1.61 ± 0.40 ppt | 1.01 | 4.0 | present |
  | TARS | 0.61 ± 0.20 ppt | 1.08 | 3.0 | absent (just: 0.61 < 1.08 - 2σ = 0.68) |
  | TGLC (12 transits) | 0.65 ± 0.23 ppt | 0.93 | 2.8 | no information |

  Every product dips at the predicted times, but three of four at only ~60% of what a real dip of the
  survey's depth would give. All reductions share the same photon noise, so this cannot prove the signal is
  astrophysical. It mostly rules out a pipeline artefact. The shallower depths are what a detection pushed just
  over threshold by noise looks like (winner's curse).
- **Against it:**
  - The pixel test is uninformative (pixel SNR 3.3). No single transit is strong enough to locate.
  - A Gaia companion 9.1″ away (G 16.88, ΔG 2.57, same parallax 14.80 mas, so bound; SIMBAD emission-line star)
    would need only a 1.3% eclipse to make the dip. ZTF has 57 good points on it but none in transit, so that is
    not tested. The neighbour-resolved pixel check gives it S 1.1 against 3.3 on the target, too weak to decide.
  - Its cell (tier A, 5+ transits) is where the far nulls put most of their survivors: 23.6 [7, 52] false alarms
    like it per survey, against ~4.3 planets expected in all 5+ transit bins together.
  - The gate-3 injections recovered 0 of 31 planets of 1.0-1.5 R_e at 30-50 d even on 13+ sector stars. A real
    planet this small would have had to be lucky to be found. At 1.67 R_e it falls in the 1.5-2 R_e cell, where
    recovery on 13+ sector stars is 8/31.
- **Catalogues:** nothing. No TOI or CTOI, no ExoFOP observations, no NASA entry.
- **Next dips:** 2026-12-18 00:47, 2027-01-21 15:21 and 2027-02-25 05:57 UTC (BTJD 4392.534, 4427.141, 4461.749;
  ± 0.4-0.5 h). Dec +67°, circumpolar from mid-northern sites.
- **TESS:** predicted transits in S118, S119, S120, S122, S123, S124, S125, S126 and later (2027-28).
- **Card:** `survey-230081659-long_0.png`. Its MES-vs-phase panel has a spike at phase -14.5 d. That is a
  plotting artefact of the panel, not a second dip; the fold and the uniqueness test show none.

### 2. TIC 447890696 (2MASS J10315803-4954073): tier A+, 16.837 d

A 19 ppt (1.9%) dip seen on three consecutive orbits in March-April 2021 (S36 and S37): BTJD 2288.41, 2305.25
and 2322.09, with SES 7.6, 8.8 and 6.2. The star has 3 real sectors (S10, S36, S37). It passes every v2 test and
the frozen pixel test (pixel SNR 12.7, neighbour 2.2).

- **Star:** 3,238 K, 0.36 R_sun, 0.35 M_sun, 75.5 pc, T = 13.54, G = 14.84. Gaia RUWE 1.395, just under the
  usual 1.4 binarity flag. Gaia RV 13.5 ± 3.5 km/s; there is only one RV epoch, so binarity is not tested.
- **It is in the pixels.** QLP (SES 10.7), TARS (10.1), TGLC PSF photometry (12.2, depth 18.6 ± 1.5 ppt against
  19.5 expected) and our TESScut aperture (8.4) all show it at the predicted times. Only the TGLC and survey depths
  match the expected depth within the pre-set "present" rule. The uncorrected apertures (QLP, TARS, TESScut) show
  it **deeper** than the crowding-diluted expectation (15.6-21.0 ppt against 12.2-12.9). That is not evidence
  against it; the rule was written with "too shallow" in mind.
- **Aliases:** P/2 to P/5 are excluded (P/2 adds 3 covered epochs, joint SES -0.5 against 10.4 injected). Three
  consecutive transits fix P = 16.837 d.
- **Against it:**
  - **Too short for the star.** The box duration is 1.0 h, and the folded dip is V-shaped, about 1 h wide.
    A central transit across this star at 16.8 d would last 2.4 h. A planet would have to cross near the limb
    (impact parameter ≳ 0.9), and a grazing planet would be larger than 5.5 R_e.
  - **Background stars inside the target pixel.** Gaia sources at 1.7″ (G 20.7), 5.0″ (G 19.1), 8.4″ (G 17.9)
    and 9.8″ (G 17.05, ΔG 2.2, parallax 0.19 mas, i.e. distant) are all unresolved by TESS's 21″ pixels. The 9.8″
    star would need a 14.5% eclipse, which is easy for an eclipsing binary. The neighbour-resolved pixel check gives
    it S 12.0 against 12.7 on the target: indistinguishable. The difference-image centroid lands nearest a
    different faint neighbour in each epoch.
  - Only 3 transits, all in one 34-day stretch.
- **If on this star:** 5.5 R_e (more if grazing), a = 0.091 AU, S = 1.60, Teq 287 K. Hotter than the habitable
  zone (1.8x its inner edge).
- **Catalogues:** nothing.
- **TESS is watching now.** S110-S113 (Oct 2026-Jan 2027) contain predicted dips at BTJD 4359.3, 4376.2, 4393.0,
  4409.9, 4426.7 and 4443.5. The timing uncertainty is ± 22 h, because three transits over 34 days do not pin the
  period for 2,000 days. Those sectors will exist only as QLP, TGLC, TARS or TESScut products (TESS-SPOC FFI light
  curves stop at S86).
- **Next dips:** 2026-12-01 16:19, 2026-12-18 12:24 and 2027-01-04 08:30 UTC (± 22 h). Dec -50°, southern.
- **Card:** `survey-447890696-long_0.png`.

### 3. TIC 353875121 (2MASS J18051671+5946519): tier B, 32.306 d. A blend, solved.

A 3.8 ppt, 2.2 h dip at 23 transits over 39 sectors (2019-2024), MES 13.5. It was the strongest small-planet-like
signal of the survey: 2.0 R_e and S = 0.61 if on the target, evenly spread over its transits (the most dominant
carries 15%), and unique on its star.

- **Pixels.** The frozen pixel test found the difference-image centroid more than 1 px off the target in 4 of
  4 strong epochs. That is the one strike that made it tier B. The neighbour-resolved check puts the centroid
  1.2-4.7″ from Gaia DR3 2152640018233483392 in all four epochs. That star sits 27.6″ away (G 17.18, BP-RP 0.74,
  parallax 0.10 mas: a distant F/G star) and has pixel S 21.1 against 15.8 on the target. It would need a 2.5% eclipse
  if all its light were in the aperture.
- **ZTF settles it.** That neighbour has 3,037 ZTF points (2018-2025). At the TESS ephemeris, 11 points on 6
  nights (BTJD 1268.9, 1688.9, 1915.0, 2819.7, 3239.6, 3885.7) are 0.19-0.29 mag faint, against a scatter of
  ~17 ppt: a 20% eclipse at SES 38. The 2.5% "needed" figure assumes all of its light is in the TESS aperture;
  at 28″ only a fraction is. **The dip comes from a background eclipsing binary.** The other three neighbours
  tested in ZTF show nothing.
- **Code note:** `ztf_neighbours.py` first called this "no information", because its "blend supported" rule
  demanded a depth *equal* to the needed depth. Fixed: a deeper eclipse than needed now counts as support. See
  *Code fixes* below.

### 4. TIC 146725602 (2MASS J02590146-4232204): tier A+, 47.876 d (and a B at 17.38 d)

A 35 ppt, 3 h dip, three transits: 2018-09-28 (S3, SES 11.1, **on a momentum dump**), S31 (SES 1.1, at a data
edge) and 2026-06-23 (S105, SES 11.5). In practice it rests on two epochs, one of them at a dump. The star has
**five** signals at or above the threshold between 17 and 52 d. That pattern belongs to the star's noise, not to
one planet. The 17.38 d B signal (#3, 13 ppt, SAP soft strike) is one of them and gets its strength from S97/S105.

- **Star:** 3,165 K, 0.37 R_sun, 94 pc, T = 14.04. A wide common-proper-motion companion, TIC 146725601 (Gaia
  4755745925980823296), sits 31″ away (G 14.18, same parallax 10.6 mas). It was searched too and shares no epochs.
  Its pixel S is 6.8 against 37.9 on the target, so the dip is not on the companion.
- **If real:** 7.6 R_e, S = 0.37, Teq 199 K. That is a Neptune/Saturn-size object, not an Earth.
- **Aliases** P/2-P/5 are excluded. There is no TESS sector through S134.
- **Next dips:** 2026-12-31 23:55, 2027-02-17 20:57 and 2027-04-06 17:58 UTC (± 0.8 h). Dec -43°.
  **A 3.5% dip on a T = 14 star is visible to a 0.4-1 m telescope in one night.** This is the cheapest kill test
  on the list.

### 5-19. The rest (likely false or already explained)

- **TIC 294328887, A+, 14.18 d, 40 transits, 6.4 ppt.**
  - The star has six near-equal signals (a noise floor).
  - The strength comes mostly from S5, where one epoch sits on a data edge with no gap margin.
  - The 1.0 h dip is less than half a central transit (2.3 h).
  - P/4 and P/5 epochs show positive SES (4.9 and 6.9): power at shorter periods.
  - A Gaia star 24″ away and 1.0 mag brighter has pixel S 27.6 against 30.8 on the target; it would need only a
    0.2% eclipse.
  - 14.18 d is 3.5% from 13.7 d.
- **TIC 171524608, A, 20.12 d, 5 transits.**
  - Two S44-45 epochs carry the signal; the S46/S47/S71 ones are SES 1.8-2.5.
  - The pixel centroid is off target in 2 of 2 strong epochs (below the frozen 3-epoch rule, so no strike).
  - One LAMOST low-resolution RV (2011-12, 32.8 ± 3.1 km/s) and the faint star's Gaia mean RV (error ~10 km/s)
    differ by 11 km/s, which is not significant (p = 0.25). That is too little for a binarity test.
- **TIC 241626984, A+, 17.16 d, 81 ppt (11 R_e if real).**
  - CROWDSAP is 0.13-0.23: most of the aperture's light is from other stars. Eight Gaia sources within 63″ are
    brighter than the target, one by 7.7 mag.
  - Six deep signals (38-104 ppt) sit on the star.
  - SIMBAD lists it as an emission-line star.
  - TESS S112 and S114 will re-test it.
- **TIC 233719663, B, 36.87 d, 12 transits on 37 sectors.**
  - The dip is weaker in SAP than in PDCSAP (z 4.3, soft).
  - One transit carries 61%.
  - The pixel check is soft.
  - TESS continues 2027-28 (S117+).
- **TIC 181110184, B, 32.60 d (2 transits, so P/n possible), 24 ppt.**
  - Untested. TESS S111 covers a predicted dip (BTJD 4373.7 ± 2.5 h, 2026-11-29), and S113/S114 more.
  - Re-run when those data are public.
- **TIC 423088347, B, 31.70 d, 3 transits.** One transit carries 77% (dom 0.77). An XMM X-ray source sits 14″
  away.
- **TIC 88762886 (LSPM J1727+4051), B, 37.09 d, 3 transits.**
  - MES 8.18, 0.13 above the threshold (low_margin strike).
  - Pixel SNR 4.2, uninformative.
  - Gaia's own RV series is constant (rv_chisq_pvalue 0.95 over 18 transits).
  - LAMOST's low-resolution RV (2013-04, -38.2 ± 2.5 km/s) differs from the Gaia mean (-45.4 ± 1.7) by 7 km/s
    (p = 0.016, below the pre-set p < 0.0027). That is a hint at most, and LAMOST low-resolution RVs of M dwarfs
    carry offsets of several km/s.
- **TIC 193729289 (LP 180-38), B, 59.47 d, 3 transits.**
  - At threshold.
  - A Gaia star 4.0″ away and only 0.5 mag fainter is unresolved and could host it.
- **TIC 365466012, B, 39.80 d, 4 transits.**
  - Red-noise strike.
  - A neighbour 1.7″ away (ΔG 3.9) is unresolved; it would need a 36% eclipse.
- **TIC 122141722, B, 16.58 d.**
  - The pixel centroid is off target in 4 of 4 strong epochs.
  - CROWDSAP 0.41-0.50. Two brighter stars sit at 41″ and 45″, and two of similar brightness (ΔG 0.4-0.5) at 11″
    and 16″.
- **TIC 349307345, B, 24.76 d, 9 transits.**
  - TIC contamination ratio 0.98; CROWDSAP 0.18-0.27.
  - A star 13″ away and 1.9 mag brighter has pixel S 41.1 against 38.2 on the target.
  - The centroid is nearest another Gaia source in 5 of 5 epochs.
- **TIC 358232435, B, 10.03 d, two transits (S29).** The pixel centroid is off target in 2 of 2 epochs; 29 ppt.
- **TIC 193136669 = TWA 34, three B signals (26.5, 31.6, 45.3 d; 33-43 ppt; secondary-eclipse soft strikes).**
  - A ~10 Myr TW Hydrae association M dwarf (42 SIMBAD references) known to host a disk.
  - Deep, repeating but inconsistent dips are typical of such stars.
- **TIC 280945693, two B signals (10.81 and 15.92 d).**
  - Gaia DR3 RV amplitude 83 km/s with rv_chisq_pvalue 0: a spectroscopic binary.
  - Gap-edge strikes on both signals.
  - SIMBAD calls it a possible young star.
- **TIC 270465315 = Wolf 129, B, 47.63 d.**
  - 27 LAMOST MRS exposures on two nights: RV -121 km/s (2018-12) and -19 km/s (2022-12). A spectroscopic binary.
  - RUWE 1.64.

Pixel check outcomes that moved signals to tier C:

- 157647954 #0 (a neighbour 55″ away with SNR 270 against 42)
- 89787749 #1
- 299486246 #1 (a neighbour 45″ away)
- 47761731 #0
- 144325909 #0
- 371108287 #0 (a neighbour 36″ away, SNR 97.6 against 66.0)
- 391745863 #0 (a neighbour 58″ away, 63.3 against 22.8)

## Community-TOI bar

The bar has four parts:

1. an informative on-target pixel result on 3+ strong transits;
2. no evidence of a binary;
3. the dip present in at least one independent reduction;
4. nothing in the catalogues that explains it.

| signal | (1) on-target pixels | (2) no binary evidence | (3) independent reduction | (4) catalogues | result |
|---|---|---|---|---|---|
| 230081659 #0 | **no**: no strong transit (SES ≤ 4.3), pixel SNR 3.3 | yes (RUWE 1.19; APOGEE + Gaia RV agree) | partly: QLP 3.8σ, TESScut 4.0σ; TARS formally absent | clean | **fails** |
| 447890696 #0 | **no**: 3 strong transits pass the frozen test, but a background star 9.8″ away has the same pixel signal | no RV test; RUWE 1.395; the 1 h V-shaped dip suggests a grazing eclipse | yes: all four products, SES 8-12 | clean | **fails** (closest) |
| 353875121 #0 | no: centroid on a neighbour | — | — | — | **fails; a blend (ZTF)** |
| 146725602 #0 | no: 2 strong transits, one on a momentum dump | no test | not run | clean | **fails** |
| all others | no | — | — | — | fail |

**No CTOI is drafted.** TIC 447890696 would come close to the bar if two things happened:

- the S110-S113 data, now being taken, show the dip at the predicted times, giving a tight ephemeris;
- seeing-limited photometry puts a 2% dip on the target rather than a 15% eclipse on the 9.8″ star.

Its 1 h V shape would still have to be argued in the submission as a grazing planet or an EB.

## What to do next

1. **Submit nothing to ExoFOP.**
2. **TIC 447890696: wait for TESS S110-S113**, released through early 2027. Measure the new transits in QLP or
   TESScut with `survey100/rereduce.py`'s method, then refit the ephemeris. If the dips are there, a 1-m class
   southern telescope (Dec -50°) can tell a 2% dip on the target from a 15% eclipse on the 9.8″ star in one transit.
3. **TIC 146725602:** the cheapest test.
   - Watch a predicted 3.5%, 3 h dip with a small southern telescope: 2026-12-31 23:55, 2027-02-17 20:57 or
     2027-04-06 17:58 UTC (± 0.8 h).
   - A flat light curve kills it.
4. **TIC 230081659: let TESS test it.**
   - S118-S126 (2027-28) add ~9 transits; re-run the frozen search then.
   - A real 1.2-1.7 ppt dip would grow to MES ~12. Noise would fall back.
   - From the ground, a 1.7 ppt, 3 h dip on a G = 14.3 star needs a 2 m-class telescope; not worth it before TESS.
   - Next dips: 2026-12-18 00:47, 2027-01-21 15:21, 2027-02-25 05:57 UTC (± 0.5 h).
5. **TIC 181110184 and 241626984:** re-run the frozen search when TESS S111-S114 are public.
6. **TIC 353875121 needs nothing more:** ZTF shows the eclipsing neighbour.

## Not checked or not possible

**Follow-up data**

- No ExoFOP spectroscopy, imaging or ground photometry exists for any of the 19 stars (all queries succeeded).
  On-target status is unproven for every signal.
- **RVs.**
  - Two or more RV epochs exist for only 5 stars: 270465315 (LAMOST MRS, a binary), and 230081659, 353875121,
    171524608 and 88762886 (one survey spectrum plus the Gaia mean).
  - Gaia RV statistics (rv_chisq_pvalue, rv_amplitude_robust) exist only for G_RVS ≤ 12, so for 2 of the 19 here.
  - The ESA Gaia TAP timed out from this machine. Gaia values come from the VizieR copy (I/355) instead.
    The TAP was the only source tried for the companion TIC 146725601, so its Gaia statistics were not checked.

**Neighbour and pixel tests**

- **ZTF** was tried only for 230081659's 9″ companion (0 in-transit points: not tested) and for 353875121's
  neighbours. Four of 353875121's fainter neighbours were not fetched (per-neighbour cap). The southern stars
  (447890696, 146725602 and others) have no ZTF coverage.
- The pixel test is still uncalibrated with pixel-level injections. A "pass" is weak evidence, and it can never
  separate sources closer than ~10″.

**Other reductions and aliases**

- Other reductions (`rereduce.py`) were run only for 230081659 and 447890696.
- TESScut photometry was limited to the 3-4 sectors holding the strongest transits.
- The alias test was run for 6 keys (the open signals and the A+/A ones).

**What the survey did not cover**

- 4,375 of the 19,827 stars (22%) have no SPOC/TESS-SPOC light curve and were not searched. 41 stars lost one or
  two sectors to unusable PDCSAP flux and were not re-searched with SAP.
- TESS-SPOC FFI light curves at MAST stop at S86, so S87+ sectors are missing except where SPOC 2-min data exist
  (3,681 star-sectors from S87+ were used).

## Ideas for vetting v3 (not applied here; v2 is frozen)

- **Neighbour lists in `vetting/catalog.py` are wrong for vetted pairs.** The batch Gaia query assigns each row to
  the nearest vetted target. So a neighbour that is itself a vetted target disappears from the other star's list,
  and from the frozen pixel test's choice of neighbour. This affects 3 pairs here (32090581/32090583,
  146725601/146725602, 452522877/452522881). `survey100/pixel_nb.py` works around it with the per-star cone.
- **Test every Gaia neighbour within 3 px**, not one at 1.5-3 px. Make "the difference-image centroid sits on the
  same neighbour in every epoch" a strike. That would have made 353875121 tier C on its own.
- **Use archival ground photometry of neighbours (ZTF) inside vetting** where it exists. One query solved the
  survey's best-looking small-planet signal.
- **Spectroscopic binarity as a strike:** a Gaia RV amplitude > 10 km/s with p < 0.01, or multi-epoch LAMOST
  variation (270465315, 280945693; and TIC 2102868 in the 50 pc survey).
- **Flag young-association members** (TWA 34) and stars where 5-6 signals sit above the threshold (146725602,
  294328887, 241626984): "signal not unique on its own star".
- **Flag dips much shorter than a central transit** (447890696 at 0.42, 294328887 at 0.43 of the central duration)
  as possible grazing EBs or blends.
- **In `rereduce.py`'s "present" rule, judge uncorrected apertures one-sided.** A dip deeper than the
  crowding-diluted expectation (447890696 in QLP/TARS/TESScut) is not a mismatch.
- **Sort future sectors by time, not number:** `dossier.py` lists S1751 (January 2026) as a future sector.

## Code fixes made during this vetting

All are outside the frozen search and the frozen v2 vetting.

- **`followup/archive/ztf_neighbours.py`:** "blend supported" now accepts an eclipse deeper than needed. Before,
  `|depth - req| < 2σ` called a 20% eclipse "no information". Earlier ZTF results (50 pc follow-up) used the old
  rule and were not re-run. They can only have missed support, never invented it.
- **`survey100/rv100.py`:** APOGEE `allvis` "RV" is VREL (not barycentric); VHelio is used instead, and the
  combined `catalog` row is skipped. The first pass flagged TIC 230081659 as an RV binary for this reason. LAMOST
  LRS RVs are read from c·z with an S/N cut, and the same exposure listed in several tables or in both DR7 and
  DR11 is counted once.
- **`survey100/alias100.py`:** injects at the raw depth from `rereduce.py` when known. At the detected depth,
  230081659's P/2 alias came out "not excluded" (injected SES 3.6 instead of 5.8).
- **`survey100/health100.py`:** the per-shard `ok` share range used min(ok)/max(n); it now uses min(ok/n).
- Earlier in this run: `survey/prep_vet.py --log`, `gate3/pixel3.py` (PIXEL3_INFO/PIXEL3_OUT),
  `survey/known_check.py --vet-dir/--targets`, `survey/research_candidates.py` (RESEARCH_DIR/RESEARCH_INFO) and
  `survey/dossier.py` (DOSSIER_VET/DOSSIER_TARGETS/DOSSIER_NOW, CROWDSAP from `sap_meta.json`). These are
  environment or path options only; defaults are unchanged.

## Reproduce

From `spikes/mdwarf_stack/`, with `PY` = the repo venv, `O = <OUT>/survey100`, `V = $O/vet`, and every heavy step
under `nice -n 19 ionice -c3` (at most 2 processes):

```
$PY -I survey100/health100.py                                   # health.txt (before any signal)
$PY -I survey/vet_rows.py --merged $O/merged/results.jsonl.gz --out $V   # rows_long / jobs_long / tics_long
$PY -I gate3/catalog3.py --targets <OUT>/gate3/targets_100pc.csv --tics $V/tics_long.txt --out $V/star_info.json
$PY -I survey100/seed_products.py                               # product lists from the survey records
bash survey100/pipeline.sh $V/batch_00 $V/batch_01 $V/batch_02  # prep -> vet100 (2 shards) -> pixel3 -> cards -> delete LCs
$PY -I survey100/tables100.py                                   # tiers, expectations -> tables.txt, graded.jsonl
$PY -I survey/known_check.py --vet-dir $V --targets <OUT>/gate3/targets_100pc.csv ; $PY -I survey100/known_all.py
RESEARCH_DIR=$V/research RESEARCH_INFO=$V/star_info.json $PY -I survey/research_candidates.py TIC ...
DOSSIER_VET=$V DOSSIER_TARGETS=<OUT>/gate3/targets_100pc.csv DOSSIER_NOW=4360.5 $PY -I survey/dossier.py KEY ...
$PY -I survey100/blend_check.py KEY ... ; $PY -I survey100/pixel_nb.py KEY ... ; $PY -I survey100/rv100.py TIC ...
MAX_TESSCUT=4 $PY -I survey100/rereduce.py KEY ; $PY -I survey100/alias100.py KEY ...
$PY -I survey100/ztf_prep.py KEY ; $PY -I followup/archive/ztf_neighbours.py $V/ztf_ephem.json $V/dossier.json \
    <DATA>/ztf100 $V/ztf_neighbours_<TIC>.json TIC --skip-cone
$PY -I survey/phone_plot.py TIC P T0 DUR_H OUT.png --title ... --sub "...|..."
```

Logs: `$V/pipeline.log`, `batch_*.{prep,vet0,vet1,pixel,cards}.log`, `rereduce_*.log`, `alias100.log`,
`rv100.log`, `pixel_nb*.log`, `ztf*.log`, `dossier.log`. The TOI table was refreshed on 2026-11-16
(`data/planets_spike/catalogs/toi_2026-11-16.csv`, 8,175 rows), and the CTOI cache on the same date. The
light curves, SAP files, ZTF downloads and cutouts were deleted after use.
