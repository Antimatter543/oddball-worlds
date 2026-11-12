# Track C vetting: from raw long-period detections to graded candidates

Status: **pre-registered 2026-11-12, before any vetting statistic was computed on any signal**, and before
any survey candidate exists. Calibration uses gate-2 material only (`docs/spikes/mdwarf_stack.md`). Nothing under
`outputs/planets_spike/survey/` or `data/planets_spike/survey/` is read by this work.

## Why
Gate 2 measured detections. Its long-range threshold tau_long = 8.054 leaves a per-star false-alarm rate on real
stars bounded only at < 9% (1 of 58 TEST real controls), so a ~4,700-star survey can expect ~15-430 false alarms
above tau_long, piling up at 45-55 d with 2-4 transits. These look like the temperate planets we are searching for.
The owner cannot judge light curves by eye. The vetting therefore has to be explicit, automatic and calibrated, and
it has to say what it cannot rule out.

## Material and split (fixed now)
Everything is re-derived deterministically from the gate-2 job specs plus the stored light curves: the vetting takes
a star's light curve (re-injected or re-transformed exactly as the gate-2 runner did) and the signal the frozen
search reported (P, t0, box width, epochs). The search is not re-run on gate-2 jobs.

| set | true signals | false alarms |
|---|---|---|
| **DEV** (thresholds chosen here) | gate-2 DEV harm-arm injections (`dev_paired`, 120) plus 200 new DEV injections (`vetting/make_jobs_v.py`, 1.0-2.0 R_e, 15-55 d, 6+ sector DEV stars, seed 20261201, frozen gate-2 search and runner, one worker). Used: top signal matches the injection (gate-2 match rule) and MES >= 7.0. | DEV long nulls (420; top signal, MES >= 6) and DEV real-control long signals (harm, up to 3 per star, MES >= 6). |
| **TEST** (measured once, after the freeze) | TEST harm-arm injections (436, `test_inj`); the headline cell is cell 0 (1.0-1.5 R_e, 15-30 d, 6+ sectors). | TEST real-control long signals (`test_real`, up to 3 per star) and **TEST long nulls** (new: 4 per TEST star, inv0 / invroll11 / roll21 / roll22, frozen search, `vetting/make_jobs_v.py`). |
| **hold-out** (measured once, after the freeze) | gate-2 hold-out planets recovered blind (`holdout_result.csv`, rec = True: 28) and the 3 gate-1 hosts' long signals. | - |

The TEST nulls and the extra DEV injections were started (search only) before this file was committed; no vetting
statistic had been computed on anything at that point, and the TEST-null outputs are not opened until the TEST stage.

Why MES >= 7.0 for DEV true signals rather than >= tau_long: the hard case is MES 8-12, and thresholds that keep
signals at 7-8 also keep them at 8-12. Why MES >= 6 for DEV false alarms: only 3 DEV nulls exceed tau_long (one
star), so the structure of the tail is studied on its shoulder (155 nulls and 39 real signals at MES >= 6); TEST
results are reported at MES >= 6, >= 7.0 and >= tau_long separately.

## Population-dependent applicability (fixed now)
- Real stars and injections: every test.
- Inverted nulls (`inv`): every light-curve test; SAP and raw flux are inverted the same way (2 - f), so time-domain
  structure (edges, dumps, scattered light) is kept. Pixel, contamination and catalogue tests are N/A (an inverted
  dip is a real brightening in the pixels).
- Rolled nulls (`roll`, `invroll`): only tests on the search-input series (odd/even, per-transit, sector
  consistency, duration, secondary, model comparison, uniqueness). Time-anchored tests are N/A, because the rolled
  flux at time t came from another time.
- N/A counts as *not a strike*. For false alarms this makes the measured rejection a **lower bound** (those tests
  might have rejected more). Every table states the applicable n per test.

## The tests (each returns pass / soft / fail, or n/a, error, info)
Notation: D = the detected box width; an *epoch* is a transit window with >= 50% data coverage (the search's
definition); per-epoch depth d_e = N_e/D_e and SES_e = N_e/sqrt(D_e) from the search's own stride-1 SES (per-sector
red-noise rescaled), so the numbers are on the same scale as MES. Default thresholds are written as soft / fail.

1. **n_transits** (info, feeds the tier): number of epochs. 2 transits = period ambiguous by construction (the true
   period may be P/n), so a duo can never be tier A.
2. **odd_even**: z = |d_odd - d_even| / sqrt(s_odd^2 + s_even^2). An EB at twice the period has unequal alternate
   eclipses. Soft z > 3, fail z > 4. N/A if one parity has no epoch (a duo with consecutive epochs has one of each).
3. **per_transit**: (a) dominance = max SES_e^2 / MES^2 (one event carrying the detection; the search caps this at
   0.8); (b) chi^2 of the d_e about their weighted mean, p-value with n-1 dof (each transit individually consistent);
   (c) the weakest epoch's SES (an epoch with no dip where one should be). Soft if dominance > 0.65 or p < 0.01 or
   min SES_e < -1; fail if p < 0.001 or min SES_e < -2.
4. **sector_consistency**: chi^2 p-value of per-sector depths (sectors that hold transits; needs >= 2 such sectors,
   else n/a). Soft p < 0.01, fail p < 0.001. *Duration* consistency across sectors is computed (best box width in
   the early vs late half of the transits) but is **info only**: at MES 8-12 the per-half width is noise. It is not
   a strike, and the doc says so here so it is not mistaken for a wired test.
5. **duration_density**: the best box width at fixed P (scan 2-36 bins, centre +-D/2) divided by the central
   circular T14 for the TIC mass and radius. Too long a transit for the star's density means a larger star (blend),
   an eccentric orbit, or noise. Soft ratio > 1.3, fail ratio > 2.0. (No lower cut: grazing transits are short.)
6. **sap**: the same rotation removal and gate-1 preprocessing applied to SAP; depth at the same epochs.
   Expected SAP depth = CROWDSAP x PDCSAP depth (PDC removes the crowding dilution).
   z_sap = (d_sap - c d_pdc) / s_sap. Soft z_sap < -3 (dip much weaker in SAP: a PDC artefact), fail z_sap < -4.
   Also the **background**: SAP_BKG (fraction of the star's flux) averaged in transit vs the local out-of-transit
   level; a background *rise* matching the dip (z_bkg > 4, and the rise > 50% of the dip) is a soft fail (scattered
   light or an object in the background annulus makes SAP dim).
   Injections are also injected into SAP, diluted by CROWDSAP (a planet on the target).
7. **systematics_window**: each epoch is flagged if its window (+-D/2 plus 1 h) holds a momentum dump (QUALITY
   bit 32, which the binned QUALITY OR keeps even for masked cadences), scattered-light bits (2048, 4096) in the
   transit bins, or lies within 0.5 d of a data-segment edge (gap > 0.5 d). share = sum of SES_e^2 over flagged
   epochs / sum over all (positive SES only). Soft share > 0.5, fail share > 0.8. Per event, on PDCSAP, as
   CLAUDE.md requires (per-star SAP shares flag most quiet stars).
8. **orbit_alias**: (a) P within 2% of n x 13.70 d (n = 1-4: 13.7, 27.4, 41.1, 54.8 d); (b) time of each epoch
   since the start of its data segment; spread = max - min over epochs. Spacecraft-orbit systematics recur at the
   same orbit phase. Soft: spread < 0.5 d; fail: spread < 0.5 d and (a). (a) alone is info (a real planet may sit at
   27 d).
9. **secondary**: the MES fold at P and D over all phases (search-input SES at stride 1, the primary +-max(2D, 0.03P)
   excluded). z_sec = (MES_sec - median) / robust sigma of the other phases; E = sqrt(2 ln(P/D)) is the expected
   maximum of that many independent phases. Soft z_sec > E + 1, fail z_sec > E + 2. A significant secondary dip
   means an EB (or the wrong period).
10. **model_compare**: windows of +-max(3D, 0.3 d) around every epoch, search-input flux, a free offset per epoch.
    Models: box (common depth at t0, D); step (common jump at mid-transit, persisting to the window edge); sinusoid
    at the star's rotation period P_rot/h, h = 1-3, coherent over all windows (P_rot = median LS peak period of
    the harm census, measured on every star whether or not harm was applied); single outlier (one in-transit
    bin, location free, +ln N_in in the penalty). dBIC = min over alternatives of BIC_alt - BIC_box.
    Soft dBIC < 6, fail dBIC < 0 (an alternative explains the dip at least as well).
11. **uniqueness**: from the same fold: MES_pos = the largest *brightening* at any phase (the inverted series at the
    same P, D), MES_2nd = the largest other dip; margin = MES - max(MES_pos, MES_2nd). Plus 100 phase scrambles
    (each data segment of the SES series circularly shifted independently, seed fixed per signal), refolded at P;
    p_scr = fraction whose best phase reaches MES. Soft margin < 2.0 or p_scr > 0.01; fail margin < 1.0 or
    p_scr > 0.05. A detection that the same period's noise reproduces is not unique.
12. **rotation_residual**: raw flux after rotation removal (no biweight), segment medians removed, transits masked:
    least-squares sinusoid power at P and P/2 compared with the same power at 100 random periods in 10-60 d (fixed
    seed): p_rot = rank fraction. Soft p_rot < 0.02 (the star's out-of-transit flux varies at the signal period,
    so spots may make the dip). Never a hard fail (red noise can do this to real planets).
13. **pixel**: TESScut FFI cutout (11x11) for each sector with an epoch (up to 4 sectors, the highest-SES epochs),
    difference image per epoch (`oddball.gallery.pixels.analyse`); combined target-3x3 SNR S_t = sum/sqrt(n), and
    the combined SNR at the brightest Gaia neighbour >= 1.5 px away (WCS per sector), S_nb. Expected pixel SNR is
    low for 1-2 R_e planets, so: **uninformative** (n/a) if max(S_t, S_nb) < 3; fail if S_nb >= 3 and
    S_nb > S_t + 2 (the dip is on a neighbour); soft if the flux-weighted difference centroid is > 1 px off target
    in at least half of the epochs with peak SNR >= 4; else pass. Cutouts are deleted after use.
14. **contamination**: Gaia DR3 sources within 21" (one TESS pixel) of the target (target identified by
    parallax > 10 mas and G closest to Tmag + 1.6 within 20"; none found = error). A neighbour with flux ratio
    r = 10^(-0.4 dG) (G band: conservative for red targets) could make the observed depth with an intrinsic
    eclipse of depth/r. Soft if any neighbour needs <= 50% (a plausible blended EB). Info: the same within 42".
15. **catalogue**: known planets (local exoplanet.eu copy), TOIs (local ExoFOP copy, 2026-11-11), CTOIs (live
    ExoFOP download), EB catalogues (Kostov+2025 J/ApJS/279/50 within 21", Gaia DR3 I/358/veb within 5", VSX
    within 21" with an eclipsing type). Planet / TOI / CTOI on the same star at P or an alias (n <= 6, 1%) ->
    tier K. EB at P or an alias -> fail; EB at another period -> soft. **A query that fails returns `error`,
    never "not known"**; an error blocks tier A and is printed in the summary.

## Calibration rule (DEV only, mechanical)
For each thresholded statistic, using the DEV true signals: if more than 3% of them would get a soft fail at the
default, the soft threshold is loosened to the value that 97% pass; if more than 1% would get a hard fail, the hard
threshold is loosened to the value that 99% pass (and never stricter than the soft one). Thresholds are **never
tightened** on DEV: tightening by looking at the DEV false alarms is the easiest way to overfit 20-odd tail events.
The DEV false-alarm power of every test is reported, including tests with no power. `vetting/freeze_vet.py` applies
the rule and writes `vetting/frozen_vet.json`; it refuses to run if any TEST or hold-out vetting output exists. The
frozen file is committed before the first TEST vetting run. Nothing changes after.

## Tiers (fixed now)
- strikes = number of soft fails + 1 if n_transits = 2.
- **K**: the catalogue test matched a known planet, TOI or CTOI (overrides; all tests are still shown).
- **C (likely false alarm)**: any hard fail, or strikes >= 2.
- **A**: strikes = 0, n_transits >= 3, no required test returned `error` (required: everything except pixel n/a).
- **B**: everything else (one strike, or a clean candidate whose checks are incomplete).

What each tier still lacks to be an ExoFOP planet candidate (written into every summary):
- A: no reconnaissance spectroscopy (an SB2 or a giant would kill it), no high-resolution imaging (a blend inside
  a few arcsec is invisible to TESS pixels), no ground-based transit to confirm the ephemeris on target, no
  statistical validation (TRICERATOPS-style FPP). For a 1-2 R_e planet the pixel test is usually *uninformative*,
  so on-target is unproven. Tier A is a "community TOI-grade" candidate at most.
- B: everything A lacks, plus the named strike: for a duo, the period (P/n aliases) is unknown and the next
  transit cannot be predicted uniquely.
- C: not a candidate. It is listed with the failed test.
- K: already known; the value is an independent recovery, not a discovery.

## Targets (TEST and hold-out; goals, not tuning knobs)
1. >= 80% of *detected* TEST injections in the headline cell (top signal matched, MES >= tau_long) in tier A or B.
2. >= 90% of false alarms above tau_long (TEST real-control signals and TEST null top signals, MES >= tau_long) in
   tier C. Because that n is tiny, the same is reported at MES >= 7.0 and >= 6 (TEST), and on the DEV tail.
3. >= 90% of recovered hold-out planets in A, B or K. Because K alone would satisfy this trivially, it is also
   reported with the catalogue test switched off (the **blind tier**), which is the meaningful number.
Wilson 95% intervals throughout. If a target is missed, the report says by how much and which test caused it.

## Projection: surviving false alarms in the survey
N_surv = 4,700 x r x s. r = per-star rate of long-search signals >= tau_long on real controls (DEV + TEST real:
gate 2 found 0 of 60 and 1 of 58). s = fraction of TEST false alarms not put in tier C, measured on the MES >= 7.0
TEST false alarms (real + null; the >= tau_long subset is reported too). Interval: Monte Carlo over Jeffreys
Beta posteriors for r and s, 2.5-97.5%. The 45-55 d, 2-4 transit subset is reported separately: test pass rates on
true signals vs false alarms in that region. If the vetting cannot separate it, the report says so.

## Implementation notes (2026-11-12 ~10:15, after the pre-registration commit, before any DEV aggregate)
Found while building the star-level inputs and a 13-signal debug run. No DEV statistic had been tabulated.
- **Gaia counterpart.** The 20" / parallax > 10 mas rule found no counterpart for 30 of 191 stars: nearby M dwarfs
  move tens of arcsec between the TIC epoch (2000) and Gaia (2016), and 13 faint controls have Gaia parallaxes of
  3-5 mas (TIC put them within 50 pc). The rule is now: within 60" with parallax > 5 mas (or a 2-parameter source
  within 5"), G within 1.5 mag of Tmag + 1.6, lowest |G - Tmag - 1.6| + sep/30; fallback, the source within 3"
  with G within 2 mag of Tmag + 1. All 191 stars now have a counterpart. Neighbour separations are measured from
  the Gaia position.
- **Kostov+2025 table2 is not an EB list** (it is the 872,720 unvetted neural-network flags). Only table3 / table4
  count. All 7 Kostov matches in our stars were table2 rows, including the TOI-270 and TOI-1266 hosts.
- **A TOI dispositioned FP or FA at P or an alias is a catalogue fail** (a known false positive), not tier K.
- Pixel positions use the Gaia DR3 (2016) coordinates of the target and neighbours.
- Pixel check on an injection (DEV, TIC 454682563, 2 sectors): S_t = 0.7, i.e. *uninformative*, as it must be,
  since the injection is not in the pixels. The pixel test therefore cannot be scored on injections at all.

## DEV-stage notes (2026-11-12 10:30, DEV only; no TEST or hold-out vetting run yet)
First DEV pass (`vet_dev_v1_unnested.jsonl`, 374 signals: 180 true, 194 false alarms, 0 errors), default thresholds:
- **model_compare was wrong as built and is changed (a test-definition change made on DEV).** Box and alternatives
  were fitted separately. On a fast rotator (TIC 233570240) a per-epoch sinusoid absorbing unrelated rotation residue
  "beat" a true injection by dBIC = 2.3e5, and lone extreme out-of-transit points gave dBIC up to 4e13 on nulls.
  The statistic is now **nested**: for each alternative X, dBIC_X = BIC(X) - BIC(X + box), i.e. how much the box
  still improves the fit once X is in (fail if < 0: X explains the dip; soft if < 6). Out-of-transit points are
  winsorised at 8 sigma. Thresholds unchanged. DEV is re-run in full with this version (`vet_dev.jsonl`).
- **No power, kept and reported:** the phase-scramble p-value is 0.00 for every DEV signal, true and false. A
  false alarm is a maximum over ~10^5-10^6 periods; at its own fixed period the scrambled folds are ordinary noise,
  so the test cannot see it. The uniqueness *margin* scales with MES the same way for true and false signals.
- **Too strict for true signals (the loosen-only rule will act):** per-transit chi^2 (97% of true signals reach
  -log10 p = 4.0; real per-epoch depths scatter more than white noise, and long-baseline period error moves the box
  off some epochs), sector consistency, SAP z, and contamination (43% of true injections sit on a star with a Gaia
  neighbour within 21" bright enough to fake the dip with a <= 50% eclipse; the rule will in effect switch it off,
  which is noted as a pre-registration weakness: contamination describes the star, not the noise, and should have
  been exempt from the rule).
- **Where the separation is:** duration vs density (true 97% at ratio 1.37; false alarms at MES >= 7 median 1.21,
  90% 2.46), single-event dominance (true 99% 0.54; false alarms at MES >= 7 median 0.60), and odd/even.

## Frozen thresholds (commit 81c7bd3, 2026-11-12 10:41:57, before any TEST or hold-out vetting)
`spikes/mdwarf_stack/vetting/frozen_vet.json` (soft / fail; DEV numbers in `vetting/analyze_dev_frozen.txt`):
odd/even z 3 / 4.38; single-event dominance 0.65 / -; per-transit -log10 p 4.04 / 10.4; min-SES (negative) 1.11 /
2.49; sector consistency -log10 p 3.82 / 10.4; duration / density ratio 1.38 / 2.0; secondary excess 1 / 2;
nested model comparison -dBIC -6 / 0; uniqueness -margin -2 / -1; scramble p 0.01 / 0.05; SAP z (negative) 3.39 /
6.72; SAP background statistic 4 / -; systematics-window share 0.5 / 0.8; orbit-alias spread soft below 0.5 d;
rotation residual p < 0.02; contamination need -0.0031 (soft only if a neighbour needs a <= 0.31% eclipse, so in
effect off). DEV with these: true signals A+B 0.97 [0.93, 0.98] (146 A, 28 B, 6 C of 180); DEV false alarms at
MES >= 7 in C 10/24, at MES >= 6 44/194.

## Results (TEST and hold-out, frozen thresholds; 2026-11-12)
Command: `cd spikes/mdwarf_stack && $PY -I vetting/analyze_vet.py all` -> `<OUT>/vetting/analyze_all.txt` (copy:
`vetting/analyze_all_frozen.txt`). Inputs: `vet_test.jsonl` (387 signals: 228 detected TEST injections, 139 TEST
false alarms at MES >= 6, the rest lower), `vet_holdout.jsonl` (82 signals, 30 recovered hold-out planets plus the
gate-1 host signals), `pixel.jsonl` (41 pixel checks: 30 hold-out planets, 10 TEST real false alarms at MES >= 7, 1
DEV debug row). 0 measurement errors. The pixel test runs on real stars only, so injections never have it.

### Tiers
| set | n | A | B | C | K | A+B (Wilson 95%) |
|---|---|---|---|---|---|---|
| TEST headline-cell detected injections | 108 | 86 | 17 | 5 | 0 | 0.95 [0.90, 0.98] |
| ... of which MES < 12 | 48 | 39 | 9 | 0 | 0 | 1.00 [0.93, 1.00] |
| TEST all detected injections | 228 | 169 | 45 | 14 | 0 | 0.94 [0.90, 0.96] |
| TEST false alarms, MES >= 6 | 139 | 65 | 38 | 36 | 0 | 0.74 [0.66, 0.81] |
| TEST false alarms, MES >= 7 | 29 | 11 | 7 | 11 | 0 | 0.62 [0.44, 0.77] |
| TEST false alarms, MES >= tau_long (8.054) | 3 | 1 | 0 | 2 | 0 | 0.33 [0.06, 0.79] |
| ... real-control false alarms, MES >= 6 | 48 | 17 | 12 | 19 | 0 | 0.60 |
| ... null (inverted / rolled) false alarms, MES >= 6 | 91 | 48 | 26 | 17 | 0 | 0.81 |
| hold-out planets (catalogue on) | 30 | 0 | 0 | 0 | 30 | - |
| hold-out planets, **blind** (catalogue off) | 30 | 17 | 8 | 5 | 0 | 0.83 [0.66, 0.93] |
| gate-1 host signals >= tau_long | 10 | 0 | 0 | 0 | 10 | - |

Headline injections by MES (A+B): 8.1-10 32/32, 10-12 16/16, 12-15 14/14, 15-20 20/22, > 20 21/24. The losses are at
high MES, where per-transit chi^2 and SAP z are most sensitive to imperfect injected-depth recovery.

### Per-test outcome (pass / soft / fail)
| test | headline injections (108) | TEST FA >= 7 (29) | hold-out planets (30) |
|---|---|---|---|
| odd_even | 0.99 / 0.01 / 0 | 0.96 / 0.04 / 0 (25) | 1.00 / 0 / 0 |
| per_transit (dominance, min-SES, chi^2) | 0.92 / 0.07 / 0.01 | **0.55 / 0.45 / 0** | 0.87 / 0.13 / 0 |
| sector_consistency | 0.97 / 0.03 / 0 | 1.00 / 0 / 0 | 0.97 / 0.03 / 0 |
| duration_density | 0.96 / 0.03 / 0.01 | 0.93 / 0.03 / 0.03 | 0.97 / 0.03 / 0 |
| secondary | 0.98 / 0.02 / 0 | 0.97 / 0.03 / 0 | 0.97 / 0.03 / 0 |
| model_compare (nested) | 1.00 / 0 / 0 | 0.97 / 0 / 0.03 | 1.00 / 0 / 0 |
| uniqueness | 1.00 / 0 / 0 | 0.93 / 0.03 / 0.03 | 1.00 / 0 / 0 |
| sap | 0.94 / 0.06 / 0.01 | **0.69 / 0.31 / 0** (13) | 1.00 / 0 / 0 |
| systematics_window | 1.00 / 0 / 0 | 0.85 / 0.15 / 0 (13) | 1.00 / 0 / 0 |
| orbit_alias | 1.00 / 0 / 0 | 0.92 / 0 / 0.08 (13) | 1.00 / 0 / 0 |
| rotation_residual | 0.98 / 0.02 / 0 | 1.00 / 0 / 0 (13) | 0.97 / 0.03 / 0 |
| pixel | n/a (injections) | **0.67 / 0 / 0.33** (6 informative) | 0.80 / 0.20 / 0 |
| contamination | 1.00 / 0 / 0 | 1.00 / 0 / 0 (10) | 0.93 / 0.07 / 0 |
| catalogue | 1.00 / 0 / 0 | 1.00 / 0 / 0 (10) | (K for all 30) |
(n in brackets where a test does not apply to every row: null false alarms on inverted or rolled light curves have
no SAP, pixel or star-level counterpart.) Scramble p is not listed: 0.00 for every signal, no power.

What rejects false alarms (TEST FA >= 6, strike counts): per-transit soft 39, uniqueness soft 13, duo strike 13,
duration soft 12 + fail 4, SAP soft 10, systematics window soft 7, model comparison soft 6.
What costs true signals (headline): per-transit soft 8, SAP soft 6, sector consistency soft 3, duration soft 3.

### Targets
1. **Met.** Headline detected injections in A or B: **0.95 [0.90, 0.98]** (103/108); goal >= 0.80.
2. **Missed, badly.** False alarms >= tau_long in C: **0.67 [0.21, 0.94]** (2/3; goal >= 0.90). At MES >= 7:
   **0.38 [0.23, 0.56]** (11/29), short of 0.90 by 52 points; at MES >= 6: 0.26 [0.19, 0.34] (36/139). Real-star
   false alarms are rejected more often (19/48 at >= 6) than nulls (17/91), mainly by SAP, systematics window and
   pixel, which see the real light curve. The cause is plain: most long-period false alarms are 2-4 epochs of
   ordinary-looking, box-shaped red noise at a plausible duration. No light-curve test separates them from a
   planet; dominance (one epoch carrying the detection) does best.
3. **Met only via the catalogue; blind missed.** Hold-out planets in A, B or K: 1.00 [0.89, 1.00] (30/30, all K).
   Blind (catalogue off): **0.83 [0.66, 0.93]** (25/30), 7 points short of 0.90. Blind C: TOI-6002 b (duration
   ratio 1.84 + pixel soft), TOI-5728 b (min-SES + pixel soft), K2-18 b (dominance 0.675 + duo), TOI-4336 Ab
   (rotation residual + contamination: it is a near-equal binary, a neighbour needs only a 0.24% eclipse),
   TOI-1266.01 (MES 46; per-transit and sector chi^2 both soft). Three of the five involve the pixel soft rule
   or contamination, both of which are flawed (below).

POST-HOC, not credited: with the pixel test also off, blind hold-out A+B is 27/30 = 0.90 [0.74, 0.97]. The pixel
*soft* rule ("difference centroid > 1 px off target in >= half of the strong epochs") fires on 6/30 real planets
because 5 of them have only one strong epoch (1 of 1), where a single noisy centroid decides it. It should need >= 3
strong epochs. The pixel *fail* rule (dip stronger on a resolved neighbour) never fired on a hold-out planet.

### The pixel check on TEST real false alarms (MES >= 7, 10 signals)
- **fail** (dip on a neighbour): testreal-long-267320484#0 (52.917 d, 3 transits; tier A without the pixel test;
  dip SNR 8.5 on a neighbour 44" away vs -3.4 on target) and testreal-long-340487632#1 (52.879 d, 4 transits;
  SNR 23.3 on a neighbour 37" away vs -6.9). Both near-53 d false alarms are caught.
- pass: 141621885#0, 154924886#2, 260972843#0, 377030976#0. Uninformative (n/a): 141621885#1, 154924886#0,
  302444623#0, 422429539#0.
- The two 52.9 d signals share no epochs (340487632: BTJD 1526.24, 2055.03, 2266.54, 3958.67, sectors 8/27/35/97;
  267320484: 1348.17, 3147.34, 3888.17, sectors 1/67/95): a period coincidence, not a common-mode systematic.
- So when the pixel test is informative it is the strongest test we have, but it was informative for 6 of 10, and
  it was never calibrated against a planet in the pixels at small depth (only the 30 hold-out planets).

### P >= 40 d with 2-4 transits (the pile-up region)
| set | n | A | B | C | A+B |
|---|---|---|---|---|---|
| true (TEST detected + DEV true >= tau) | 16 | 11 | 5 | 0 | 1.00 [0.81, 1.00] |
| TEST false alarms >= 6 | 35 | 11 | 12 | 12 | 0.66 [0.49, 0.79] |
Per test, only dominance (true 0.88 pass, FA 0.60), SAP (0.88 vs 0.76), systematics window (1.00 vs 0.71) and the
pixel fail (2/2 informative FA) separate them. **The vetting cannot separate most of this population**: two thirds
of 40-55 d false alarms survive as A or B. Only the pixel check, more sectors (a third or fourth transit that
must line up) and a ground-based transit at the predicted time can.

### Projection: surviving false alarms per 4,700 survey stars
r = 1/118 (real controls with a long signal >= tau_long), Jeffreys posteriors, Monte Carlo 2.5-97.5%.
- before vetting: 47 [4, 183]
- after, survival s from TEST FA >= 7 (18/29 not C, the pre-registered choice): **28.6 [2.6, 115]**
- s from TEST FA >= tau_long (1/3): 14.2 [0.7, 89]; s from TEST FA >= 6 (103/139): 34.6 [3.2, 136]
Read: expect roughly 3-100 false alarms in tier A/B from the survey, against an expected handful of real planets.
Most surviving A/B candidates will be false. Pixel checks on every A/B candidate will remove some of them (2 of the 6
informative real-star FA here), which this projection does not include because the pixel test was not run on null
false alarms.

### Weak or no-power tests
- Phase-scramble p: no power (0.00 for everything).
- Uniqueness margin: little (scales with MES for true and false alike).
- Contamination: switched off by the loosen-only rule (pre-registration weakness: it describes the star, not the
  noise, and should have been exempt). Where it does fire (TOI-4336 Ab, TOI-1452 Ab) it is a real near-equal binary.
- Per-transit chi^2 and sector consistency: loosened to near-no power; dominance carries the per-transit test.
- Rotation residual, orbit alias: rarely fire.
- Pixel soft rule: unreliable with 1 strong epoch (see above); not calibratable on injections.
- Duration consistency per half: info only.
Strongest: dominance, duration vs density, SAP, systematics window, pixel fail.

### POST-HOC finding (found on TEST, not credited; a v2 candidate)
The share of SES^2 in epochs within 1 h of a data gap >= 20 min exceeds 0.5 for 0/228 detected TEST injections but
4/13 time-anchored TEST false alarms at MES >= 7 (267320484 0.84, 377030976 1.00, 422429539 0.65, 154924886#0
0.60). It looks like a useful gap-edge test, but it was found by looking at TEST and must be validated on fresh
material (e.g. the survey's own null runs) before use.

### Cards
`<OUT>/vetting/cards/test/`: testinj0-074-41358832#0 (A, a true injection), testreal-long-267320484#0 (C, pixel
fail), testreal-long-340487632#1 (C, dominance + systematics + pixel fail), each with `cards.md` text.

### Changes after the freeze
Display only: the contamination reason printed the needed eclipse depth with `.0f`, so 0.24% showed as "0%"; it now
uses `.2g`. No threshold, test or tier rule changed; the tier tables are identical before and after.

## Running the vetting on survey output (v1 procedure; superseded, see "Running the vetting on survey output (v2 decision)" at the end)
`PY` = any repo venv python with wotan/numba (e.g. the gate-2 one). From `spikes/mdwarf_stack/`, ONE process:
1. `$PY -I -c "import json,sys; tau=json.load(open('<OUT>/gate2/frozen.json'))['tau_long']; print('\n'.join(sorted({str(r['tic']) for r in map(json.loads, open(sys.argv[1])) if r.get('range','long')=='long' and any(s['mes']>=tau for s in r.get('signals',[]))})))" SURVEY_ROWS.jsonl > tics.txt`
   (the survey runner's long-search rows, same format as `gate2/run2.py` output).
2. `VET_LC_DIR=<survey lc dir, if not data/planets_spike/lc> $PY -I vetting/fetch_sap.py list --file tics.txt`
3. `$PY -I vetting/catalog.py tics=tics.txt` (Gaia, EB catalogues, TOI, CTOI; any failure is recorded and
   blocks tier A, never read as "not known"; refresh `data/planets_spike/catalogs/toi.csv` first if it is old).
4. `VET_LC_DIR=... nice -n 19 ionice -c3 $PY -I vetting/run_vet.py survey --rows SURVEY_ROWS.jsonl [--jobs JOBS.jsonl]`
   -> `<OUT>/vetting/vet_survey.jsonl` (mass/radius from the job file, else `gate2/survey_targets.csv`).
5. Pixel check for every tier A/B signal (network, ~10-40 s per sector):
   `$PY -I vetting/pixel_check.py <OUT>/vetting/vet_survey.jsonl KEY ...`
6. `$PY -I vetting/cards.py survey --tiers A,B,K` -> `<OUT>/vetting/cards/survey/` (one PNG + `cards.md` with the
   plain-English summaries, graded with `frozen_vet.json`). Use `--tiers C` to look at what was rejected and why.

## Vetting v2: pre-registration (2026-11-12 ~12:10 AEST, before any v2 statistic on any signal)
Why: v1 (frozen, TEST) keeps 95% of detected headline injections but sends only 38% of long-period false alarms
at MES >= 7 to tier C (goal 90%), and two thirds of 40-55 d, 2-4 transit false alarms grade A/B. v1 stays frozen
and unchanged (`vet.py`, `frozen_vet.json`); v2 is new code (`vet2.py`, `frozen_vet2.json`) that calls v1's
`measure` and adds statistics. TEST is used up: v2 is chosen on DEV and validated on a **fresh** set of survey stars.

What v2 knows from TEST (written in the v1 results above, and nothing else): the per-test TEST table, the
40-55 d pile-up, the dead tests, the pixel soft-rule misfire on single-strong-epoch planets, and the post-hoc gap
idea (0/228 injections vs 4/13 time-anchored false alarms). No other TEST number is opened for v2.

### v2 tests
Kept from v1 with **v1's frozen thresholds, unchanged**: odd_even, per_transit (dominance and min-SES only),
duration_density, secondary, model_compare, uniqueness (margin only), sap (z and background),
systematics_window, orbit_alias, rotation_residual, pixel (rule revised below), catalogue.

Dropped (v1 DEV report in `frozen_vet.json`, after its loosening; soft-or-worse rate on DEV true vs DEV false
alarms): **uniqueness scramble p** (0 of 180 vs 0 of 194: no power, and it is the most expensive statistic);
**per-transit chi^2** (2.8% vs 0.5%: it strikes planets more often than false alarms); **sector consistency**
(2.8% vs 0.5%, same). Kept although DEV shows no power: odd_even and secondary, because they target eclipsing
binaries, which the DEV false-alarm material (noise nulls and controls without known EBs) does not contain.

**Contamination becomes info only** (printed in every A+/A/B summary, never a strike, its error does not block A).
v1 DEV: at the 50% default, 88.6% of DEV true signals (n = 88) would be struck, so exempting it from the
loosen-only rule makes it fire on almost every planet; it describes the star, not the signal, so it cannot tell
a noise false alarm from a planet on the same star. The question it asks (a blended EB on a neighbour) is
answered by the pixel test, which A+ requires.

New statistics (all computed on the search input series, the same scale as MES):
1. **gap_edge.share**. A gap is >= 20 min without data in the search input (consecutive present bins >= 3 bins
   apart; segment edges, momentum-dump gaps, downlinks and flare-clip holes all count). An epoch is flagged if its
   box window [c - D/2, c + D/2] lies within 1 h of a gap. share = sum of max(SES_e, 0)^2 over flagged epochs /
   sum over all. Default soft > 0.5, fail > 0.8. It uses only the time sampling, which every null keeps, so it
   applies to **every** mode (real, inv, roll, invroll).
2. **red_noise.neg_ratio**. For each epoch, sigma_e = 1.4826 x MAD of the stride-1 SES (same width) with
   D < |t - c| <= 1.5 d (>= 30 values, else sigma_e = 1, the per-sector scale). MES_red = sum_e (SES_e / sigma_e)
   sqrt(D_e) / sqrt(sum D_e); statistic -MES_red / MES_vet. Default soft > -0.8 (local noise makes the signal 20%
   weaker), fail > -0.6. Every mode.
3. **low_margin.neg_excess** (the transit-count-dependent cut, as a vetting-stage strike). The few-transit class
   is F = {n_tr <= 3} or {P >= 40 d and n_tr <= 4} (n_tr = the vetting's epoch count). For a signal in F,
   excess = MES - tau_det, where tau_det is the detection threshold that selected the signal (tau_long = 8.054
   for the survey and for injections); statistic -excess, default soft > -1.0 (an F signal needs MES >= tau_det
   + 1). Signals not in F pass. **Proxy rule:** the false-alarm populations selected at MES >= 7 (or >= 6) are
   stand-ins for the >= tau_long tail, so for them tau_det = 7.0 (or 6.0). Reason: a cut at a fixed MES above
   tau_long would "reject" every 7-8 MES false alarm trivially, although those are below the detection threshold
   and never reach vetting in the survey; measuring the margin above the selecting threshold is fair if the tail
   is roughly exponential, which DEV nulls show (155 at >= 6, 19 at >= 7, 3 at >= 8). The >= tau_long rows are
   also reported with tau_det = tau_long. A fixed per-bucket MES threshold from the DEV nulls alone was
   considered and rejected: the DEV tail by transit count is 3 / 3 / 4 / 3 / 6 nulls at MES >= 7 for 2 / 3 / 4 /
   5-6 / 7+ transits, too few to set a threshold per bucket.
4. **pixel (revised rule)**: the soft rule (difference centroid > 1 px off target in at least half of the strong
   epochs) needs **>= 3 strong epochs** (peak SNR >= 4); with fewer it is not applied. Fail and uninformative
   rules unchanged.

Calibration (DEV only, the v1 rule, mechanical, `vetting/freeze_vet2.py`): for each new statistic, on the 180 DEV
true signals, soft is loosened to the 97% value if > 3% exceed the default, fail to max(99% value, soft) if > 1%
exceed it; never tightened. For low_margin the denominator is the DEV true signals with MES >= tau_long (the
signals it is applied to in the survey), non-F signals counting as passes. The DEV false-alarm rate of every
statistic is reported but not used. `freeze_vet2.py` refuses to run once any validation vetting output exists.
The frozen file is committed before the first validation vetting run.

### v2 tiers
strikes = soft fails (v2 tests, including low_margin) + 1 if n_tr = 2. **K** catalogue match (as v1). **C** any hard
fail or strikes >= 2. **A** strikes = 0, n_tr >= 3, no required test returned `error` (pixel and contamination
not required). **A+** = A and the pixel test passed on target with >= 3 strong epochs (informative). **B** the rest.

### Fresh validation set (`vetting/select_val2.py`, seeds 20261212 stars / 20261213 injections)
- Pool: `gate2/survey_targets.csv` with >= 6 predicted sectors (4,681), minus every gate-2 DEV/TEST control,
  hold-out host and gate-1 host (101), TOI hosts of any disposition (34) and exoplanet.eu hosts within 30" (31):
  4,515 stars. Strata 6-12 / 13-25 / 26+ predicted sectors, proportional: **150 stars = 130 / 8 / 12**. Each
  stratum in a seeded random order; a star with no TESS product (or a failed download) is replaced by the next
  in that order.
- Per star: 4 long-range nulls (inv0, invroll11, roll21, roll22, the v1 TEST-null set) = 600 nulls. v1 TEST had
  19 of 232 nulls at MES >= 7 (8.2%; DEV 4.5%), so ~49 are expected (~27 at the DEV rate). 2 headline injections
  (1.0-1.5 R_e, 15-30 d) and 1 long injection (1.0-2.0 R_e, 30-55 d): 450 injections; at the TEST detection rates
  (~0.4 headline on 6-12 sector stars, ~0.5-0.7 elsewhere) ~120 headline and ~70 long detected are expected.
- Frozen gate-2 search (`gate2/run2.py`, rot = harm, long range, top signal), 2 niced workers, light curves
  from the survey's S3 route (`vetting/fetch_val2.py`; PDCSAP and SAP from the same bytes). Deleted afterwards.
- **Detected injection** = top signal matched by the gate-2 rule with MES >= tau_long; headline = 1.0-1.5 R_e,
  15-30 d. **Null false alarm** = a null's top signal with MES >= 7 (primary), also >= tau_long and >= 6.
- If compute runs short the set is the first N stars in the draw order (strata interleaved) whose jobs all
  finished; the rule is not changed, the intervals widen.
- **Blinding.** These are survey stars. Only null realisations and injected copies are searched; the job maker
  refuses a real unaltered job. The real survey search output and `outputs/planets_spike/survey/` are never
  read. For injected copies only the matched top signal is used: an unmatched top signal on an injected copy may
  be a real signal of the star and is neither vetted nor printed. **Limitation:** null false alarms may differ in
  structure from real-star false alarms (no pixel, SAP or catalogue view on rolled nulls; no real EBs); the
  gate-2 DEV real-control signals are the only real-star check, and they are DEV.

### Paired comparison and decision rule (fixed now)
v1 (frozen `vet.grade`, unchanged) and v2 are graded on the same rows. Injections are graded with the catalogue
on; K counts as kept. Intervals: cluster bootstrap over stars (10,000 resamples, seed 20261214), 2.5-97.5%;
Wilson for single proportions.
- **Adopt v2 for the survey only if** (i) on the fresh nulls at MES >= 7, v2 sends more false alarms to tier C
  than v1 (paired difference > 0; if its interval spans zero the report says the set cannot tell them apart),
  and (ii) v2 keeps >= 85% of detected headline injections in A+/A/B, and (iii) v2's headline retention is no
  more than 5 points below v1's. Otherwise v1 stays.
- Reported for both: tier tables, paired differences by transit count (2, 3, 4, 5+) and period range (10-30,
  30-40, 40+ d), the per-test power table (fresh injections vs fresh nulls), the headline cost, and the 40 d+
  2-4 transit injections.
- Projection for the winner: N_surv = 4,700 x r x s, r = 1/118 (v1's real-control rate, Jeffreys), s = fresh null
  false alarms at MES >= 7 not in C; split by tier (A+, A, B) and by transit count, and for 40 d+ with 2-4 transits;
  Monte Carlo over Jeffreys posteriors, 2.5-97.5%.
- Candidate summaries quote the fresh-set false-alarm survival and the projected false-alarm count for the
  candidate's own tier and transit-count bin.

## Vetting v2: frozen thresholds (DEV only; before any validation vetting run)
(The pre-registration header says "~12:10"; its commit 85d4831 is timed 12:00:18. The text is unchanged.)
`$PY -I vetting/run_vet2.py dev` (v2-only statistics for the 374 v1 DEV signals, 0 errors, 320 s, one niced
process) then `$PY -I vetting/freeze_vet2.py` -> `vetting/frozen_vet2.json`. **No new threshold was loosened**;
all three keep their defaults. Kept v1 tests carry v1's frozen thresholds.

| statistic | frozen soft / fail | DEV true soft+ (n) | DEV FA >= 6 soft+ / fail (n) | DEV FA >= 7 soft+ / fail (n) |
|---|---|---|---|---|
| gap_edge.share | 0.5 / 0.8 | 0.011 (180) | 0.067 / 0.015 (194) | 0.21 / 0.04 (24) |
| red_noise.neg_ratio | -0.8 / -0.6 | 0.000 (180) | 0.010 / 0 (194) | 0 / 0 (24) |
| low_margin.neg_excess | -1.0 / - | 0.000 (166 at MES >= tau_long) | 0.19 (194, tau_det 6) | 0.375 (24, tau_det 7) |

DEV tiers, v1 vs v2 (information only; thresholds were already fixed by the rule): DEV true signals at MES >=
tau_long (166) v1 A/B/C 134/26/6, v2 140/23/3 (dropping the chi^2 tests recovers 3); DEV false alarms at MES >= 7
(24) in C: v1 10, v2 14; DEV real-control signals >= 6 (39, the only real-star check): v1 C 12, v2 C 13; DEV
nulls >= 6 (155): v1 C 32, v2 C 49. red_noise has almost no DEV power; it is kept as pre-registered (no test is
dropped after seeing its DEV power, because that would be a tightening-by-selection on DEV false alarms).
DEV holds only a few true signals in the few-transit class, so low_margin's cost on planets is measured on the
fresh set (40 d+ injections), not here.

## Vetting v2: validation results on the fresh set (2026-11-12; v1 and v2 frozen, paired)
**Order of events.** Pre-registration 85d4831 (12:00:18). Light curves of the 150 stars downloaded 11:56-12:10
(S3 route; 160 drawn, 10 had no SPOC/TESS-SPOC product and were replaced in draw order; 59 MB). Frozen v2
thresholds d58780e (12:08:46). Frozen gate-2 search on the 1,050 validation jobs 12:13-13:29 (2 niced workers, 0
errors, 0 real unaltered jobs). Validation vetting (v1 + v2 statistics, 282 signals, 2 niced shards, 0 errors)
13:30-13:35, then `analyze_vet2.py` once. Catalogue / Gaia info for the 150 stars: `catalog.py tics=...`, all
sources ok. Text: `vetting/analyze_val2_frozen.txt` (copy of `<OUT>/vetting/v2/analyze_val.txt`).

**Set.** 150 stars (130 / 8 / 12 by predicted-sector stratum; real sectors median 6, 10-90% 3-18), 600 nulls, 450
injections. All 150 completed, so no shrinking. **Shortfall against the pre-registered size:** only 81 injections
were detected (55 headline, 26 others) against >= 150 (>= 75 headline) planned, because survey stars detect far
fewer injections than the gate-2 controls did (headline 0.18 [0.14, 0.23] vs 0.54 on TEST; post-hoc below).
Null false alarms at MES >= 7: 48 (planned >= ~40). The rule was not changed; the intervals are wider.

### Paired v1 vs v2
| population | n | v1 in C (or kept) | v2 in C (or kept) | v2 - v1, cluster bootstrap 95% |
|---|---|---|---|---|
| null false alarms, MES >= 7 (to C) | 48 | 15 (0.31 [0.20, 0.45]) | **25 (0.52 [0.38, 0.66])** | **+0.21 [+0.09, +0.35]** (11 only v2, 1 only v1) |
| null false alarms, MES >= tau_long (to C) | 16 | 10 (0.62) | 10 (0.62) | +0.00 [-0.18, +0.20] (1 / 1) |
| null false alarms, MES >= 6 (to C) | 201 | 39 (0.19) | 66 (0.33) | +0.13 [+0.09, +0.18] |
| detected headline injections (kept A+/A/B/K) | 55 | 47 (0.85 [0.74, 0.92]) | **52 (0.95 [0.85, 0.98])** | **+0.09 [+0.02, +0.17]** |
| all detected injections (kept) | 81 | 69 (0.85) | 78 (0.96) | +0.11 [+0.05, +0.18] |
| 40 d+ injections with 2-4 transits (kept) | 5 | 5 | 5 | 0 |

Tiers. Nulls >= 7: v1 A/B/C 14/19/15, v2 8/15/25. Headline injections: v1 44/3/8, v2 48/4/3. No A+ anywhere (the
pixel test cannot run on nulls or injections), no K.

| bin | nulls >= 7: n | v1 C | v2 C | detected injections: n | v1 kept | v2 kept |
|---|---|---|---|---|---|---|
| 2 transits | 13 | 7 | 11 | 1 | 1 | 1 |
| 3 transits | 15 | 5 | 11 | 2 | 2 | 2 |
| 4 transits | 9 | 3 | 3 | 11 | 10 | 11 |
| 5+ transits | 11 | 0 | 0 | 67 | 56 | 64 |
| P 10-30 d | 15 | 2 | 7 | 55 | 47 | 52 |
| P 30-40 d | 15 | 4 | 7 | 14 | 12 | 14 |
| P 40+ d | 18 | 9 | 11 | 12 | 10 | 12 |

### Decision (pre-registered rule): **ADOPT v2 for the survey**
(i) v2 sends more null false alarms at MES >= 7 to C: +0.21 [+0.09, +0.35], the interval excludes zero; (ii) v2
keeps 0.95 of detected headline injections (>= 0.85); (iii) v2 is 9 points *above* v1 on the headline cell, not
below. What the win is made of, stated plainly:
- **The rejection gain is the few-transit margin strike.** Of the 11 nulls only v2 rejects, 10 carry low_margin
  (4 of them duos). That strike is measured above the selecting threshold (7.0 for this population, the
  pre-registered proxy rule). On the 16 nulls that actually pass tau_long, v1 and v2 reject the same number (10),
  and the interval spans zero: **at the real detection threshold this fresh set cannot tell v1 and v2 apart.**
- **The retention gain is from dropping the chi^2 tests**, a DEV decision: on the fresh set v1's per-transit test
  struck 13% and sector consistency 9% of headline injections (TEST had shown 5% losses). No injection is
  rejected by v2 that v1 keeps.
- **The cost of low_margin on true 40 d+ few-transit planets is barely measured:** 5 such injections were
  detected, all kept by both versions. 3 detected injections had <= 3 transits. The strike's planet cost rests on
  DEV (0 of 166) and these 5.
- The surviving false alarms moved to many-transit signals: neither version rejects any of the 11 nulls >= 7 with
  5+ transits (v2: 5 A, 6 B).

### Per-test outcome on the fresh set (v2 grading; soft-or-worse rate, n applicable)
| test | headline inj. (55) | all detected inj. (81) | nulls >= 7 (48) | inv nulls >= 7 (8) |
|---|---|---|---|---|
| low_margin (new) | 0.00 | 0.00 | **0.38** | 0.50 |
| per_transit (dominance, min-SES) | 0.05 | 0.04 | **0.35** | 0.12 |
| odd_even | 0.02 | 0.03 | 0.22 (41) | 0.14 (7) |
| duration_density | 0.04 | 0.02 | 0.17 | 0.25 |
| gap_edge (new) | 0.00 | 0.04 | 0.15 | **0.38** |
| sap | 0.02 | 0.02 | 0.25 (8, inv only) | 0.25 |
| systematics_window | 0.02 | 0.01 | 0.12 (8, inv only) | 0.12 |
| red_noise (new) | 0.00 | 0.00 | 0.06 | 0.00 |
| secondary | 0.00 | 0.00 | 0.04 | 0.00 |
| model_compare, uniqueness margin, orbit_alias, rotation_residual | 0.00 | 0.00 | 0.00 | 0.00 |
| catalogue | 0.00 | 0.00 | n/a | n/a |
| pixel | n/a | n/a | n/a | n/a |
v1-only tests on the same rows: per-transit (with chi^2) 0.13 headline / 0.38 nulls; sector consistency 0.09 /
0.06; scramble p 0 / 0; contamination (v1 frozen value) 0 / n/a.

### Projection with v2 (pre-registered: r = 1/118 from gate-2 real controls, s from fresh nulls >= 7)
| part | k / 48 | false alarms per 4,700 stars [95%] |
|---|---|---|
| before vetting | 48 | 59 [4, 181] |
| **surviving (not C)** | 23 | **28 [2, 90]** |
| tier A+ / A / B | 0 / 8 / 15 | 0.6 [0, 3.8] / 10 [0.7, 35] / 19 [1.3, 61] |
| 2 / 3 / 4 / 5+ transits, not C | 2 / 4 / 6 / 11 | 3.0 [0.1, 12] / 5.4 [0.3, 20] / 7.9 [0.5, 28] / 14 [0.9, 46] |
| 40 d+ with 2-4 transits, not C | 6 | 7.9 [0.5, 28] |
The point estimate (28) equals v1's TEST projection (28.6): v2 moves which false alarms survive more than how many
at the real threshold. The A+ row is a placeholder: no false alarm can reach A+ in this test because the pixel
test never runs on nulls, so its 0.6 is not evidence that A+ is clean.

### Odds quoted by the candidate summaries (v2, fresh nulls >= 7 and detected injections)
| tier, transits | nulls in this cell (of 48) | injections in this cell (of those with this transit count) | false alarms per 4,700 stars |
|---|---|---|---|
| A, 3 | 0 | 1 of 2 | 0.6 [0, 3.8] |
| A, 4 | 3 | 9 of 11 | 4.2 [0.2, 16] |
| A, 5+ | 5 | 58 of 67 | 6.6 [0.4, 24] |
| B, 2 | 2 | 1 of 1 | 3.0 [0.1, 12] |
| B, 3 | 4 | 1 of 2 | 5.4 [0.3, 20] |
| B, 4 | 3 | 2 of 11 | 4.2 [0.2, 16] |
| B, 5+ | 6 | 6 of 67 | 7.9 [0.5, 27] |
Example (a detected injection, `<OUT>/vetting/v2/cards/val/cards.md`): "Signals like this one were false 5 times out
of 63 in testing, but the test planted about three planets per star, far more than nature does, so that ratio
flatters the candidate. Scaled to the survey, about 6.6 [0.4, 23.7] false alarms like this are expected per 4,700
stars. If that is comparable to the number of real planets the survey can find (a handful is the rough
expectation, not a measurement), a signal like this one is more likely false than real."

### Weak, cut or unverified
- **No power on the fresh nulls:** model_compare, uniqueness margin (it struck 12% of DEV false alarms), orbit_alias,
  rotation_residual; red_noise is weak (0.06, 3 of the 11 v2-only rejections, never alone). secondary 0.04.
- **A+ and the revised pixel rule are untested here:** no pixels for nulls or injections. Their only evidence is
  v1's TEST pixel findings (2 of 2 informative 53 d real false alarms failed on a neighbour; the soft rule misfired
  on single-strong-epoch hold-out planets).
- **Null false alarms are not real-star false alarms.** 40 of the 48 at MES >= 7 are rolled copies, which have no
  SAP, systematics-window, orbit or rotation view; real false alarms were rejected more often than nulls in v1
  TEST. No eclipsing binaries are in the null material, so odd/even and secondary are untested against EBs. The
  DEV real-control signals (DEV) are the only real-star check: v1 C 12 of 39, v2 C 13 of 39.
- The proxy rule (margin above 7.0 for the >= 7 population) is what makes low_margin count; it assumes the null
  tail is roughly exponential above any threshold, which DEV supports only coarsely.
- Detected injections were half the planned number; few-transit and 40 d+ planets are almost absent from it.
- Display: the v1 card's "MES at every phase" panel can show a spike at a sparsely covered phase bin (seen at
  phase -9 d on an injection card); this is the plot, not a statistic.

### POST-HOC findings (after the decision; not credited) - `vetting/posthoc_val2.py`, `vetting/posthoc_val2.txt`
- **The survey-star null tail is fatter than DEV's above tau_long:** 16 of 600 fresh nulls (0.027 [0.016, 0.043])
  reach tau_long vs 3 of 420 DEV nulls (0.007), on 9 of 150 stars; 14 of the 16 are rolled (non-inverted) copies,
  8 are duos, and 12 come from stars with 6-12 real sectors. With r taken from this fresh per-realisation rate
  instead of 1/118, the surviving count becomes ~50-60 per 4,700 stars. Rolled copies of real stars may produce more
  false alarms than the real stars do (real controls: 1 of 118), so this is an upper-side scenario, not a
  measurement of r.
- **Survey stars detect fewer injections:** headline 0.18 [0.14, 0.23] (55 of 300) vs 0.54 on the TEST controls;
  by real sectors 1-3: 0.03, 4-5: 0.10, 6-12: 0.20, 13+: 0.58. 58 of 450 injections had < 2 transits in the data.
  Predicted sectors overstate real ones on the survey list (median 6 real for a list cut at >= 6 predicted). The
  survey's expected yield should be re-estimated with real sector counts; gate 2's ~0.37 for the 6-12 bucket is
  optimistic.

## Running the vetting on survey output (v2 decision): run **v2** (`run_vet2.py`, `frozen_vet2.json`)
`PY` = a repo venv python with wotan/numba (e.g. the gate-2 one); from `spikes/mdwarf_stack/`; at most 2 niced
processes. Light curves: the survey deletes its NPZs, so first rebuild them with
`$PY -I survey/get_lightcurve.py TIC ... --max-mb 2000` (into `data/planets_spike/lc_cache/lc/`) and set
`VET_LC_DIR` to that directory.
1. Signals to vet: `tics.txt` exactly as in step 1 of the v1 procedure above (every long-search signal >= tau_long).
2. `VET_LC_DIR=... $PY -I vetting/fetch_sap.py list --file tics.txt` (SAP + background).
3. `$PY -I vetting/catalog.py tics=tics.txt` (refresh `data/planets_spike/catalogs/toi.csv` first; a failed
   query blocks A, never reads as "not known").
4. `VET_LC_DIR=... nice -n 19 ionice -c3 $PY -I vetting/run_vet2.py survey --rows SURVEY_ROWS.jsonl
   [--jobs JOBS.jsonl] [--shard 0/2 and 1/2 in two processes]` -> `<OUT>/vetting/v2/vet2_survey*.jsonl`. Each row
   holds the v1 statistics too, so v1 can still be graded from it (`vet.grade` with `frozen_vet.json`).
5. List the A/B signals: `$PY -I vetting/cards2.py survey --tiers A,B --text-only --max 10000` (prints key and
   tier), then the pixel check for each: `$PY -I vetting/pixel_check.py <OUT>/vetting/v2/vet2_survey.jsonl KEY ...`
   (with shards, pass the shard file that holds the key). The v2 pixel rule is applied at grading time.
6. `$PY -I vetting/cards2.py survey --tiers A+,A,B,K` -> `<OUT>/vetting/v2/cards/survey/` (PNG + `cards.md`;
   each summary quotes the fresh-set odds for its tier and transit count from `<OUT>/vetting/v2/analyze_val.json`,
   or its committed copy `vetting/analyze_val2.json`). `--tiers C` shows the rejected ones and why.
Read every A/B card as "more likely false than real unless shown otherwise": about 28 [2, 90] surviving false
alarms are projected for 4,700 stars, against a handful of real planets.

## Cut / known limitations (stated in advance)
- No injection into pixels, so the pixel test cannot be calibrated on injections; it is run on real stars only
  (hold-out planets, real-control false alarms) and is the first thing dropped if time runs short.
- Injections go into PDCSAP/SAP after PDC; PDC's own effect on transits is not measured.
- The false-alarm tail is small (a few events above tau_long); rejection rates there have wide intervals.
- No eccentric orbits, no TTVs, no spectroscopy, no centroid time series (MOM_CENTR is not in our files).
