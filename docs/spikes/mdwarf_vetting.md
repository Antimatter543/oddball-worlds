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

## Cut / known limitations (stated in advance)
- No injection into pixels, so the pixel test cannot be calibrated on injections; it is run on real stars only
  (hold-out planets, real-control false alarms) and is the first thing dropped if time runs short.
- Injections go into PDCSAP/SAP after PDC; PDC's own effect on transits is not measured.
- The false-alarm tail is small (a few events above tau_long); rejection rates there have wide intervals.
- No eccentric orbits, no TTVs, no spectroscopy, no centroid time series (MOM_CENTR is not in our files).
