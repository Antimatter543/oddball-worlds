# Track C gate 3: does the frozen survey still work on 50-100 pc M dwarfs?

Status: **pre-registered 2026-11-13 (commit time is the authority), before any catalogue query,
light curve, search or vetting on any 50-100 pc star.** Before this commit the only far-star
material that exists is the follow-up's 30-cone sample and its 200-star coverage check
(`docs/spikes/mdwarf_followup.md` 2.5). One route check was run on **< 50 pc** stars only: the MAST
TIC criteria query below, for RA 0-10 deg and d 0-50 pc, returned 419 stars in 20 s and contains
all 410 gate-1 catalogue stars in that slice (9 extra).

Nothing frozen is changed: the gate-2 search (`gate2/frozen.json`, sha256 4194793b, tau_long =
8.054, rot = harm, `gate2/run2.py`), the survey's product rule and S3 route (`survey/download.py`),
and vetting v2 (`vetting/frozen_vet2.json`, sha256 d5076c08; `vetting/run_vet2.py`, `vet2.grade2`).
New code lives in `spikes/mdwarf_stack/gate3/` and only calls those.

## Question
The 100 pc estimate (follow-up 2.5) assumed the 50 pc per-sector-bucket detection rates carry
over to fainter, more blended, less covered stars. Measure the far stars directly, project the
yield and false alarms of the full extension, and recommend it only if the criteria below pass.

## 1. Target list (the full 50-100 pc list)
- **Route:** MAST TIC v8 (`Catalogs.query_criteria`, catalog "Tic"), criteria only, no cones:
  Teff 2700-3400 K, d 50-100 pc, Tmag -5 to 15, objType STAR (the survey's cuts with d moved),
  split into 36 RA slices of 10 deg, each under a 180 s wall-clock alarm, 3 tries with back-off.
  A slice that still fails is retried once more at the end; if it fails again it is "not
  checked" and the list size is reported as a lower bound with the missing slice named.
- Duplicates by TIC ID dropped; any TIC in the gate-1 50 pc catalogue
  (`tic_mdwarfs_50pc.csv`) or with d < 50 dropped.
- Sectors: tess-point (`select_controls.sectors_for`, the survey's function) through **S107**
  (the survey's `LAST_OBSERVED`); kept: **>= 6 predicted sectors** (the survey selection).
- Written to `<OUT>/gate3/targets_100pc.csv` (frozen list; sha256 recorded in the results).
  Reported by Tmag (< 12, 12-13, 13-14, 14-15) and predicted-sector bucket (6-12, 13-25, 26+).

## 2. Sample
- Pool: the full list minus TOI hosts of any disposition (local `toi.csv`, 2026-11-11) and
  exoplanet.eu hosts within 30" (as the v2 validation pool: rolled copies of a star with a real
  transit make planet-shaped nulls). The pool and full-list sizes are both reported.
- **300 stars**, stratified by predicted-sector bucket (6-12, 13-25, 26+) x Tmag (< 13, 13-14,
  14-15): 9 strata, allocation proportional to the pool (largest remainder), each stratum in a
  seeded random order (seed **20261116**). Draw order interleaves strata by rank / quota, so any
  prefix stays proportional.
- **Stars without a SPOC / TESS-SPOC product are kept, not replaced**: they are the coverage
  measurement and count as zero recovery and zero signals.
- Representativeness: Tmag and predicted-sector distributions of the sample vs the full list
  (two-sample KS p-values and the stratum table) are reported.
- **If Kaggle time runs short**, the measured set is the first N stars in draw order whose jobs
  all finished; intervals widen; no criterion changes.

## 3. Measurements (all on Kaggle, frozen code)
Per sample star, light curves by the survey's product rule (SPOC 2-min > TESS-SPOC, MAST
observation query for the list, bytes from the S3 bucket, MAST fallback), PDCSAP and SAP from the
same bytes (`vetting/fetch_val2.fetch_one`). Jobs (frozen `gate2/run2.py`, rot from frozen.json):
- **real:** long (10-60 d) and short (1-10 d), 6 signals each (the survey setting);
- **nulls:** inv0, invroll11, roll21, roll22, long, top signal (the v2 validation set);
- **injections**, into the RAW binned flux before rotation removal and detrending (run2), at
  the raw model depth, long range, top signal; b U(0, 0.9), P log-uniform, t0 uniform in the first
  period of the data; per star (rng seeded by (20261117, TIC)):
  2 x (1.0-1.5 R_e, 15-30 d) [headline], 1 x (1.0-1.5, 30-50 d), 1 x (1.5-2.0, 15-50 d),
  1 x (2.0-3.0, 15-50 d).
- **50 pc reference** for the new cells: the 150 v2-validation survey stars (`val_set.csv`) get
  2 injections each (1.5-2.0 and 2.0-3.0 R_e, 15-50 d; rng (20261118, TIC)), detection level
  only, plus the noise metric. Their headline-cell and null values already exist (v2 validation).

Measured quantities and their 50 pc values:
1. **Coverage:** real / predicted sectors; share of stars with no product (50 pc survey: 0.78
   over searched stars, 331 / 4,681 = 7.1% with none; follow-up far check 0.647, 34 / 200).
2. **Noise on the transit timescale:** per sector, robust (MAD) scatter of 2-h bins of the
   frozen-preprocessed light curve (`followup/noise.sigma_2h`); per star, the median over sectors.
   Far sample vs the 150 v2-validation (50 pc) stars, overall and in matched Tmag bins.
3. **Null false-alarm rate** per realisation at tau_long: share of the 4 x n nulls whose top
   signal has MES >= 8.054. 50 pc (fresh v2 survey stars): **16 / 600 = 0.027**. Per star (any of
   4): 50 pc 9 / 150.
4. **Injection recovery:** detected = top signal matched (gate-2 rule) with MES >= tau_long; by
   real-sector bucket (1-3, 4-5, 6-12, 13+) and by cell. 50 pc headline: 2/70, 7/70, 25/124, 21/36
   (55/300). **Detectable** (used in the yield) = detected AND v2 tier not C (A+/A/B/K; graded
   with frozen_vet2, catalogue on). 50 pc: 52 / 55 detected headline kept.
5. **Real search:** long signals >= tau_long per searched star (50 pc survey: 510 / 4,350 =
   0.117; stars with any: 252 / 4,350); all are vetted with v2 (survey task definition), the A/B
   ones get the pixel check (locally), tiers reported. 50 pc: K 48, A+ 0, A 3, B 10, C 449.

## 4. Projection to the full list
- **Occurrence (changed from the follow-up, justified):** Dressing & Charbonneau 2015 (ApJ 807,
  45), Table 4 (Huber et al. stellar parameters), the 5.5-18.2 d and 18.2-60.3 d bins, log-uniform
  in P inside each bin, mapped onto 15-50 d. The follow-up took 0.56 (their P < 50 d total) x a
  guessed 15-30 d share; the period-binned table measures the long-period rate directly.
  For 1.0-1.5 R_e the 5.5-18.2 d bin is an upper limit only, so the 18.2-60.3 d density is used
  for 15-18.2 d. Planets per star (central [low = -1 sigma, high = +1 sigma], errors added
  linearly over sub-bins):
  - 1.0-1.5 R_e, 15-30 d: 0.134 [0.099, 0.188]; 30-50 d: 0.098 [0.073, 0.138]
  - 1.5-2.0 R_e, 15-50 d: 0.258 [0.200, 0.343]
  - 2.0-3.0 R_e, 15-50 d: 0.323 [0.233, 0.468]
  Caveat: DC15's Kepler M dwarfs are mostly early M (Teff ~3,500-4,000 K); ours are 2,700-3,400 K.
  Mid-to-late M dwarfs may host fewer 2-3 R_e planets, so (a) is also reported with the 2.0-3.0
  R_e cell set to zero.
- **Expected detectable planets** per cell c: sum over strata h of N_h x mean over sample stars
  in h of y_j,c x eta_c, where y_j,c = mean over star j's injections in cell c of
  (detectable_i x R*/a(P_i)) (geometric transit probability, TIC R* and M*, fallback 0.3 / 0.3);
  no-data stars have y = 0; stars whose jobs errored are left out of the mean (counted).
  N_h from the **full list** (TOI hosts included; they are ~1%). Interval: stratified bootstrap
  over sample stars (2,000 draws) at central eta; low and high eta cases reported beside it.
- **Expected false alarms surviving v2** over the full list: F = max(F_direct, F_scaled):
  - F_direct = sum_h N_h x mean_h(k_j), k_j = the star's real long signals >= tau_long graded
    A+/A/B (not C, not K) after the pixel check;
  - F_scaled = sum_h N_h x mean_h(m_j) x s_50, m_j = its non-K signals >= tau_long and
    s_50 = 13 / 462 (50 pc survey: not-C survivors / non-K signals; i.e. the 50 pc survival rate),
    which protects the estimate from a lucky zero in 300 stars.
  Intervals: bootstrap over sample stars, and Jeffreys on s_50.
- **Vetting load** = N_full x (all signals >= tau_long per sampled star).

## 5. Criteria (all fixed now; nothing loosened afterwards)
- **(a)** Expected detectable planets over the full list, summed over 1.0-1.5 (15-30 + 30-50),
  1.5-2.0 and 2.0-3.0 R_e at 15-50 d, central estimate (point recovery, central eta) **>= 1.0**.
- **(b)** Null false-alarm rate per realisation at tau_long **<= 2 x 16/600 = 0.0533** (point).
- **(c)** F (central) **<= 10 x** the (a) central estimate. If (c) fails while (a) and (b) pass,
  these stricter cuts are evaluated, each applied to both the detected injections (yield) and the
  real signals (F): (i) tier A/A+ only; (ii) >= 4 transits; (iii) both. The report gives, for
  each, the restored ratio and the yield it costs. They are not part of the verdict.
- **(d)** Cost: Kaggle wall time for the full list < ~5 days at 5 concurrent CPU sessions of 3
  workers (measured CPU per far real search x N_full, Kaggle factor included, plus setup),
  local disk < 500 MB (result records per star x N_full; light curves never kept), and the vetting
  load stated.
- **Verdict:** all four pass = recommend the extension; (a) fails = stop, the extension cannot
  find enough; otherwise name the failing criterion. Everything is reported either way.
- If it passes: the frozen target list, a shard plan (shards, hours each, waves of 5), the exact
  build / push / monitor / fetch commands with the existing wrapper, and the vetting procedure.
  No real shard is pushed.

## 6. Failure modes audited before reporting
Yield from 50 pc rates instead of far measurements; an occurrence rate picked to pass (low and
high shown); an unrepresentative sample; injections after detrending; unvetted detections
counted as planets; ignoring that fainter hosts are harder to confirm (the telescope class needed
for a typical far candidate is stated); anything frozen retuned.

## Compute
Kaggle: 5 private CPU script kernels (embedded `git archive`, as the smoke test), 60 sample stars
+ 30 reference stars each, 3 workers (run2's cap), results only come back. Locally: catalogue
queries (target list, catalogue info for vetting), the pixel check on A/B signals, analysis; at
most 2 niced processes.

---

# Results (2026-11-13; everything above this line is the pre-registration, unchanged)

**Verdict (pre-registered criteria): recommend the extension. All four criteria pass.** The
yield is carried by 1.5-3 R_e planets around stars in or near the TESS continuous viewing zones.
For Earth-size planets (1.0-1.5 R_e, the survey's headline cell) the extension adds about 0.2
expected planets. A post-hoc check (R6, not part of the verdict) finds that the same yield model
predicts about 2.7x more 1.5-3 R_e planets than the 50 pc survey actually found. Corrected by
that factor, the far yield is about 4-5 planets (range ~1.6-10), still above 1.

## R0. Order of events
| time (AEST) | event |
|---|---|
| 13:49:32 | pre-registration committed (`10071fd`), before any 50-100 pc query |
| 13:51:16 | first far-star file written (TIC slice RA 0-10) |
| 13:58:19 / 14:33:24 | gate-3 code (`9e0b2b4`), fetcher and parallel slice fetch (`6fef4b7`); the kernels embed `git archive 6fef4b7` |
| 14:29:54 | target list frozen (all 36 slices) |
| 14:39 | 5 Kaggle kernels pushed |
| 15:28-16:13 | kernels finished (0.75, 1.20, 1.23, 1.43, 1.56 h). No search deadline was reached, every job row came back, 0 error rows |
| 15:35:16 | cost/disk packing in `analyze3.py` and the `vet100.py` wrapper committed (`538aec9`). No analysis output existed yet: a preview on shard 3 alone stopped at the prefix rule with 0 measured stars |
| ~16:15 | `analyze3.py` run on all five shards; output in `<OUT>/gate3/analyze3.txt` / `.json` |

The sample produced no real A/B signal, so the local pixel check had nothing to run on. Nothing
frozen and no criterion was changed.

## R1. Target list
All 36 slices answered: 60,303 stars at 50-100 pc (0 duplicates), of which **19,827 have >= 6
predicted sectors** (`targets_100pc.csv`, sha256
`31f26b01d5b2aa3a8e67dbbb118094882a5400bf73901a8cfe55d0bc96c08565`). Tmag median 14.16;
predicted sectors median 8, mean 11.0.

| Tmag \ predicted sectors | 6-12 | 13-25 | 26+ | all |
|---|---|---|---|---|
| < 12 | 76 | 3 | 6 | 85 |
| 12-13 | 1,085 | 87 | 95 | 1,267 |
| 13-14 | 5,771 | 427 | 629 | 6,827 |
| 14-15 | 9,871 | 750 | 1,027 | 11,648 |
| all | 16,803 | 1,267 | 1,757 | 19,827 |

That is 4.2x the 50 pc survey (4,681 stars), and fainter (median T 14.16 vs 13.05).

## R2. Sample
Pool 19,785 (the list minus 22 TOI hosts and 20 exoplanet.eu hosts). 300 stars drawn in 9
strata. All 300 finished, so the prefix rule was not needed: **235 searched, 65 with no SPOC /
TESS-SPOC product** (kept and counted as zero). Sample vs full list: Tmag KS p = 0.73,
predicted-sector KS p = 1.00; medians 14.13 vs 14.16 and 8 vs 8. Every stratum's share is within
0.5 percentage points of the list's (`analyze3.txt` section 2).

## R3. Measurements (far sample vs 50 pc)
| quantity | 50-100 pc (this test) | 50 pc |
|---|---|---|
| stars with no product | 65/300 = 0.217 [0.174, 0.267] | 331/4,681 = 0.071 |
| real / predicted sectors (stars with data) | 0.666 | 0.778 |
| real / predicted sectors (all stars) | 0.560 | 0.734 |
| real sectors per searched star | median 5 (138 of 235 have < 6) | median 6 |
| sigma_2h (median over stars, ppm) | 1,882 (n 235) | 1,019 (n 150); ratio 1.85 [1.54, 2.14] |
| sigma_2h at Tmag 12-13 / 13-14 / 14-15 | 808 / 1,422 / 2,524 | 625 / 1,177 / 2,848 (ratios 1.29, 1.21, 0.89) |
| null rate per realisation at tau_long | **9/940 = 0.010 [0.005, 0.018]** | 16/600 = 0.027 |
| nulls per star (any of 4) | 8/235 = 0.034 | 9/150 = 0.060 |
| nulls >= 7 not rejected by v2 | 20/42 = 0.48 | 23/48 = 0.48 |
| real long signals >= tau_long per searched star | **3/235 = 0.013** (2 stars; all 3 tier C) | 510/4,350 = 0.117 (252 stars) |

Injection recovery. "Detected" means the top signal matched the injection with MES >= tau_long;
intervals are 95% Wilson. Far injections went into the raw flux before detrending, as
pre-registered.

| cell | far: detected | far: kept (not C) | 50 pc |
|---|---|---|---|
| 1.0-1.5 R_e, 15-30 d (headline) | 3/470 = 0.006 [0.002, 0.019] | 3/470 | 55/300 = 0.183 (v2 validation) |
| 1.0-1.5 R_e, 30-50 d | 0/235 [0, 0.016] | 0 | (not measured at 50 pc) |
| 1.5-2.0 R_e, 15-50 d | 9/235 = 0.038 [0.020, 0.071] | 9 | 59/150 = 0.393 (reference stars, this run) |
| 2.0-3.0 R_e, 15-50 d | 36/235 = 0.153 [0.113, 0.205] | 36 | 78/150 = 0.520 (reference stars, this run) |

The far stars do worse at the same real-sector count too: headline cell with 13+ sectors 3/62 =
0.048 vs 21/36 = 0.58; 2-3 R_e with 6-12 sectors 0.15 vs 0.69. Every detected far injection was
vetted: 41 A, 7 B, 0 C.

**Why recovery is so much lower** (checked with `gate3/posthoc/inj_diag.py`). Plotted against
the injected white-noise SNR, recovery is the same for far and 50 pc stars:

| injected SNR | far | 50 pc |
|---|---|---|
| 12-20 | 5/9 (1.5-2 R_e), 15/34 (2-3 R_e) | 23/31, 23/41 |
| > 20 | 22/22 | 87/101 |

So the search behaves the same, and the loss is physical. At the same Tmag cut, the far stars
are:
- larger, because they are intrinsically brighter: TIC R* median 0.29 vs 0.23 R_sun, so transits
  are ~40% shallower;
- noisier, because they are fainter: x 1.85;
- covered by fewer transits in the data: median 2 vs 4;
- more blended: median CROWDSAP 0.82 vs 0.91.

The median injected SNR is 3.3 vs 10.1 at 1.5-2 R_e, and 7.1 vs 18.7 at 2-3 R_e. The far
injections that were detected sit on stars with many sectors (median 31.5 real sectors, median
18.5 transits).

The follow-up's 100 pc estimate (2.5) assumed the 50 pc recovery at each Tmag. The measured far
headline recovery is about an eighth of that (0.005 vs 0.041 per listed star), which gives 0.21
expected headline planets instead of the follow-up's 0.65-1.73.

## R4. Projection to the full list (19,827 stars)
Expected detectable planets (detected and not C, central eta from DC15), with 95% bootstrap
intervals over sample stars and the low / high eta cases:

| cell | planets [95%] | eta low / high |
|---|---|---|
| 1.0-1.5 R_e, 15-30 d | 0.21 [0.00, 0.44] | 0.15 / 0.29 |
| 1.0-1.5 R_e, 30-50 d | 0.00 [0, 0] | 0 / 0 |
| 1.5-2.0 R_e, 15-50 d | 2.20 [1.25, 3.27] | 1.71 / 2.92 |
| 2.0-3.0 R_e, 15-50 d | 9.78 [7.60, 12.22] | 7.06 / 14.14 |
| **sum** | **12.19 [9.41, 15.12]** | 8.92 / 17.35 |
| sum without 2-3 R_e (DC15 early-M caveat) | 2.41 | low eta 1.86 |

Yield and CPU by predicted-sector bucket (a post-hoc split, `gate3/posthoc/yield_by_bucket.py`):

| predicted sectors | share of stars | share of yield | share of search CPU |
|---|---|---|---|
| 26+ | 9% | 64% (7.8 planets) | 28% |
| 13-25 | 6% | 12% | 9% |
| 6-12 | 85% | 24% | 63% |

Expected false alarms surviving v2:
- F_direct = 0.0, bootstrap [0, 0]: the 3 real signals are all C.
- **F_scaled = 5.5 [0.0, 16.0]**: non-K signals per star x s_50 = 13/462.
- So F = 5.5, and F / planets = 0.4.

The vetting load is ~195 signals >= tau_long over the full list. The follow-up had guessed
2,000-3,000 from the 50 pc rate.

Stricter cuts (not part of the verdict):

| cut | planets | F |
|---|---|---|
| A/A+ only | 10.3 | 1.3 |
| >= 4 transits | 11.8 | 3.0 |
| both | 10.1 | 0.4 |

Cost (measured):
- **CPU:** the far real search took 51.1 CPU-s per searched star (median 39.1, max 463; the 50 pc
  survey took 74.4 s). With 0.78 of the list having data, that is **221 CPU-h**.
- **Wall time:** at 3 workers and <= 3.0 CPU-h per worker, **25 shards of ~3.8 h in 5 waves of
  5, ~21 h wall** (0.9 d).
- **Disk:** result records 258 MB (16.6 kB per star) plus a 79 MB vetting working set (PDCSAP +
  SAP for ~132 stars, deleted after each batch) = 337 MB at peak.

## R5. Criteria
| criterion | value | limit | result |
|---|---|---|---|
| (a) expected detectable planets, sum 1.0-3.0 R_e, 15-50 d | 12.19 [9.41, 15.12] | >= 1.0 | PASS |
| (b) null rate per realisation | 0.0096 [0.005, 0.018] | <= 0.0533 | PASS |
| (c) F <= 10 x planets | 5.5 | <= 121.9 | PASS |
| (d) wall / disk | ~21 h / 337 MB; vetting load ~195 | < 120 h / < 500 MB | PASS |

**Verdict: recommend the extension.**

## R6. Post-hoc checks (not part of the verdict)
**Does the yield model over-predict?** The check: apply the same model to the 50 pc survey
(`gate3/posthoc/calib50b.py`; K-tier list from `calib50c.py`). It uses the 150 reference stars' injections in the 1.5-2 and
2-3 R_e cells (run in the same kernels), x 4,350 searched stars, with the same DC15 rates.
- **Predicted:** 13.7 detectable 1.5-3 R_e planets at 15-50 d (eta low 10.2, high 19.3).
- **Found:** about **5 known planets** in the K tier (TICs 336128819, 278661431, 299798795,
  141608198 and 271971130, all at 15.7-19.3 d; radii estimated roughly from the search depth and
  TIC R*). There were also 13 A/B survivors, all judged more likely false (noise alone predicted
  ~26).
- **Ratio:** the model over-predicts by ~2.7x (observed / expected 0.37, Poisson 95% range
  0.12-0.85).

Two explanations fit, and this check cannot separate them:
- the pre-registered caveat: DC15's Kepler hosts are mostly early M and have more 1.5-3 R_e
  planets than mid-to-late M dwarfs;
- real planets are harder to keep than injections.

Scaled the same way, the far projection is **0.21 + 11.98 x 0.37 = ~4.6 planets (range
~1.6-10)**, still above criterion (a)'s 1.0. In the headline cell the model agrees with the
survey: it expected 0.86-1.44 and the survey found one known planet near 1.4 R_e (TIC 153065527,
16.5 d).

**Are the far planets already known?** 22 hosts on the list have TOIs (25 TOIs). None of them is
at 1-3 R_e and 15-50 d; the longest-period small one is at 13.7 d. So the projected planets would
be new (`gate3/posthoc/toi_known.py`).

## R7. Typical far candidate and what confirming it takes
The typical kept far injection:
- host Tmag median 13.5 (16-84%: 13.1-14.6), R* 0.31 R_sun;
- depth median ~6 ppt;
- on a star with 20-50 sectors, so the ephemeris is good (median 18 transits);
- RV semi-amplitude ~4 m/s under a rough mass-radius relation (median 4.3 m/s; 1.5-3 R_e at
  15-50 d).

What confirming one takes:
- **Transit photometry:** a 6 ppt transit on a T ~ 13.5 M dwarf needs ~1-2 ppt per 10-min bin in
  the red / near-IR, which means **1-2 m class telescopes** (SPECULOOS-, LCO 1-m- or MuSCAT-type).
  Feasible, but each transit lasts hours and comes once every 15-50 d.
- **Mass:** K ~ 4 m/s on a V ~ 16-17 star needs **8-10 m class red / near-IR spectrographs**
  (MAROON-X, ESPRESSO, IRD) and many nights per target, so it is marginal. For most targets the
  realistic route is statistical validation (TRICERATOPS, high-resolution imaging).

Compared with 50 pc candidates (T ~ 11-13), these are about 1-2 magnitudes harder.

## R8. What was cut or not checked
- **Not checked:**
  - 273 SPLIT and 256 DUPLICATE TIC dispositions (counted among the first ~42k rows) stay in the
    list, as in the survey.
  - The 50 pc reference stars were measured at detection level only, not vetted.
  - There is no 50 pc number for 1.0-1.5 R_e at 30-50 d (the far value is 0/235 anyway).
  - DC15's occurrence rates are for early M dwarfs (R6).
  - Stratum interleaving used (rank + 0.5) / quota.
- **Coverage** runs through TESS-SPOC S86 at MAST (S81 in the bucket). S87+ would raise real /
  predicted sectors; the list does not change.
- **The real-signal false alarms rest on 3 signals in 235 stars**, so F_direct = 0 is a
  small-number zero. F uses F_scaled, which takes the 50 pc survival rate.
- **Nothing was cut for time:** all 300 sample stars and 150 reference stars finished.

## R9. Launch package
Frozen list: `<OUT>/gate3/targets_100pc.csv` (19,827 stars, sha256 `31f26b01...`). It has the
same columns as `survey_targets.csv`, so `run_survey.py` reads it unchanged. **Nothing was
pushed.**

**Optional cheaper first wave (a suggestion, not tested):** the 3,024 stars with >= 13 predicted
sectors carry ~76% of the projected yield for ~37% of the CPU (R4). To skip the full 21 h,
build them as a filtered list (`n_sectors >= 13`) and run them first.

**Before launching:**
- Merge this branch into main and push main to the private origin: `--mode git` kernels clone
  the repo at a pinned, pushed commit. Git mode was never tested. Gate 3 and the smoke test used
  `--mode embed`, which works without a push.
- Re-check that TESS-SPOC at MAST still stops at S86.

**Shard plan.** 25 shards, using the runner's position-mod-N split of the list sorted by sector
count, so each shard gets the same mix. 3 workers each, ~3.8 h per shard from the measured far
CPU, guard `--budget-hours 11`. 5 waves of 5 (the account's session cap), ~21 h wall in total.

**Commands** (main checkout; `PY=.venv/bin/python`, `KPY=~/miniconda3/bin/python`):
```
T=outputs/planets_spike/gate3/targets_100pc.csv
$PY -I spikes/mdwarf_stack/survey/kaggle/build_kernel.py --of 25 --tag survey100 \
    --targets $T --min-sectors 6 --slug transit-survey100 --mode embed
#   (or --mode git --commit $(git rev-parse HEAD) once main is pushed)
TAG=survey100 SLUG=transit-survey100 spikes/mdwarf_stack/survey/kaggle/shards.sh push 25 0 1 2 3 4
TAG=survey100 SLUG=transit-survey100 spikes/mdwarf_stack/survey/kaggle/shards.sh status 25
#   push 5-9, 10-14, 15-19, 20-24 as each wave's slots free (a refused push prints "NOT pushed")
$KPY -I spikes/mdwarf_stack/survey/kaggle/fetch_outputs.py --of 25 --tag survey100 \
    --slug transit-survey100 --targets $T
#   (not `shards.sh fetch`: it does not pass --targets, so the merge would check the 50 pc list)
#   a shard stopped by the guard: build_kernel.py ... --shards I --skip-from-local, then push it again
```
Outputs land in `outputs/planets_spike/survey100/` (shards/, merged/), separate from the 50 pc
survey.

**Vetting afterwards.** Frozen v2, unchanged. At most 2 niced local processes. Work in batches
of <= 200 stars and delete the light curves after each batch. Expect ~200 signals in total.
```
O=outputs/planets_spike/survey100
$PY -I spikes/mdwarf_stack/survey/vet_rows.py --merged $O/merged/results.jsonl.gz --out $O/vet
split -l 200 $O/vet/tics_long.txt $O/vet/batch_
# per batch B:
$PY -I spikes/mdwarf_stack/survey/prep_vet.py $O/vet/batch_B --merged $O/merged/results.jsonl.gz --max-mb 150
VET_LC_DIR=data/planets_spike/lc_cache/lc $PY -I spikes/mdwarf_stack/vetting/fetch_sap.py list --file $O/vet/batch_B
$PY -I spikes/mdwarf_stack/gate3/catalog3.py --targets $T --tics $O/vet/batch_B --out $O/vet/star_info.json
VET_LC_DIR=data/planets_spike/lc_cache/lc nice -n 19 ionice -c3 $PY -I spikes/mdwarf_stack/gate3/vet100.py \
    --info $O/vet/star_info.json --rows $O/vet/rows_long.jsonl --jobs $O/vet/jobs_long.jsonl \
    --out $O/vet/vet2_100pc.jsonl
#   then delete the batch's light curves (lc_cache, vetting/sap/tic*.npz of the batch)
```
Then:
- run the pixel check on every A/B signal (`gate3/pixel3.py` style; keep the results apart from
  the 50 pc pixel file);
- grade with `vet2.grade2` + `frozen_vet2.json`;
- make the cards.

Expect roughly 10 A/B signals, about half of them false alarms (R4, R6). Treat each one as
uncertain until the pixel check, the catalogues and follow-up have all been done.

Kaggle kernels used here: `antimatter543/transit-gate3-000of005` to `-004of005` (private). They
are safe to delete; their outputs are fetched to `<OUT>/gate3/kaggle_out/`.
