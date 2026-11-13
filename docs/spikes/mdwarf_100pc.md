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
