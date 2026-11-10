# Spike: all-sector stacked transit search for temperate planets around nearby M dwarfs

Status: **pre-registered 2026-11-11, before any search was run on any star.** This is a time-boxed (~3 h) kill test.
A clear "no" is an acceptable outcome.

## Question
Does a blind box search that stacks **all** TESS sectors per star (not only consecutive ones) recover the long-period
planets the deepest published M-dwarf search missed? And does its compute cost scale to a few thousand stars on one
12-core machine?

The reference is Gillis, Cloutier & Pass (arXiv:2602.23364). They searched 8,134 M dwarfs in sectors 1-86 with TLS
over 0.2-30 d, concatenating at most four consecutive sectors. Their 1.0-1.5 R_earth, 15-30 d target-weighted
recovery is 6.5% (Table 4). LHS 1140 b was undetectable because sectors 3 and 30 are not consecutive. Sensitivity to
Gliese 12 b was below 10% survey-averaged. Their "recovered" means a *vetted candidate* with P, nP or P/n (n = 2-6)
within 0.2%. Our "recovered" is a *detection* (signal above the threshold) with no TRICERATOPS or centroid vetting,
so our number is an upper bound on theirs at the same stage. This is stated next to every comparison.

## Pass/fail criteria (all three required)
- **(a)** At least 2 of {LHS 1140 b, TOI-700 d, Gliese 12 b} are recovered blind, above the threshold τ.
- **(b)** Injection recovery for 1.0-1.5 R_earth at 15-30 d, on control stars with 6+ sectors of data, is **≥ 20%**.
- **(c)** The extrapolated full-survey search cost is **< ~1,500 CPU-hours** for 8,000 stars (3,000 also reported).

**Fail** otherwise. τ is never adjusted to make a criterion pass. The numbers are reported either way.

## Fixed method (frozen here; nothing below may change after a known-host result is seen)

### Data
- For every sector: SPOC 2-min PDCSAP if it exists, else the TESS-SPOC FFI HLSP. 20-s data are not used (the 2-min
  file always exists alongside). QLP is not used.
- QUALITY bitmask 17087 (the lightkurve default, `oddball.tess.lightcurve`). Each sector is divided by its own
  median, then binned to 10 min on the global grid. FITS files are never written to disk.

### Preprocessing (identical for hosts, controls, nulls and injections)
1. Split into segments at gaps > 0.5 d. Drop the first and last 0.25 d of each segment (orbit-start ramps).
2. Flare clipping: fit a biweight trend (wotan, window 0.5 d). Remove bins more than 3σ *above* the trend, plus the
   next 3 bins (30 min of flare decay). Repeat twice. σ is the robust (MAD) scatter for the sector.
   No downward clipping, because transits are downward.
3. Detrend: divide by a wotan biweight with a **0.5 d window**. The longest transit searched is 4 h, so the window is
   3× that. After detrending, clip once more at 3σ upward.
4. Weights: per-sector robust σ of the detrended 10-min flux.

### Search (numba box search; the same code for every run)
- **Durations:** 3, 4, 6, 8, 10, 12, 15, 18, 24 bins (0.5-4 h). Each duration D is searched only over the periods
  where 0.33 ≤ D / D_c(P) ≤ 1.5. D_c is the central, circular transit duration for the star's TIC mass and radius
  (fallback 0.3 M_sun, 0.3 R_sun).
- **Single-event statistic (SES):** a box of width D, sampled at stride s = max(1, round(D/3)) bins. Windows less
  than 50% covered by data are dropped. For each sector and duration, the SES is divided by its own robust scatter.
  This is a red-noise (CDPP-like) correction.
- **Period grid:** 1-60 d, geometric, ΔP/P = s·Δt / T_base. T_base is the star's full first-to-last baseline, so
  epochs across non-consecutive sectors stay phase-coherent. Phase bins have width s·Δt.
- **Valid peak:** at least 2 transit epochs with data, and single-event dominance max(SES_i²)/MES² ≤ 0.8.
  MES = ΣN/√ΣD over epochs.
- **Refinement:** around the best coarse peak, ±3 coarse steps at 5× finer period step and stride 1, all allowed
  durations.
- **Iteration:** record the signal, mask ±1 D around every epoch, and search again. Stop when MES < 6.0, or after 6
  signals for host and real-control runs, or 3 for null and injection runs.

### False-alarm threshold τ (control set only; never the three hosts)
- **Control set:** 30 M dwarfs with Teff 2700-3400 K and d < 50 pc. Each has |ecliptic latitude| > 60° (for many
  sectors), no TOI of any disposition, and no exoplanet.eu planet within 30″. They are drawn with a fixed seed,
  stratified by predicted sector count: 6 stars with 2-5 sectors, 8 with 6-12, 8 with 13-25 and 8 with 26+
  (`select_controls.py`).
- **Null realisations per control (4):**
  1. Inverted: the detrended residual flipped, after the flare clip, so flares are not turned into dips.
  2. Inverted plus a per-sector circular roll (random shift).
  3. and 4. Per-sector circular roll with two seeds. This keeps within-sector systematics, including the ~13.7 d
     orbit pairs, but breaks cross-sector coherence.
- **Per-realisation statistic:** the maximum MES over its signals.
- **τ** is the 2nd-highest statistic among the ≥ 120 realisations, so 1 in 120 (0.8%) exceeds it.
- **Diagnostic:** the identical search on the real (unaltered) control light curves. If more than 1 real control
  exceeds τ, the threshold is flagged as under-calibrated. All results are then also reported at
  τ_cons = the highest real-control statistic. The periods of all null and real-control peaks are listed, to check
  for clustering near 13.7 and 27.4 d.

### Recovery and transit-time matching (hosts and injections)
- **Detected epochs:** epochs of the signal that have ≥ 50% coverage in the data. **Known epochs:** those of the
  planet, from the ExoFOP ephemeris, in the data.
- **Tolerance:** max(the known T14, the detected D).
- **Match:** P_det/P_true or P_true/P_det is within 1% of an integer n ≤ 6, and either ≥ 80% of the detected epochs
  are within the tolerance of a known transit, or ≥ 80% of the known transits are within the tolerance of a
  detected epoch.
- **Alias:** a match with n ≠ 1 is reported as an alias.
- A planet is **recovered** if any signal with MES ≥ τ matches it. A match below τ is reported but does not count.
  Masking inner planets is part of the blind iteration; the known periods are never used.

### Injections (control set, into the RAW binned flux before preprocessing)
- **Cells:** R_p ∈ [1.0, 1.5) or [1.5, 2.0] R_earth, uniform. P ∈ [15, 30) or [30, 50] d, log-uniform.
- **Geometry:** b uniform on [0, 0.9], t0 uniform, circular orbits, TIC M* and R*.
- **Model:** small-planet limb-darkened model, quadratic (u1, u2) = (0.25, 0.35), with a linear ingress/egress
  ramp and 10× supersampling inside each 10-min bin.
- **Target:** 200 per cell (800 total), assigned round-robin over the 30 controls with a fixed seed. If the budget
  runs out, the count is cut, never the criteria, and binomial 95% (Wilson) intervals are reported.
- **Recovered:** a match among the run's signals with MES ≥ τ. The denominator is all injections, including those
  with < 2 transits in the data; the fraction among ≥ 2 in-data transits is also reported.
- **Buckets:** by sectors of data: 2-5, 6-12, 13-25, 26+. Criterion (b) uses all injections on 6+ sector stars.

### Compute
- Each worker process runs one star at a time with numba on 1 thread. Up to 6 processes, so ≤ 6 cores.
- CPU seconds per search are recorded with the sector count, T_base and number of bins.
- **Cost model:** CPU seconds against (bins × periods), fitted on the measured searches over all controls, so it
  is not drawn from the cheapest stars.
- **Extrapolation:** applied to tess-point sector coverage for a TIC sample of Teff 2700-3400 K, d < 50 pc stars,
  with random draws of 3,000 and 8,000. The mean number of signals per star comes from the real-control runs.

## Results log
(Filled in after the runs. Amendments made before any host result is seen are listed here with their reason.)
