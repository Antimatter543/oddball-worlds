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

## Amendments (made before any null or host search was run)
All three are compute changes. They were made after timing searches on two real **control** stars,
TIC 441721594 (40 sectors) and TIC 278078111 (10 sectors).

1. **Search engine.** Brute-force folding cost about 1,500 CPU-s per iteration for the 40-sector control. The coarse
   search now uses a **Fast Folding Algorithm (FFA)**. Its tree carries ΣN, ΣD and the epoch count (added) and the
   max SES² (max-combined), so the valid-peak statistic is unchanged. The cyclic-shift FFA mis-aligns epochs by
   about T·binw/P² samples once the drift exceeds a period. Below P_sw = √(T·binw), where that error would exceed
   1 sample, the exact brute fold is used. A unit test (`test_ffa.py`) checks FFA against brute force at integer
   periods and checks blind recovery of a synthetic transit across non-consecutive chunks.
   The **0.5 h and 0.67 h boxes are dropped**: at stride 1 they cost two-thirds of the search. Grazing transits of
   the latest M dwarfs lose some SNR in a 1 h box; the injections (b up to 0.9) measure that.
2. **Coarser coarse-grid.** 5 durations (1, 1.5, 2.17, 3, 4 h), spaced about 1.4× apart (was 7). Coarse stride and
   phase bin are D/2 (was D/3). The period step follows the same rule, ΔP/P = s·Δt/T_base. Refinement is unchanged:
   ±3 coarse steps, 5× finer, stride 1. This cut the 10-sector control from 39 s to 13 s of CPU per iteration.
3. **Injection count.** 100 per cell (400 total) instead of 200, to fit the time budget. The criteria are unchanged,
   and Wilson intervals are reported.

4. **Injection count, second cut.** 60 per cell (240 total; 8 per control) instead of 100. The sector-count
   ordering of the run put the expensive 26+ sector controls last, and they made 400 too slow.

## Threshold (fixed from the nulls; committed before the host search)
- **τ = 12.27.** This is the 2nd-highest of 120 null statistics, by the pre-registered rule. The maximum is 31.24.
- All 5 top nulls are **inverted** light curves with best periods of 1.17-1.39 d and 93-596 epochs. The
  inverted-plus-rolled copies of the same stars do not show this. A coherent ~1.2-1.4 d signal across years, in
  different stars, is consistent with fast rotation of mid-to-late M dwarfs leaking through the 0.5 d biweight
  (failure mode 5).
- The pre-registered τ is what the pass/fail uses. A **post-hoc diagnostic**, restricted to signals with P ≥ 10 d,
  is reported separately and is labelled as not pre-registered.

## Mid-run change from the owner (2026-11-11, after the host search, before the injections)
- **CPU:** the machine has 6 physical cores, so the work was cut to ≤ 3 worker processes under
  `nice -n 19 ionice -c3`. The real-control and host runs were restarted at 3 workers. `run_jobs` resumes, so no
  results were lost.
- **GPU:** the coarse search was ported to torch/CUDA (`gpu_search.py`). It does an exact brute fold of every
  period on the pre-registered grid, with no FFA approximation. Phases are integer plus float32 fraction. ΣN, ΣD
  and the count come from one `index_add`. Dominance is two-pass (a bin is invalid if any of its samples has
  SES² > 0.8·MES²), which is identical to the max test. Refinement, masking and iteration are the same CPU float64
  code.
- **Validation, tolerance fixed before comparing** (|ΔP/P| < 1e-3 or the same alias, and |ΔMES|/MES < 2%), on 4
  controls and 4 injections (`validate_gpu.py`):
  - The per-period coarse MES on the same grid agrees to median 0 and p99 < 1e-6 relative.
  - 1-6 of 15-30k periods flip validity at the float32 dominance tie.
  - 7 of 8 full searches agree within tolerance. All 4 injections give identical match outcomes.
  - The exception is control TIC 278078111. CPU top: 53.877 d, MES 6.17. GPU top: 37.661 d, MES 6.48. Both
    final MES values come from the same CPU refinement, so the GPU found a higher, real noise peak that the CPU's
    approximate FFA coarse grid ranked lower. Both are far below τ.
  - The nulls that set τ (P < 1.4 d) lie in the CPU's exact-brute region, so τ is unaffected by the engine.
- **Speed:** one GPU process is 5-7× faster per search than one CPU worker.
- **Injection run:** to finish in budget, the GPU ran the 240 injections from the largest stars down while 3 niced
  CPU workers ran from the smallest up. Each row records its engine (`inj_gpu.jsonl`, `inj_cpu.jsonl`), and rows
  are de-duplicated by id. Hosts, nulls and real controls are all CPU.
- **GPU load:** utilisation was ~97% during GPU jobs, not short bursts. Peak 54 C, 2.2 GB VRAM in total
  including the desktop. No CUDA errors and no `nvidia-smi` failures.

## Results log

### 2026-11-11: kill test. Formal PASS at the pre-registered τ; the threshold is flagged under-calibrated (rotators)
Commands, run from the worktree (outputs in `outputs/planets_spike/`):
`fetch.py`, `select_controls.py`, `make_jobs.py`, then `run_jobs.py jobs/{nulls,controls_real,hosts_only}.jsonl`
(CPU), then `run_jobs.py jobs/inj240.jsonl --gpu` together with `jobs/inj240_rev.jsonl` (CPU), then
`analyze.py all` and `validate_gpu.py compare`. Data: SPOC 2-min PDCSAP (TESS-SPOC FFI for 5 control sectors),
downloaded 2026-11-11. Provenance is in each NPZ's `meta`.

**Threshold.** τ = 12.27, from 120 nulls. 4 of the 30 *real* controls exceed it, at P = 1.18-1.39 d: TIC 233570240
(24.0), 339607562 (15.5), 188777432 (14.3) and 284476342 (13.3). These are the same stars whose inverted nulls
set τ. They are fast rotators, and the 0.5 d biweight does not remove rotation at P ≈ 1.2-1.4 d. By the
pre-registered rule the threshold is **flagged**, and results are also given at τ_cons = 23.99.

The orbit-period check is clean. 13% of null signals fall within 3% of a 13.7/27.4 d multiple or fraction, which
is about the chance rate for those windows. No top null or real-control peak is orbit-related.

A **post-hoc diagnostic (not pre-registered)** restricted to P ≥ 10 d: the null maximum is 8.27 and the 2nd
highest is 7.56. The real-control maximum is 8.27 (TIC 233903720, 18.08 d, 5 epochs; sub-threshold, not vetted).

**(a) Known planets: blind, all sectors, CPU engine.**

| planet | sectors used | P_true | P_det | order | MES | ≥ τ (12.27) | ≥ τ_cons (23.99) | epoch match (det/known) | alias |
|---|---|---|---|---|---|---|---|---|---|
| LHS 1140 b | 3, 30, 107 | 24.7373 | 24.7372 | 1st | 33.07 | yes | yes | 1.00 / 1.00 (3 transits) | none |
| LHS 1140 c | same | 3.7779 | 3.7779 | 2nd | 25.47 | yes | yes | 1.00 / 0.93 | none |
| TOI-700 d | 37 sectors, S1-S97 | 37.4237 | 37.4236 | 3rd (after c, b) | 15.12 | yes | no | 1.00 / 0.92 (22 epochs) | none |
| TOI-700 e | same | 27.8097 | 27.8100 | 4th | 9.90 | no | no | 1.00 / 0.93 | none |
| TOI-700 b / c | same | 9.977 / 16.051 | exact | 2nd / 1st | 16.98 / 54.25 | yes | c only | ≥ 0.91 | none |
| Gliese 12 b | 42, 43, 57, 70, 84 | 12.7615 | 12.7614 | 1st | **12.16** | **no (by 0.11)** | no | 1.00 / 1.00 (7 transits) | none |

**(a): 2 of 3 at τ, PASS. 1 of 3 at τ_cons, FAIL.** Gliese 12 b is the top blind signal at the exact period,
and it sits above every null with P ≥ 10 d (8.27). It misses only the rotator-set τ.

**(b) Injections:** 240 total, 60 per cell. They went into the raw binned PDCSAP of the 30 controls, 8 per star.
The engines are 176 CPU and 64 GPU (validated equivalent). 95% Wilson intervals; n in brackets.

| cell | all stars | 6+ sectors | 1-5 sectors | 26+ sectors | 6+ sectors at τ_cons | 6+ sectors at post-hoc τ_long = 7.56 |
|---|---|---|---|---|---|---|
| 1.0-1.5 R_e, 15-30 d | 0.17 [0.09, 0.28] (60) | **0.23 [0.13, 0.37] (44)** | 0.00 [0.00, 0.19] (16) | 0.50 [0.22, 0.78] (8) | 0.05 [0.01, 0.15] | 0.39 [0.26, 0.53] |
| 1.0-1.5 R_e, 30-50 d | 0.10 [0.05, 0.20] (60) | 0.18 [0.08, 0.36] (28) | 0.03 [0.01, 0.16] (32) | 0.42 [0.19, 0.68] (12) | 0.14 [0.06, 0.31] | 0.50 [0.33, 0.67] |
| 1.5-2.0 R_e, 15-30 d | 0.38 [0.27, 0.51] (60) | 0.43 [0.30, 0.58] (44) | 0.25 [0.10, 0.49] (16) | 0.50 [0.22, 0.78] (8) | 0.20 [0.11, 0.35] | 0.57 [0.42, 0.70] |
| 1.5-2.0 R_e, 30-50 d | 0.38 [0.27, 0.51] (60) | 0.71 [0.53, 0.85] (28) | 0.09 [0.03, 0.24] (32) | 1.00 [0.76, 1.00] (12) | 0.21 [0.10, 0.40] | 0.86 [0.69, 0.94] |

Every 6+ sector injection had at least 2 transits in the data.

**(b): 0.23 vs ≥ 0.20, PASS on the point estimate**, but the interval [0.13, 0.37] includes 0.20. At τ_cons it is
0.05, FAIL.

Comparison with Gillis (6.5%, all stars, vetted-candidate level): ours over all stars is 0.17 [0.09, 0.28], at
*detection* level with no vetting, so it is an upper bound on a like-for-like number.

**Where the misses go:** 134 of 240 were never matched by any signal.
- 48 are the 8 injections on each of the **6 fast rotators**: all 3 signal slots were spent on P < 3 d rotation
  (failure mode 5).
- 78 never produced a signal above the MES floor of 6, mostly on stars with 1-5 sectors.

**(c) Compute.** The cost per iteration was fitted on all measured searches (333 CPU, 72 GPU; cheap and 40-sector
stars alike). It was applied to tess-point coverage of 14,471 TIC M dwarfs (Teff 2700-3400 K, d < 50 pc). That
coverage includes pointings up to S132, beyond today; on the controls, real data cover 0.74× the predicted
sectors, so the estimate is conservative. 2.77 signals per star, from the real controls.
- **CPU:** mean 0.029 CPU-h per star (max 0.33). **3,000 stars: 87 CPU-h; 8,000 stars: 230 CPU-h**
  (3.2 days on 3 niced workers). Limit 1,500: **PASS**.
- **GPU:** mean 0.0054 GPU-h per star. **3,000 stars: 16 GPU-h; 8,000 stars: 43 GPU-h** (1.8 days on the
  1650S). Limit 2 weeks: **PASS**.
- **What dominates:** the period count grows as T_base/binw. Brute fold costs N_samples × N_periods. The FFA
  barely helped, because non-consecutive sectors leave the dense series ~83% empty. What actually cut the cost:
  10-min bins, a D/2 coarse stride, 5 boxes, and the GPU exact fold (about 1.4 G fold-elements/s, ~5-7× one core).

### Verdict: continue-with-changes
All three criteria pass at the pre-registered τ. But criteria (a) and (b) pass by margins smaller than their own
uncertainty, and both fail at the τ_cons fallback the pre-registration required once the threshold was flagged.
The cost question is settled: an 8,000-star all-sector search costs ~230 CPU-h, or ~2 days on the GPU. The
sensitivity question is not. A single global threshold is set by M-dwarf fast rotation at ~1.2-1.4 d, which also
eats the signal slots on 20% of the controls.

**Changes before continuing:**
1. Remove rotation explicitly: fit and subtract P_rot and its harmonics, or use a GP, before the box search, and
   test it against injections.
2. Pre-register a period-dependent threshold (for example 1-10 d and 10-60 d) calibrated on several hundred null
   realisations.
3. Enlarge the control set, so the 1% false-alarm level is measured rather than extrapolated.

**Biggest risk:** after rotation is handled, the long-period false-alarm tail is unknown. It comes from 13.7 d
pairs and edge systematics, and from 30 controls and 120 realisations the post-hoc τ_long is only an estimate.
If the true long-period tail is near 9-10 rather than 7.6, Gliese 12 b-like and 1.0-1.5 R_e signals sit right
at it.

## Gate 2 pre-registration (2026-11-11, committed before any TEST-star or host search)
Question: with rotation handled and thresholds set per period range, is the all-sector search sensitive enough at
10-60 d to justify a full survey? A clear "no" is acceptable. Everything not stated here is the gate-1 method above.

### Engine (forced change)
The GPU fell off the bus at 11:55 (Xid 79) before gate 2 used it. Every gate-2 search uses the gate-1 **CPU
engine** (exact brute fold below P_sw = sqrt(T·binw), FFA above, then the same CPU refinement), at most 3 workers
under `nice -n 19 ionice -c3`. Nulls, controls, injections and hosts all use this one engine, so the threshold and
the recoveries are self-consistent. For P >= 10 d the whole long range is in the FFA regime (its phase error
T·binw/P^2 is below 1 sample there). Gate 1 showed the FFA coarse grid can rank a real noise peak lower than the exact
GPU fold (TIC 278078111), so CPU null maxima may be slightly *lower* than exact ones; this is listed as a risk.

### Splits (`gate2/select_g2.py`)
120 controls, same rules and strata as gate 1 (24/32/32/32 for 2-5/6-12/13-25/26+ predicted sectors), drawn with
seed 20261121. DEV = the 30 gate-1 controls + 30 new; TEST = 60 new (one third / two thirds of each new stratum, seed
20261122). Every method choice below is made on DEV only.

### Two independent searches
- **Short:** 1-10 d. **Long:** 10-60 d. Each is `search_star(..., prange=)` on the same preprocessed light curve,
  with its own signal slots. Nothing found by one search is masked in the other.
- Slots: null and injection runs keep the **top** signal only (max_sig 1; the run statistic is that signal's MES,
  0 if no peak reaches the MES floor of 6). Real-control runs keep 3, host and hold-out runs 6.

### Rotation removal: candidates and DEV choice rule
- **none:** gate-1 preprocessing.
- **harm** (`gate2/search2.py`): per sector, Lomb-Scargle of the raw normalised flux over 0.1-5 d. If the peak's
  normalised power is >= 0.02, fit 3 harmonics of that period with amplitudes that vary as linear splines (knots
  every 3 d within each segment), plus a spline baseline, by least squares with 3-sigma two-sided clipping of the fit
  points (3 passes). Divide the flux by the model, then run the unchanged gate-1 preprocessing.
- Not tested: a biweight window scaled to P_rot (a window of P_rot/3 still passes ~17% of the rotational amplitude at
  P_rot = 1.3 d, while 0.43 d is under 3× the longest box, so it would cut 3-4 h transits), and a GP (celerite2 is
  not installed, and a fitted GP per injection does not fit the CPU budget).
- **Choice rule (DEV):** 120 paired injections (2 per DEV star; R_p 1.0-2.0 R_e, P 15-50 d; the same injection under
  both arms; long search). Recovery counts at a provisional MES of 7.5 (only for this comparison). Choose harm if,
  on DEV slow rotators, it loses <= 3 points against none and, on DEV fast rotators, it recovers at least as many.
  Otherwise choose none. The DEV real controls under both arms are reported too.
- **Fast/slow** is fixed from DEV, before TEST, using the rotation measured on each star's real (uninjected) light
  curve. It is recorded in the frozen-method entry and not changed afterwards.

### Thresholds (DEV nulls only)
- Null realisations per DEV star: long, 7 (inverted; inverted plus roll, 2 seeds; roll, 4 seeds), giving 420; short,
  5 (inverted; inverted plus roll; roll, 3 seeds), giving 300.
- **tau_long** = the (floor(0.005 N) + 1)-th highest long-null statistic, so at most 0.5% of DEV null realisations
  exceed it (3rd highest of 420). **tau_short** uses the same rule on the short nulls (2nd highest of 300).
- If fewer than 400 long nulls finish, tau_long is not computed and criterion (a) fails.

### Prediction for known planets
MES_pred = eff × depth × sqrt(sum of 1/sigma^2 over in-transit bins in the preprocessed data), with depth and T14
from the TOI table (else exoplanet.eu, else computed from R_p, R*, M*, b = 0.3). eff is the median of
MES_measured / SNR_white over DEV harm-arm injections with SNR_white >= 12 that were matched (fixed in the frozen
entry). It is 0 if fewer than 2 transits are in the data. **Predicted detectable** means MES_pred >= 1.25 × tau_long.

### Criteria (all five needed to recommend the survey)
- **(a) Honest threshold:** tau_long comes from >= 400 DEV null realisations, and at most 2% of the 60 TEST real
  controls (so at most 1) have a long-search signal (any of 3) with MES >= tau_long. If more do, the gate fails on
  (a), whatever the rest shows.
- **(b) Sensitivity:** 200 TEST injections, R_p 1.0-1.5 R_e, P 15-30 d, all on TEST stars with 6+ sectors of data.
  Each goes into the raw binned flux before rotation removal and detrending. It is recovered if the top long signal
  matches it (the gate-1 match rule) with MES >= tau_long. Pass if the 95% Wilson lower bound is >= 0.20. Also
  reported: 80 at 1.0-1.5 R_e and 30-50 d, 60 in each 1.5-2.0 R_e cell, 36 in the headline cell on 2-5 sector stars,
  per sector bucket, and fast vs slow.
- **(c) Known planets:** Gliese 12 b, LHS 1140 b and TOI-700 d each have a matching long-search signal >= tau_long
  (reported as non-blind). AND at least half of the predicted-detectable hold-out planets (`holdout_g2.csv`, 10 < P <=
  60 d) are recovered blind: some long-search signal >= tau_long matches the planet (P or an n <= 6 alias, plus
  epochs). If fewer than 3 hold-out planets are predicted detectable, (c) is not passed.
- **(d) No harm:** for every TEST injection the same injection is also run with rot = none. On TEST slow rotators,
  recovery(harm) − recovery(none) >= −0.03 over all cells. On TEST fast rotators, recovery under harm is above 0;
  both arms are reported.
- **(e) Cost:** the CPU cost per star (rotation step, long and short search, measured on every gate-2 search) is
  extrapolated to 8,000 stars from the gate-1 TIC sample. The GPU cannot be measured now. GPU time = CPU time / 5,
  the low end of the GPU speed-up measured in gate 1. Pass if that is under 14 days; 3-worker CPU wall time is also
  reported.
- **Verdict:** all five pass means recommend the survey. If (a) or (b) fails, recommend stopping, or name the one
  change that could plausibly fix it. Nothing is adjusted after a TEST or host result is seen.

## Gate 2: resume point (updated after each stage; a reboot may kill the run without warning)
**State at 2026-11-11 ~12:10 AEST.** The GPU fell off the bus at 11:55:21 (Xid 79, reset required) during the
control downloads, before any gate-2 search touched it. Gate 2 runs on CPU only (the validated gate-1 CPU engine:
exact brute fold below P_sw, FFA above), at most 3 workers under `nice -n 19 ionice -c3`.

Done:
- Control set enlarged to 120 (`gate2/select_g2.py`, seeds 20261121 draw / 20261122 split): 24/32/32/32 per
  sector stratum. DEV = the 30 gate-1 controls plus 30 new; TEST = 60 new. Written to
  `data/planets_spike/catalogs/controls_g2.csv`. No TEST star has been searched.
- Hold-out list built (`gate2/holdout.py build`) from the cached exoplanet.eu and ExoFOP TOI tables (the NASA
  Exoplanet Archive TAP returned a maintenance page): `data/planets_spike/catalogs/holdout_g2.csv`, 84 planets with
  10 < P <= 60 d on 68 hosts. No hold-out host has been searched.
- `search.py`: `search_star(..., prange=)` for separate short (1-10 d) and long (10-60 d) searches (None = gate 1).
- `gate2/search2.py`: rotation removal (per-sector Lomb-Scargle, harmonic fit with spline amplitudes), being
  tested on DEV fast rotators only.

Running / cached:
- Downloads of the 90 new controls (`outputs/planets_spike/gate2/fetch_controls.log`) and the 68 hold-out hosts
  (`fetch_holdout.log`) into `data/planets_spike/lc/` (one NPZ per star; `fetch.py` skips stars already present, so
  re-running the same command resumes).

Next commands (from the worktree; Python = a venv with wotan, e.g. the gate-1 worktree's `.venv`):
`nice -n 19 ionice -c3 python -I spikes/mdwarf_stack/fetch.py $(cat data/planets_spike/catalogs/g2_fetch_order.txt)`
and the same with `holdout_fetch.txt`; then the DEV rotation tests (next commit).
