# Evaluation: injections, classical baseline, harness

The measuring stick for Track A (docs/ML_DESIGN.md §6, interfaces §9). Any scorer that writes
§9 score files is judged by the same code.

## Run it

```bash
export ODDBALL_DATA=/path/to/data
uv run python -I scripts/make_injections.py --sector 32 --name v1      # ~3 min
uv run python -I scripts/score_classical.py --sector 32 --inj v1      # ~16 min (2 passes)
uv run python -I scripts/evaluate.py --scorer classical
uv run python -I scripts/evaluate.py --compare classical classical_cm <your_scorer>
uv run python -I scripts/plot_injections.py --name v1 --split val     # visual sanity
```

## Plugging in a scorer

Write `outputs/scores/<scorer>/s0032/real.parquet` (every real light curve) and
`inj_v1.parquet` (every light curve under `data/injections/v1/s0032/{val,test}/`) with columns
`tic, sector, inj_id (null for real), score` (higher = weirder). Injection shards use the
ingest schema plus `inj_id` and `host_tic` (`tic == host_tic`), so read them like a sector:
`oddball.tess.arrays.iter_batches(dir, sector_grid(32))` gives NaN-padded `(n, L)` batches
on the global bin grid, for real and injected directories alike. Then run `evaluate.py`.

You do not need to calibrate: the harness ranks each injection against the real stars of
its own split and stratum. Calibrating per stratum (as §4 asks) is still what makes your
`real` ranking meaningful for the gallery.

## Injection set v1 (`oddball.inject`)

- **Hosts:** only from `split_of(tic) == split` (val or test); `known` never. A proportional
  stratified random sample over Tmag bin x camera/CCD (`oddball.common.strata`), with no
  quietness or quality cut. One injection per host. Seed 20261109.
- **Model:** multiplicative. PDCSAP `*= m(t)`. SAP `*= 1 - CROWDSAP * (1 - m)`, since SAP is
  not crowding-corrected. Errors unchanged. Averaged over each 30-min bin (5 sub-samples).
  `params_json` rebuilds the model exactly (`model_from_params`).
- **Kinds (500 each per split):** aperiodic_dipper, quasiperiodic_dipper, exocomet (linear
  ingress, exponential egress), single_dip (Gaussian), boyajian (3-7 overlapping asymmetric
  dips), outburst. Depth is log-uniform over 0.1-30% (peak |m - 1|), and duration is
  log-uniform over 2 h-5 d (see `sampling.py` for what duration means per kind). The peak bin
  always holds data, and >= 75% of the event window does.
- **Nulls (300 per split, `is_null=True`):** 10 events on 10 distinct CCDs with 30 hosts
  each. Every host of an event gets the identical model. 7 `null_ccd` (Gaussian, sharp-edged
  box, or ramp with instant recovery; 2 h-2 d) and 3 `null_dump` (a sharp drop at a real
  momentum dump read from QUALITY bit 32, then an exponential recovery of 2-8 h). Event depths
  are spread evenly in log over 0.1-30%.

## Classical baseline (`oddball.score.classical`)

Per channel (PDCSAP and SAP): robust skewness; Cody+2014 M (flux sign, so dippers < 0);
simplified Q (FFT peak 0.25-10 d, 20-phase-bin fold, white noise from point-to-point
scatter; scored as `Q * var_frac`); the largest dip from a box search (2 h-2.7 d) on a
3.25-d running-median-detrended residual, with its depth, duration and asymmetry;
a brightening box search; excess variance over the per-bin errors; and the fraction of bins
below -3 sigma. That gives 20 sub-scores, each turned into a stratum percentile against the
real stars. `score` is the stratum percentile of their mean.

Variants: `classical` (as specified), `classical_tail` (mean of -log(1-p): extremes dominate),
and `classical_cm`, the systematics filter v0. `classical_cm` removes common-mode bins and
momentum-dump-adjacent bins, recomputes everything, then rank-averages.

**Common-mode map:** for each channel x CCD x 30-min bin, the fraction of stars whose centred
2-h box residual is below -3 sigma. A cell is common-mode when that fraction exceeds the CCD's
median + 8 x robust sigma (floor 1%) and at least 5 stars dip. For an injection set, the map
is the real map OR the set's own map, thresholded against the set's own typical level (floor
5%). Every light curve in an injection set carries a signal, so a fixed fraction would flag
the injections themselves. Only time-aligned excesses survive, which is what an instrument
event looks like.

**Per-light-curve systematics feature:** `cm_frac`, `dump_frac` and `sys_frac` are the shares
of the star's anomalous-bin evidence (weighted by SNR^2) that sit in common-mode bins, next to
a momentum dump, or in either. `sys_flag = sys_frac >= 0.5 and n_anom >= 4`. The report shows
null FPR and astrophysical recall with and without vetoing `sys_flag`.

## Harness (`oddball.eval`)

- Each injection's percentile is its mid-rank among real stars of the same split and the same
  stratum. Top 1% means a percentile >= 0.99. Smallest strata hold ~40 real stars per split, so
  the cut is coarse there and never looser than 1%.
- Recall at top 1% and top 0.1% by kind x depth x Tmag (and x duration). The depth bins are
  half-decades centred on 0.3 / 1 / 3 / 10%.
- AUROC is the mean stratum-matched percentile.
- Null FPR is the fraction of nulls in the top 1%, per kind and per event.
- Known positives (`known` split) are ranked against all real non-known stars in their stratum.
- **Gate:** recall in top 1% >= 50% for non-null kinds in the ~0.3% bin (0.17-0.55%), **and**
  at least half the known positives in the top 1%. "All known" is reported as
  `known_pass_all`. The headline is test; val is for tuning.

## Known limitations (v1)

- Durations are log-uniform up to 5 d, but the dip box search stops at 2.7 d, and the 3.25-d
  detrending window absorbs much of a dip longer than ~2 d. M, skew and excess variance see
  them undetrended.
- Null events are applied only to their 30 hosts. Real CCD-wide events hit every star, so the
  injection set's own common-mode map (30 of ~220 stars per CCD) is a weaker signal than the
  real one would be.
- The classical Q uses an FFT of gap-zeroed data, not a Lomb-Scargle, and has no smoothing
  before folding.
- Known positives in S32 are mostly Capistrant+2022 dipper candidates, many faint (T 13-16) and
  in one crowded field (cam 1, CCD 2, near PDS 110). They may not dip in S32 at all, so a low
  percentile is not necessarily a miss.
