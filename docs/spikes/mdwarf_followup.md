# Track C follow-up: other data for the leftover signals, and whether gap-filling the survey pays

Status: **pre-registration committed 2026-11-13 12:11:48 +1000 (commit 359e1c1; the text below
says 12:15, the commit time is the authority), before any light curve, image or catalogue other
than the survey's own SPOC / TESS-SPOC light curves was looked at for these stars.** Results were
added below the pre-registration on the same day ("Part 1 results", "Part 2 results"); the
pre-registered text is unchanged except for this paragraph. The v2 vetting and its tiers are not
changed; everything here is extra evidence on top. Default for every signal stays "probably not a
planet", and it still is for all five.

Code: `spikes/mdwarf_stack/followup/` (`fu.py` shared statistic, `ephem.py` the tables below).
Outputs: `outputs/planets_spike/followup/`.

## Part 1 pre-registration

### Signals and ephemerides (`followup/ephem.py`, survey data only)
Weighted linear fit to the survey's detected epochs. Per-epoch timing error
sigma_i = max(D/4, D / max(SES_i, 1)) (conservative: Carter et al. 2008 give ~3x smaller for a
box with a 15 min ingress); covariance multiplied by chi^2/dof when that exceeds 1. The
prediction window for a dip centre is **+-3 sigma_pred** around the predicted time.

| signal | P (d) | T0 (BTJD, epoch n = 0) | D (h) | depth (ppt) | sigma_P | sigma_T0 |
|---|---|---|---|---|---|---|
| TIC 2102868 #0 | 39.39854 | 2476.1444 | 1.5 | 1.44 | 0.016 h | 0.27 h |
| TIC 234295610 #2 | 49.16226 | 1326.6751 | 4.0 | 4.96 | 0.024 h | 0.81 h |
| TIC 146636926 #1 | 54.64927 | 1400.6174 | 4.0 | 1.30 | 0.041 h | 1.99 h |
| TIC 141816993 #0 | 22.13831 | 1729.2381 | 4.0 | 0.52 | 0.025 h | 1.49 h |
| TIC 57830249 #2 | 45.06806 (or 1802.722 / n) | 2291.1701 | 1.5 | 9.73 | 0.013 h | 0.38 h |

Every predicted epoch while the star was on silicon (tess-point 0.9.5 plus TESScut's own sector
list), through the last released sector:

| signal | n | BTJD | sigma (h) | window +- (h) | sector | in the survey light curve? |
|---|---|---|---|---|---|---|
| 2102868 | 0 | 2476.144 | 0.27 | 0.81 | 43 | yes (detected) |
| 2102868 | 1 | 2515.543 | 0.26 | 0.78 | 44 | yes (detected) |
| 2102868 | 11 | 2909.528 | 0.22 | 0.65 | 59 | no: gap in a searched sector |
| 2102868 | 21 | 3303.514 | 0.27 | 0.82 | 73 | no: gap |
| 2102868 | 30 | 3658.101 | 0.37 | 1.12 | 86 | yes (detected) |
| 2102868 | 40 | 4052.086 | 0.51 | 1.53 | 1751 | no: gap (the S1751 data start ~4056) |
| 234295610 | 0 | 1326.675 | 0.81 | 2.44 | 1 | yes (detected) |
| 234295610 | 15 | 2064.109 | 0.62 | 1.85 | 28 | yes (detected) |
| 234295610 | 52 | 3883.113 | 0.88 | 2.65 | 95 | no: gap |
| 234295610 | 57 | 4128.924 | 0.98 | 2.93 | 102 | yes (detected) |
| 234295610 | 58 | 4178.086 | 1.00 | 2.99 | 103/104 boundary | no: gap |
| 146636926 | 0 | 1400.617 | 1.99 | 5.96 | 3 | yes (detected) |
| 146636926 | 47 | 3969.133 | 0.58 | 1.74 | 97 | yes (detected) |
| 146636926 | 51 | 4187.730 | 0.62 | 1.85 | 104 | yes (detected, window 57% covered) |
| 146636926 | 52 | 4242.380 | 0.63 | 1.90 | 106 | yes (detected) |
| 141816993 | 0 | 1729.238 | 1.49 | 4.47 | 15 | yes (detected) |
| 141816993 | 1 | 1751.376 | 1.47 | 4.40 | 16 | no: gap |
| 141816993 | 8 | 1906.345 | 1.30 | 3.91 | **22 (TESScut only; not in tess-point, never searched)** | no |
| 141816993 | 40 | 2614.770 | 0.67 | 2.02 | 48 | yes (detected) |
| 141816993 | 41 | 2636.909 | 0.66 | 1.97 | 49 | no: gap |
| 141816993 | 42 | 2659.047 | 0.65 | 1.94 | 49 | yes (detected) |
| 141816993 | 73 | 3345.334 | 0.73 | 2.19 | 75 | no: gap |
| 141816993 | 74 | 3367.473 | 0.75 | 2.24 | 76 | yes (detected, 59% covered) |
| 141816993 | 75 | 3389.611 | 0.76 | 2.29 | 76 | yes (detected) |
| 57830249 | -16 | 1570.081 | 0.55 | 1.64 | 10 | no: gap |
| 57830249 | 0 | 2291.170 | 0.38 | 1.12 | 36 | yes (detected) |
| 57830249 | 40 | 4093.892 | 0.37 | 1.12 | 100 | yes (detected) |

Future windows (not testable now): 2102868 none planned through S134; 141816993 S121-122,
S127-128 (+-6 h at 3 sigma); 57830249 S114 (BTJD 4454.437 +- 1.4 h at 3 sigma, for P = 45.068).

Every sector in which these five stars fell on silicon through the released sectors was already
searched, except S22 for TIC 141816993. So the new-photon information from TESS is limited to the
**epochs that fell in gaps of the SPOC / TESS-SPOC light curves** (other products keep different
cadences) and the one unsearched sector. The detected epochs can only be **re-reduced** (same
photons, other pipeline): that tests "SPOC/PDC artefact", not "noise".

**TIC 57830249 aliases.** The two epochs are 1802.722 d apart, so P = 1802.722 / n. Aliases with
no other epoch in the survey light curve: n = 1-10, 12-15, 19, 20, 24-26, 30, 35, 40 (45.068 d),
45 (40.06 d). Aliases with other searched epochs are tested on the survey light curve with the
joint rule below (same data, new question), then on new data.

### The statistic (fixed now; `fu.Meter`)
Any light curve (SPOC, QLP, TGLC, TESScut aperture photometry, ...) is normalised per sector,
binned to the survey's 10-min grid and run through the **frozen gate-2 preprocessing**
(`search2.preprocess2`, rot = harm). At a trial centre t, a box of the signal's duration D gives
depth = weighted mean deficit, sigma = sigma_white x beta (beta = robust scatter of the white-noise
box SES at every other position in the same sector, >= 1, i.e. the search's red-noise rescaling),
SES = depth / sigma. A position counts as covered if >= 50% of its bins have data. Positions are
stepped by 10 min across the +-3 sigma_pred window.

- **Expected SES** = signal depth / sigma at the predicted time.
- **Injection:** a box of the signal's depth and duration is multiplied into the raw normalised
  flux at 5 positions spread evenly over the window (one at a time), the light curve is
  preprocessed again, and each is **recovered** if SES >= 3 at the injected position.
- **Random windows:** 400 windows of the same width at random covered times > 2 d from every
  predicted epoch; their maximum SES gives the light curve's empirical false-support rate.

### Per-epoch outcome
- **Supported:** the maximum SES in the window is >= max(3.0, the 95th percentile of the
  random-window maxima), and that position's depth is consistent with the signal depth
  (|depth - signal depth| < 2 sigma), and the position is not within 1 h of a segment edge.
  So by construction at most ~5% of random windows would "support" a signal.
- **Refuted:** >= 80% of window positions covered, all 5 injections recovered, and at every
  covered position depth < signal depth - 2 sigma (the signal's dip is excluded at 2 sigma at
  every allowed time). This needs expected SES of about 4 or more in practice.
- **No information:** everything else (window not covered, injections lost, noise too high, or a
  dip that is significant but of the wrong depth, which is reported separately).
- A dip found **outside** the window is never counted, and no window is widened after looking.

### Per source and per signal
- A source **supports** a signal if >= 1 new-photon epoch is supported and none refuted;
  **refutes** if >= 1 is refuted and none supported; both = conflicting, reported as no
  information. Re-reductions of detected epochs are reported in their own row ("artefact test").
- **Joint test** over all new-photon epochs of a source: summed (weighted) depth and its sigma;
  refuted if expected joint SES >= 4 and observed < expected - 3; supported if observed >= 3 and
  consistent with the signal depth within 2 sigma.
- **TIC 57830249:** each alias n is tested with the joint test on its other epochs (survey light
  curve first, then new data). An alias is excluded only by the refute rule; a dip at an alias
  epoch counts as support only if it passes the per-epoch support rule with the threshold raised
  for the number of alias windows tested (Bonferroni on 5%).

### Other data sources (rules fixed now)
- **K2 / Kepler:** in a campaign field AND with a target pixel file or light curve. Ephemerides
  propagated backwards with the same covariance; the same per-epoch rules.
- **CHEOPS archive:** any visit; same rules. If the archive needs an account, skip and say so.
- **MEarth public releases:** is the star a MEarth target; nights covering any window (all
  epochs, not only TESS-era ones). Same rules on MEarth's light curve (per-night median removed,
  then the box statistic; injections into the same data). A night that covers a window but whose
  per-point scatter makes expected SES < 3 is "no information".
- **Neighbour eclipse test (ZTF / ATLAS / ASAS-SN / Gaia DR3 epoch photometry):** for each Gaia
  neighbour that could produce the dip (required eclipse depth = signal depth x F_target /
  F_neighbour <= 100%, inside 63"), fold its light curve on P (and on 2P). **Blend supported** if
  in-window points show a dip of the required depth at >= 5 sigma; **blend excluded for that
  neighbour** if >= 5 in-window points exist and the required depth is excluded at 5 sigma; else
  no information. ATLAS forced photometry needs an account: skipped.
- **Binarity (RVs: APOGEE, LAMOST, GALAH, RAVE; Gaia DR3 RV scatter; astrometric flags):**
  multi-epoch RV variation > 3 sigma and > 2 km/s = **binary supported** (the signal may come from
  or be diluted by a companion; it does not refute the dip). One epoch, or none, is no information.
- **Background star at the current position (high proper motion; DSS/POSS, 2MASS, Pan-STARRS):**
  a source at the star's current (2026) position in an early plate, bright enough to make the dip
  (mag <= target mag + 2.5 log10(1/depth)), = **background-blend scenario supported**; a clean
  position to deeper than that = **excluded to the plate limit**; otherwise no information.

A failed or timed-out query is "not checked", never "nothing found".

## Part 2 plan (fixed now)
1. Gap: star-sectors on silicon (tess-point, released sectors) vs searched, over the 4,681 targets;
   causes from the MAST product lists (every product, batch observation queries) and the survey
   records.
2. Noise: ~30 survey stars, Tmag 10-15, sectors with both SPOC/TESS-SPOC and an alternative
   product: robust scatter of 2-h bins of the frozen-detrended light curve, product vs product,
   same star and sector.
3. Projection: per-sector-bucket recovery from the v2 validation (1-3: 0.03, 4-5: 0.10, 6-12: 0.20,
   13+: 0.58) applied to the new sector counts, with the added sectors down-weighted by the
   measured noise ratio (an added sector with noise ratio r counts as 1/r^2 SPOC sectors), and
   intervals from the validation counts.

---

## Implementation notes (changes after the pre-registration, all made before the windows they affect were read)
- **Coverage at the product's own cadence.** `fu.Meter` counts a box as covered relative to the
  light curve's cadence (a 30-min FFI light curve fills only every third 10-min bin; with the
  10-min count no 30-min window could ever be "covered").
- **TESScut aperture fixed at r = 1.5 px.** Choosing the radius by the lowest relative scatter was
  tried first and dropped before any window was read: relative scatter falls as neighbours' flux
  is added, so that rule picks the most diluted aperture (it also had a NaN bug on 30-min cubes).
  TESScut and QLP SAP depths are not crowding-corrected, so they are diluted relative to PDCSAP.
- **TGLC S12-S55** are not in MAST's API listing; they were read from the public bucket (key
  layout "00" + Gaia DR3 id, split into 4-digit groups; `products.tglc_key`). T16 uses
  `TMID_BJD` / `IFL2`; TARS (CPM residual flux centred on 0) is shifted by +1 before
  normalising.
- **Joint test** also requires the injections to be recovered (the owner's rule: a non-detection
  refutes only if an injected dip of the same depth and duration is recovered there).

## Part 1 results

### What new photons exist at all
Every predicted epoch was checked in every product MAST lists for the star (SPOC, TESS-SPOC,
QLP, TGLC from the bucket, eleanor-lite, T16, TARS, TASOC/CDIPS where listed) and in a TESScut
cutout. **All gap epochs are true downlink / data gaps in every product** except two:
TIC 234295610 n = 52 (S95: data in TARS and TESScut only; S95 has no SPOC / TESS-SPOC light
curve for this star, so the survey never had it, see Part 2) and TIC 141816993 n = 73 (S75: QLP
and TESScut). The one "unsearched sector" (S22 for TIC 141816993) is not usable: TESScut returns
a cube, but the 1.5-px aperture holds 0.2 e-/s (a T = 10.2 star gives thousands), so the star is
not on science pixels there, as tess-point says; sigma at the prediction is 147 ppt.
TIC 2102868 and TIC 57830249 have **no new TESS photons in any product**; TIC 146636926 has no
gap epochs at all.

Full per-epoch tables: `outputs/planets_spike/followup/tess_tables.txt`; figures
`outputs/planets_spike/followup/figures/tess_*.png`; records `tess_epochs.jsonl`
(`followup/tess_test.py`, `followup/tess_summary.py`).

### New-photon epochs (decide supported / refuted / no information)

| signal | epoch (sector) | product | coverage | depth at prediction (ppt) | expected SES | injections recovered (pre-reg) | pre-registered outcome |
|---|---|---|---|---|---|---|---|
| 234295610 (4.96 ppt, 4 h) | n=52 (S95) | TARS | 100% | 0.42 +- 0.70 | 7.1 | 0/5 | no information |
| 234295610 | n=52 (S95) | TESScut | 100% | 0.05 +- 0.46 | 10.7 | 0/5 | no information |
| 141816993 (0.52 ppt, 4 h) | n=73 (S75) | QLP | 100% | 0.11 +- 0.15 | 3.5 | 0/5 | no information |
| 141816993 | n=73 (S75) | TESScut | 100% | -0.17 +- 0.21 | 2.4 | 0/5 | no information |
| 141816993 | n=8 (S22, never searched) | TESScut | 100% | 146 +- 147 | 0.0 | 0/5 | no information |
| all others | (S10, S16, S49, S59, S73, S103/104, S1751) | every product | 0% | - | - | - | no information (no data) |

Joint tests per source: TARS (234295610) 0.42 +- 0.70 ppt; TESScut (234295610) 0.05 +- 0.46;
QLP (141816993) 0.11 +- 0.15; TESScut (141816993) -0.17 +- 0.21: all **no information**, because
the injections were not recovered.

**Why the injections fail where the expected SES is 7-11.** The frozen detrend (0.5-d biweight
plus rotation harmonics) removes most of a 4-h box: in S95 only 22% of an injected 4-h box
survives (median over 5 positions), versus about 50% at the detected epochs in the survey light
curve. The pre-registered injection used the signal's *post-detrend* depth as the *raw* depth,
so it is doubly weakened. This is a flaw in the pre-registered injection design, and the
pre-registered outcomes above stand.

**Post-hoc (not pre-registered; does not change any verdict): matched-depth injection**
(`followup/posthoc_inj.py`, `posthoc_inj.json`). Raw depth = signal depth / survival at the
detected epochs in the survey light curve (s_det); the dip expected in the new light curve =
raw depth x local survival; injections at the raw depth; refute rule re-applied against the
expected local depth.

| signal, epoch, product | s_det | local survival | expected local depth (ppt) | observed at prediction | injections (SES) | post-hoc outcome |
|---|---|---|---|---|---|---|
| 234295610 n=52 S95 TESScut | 0.50 | 0.22 | 2.22 | 0.05 +- 0.46 | 4.7, 4.0, 4.8, 4.2, 4.3 (5/5) | refuted (post-hoc) |
| 234295610 n=52 S95 TARS | 0.50 | 0.22 | 2.21 | 0.42 +- 0.70 | 3.9, 3.2, 3.5, 2.8, 3.7 (4/5) | no information |
| 141816993 n=73 S75 QLP | 0.55 | 0.38 | 0.36 | 0.11 +- 0.15 | 2.6, 3.2, 3.3, 2.1, 2.7 (2/5) | no information |
| 141816993 n=73 S75 TESScut | 0.55 | 0.50 | 0.47 | -0.17 +- 0.21 | 0.6-3.9 (1/5) | no information |

A cruder post-hoc variant (injection scaled so its post-detrend depth equals the signal's quoted
depth) "refutes" both S95 light curves and the S75 QLP one; it is not used, because it compares
against a deeper dip than a real signal would leave after this detrend in this sector. Caveats
for the one post-hoc refutation: TESScut photometry is not crowding-corrected, so any dilution
by neighbours lowers the dip expected there and weakens the exclusion (not measured for this
star); the S95 survival of 0.22 vs 0.50 at the detected epochs is unexplained (same detrend, a
200-s FFI light curve); and it is one epoch from one reduction, post-hoc.

### Re-reductions of detected epochs (same photons, other pipelines: tests "SPOC/PDC artefact", not "noise")

| signal | epoch (sector, survey depth) | reproduced (supported) | not reproduced at the predicted time | other |
|---|---|---|---|---|
| 2102868 | n=0 (S43, 1.34 +- 0.25) | QLP 0.72 +- 0.31 (max 1.33 at -0.6 h), TESScut 1.74 +- 0.31, TGLC 0.80 +- 0.29 | - | TARS 0.56 +- 0.38 (no info). Every reduction shows the dip followed by a ~+1.5 ppt bump within 2 h |
| 2102868 | n=1 (S44, 1.01 +- 0.27) | QLP 0.94 +- 0.30, TARS 1.50 +- 0.39, TGLC 1.04 +- 0.30 | TESScut 0.77 +- 0.32 (refute rule met, but only 0.8 sigma from the survey value) | |
| 2102868 | n=30 (S86, 2.05 +- 0.27; strongest epoch) | - | QLP -0.84 +- 0.30, TARS 0.01 +- 0.44, TESScut 0.13 +- 0.36 | TESScut has a 1.28-ppt dip 1.0 h *earlier* (SES 3.6, inside the window, so "supported" by the rule); the same pixels cannot give a real dip at two different times. QLP and TARS start ~4 h before the prediction (QLP ramps upward there) |
| 234295610 | n=0 (S1, 6.91 +- 0.98) | eleanor 3.32 +- 0.96, TARS 4.53 +- 1.31 | - | QLP 3.98 +- 1.62, T16 3.75 +- 1.81, TESScut 3.81 +- 0.80 (support threshold 9.6 there), TGLC 7.4 +- 2.8: dips present at lower SES |
| 234295610 | n=15 (S28, 3.89 +- 0.87) | TARS 3.51 +- 0.83 | - | TESScut 1.39 +- 0.48 (a dip of a third of the depth), QLP 4.15 +- 1.68 |
| 234295610 | n=57 (S102, 4.73 +- 1.03) | - | - | QLP 4.43 +- 1.29, TESScut 3.18 +- 1.24 (below the support thresholds) |
| 146636926 | n=47 (S97, 1.40 +- 0.28) | - | QLP 0.73 +- 0.24 (half depth, SES 3.2) | TESScut 0.67 +- 0.47 |
| 146636926 | n=52 (S106, 1.23 +- 0.23) | TESScut 1.09 +- 0.29 | - | |
| 146636926 | n=0 (S3), n=51 (S104) | - | - | weak everywhere (expected SES 2-5, injections mostly lost) |
| 141816993 | n=40 (S48, 0.80 +- 0.12) | - | - | TESScut 1.86 +- 0.20 and TGLC 1.45 +- 0.31: dips 2x deeper than the survey's (wrong depth for a 0.52-ppt signal) |
| 141816993 | n=42 (S49, 0.33 +- 0.13) | TGLC 0.69 +- 0.24 | - | QLP 0.37 +- 0.13, TESScut 0.35 +- 0.16 (same as survey, below threshold) |
| 141816993 | n=0 (S15), n=75 (S76) | - | QLP 0.14 +- 0.13 at S76 | others weak |
| 57830249 | n=0 (S36, 18.8 +- 2.6) | QLP 14.6 +- 2.9, TGLC 12.7 +- 3.0 | - | TARS 16.6 +- 3.3 and TESScut 18.6 +- 3.3 (SES 5.2-5.7, below raised thresholds) |
| 57830249 | n=40 (S100, 8.1 +- 1.1) | - | - | QLP no data; TESScut window 50% covered (6.3-ppt dip at -0.1 h) |

**TIC 57830249 aliases** (`followup/alias.py`, `alias_57830249.json`; survey light curve, the
pre-registered joint rule): of the 180 aliases P = 1802.722 / n (n = 1-180), **137 are
excluded and 43 allowed**. Allowed in 15-50 d: n = 37 (48.72 d), 40 (45.07 d, the nominal
period), 41 (43.97), 43 (41.92), 45 (40.06), 46 (39.19), 47 (38.36), 50 (36.05), 51 (35.35),
52 (34.67), 53 (34.01), 54 (33.38), 57 (31.63), plus n = 92 (19.60 d); the rest of the allowed
ones are > 50 d. Of these, n = 40 and 45 have no other epoch in the data; the others have 1-3
epochs where the joint injection does not reach SES 4 (n = 37, 43, 47, 50 come close: injected SES
3.3-3.4 with observed -1.2 to -2.4). No alias epoch shows a dip (largest observed joint SES 2.6,
n = 31, on a 66-ppt-sigma point). The two detected depths disagree: 18.8 +- 2.6 (S36) vs
8.1 +- 1.1 ppt (S100), 3.8 sigma apart in the survey light curve; S36 is ~13-19 ppt in every
re-reduction. One planet gives the same depth at every transit; both depths are from the survey's
crowding-corrected PDCSAP light curves, so a sector-to-sector crowding difference would have to be
a factor of two. Unequal depths point to two unrelated events, a systematic in one of them, or a
variable / eclipsing neighbour in the aperture rather than one planet.

### Archive sources (2-4)
<<ARCHIVE>>

### Per-signal verdicts (all: probably not a planet)

| signal | new photons (TESS) | artefact test (same photons) | other archives | verdict |
|---|---|---|---|---|
<<VERDICTS>>

## Part 2 results: is gap-filling worth it?

Code: `followup/gap_query.py` (MAST batch product lists for all 4,681 targets, 179,690 product
rows, 2026-11-13), `gap_analysis.py`, `tglc_check.py`, `noise.py`, `projection.py`, `tic100.py`.
Outputs: `gap_tables.txt`, `gap_stars.csv`, `tglc_check.json`, `noise.jsonl`, `projection.txt`,
`tic100_sample.csv` in `outputs/planets_spike/followup/`.

### 2.1 The gap
Star-sectors on silicon through S107 (tess-point) for the 4,681 targets: **50,420; searched
37,007 (0.734; 0.778 over the 4,350 searched stars); missing 13,413.** Of the missing, **13,401
have no SPOC or TESS-SPOC light curve at MAST** and 12 had one that failed conversion. No
searched star-sector lies outside the tess-point list.

| cut | searched / predicted |
|---|---|
| Tmag < 11 / 11-12 / 12-13 / 13-13.5 / 13.5-14 / 14-15 | 0.84 / 0.84 / 0.76 / 0.74 / 0.70 / 0.62 |
| S1-13 / 14-26 / 27-39 / 40-55 / 56-69 / 70-83 / 84-96 / 97-107 | 0.85 / 0.93 / 0.89 / 0.94 / 0.80 / 0.65 / 0.45 / 0.24 |
| contamination ratio < 0.01 / 0.01-0.1 / 0.1-0.5 / > 0.5 / not in TIC | 0.87 / 0.82 / 0.61 / 0.44 / 0.20 |

**Causes.** (1) Time: the TESS-SPOC FFI HLSP at MAST stops at S86 today; S87-S107 have only the
SPOC 2-min targets, so 6,691 of the missing star-sectors (50%) are S87-S107. This part closes by
itself if TESS-SPOC catches up. (2) Faint and crowded stars: TESS-SPOC selects by brightness and
the early-sector selection missed many faint / crowded M dwarfs (0.44 searched at contamination
> 0.5). (3) 331 targets (median T = 13.83, 2,882 predicted star-sectors) have no SPOC/TESS-SPOC
light curve at all.

**What else exists for the missing 13,413.** MAST lists QLP for 8,011 (T < 13.5 only; S1-S104,
the S99-S104 deliveries with timestamp errors, the public bucket holds S1-S98), TARS for 9,150
(Boyle et al. 2026 "unpopular" CPM, T < 16, S1-S96, MAST only, no bulk scripts), eleanor-lite,
T16 (S1-S10) and TGLC; TGLC S12-S55 is in the public bucket but not MAST's API (bucket check:
present for 140/150 missing and 48/50 searched star-sectors sampled). **Union of QLP + TARS +
TGLC (<= S55): 9,833 of the 13,413 (73%).** The rest (3,580) has no product at all, mostly
S97-S107; only TESScut (own photometry from the FFIs) could reach those. For the 331 no-data
stars: QLP 304 stars (1,553 star-sectors), TARS 328 (2,338), nothing 177 stars (468).

### 2.2 Noise of the alternatives
30 survey stars (T 10-15, 24 with SPOC 2-min and 6 with TESS-SPOC as the survey product), one
sector each with the survey product and the alternative; the same frozen detrend; robust scatter
of 2-h bins; ratio = alternative / survey product, same star and sector.

| product | median ratio [16-84%] (n) | T < 12 (12) | T 12-13.5 (11) | T > 13.5 (7) |
|---|---|---|---|---|
| QLP | 1.11 [0.83, 1.36] (30) | 0.91 | 1.11 | 1.34 |
| TARS | 1.23 [0.58, 1.65] (30) | 1.30 | 1.26 | 0.95 |
| TESScut (own) | 1.12 [0.57, 1.51] (29) | 0.89 | 1.15 | 1.09 |
| TGLC | 1.18 [0.98, 1.43] (21; 8 not available) | 1.24 | 1.16 | 0.99 |

The alternatives are 10-25% noisier at 2 h, so an added sector is worth 1 / r^2 = 0.64-0.83 SPOC
sectors. Ratios < 1 at T > 13.5 mostly reflect PDCSAP's crowding blow-ups (CLAUDE.md), not
better photometry. Not measured: whether each product **keeps** a 1-4 h transit. TARS is a CPM
(pixel-regression) light curve and can absorb part of a transit; QLP SAP and TESScut are not
crowding-corrected (diluted depths). Measuring that needs pixel-level injections, not done here;
it is part of the kill test below.

### 2.3 Projected gain (`followup/projection.py`, `projection.txt`)
Method: the 300 headline injections of the v2 validation (1.0-1.5 R_e, 15-30 d, on 150 survey
stars; 55 detected at tau_long) give P(detect) as a logistic in log white-noise SNR (50% at SNR
12.3; reproduces the validation buckets: N 1-3 0.03 vs 0.03 measured, 4-5 0.09 vs 0.10, 6-12 0.21
vs 0.20, 13+ 0.60 vs 0.58). Each target gets the mean P(detect) of the injections within 0.5 mag
of its Tmag, moved to its sector count (SNR x sqrt(N_new / N_inj)); an added sector from a product
with noise ratio r counts as 1 / r^2 sectors (median r per product from 2.2). Intervals: 200
bootstrap draws over validation stars (logistic refit) and over the noise sample. They do **not**
include the untested assumptions: that the alternative products keep a transit as well as
PDCSAP, that the frozen search and vetting behave the same on mixed-product light curves, and
that recovery scales with sqrt(sectors).

| scenario | recovery per searched star | star-equivalents (all 4,681) | gain over survey [95%] | expected headline-cell detections* |
|---|---|---|---|---|
| A: survey as run | 0.164 [0.144, 0.186] | 715 | - | 0.86-1.44 |
| C: TESS-SPOC delivers S87-S107 at its S56-S86 rate (same product, no re-validation) | 0.180 [0.157, 0.205] | 790 | +76 [63, 89] (+11%) | 0.95-1.59 (+0.09-0.15) |
| B: fill with QLP + TARS + TGLC (S1-S96 / S55) | 0.184 [0.159, 0.209] | 818 | +103 [81, 135] (+14%) | 0.99-1.64 (+0.12-0.21) |
| B2: B plus TESScut photometry for everything else | 0.194 [0.168, 0.220] | 865 | +151 [123, 200] (+21%) | 1.04-1.74 (+0.18-0.30) |

\* star-equivalents x geometric transit probability (mean 0.0129 over 15-30 d) x an assumed
0.084-0.14 planets of 1.0-1.5 R_e per M dwarf in 15-30 d (Dressing & Charbonneau 2015, ApJ 807,
45: 0.56 per star at P < 50 d; 15-30 d share 0.15-0.25 for log-uniform periods). The survey's
own expectation (0.9-1.4) and its 0 detections are compatible (Poisson P(0) = 0.24-0.42). The
absolute level is model-dependent: weighting the validation buckets by sector count gave ~0.19
per searched star in the survey write-up, this Tmag-matched model gives 0.164; the gains are
relative and the more robust number.

What the numbers say: N_eff per searched star goes from median 6 to 6.6 (stars with < 6 sectors:
1,994 -> 1,418); the 331 no-data stars add only 19-23 star-equivalents (faint, few sectors).
B and C overlap (both fill S87-S96), so **B's gain over simply waiting for TESS-SPOC is about
+27 star-equivalents, ~0.03-0.06 expected planets.** Even B2 adds about a fifth of a planet.

### 2.4 Cost and risk
- **Download:** 9,833 star-sectors for B (QLP 8,011, TARS 9,150, TGLC <= S55 2,488; overlap
  removed by taking the lowest-noise product first), ~0.15-0.6 MB each, ~3-4 GB transfer. QLP and
  TGLC come from the bucket (fast, Kaggle-friendly); TARS is MAST-only with no bulk script (the
  ~30 files/s MAST ceiling: ~6 min of transfer at best, realistically an hour or more with
  retries). B2's TESScut part (3,580 star-sectors with no product, plus re-doing the rest) is
  slow through the TESScut service (median ~9 s per cutout plus photometry here, so ~1.5 days
  serial for 13k cutouts, and a public service should not be hammered in parallel); at scale it
  would need FFI cubes from the bucket and astrocut on Kaggle, plus validating our own
  photometry (crowding, apertures): a project of its own.
- **Search:** the survey cost 89.9 CPU-h on Kaggle. Every star with an added sector must be
  re-searched (most of them), with ~27% more data: ~110-120 CPU-h, about one day of Kaggle shards.
- **Re-validation (the real cost):** the frozen search, the noise model behind tau_long and the
  v2 vetting were all validated on SPOC/TESS-SPOC PDCSAP only. Mixed-product light curves change
  the noise and systematics (QLP SAP is scattered-light heavy, TARS is CPM-detrended, no crowding
  correction), so before any result counts: new nulls (inverted / scrambled light curves) to
  re-check the false-alarm rate at tau_long, and injections into the new products' light curves
  (pixel-level for TARS) to measure recovery. ~1,000-1,500 searches, 20-30 CPU-h, plus a
  pre-registered gate. Freezing rules without peeking at the new real-star results is the hard part.
- **Risk:** the product's transit survival is the main unknown. If TARS (the largest single
  source of added sectors) suppresses a 4-h dip like the frozen detrend did in S95 (Part 1), its
  sectors add noise and almost no signal, and B falls toward C. Mixed products also add a new
  false-positive channel (product-specific systematics at the same time in different sectors
  don't line up, but a single-product glitch plus a SPOC dip can make a 2-epoch "signal").

### 2.5 Compared with extending the survey to 100 pc (`followup/tic100.py`, `far_cov.py`)
Sample: TIC v8 stars with Teff 2,700-3,400 K, T < 15, d < 100 pc in 30 random 6-degree cones
(seed 20261113; all 30 answered after retries; 8.2% of the sky), tess-point sectors through
S107, the survey's selection (>= 6 predicted sectors). Scaling check: < 50 pc, T < 15 in the cones
scales to 13,760 all-sky (gate-1 catalogue: 14,471), but the >= 6-sector subset scales to only
2,945 (survey: 4,681), because the sector count depends on ecliptic latitude and 30 cones sample
it unevenly. So two scalings are given: by area, and calibrated on the survey (x 4,681 / 242).

**Do faint 50-100 pc stars get SPOC / TESS-SPOC light curves?** Checked directly for 200 random
far stars (all 200 queries answered): have / predicted 0.647 (T 12-13 0.82, 13-13.5 0.65,
13.5-14 0.73, 14-15 0.60; 34 of 200 have none), close to the survey's own fractions at the same
Tmag (0.76, 0.74, 0.70, 0.62). The projection uses the survey's fractions.

| | stars (area / survey-calibrated) | Tmag median | recovery per star | star-equivalents | expected headline-cell detections* |
|---|---|---|---|---|---|
| survey (< 50 pc), as run | 4,681 | 13.05 | 0.164 | 715 | 0.86-1.44 |
| 50-100 pc, existing SPOC / TESS-SPOC only | 12,350 / 19,630 | 14.15 | 0.041 | 502 / 798 | 0.65-1.09 / 1.04-1.73 |

Recovery per far star by Tmag: T < 12 0.25 (6 in cones), 12-13 0.14 (70), 13-14 0.06 (346), 14-15
0.016 (593). The mean geometric transit probability is a little higher (0.0142 vs 0.0129) because
the T < 15 cut keeps the larger stars at 50-100 pc (median TIC radius 0.28 vs 0.24 R_sun). In the cones the < 50 pc model gives 465 star-
equivalents where the survey has 715 (ratio 0.65, the same as the count shortfall), which supports
the calibrated column.

**What it costs.** Same products and the same frozen search and v2 vetting (no re-validation of
a new product type), but 3-4x the survey: ~12-20k stars x ~7.7 real sectors, ~250-400 CPU-h of
Kaggle shards (survey: 89.9 CPU-h for 4,350 stars) and ~100-150k light-curve downloads from the
bucket / MAST. The vetting load grows with the star count (the survey had 510 signals >= tau_long
on 4,350 stars; expect ~2,000-3,000), and 60% of the far stars are at T 14-15 where PDCSAP
crowding blow-ups are common; the noise model behind tau_long was validated on the < 50 pc mix.

**Comparison.** Extending to 100 pc with the existing products roughly doubles the survey's
expected headline-cell yield (+500-800 star-equivalents, +0.65-1.7 planets), versus +76 (waiting
for TESS-SPOC) or +103 (HLSP gap-fill) star-equivalents. Per CPU-hour it is also better (~2
star-equivalents per CPU-h vs ~0.7 for gap-fill including re-validation), and it needs no new
product validation. It is still at most about one more expected planet in the headline cell, and
the far planets would be around faint stars (T ~ 14), much harder to confirm or characterise.

\* same assumptions as 2.3 (Dressing & Charbonneau 2015 occurrence, 15-30 d share 0.15-0.25).

### 2.6 Recommendation: **no-go on gap-filling now**
The best case (B2, everything filled, every assumption holding) adds ~150 star-equivalents
(+21%) and ~0.2-0.3 expected planets in the headline cell; the realistic case (B) +14% and ~0.1-0.2,
of which more than two thirds also arrives for free if TESS-SPOC delivers S87+. That is a day of
Kaggle search plus a full re-validation for a fraction of a planet. **Cheapest path: wait for
TESS-SPOC S87+ (same product, no re-validation), then re-run the frozen survey on the stars that
gain sectors.** Check the TESS-SPOC listing at MAST every few months; nothing to build now. If more
yield is wanted, extending to 50-100 pc with the same products (2.5) is the bigger lever (about 5x
the gain of gap-filling, no new product to validate), with its own cheap test first: run the
frozen survey on ~300 random far stars (~6 CPU-h) and stop if their rate of signals >= tau_long
per star is well above the survey's 0.12 (510 / 4,350), since that sets the vetting load and the
false-positive budget.

**Cheap kill test (if gap-filling is ever reconsidered), ~2-3 CPU-h, before any download at
scale:** take 50 survey stars that have both TESS-SPOC and QLP / TARS for the same sector;
inject headline boxes at the pixel level (TESScut) or into the raw flux, run the frozen
detrend + `fu.Meter` on each product, and compare per-sector injected SES to TESS-SPOC's.
**Kill if** the median SES ratio for QLP and TARS is < 0.7 (an added sector then counts
for < 0.5 of a TESS-SPOC sector and B's gain over C shrinks to noise), or if the frozen search on
null (inverted) QLP / TARS light curves of the same 50 stars throws more tau_long signals than on
TESS-SPOC.
