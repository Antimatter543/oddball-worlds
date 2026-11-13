# Track C follow-up: other data for the leftover signals, and whether gap-filling the survey pays

Status: **pre-registration, 2026-11-13 12:15 AEST, committed before any light curve, image or
catalogue other than the survey's own SPOC / TESS-SPOC light curves was looked at for these
stars.** The v2 vetting and its tiers are not changed; everything here is extra evidence on top.
Default for every signal stays "probably not a planet".

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
