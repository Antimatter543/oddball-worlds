# Plan

## Goal
A completeness-calibrated **anomaly hunt** across TESS extended-mission (Sector ≥27) full-frame-image light curves, aimed at the weirdest stars. The named classes are young stars with disks (dippers), Boyajian-like irregular dimmers, exocomets and disintegrating planets. Unclassifiable anomalies are a first-class output, not a by-product.

**Deliverable:** a catalogue with injection-recovery completeness, plus a public gallery. Astronomers will judge it on completeness and false positives, not on the model architecture.

## Hardware constraints (real, design around them)
- GPU: GTX 1650 Super, **4 GB VRAM**. Small models only; batch accordingly.
- RAM: 15 GB total, often only ~3–5 GB free. Stream; never load a whole sector into memory.
- Disk: ~189 GB free. **Never keep raw FITS.** Download, bin to compact Parquet, delete the FITS.

## Milestone 1 (2–4 weeks): prove the search works

| Step | What | Done when |
|---|---|---|
| 1a | **Data pipeline.** Stream TESS-SPOC FFI light curves for one sector, then ~10 sectors (include S32/S33 for β Pic and S55 for BD+05 4868). Quality-mask, normalise, bin to ~30 min, write Parquet shards. | 1 sector ingested end to end, with sizes and timings reported. |
| 1b | **Known-positive list.** TIC IDs and sectors for β Pic, the Norazman & Kennedy exocomet hosts, BD+05 4868 Ab and the Capistrant dippers that have S≥27 data. | `data/known_positives.csv` builder script plus a committed small CSV. |
| 1c | **Injection library.** Exocomet (asymmetric exponential tail), dusty-tail transit (variable depth), dipper (aperiodic). Inject into real light curves. | Unit tests on profile shapes; visual sanity plots. |
| 1d | **Baseline detector.** Classical asymmetric-dip matched filter / single-event search. | Scores every light curve in a sector. |
| 1e | **Learned anomaly score.** EncoTESS (or a small self-supervised encoder) embeddings plus kNN density outlier score. | Scores every light curve in a sector. |
| 1f | **Systematics filter.** Momentum dumps, scattered-light windows, CCD-neighbour coincidence. | Flags and removes shared events. |
| 1g | **Evaluation.** Recovery against depth and duration; blind rank of known positives; baseline vs learned. | Gate numbers reported. |
| 1h | **Gallery.** Top 20 un-catalogued oddballs, hand-vetted against SIMBAD, VSX and the TOI list. | Static HTML gallery. |

**Gate:** ≥50% recovery of injected 0.3%-depth events **and** known positives in the top 1%. Otherwise stop and rethink.

## Validation strategy
1. **Injection-recovery into real light curves** (not quiet, pre-cleaned ones), stratified by magnitude, crowding and stellar variability. Report recovery per stratum.
2. **Blind known positives.** No tuning on them. Freeze the method, then look.
3. **Null tests.** Score light curves with injected *systematics-like* events (momentum-dump-shaped dips, CCD-wide dips) and measure how often they get flagged as astrophysical.
4. **Baselines.** The classical matched filter must be reported alongside the learned score. If ML doesn't beat the baseline, say so.
5. **Frozen data versions.** Record the sector list, product version and download date for every run.
6. **Honest outputs.** Every gallery object gets the raw light curve, the pixel-level check, neighbour-star check and catalogue cross-match shown.

## Failure catalogue (audit against this before claiming results)
1. **Systematics masquerading as astrophysics:** momentum dumps, scattered light at orbit edges, CCD-wide events. Most "anomalies" in TESS are this.
2. **Blends:** a 21″ pixel contains other stars, so the dip may belong to a neighbour.
3. **Optimistic injections:** injecting into clean, quiet stars inflates completeness.
4. **Leakage:** tuning thresholds while looking at known positives.
5. **Known-already:** "new" objects that are in VSX, SIMBAD or the TOI list. Always cross-match.
6. **Detrending eats the signal:** aggressive flattening erases long exocomet tails. Test detrending against injections.
7. **Disk/RAM blow-ups:** keeping FITS, or loading whole sectors.

## Track B (parked): Gaia DR4 neural orbit fitter
Gaia DR4 is released 2 Dec 2026. The idea is amortized neural posterior estimation for stars with 0, 1 or 2 hidden companions, plus a per-source detection-completeness map for the 100 pc volume. Novelty: no published NPE on Gaia orbits (as of Nov 2026). Competition: Flatiron "Epochalypse" (arXiv:2611.06915), DPAC's own NSS tables. Revisit in December.

## Results log

### 2026-11-09: classical baseline on S32 (test split) — gate FAIL
- Recall within the top 1% at ~0.3% / ~1% / ~3% depth: **0.03 / 0.09 / 0.28**. AUROC 0.87. Even at 17–30% depth, recall only reaches ~0.5–0.65. Report: `outputs/eval/classical/s0032/report.md`.
- Known positives: 11 of 18 in the top 1% of their stratum. PDS 110 is at 0.994, β Pic at 0.834, and the lowest at 0.787.
- Null (instrument-like) false-positive rate: 17.7%. A `sys_flag` veto cuts it to 1.3%.
- **Why:** the real top 1% (~1,600 stars) is dominated by genuine eclipsing binaries and big-amplitude variables, plus a scattered-light ramp in SAP at BTJD 2185.5. A score that can't explain away known variable classes can't rank a 1% dip above a 40%-deep eclipsing binary. Rank-*averaging* sub-scores also penalises events that are extreme in only one feature.

**Gate clarifications, fixed before any ML result was seen:**
1. The known-positive criterion is the strict reading: **all** S32 known positives in the top 1%. The fraction is reported alongside.
2. The gate itself is unchanged. Recall against **injected SNR** (depth ÷ robust scatter × √bins in the event) is added as a diagnostic, because a 0.3% dip sits below the noise for T ≳ 12 stars and no method can find it there.
3. The known-class explain-away score (ML_DESIGN §4.5) is a prerequisite for a fair gate test. The gate is re-run once it exists, and both numbers will be reported.

### 2026-11-09: first ML scorer `ssl_v1` on S32 (test split) — gate FAIL, complementary to classical
- Patch-MAE transformer (2.0M params, 6 epochs, not converged). Report: `outputs/eval/compare/s0032/classical__ssl_v1/report.md`.
- Recall within the top 1% at ~0.3% / ~1% / ~3% depth: **0.23 / 0.53 / 0.52** (classical 0.03 / 0.09 / 0.28). AUROC 0.91 (classical 0.87).
- Null (instrument-like) false-positive rate **0.41** (classical 0.18, 0.013 with its `sys_flag` veto). No veto applied to ssl_v1 yet.
- Known positives: **2 of 18** in the top 1% (β Pic, TIC 43488669); classical 11 of 18. Real dippers sit inside the cloud of ordinary variables in embedding space, so kNN density misses them.
- **Caveats:** the reconstruction head collapsed to "predict the median" (now fixed with sin-cos positions, not yet retrained), so `score_recon` is really a learned-scatter deviation score. The embedding still reads camera at 55% vs 26% chance, mostly via the SAP channel. Combined-score top-100 overlap across two seeds is 0.28, below the §6 bar, so no ranking from it is reported. The reconstruction sub-score alone is stable (0.71).
- **Reading:** ML finds injected events far better; classical finds real dippers and rejects instrument events better. Neither passes. Next: retrain with sin-cos positions and no SAP channel, apply the common-mode veto to ML scores, then the known-class explain-away score (§4.5).

### 2026-11-10: known-class explain-away `classical_ka` on S32 (test split): small gain, gate FAIL
- Details: ML_DESIGN §11. Reports: `outputs/eval/compare/s0032/classical__classical_ka/report.md`, `outputs/knownclass/kc_v1/s0032/report_classical_ka.md`.
- 58,765 catalogue-labelled stars (VSX, Gaia DR3 vclassre, Prša+2022 EBs). Young-star and irregular-dimmer types are hard-excluded. Val recall: EB 0.89, δ Sct/γ Dor 0.91, rotators 0.95. The rare classes are too small to learn.
- Recall within the top 1% at ~0.3% / ~1% / ~3% depth: **0.042 / 0.13 / 0.33** (classical 0.027 / 0.09 / 0.28). With the `sys_flag` veto: 0.04 / 0.15 / 0.43. Null FPR 0.20 (0.02 with the veto). Known positives in the top 1%: **10 of 18** (classical 11; TIC 4350339 was explained away as a rotator).
- **Why it's small:** about 70% of the real top 1% has no catalogue entry. And in these hand-made features the classifier sees the real dippers as "typical rotators or EBs" (15 of 18 have E ≥ 0.9). So the explain-away can't be pushed harder without also explaining away the targets.
- **Next:** feed the classifier ML embeddings instead of hand-made features (the combiner is scorer-agnostic), and apply it to ssl_v2.

### 2026-11-10: `ssl_v2`, trained on S33/38/43/44/55, scored on held-out S32 (test split) — gate FAIL
- Hybrid conv-stem + transformer patch-MAE (1.91M params), no SAP channel, 12k steps, 3 seeds. Details and deviations: ML_DESIGN §10. Report: `outputs/eval/compare/s0032/classical__ssl_v1__ssl_v2__ssl_v2_recon__ssl_v2_iforest/report.md`.
- **The reconstruction now uses context.** On held-out S32 signal-dominated stars, masked-bin MSE 0.93 vs 1.52 for predicting the median (ssl_v1: 1.64 vs 1.70); hidden 0.7-d sinusoid correlation 0.87-0.89 (ssl_v1 ~0). The pure transformer still collapsed with sin-cos positions; a 3-block conv stem fixed it.
- Recall within the top 1% at ~0.3% / all depths: `ssl_v2` **0.23 / 0.52** (ssl_v1 0.23 / 0.39, classical 0.03 / 0.25); top 0.1% 0.32. Val-selected `ssl_v2_iforest`: **0.30** / 0.47. AUROC 0.92.
- Null false-positive rate: `ssl_v2` 0.53, **0.16 with the classical `sys_flag` veto**; `ssl_v2_iforest` 0.25, 0.03 with veto (0.25-0.48 across seeds). Veto costs recall: all depths 0.52 -> 0.45.
- Known positives in the top 1%: `ssl_v2` 6/18 (ssl_v1 2, classical 11). PDS 110 drops to 0.86 without SAP.
- Nuisance probe: camera 0.37 vs chance 0.26 (ssl_v1 0.55), camera+CCD 0.16 vs 0.07.
- Stability: combined top-100 overlap 0.41 (fails, kNN 0.20); `ssl_v2_recon` 0.76 and `ssl_v2_iforest` 0.70 pass. No combined ranking is reported.
- **Reading:** ML leads on injections at every depth; on nulls it still needs the veto and stays worse than vetoed classical (0.16 vs 0.013). Nothing reaches 50% at 0.3%, and known dippers still sit with the variables. Next: the known-class explain-away score (§4.5), then the gate re-run.

### 2026-11-10: per-event veto `cmv2` and ensemble `ens_v1` on S32 (test split): gate FAIL
- Details are in ML_DESIGN §12. Reports:
  - `outputs/veto/cmv2/s0032/report.md` (the veto)
  - `outputs/eval/compare/s0032/classical__ssl_v2_iforest__ens_v1__ens_v1_ka/report.md` (the compare)
  - `outputs/eval/compare/s0032/ens_v1__ens_v1_seed1__ens_v1_seed2__ens_v1_2c__ens_v1_max/report.md` (seeds and ablations)
- **The v0 veto was broken.** `sys_flag` flagged 77.5% of real stars, because SAP scattered light (22-26% of bins on every CCD) outweighs most stars' other dip evidence. Every earlier "with veto" number ranked only the unflagged 22%. PDS 110 was flagged because of its SAP orbit-start ramp (79% of its dip chi^2), and β Pic because of orbit-edge common-mode dips.
- **`cmv2` works per event.** It labels each event as systematic or not: common-mode on the star's CCD or its 512-px cell, or inside the scattered-light and dump windows from QUALITY. A star is flagged only if a strong systematic event remains and no astrophysical event does. The rule uses PDCSAP with t_sys = t_res = 10, chosen on val.
  - **Flagged real stars: 4.6%** (v0: 77.5%).
  - **Test null FPR:** classical 0.177 -> **0.007** (v0 0.013); `ssl_v2_iforest` 0.247 -> **0.027** (v0 0.027).
  - **Recall stays intact.** `ssl_v2_iforest` at ~0.3 / 1 / 3% depth: 0.299 / 0.422 / 0.574 with no veto, 0.286 / 0.413 / 0.574 with cmv2, 0.144 / 0.311 / 0.506 with v0. Val agrees.
  - **Known positives vetoed:** 1 of 18 (β Pic); v0 vetoed 2.
- **`ens_v1`** takes a weighted max of the surprises from classical, `ssl_v2_iforest` and `ssl_v2_recon` (weights 1 : 1.5 : 1.5), then applies the cmv2 veto. The rule was chosen on val from rank-average, max, weighted max, Fisher and Stouffer.
  - Test recall at ~0.3 / 1 / 3%: **0.303 / 0.574 / 0.788**; all depths 0.614; null FPR 0.040.
  - With `explain.downweight` (`ens_v1_ka`): **0.324 / 0.607 / 0.788**; null FPR 0.050.
- **But only 2 of 18 known positives reach the top 1%** (classical: 12/18 with the veto). The ML weights above 1 lock classical's top stars out of the ensemble's top 1%, and val injections can't show that cost. An equal-weight max (built after seeing this, not selected) gives 0.259 / 0.510 / 0.752 and 6/18.
- **Seeds 0 / 1 / 2:** recall at ~0.3% 0.303 / 0.286 / 0.237; known positives 2 / 4 / 2 of 18; top-100 overlap 0.68 (passes).
- **Recall vs injected SNR** (~0.3% depth bin, test):
  - SNR < 10: 20% of injections, recall 0.03-0.07.
  - SNR 10-30: 41%, recall 0.17.
  - SNR 30-100: 33%, recall 0.54.
  - Reaching 50% overall would need ~0.64 in the SNR 10-30 bin.
- **Gate, strict reading: fails on both counts.** The best recall at ~0.3% is 0.32 (needs 0.5). Only 2 of 18 known positives are in the top 1% (needs 18; even a majority fails).
- **Reading:** the veto is fixed, and the ML scores and the ensemble now lead on injections at every depth. But injection recall and real-dipper recall pull in opposite directions, and tuning on val injections alone chooses against the dippers.
- **Next:**
  - A val target that represents real dippers without leaking the known positives. Options: injections into catalogued YSO hosts, or a held-out dipper set from another sector.
  - Better recall at SNR 10-30.

### 2026-11-10: first look at the candidates (S32 galleries, model-assisted eyeballing)
- 96 unique stars (the ens_v1 and ssl_v2_iforest galleries, 60 cards each): **53 EB, 29 artefact, 14 ordinary variable (incl. 6 flare stars), 0 weird**. The known positives are excluded from galleries by design.
- **Artefacts:** asteroid crossings of the background annulus (8 in the top picks: the target dips while neighbours brighten in sequence), and orbit-start / scattered-light SAP ramps, which dominate the exploration picks.
- **35 EBs have no VSX/Gaia/Prša entry.** They still need checking against the FFI EB catalogues before anything is called new. Notable: 77838934 (flat 4% box dips 16.4 d apart), 375150432 (flat 6% box dips, P≈12.2 d), 415401950 (very different eclipse widths).
- **Leads outside S32:** 339670265 has a single ~15% dip in S38 near BTJD 2343.3. 350298319 is a blended EB whose source is unclear.
- **Reading:** "big dip" ranks above "weird dip". Next: automated EB rejection and asteroid / scattered-light checks (docs/VETTING.md), then look deeper than the top 60.

### 2026-11-10: per-event veto `cmv3` (asteroid, step, column and background labels) and `ens_v2` on S32 (test split): gate FAIL, artefact nulls mostly gone
- Details are in ML_DESIGN §13. Reports:
  - `outputs/veto/cmv3/s0032/report.md` (before / after per null type, ablation, known positives, triage)
  - `outputs/eval/compare/s0032/classical__ens_v1__ens_v2__veto-cmv3/report.md` (the compare; `..._ens_v2/` is the same with the cmv2 veto)
- **`cmv3` = `cmv2` + eight per-event labels:** `bkg`, `pdc`, `gap`, `qbits`, `dumpedge`, `track`, `flank` (dip-spike-dip asteroid crossing), `column`. These are VETTING §4 items 1 and 3.
  - **New null set `n2`:** asteroid tracks across real stars, steps at momentum dumps, single-column events. Val 329 evaluated, test 331.
  - **Rule chosen on val:** all labels, PDCSAP only, t_sys 15, t_res 10. It flags **2.4%** of real stars (`cmv2` 4.6%).
- **Test null FPR in the top 1%** (none -> `cmv2` -> `cmv3`):
  - `ens_v2`: asteroid 0.51 -> 0.51 -> **0.04**; step 0.75 -> 0.60 -> **0.20**; column 0.69 -> 0.68 -> **0.08**; CCD 0.67 -> 0.05 -> 0.02; dump 0.43 -> 0.02 -> 0.02.
  - classical: asteroid 0.04 -> 0.06 -> 0.00; step 0.31 -> 0.32 -> 0.02; column 0.26 -> 0.31 -> 0.01.
  - `ssl_v2_iforest`: asteroid 0.13 -> 0.13 -> 0.04; step 0.70 -> 0.54 -> 0.20; column 0.26 -> 0.29 -> 0.06.
- **Recall is unchanged within noise.** `ens_v2` at ~0.3 / 1 / 3%: 0.303 / 0.590 / 0.788 with cmv2, 0.321 / 0.590 / 0.794 with cmv3. Astrophysical injections flagged: 0.04 (cmv2 0.03).
  - **Classical alone loses a little** (~3%: 0.330 -> 0.291; v1 null FPR 0.007 -> 0.013; known positives in the top 1% 12 -> 11 of 18), because cmv3 removes half as many real stars.
- **Known positives flagged:** 1 of 18 (β Pic), the same as `cmv2`.
- **Eyeballed S32 triage** (evaluation only):
  - 11 of 29 artefacts flagged (`cmv2`: 0): **9 of 10 asteroids**, but only 2 of 19 scattered-light cards. Those are quiet stars with no event above SNR ~5, picked by the ML for shape.
  - EBs: 1 of 53 flagged (the same star as cmv2). Ordinary variables: 0 of 14.
  - **Caveat:** the `flank` label and the half-maximum point width were added after a first pass flagged only 3 of the 10 asteroids, so the triage number is not independent. Val and test nulls are.
- **`ens_v2`** = `ens_v1` re-selected on val with the cmv3 veto. It is a weighted max 1 : 2 : 2.
  - Test, cmv3 veto: recall **0.321 / 0.590 / 0.794**, all depths 0.624, null FPR 0.020.
  - Known positives in the top 1%: **3 of 18**. Still the §12 problem.
- **SkyBoT (C57)** on the `ens_v2` top 200 that survive the veto, plus the triage stars (296 anonymous queries, ~20 min):
  - **2 of the top 200** have a known asteroid with V ≤ 19 within 6 px during their strongest event (ranks 52 and 92). Both are PDC bumps that `cmv3` missed.
  - **5 of 29 triage artefacts** are explained: Nemausa (V 10.9) accounts for 3 of them, including one card called scattered light.
  - Five bright "asteroid" crossings have no V ≤ 19 body nearby. They may be comets, uncatalogued bodies or the wrong epoch; not resolved.
- **Gate: fails** (recall at ~0.3% is 0.32, needs 0.5; 3 of 18 known positives).
- **Reading:** camera artefacts no longer reach the top on S32 nulls. What's left is the shallow step tails, scattered-light shapes the ML likes, and the known-positive loss in the ensemble.
- **Next:** run cmv3 on S43/S44 (ecliptic) to size the asteroid problem; label the eyeballed galleries per star from now on (`labels/`).
