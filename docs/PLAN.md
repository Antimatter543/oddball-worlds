# Plan

## Goal
A completeness-calibrated search of TESS extended-mission (Sector ≥27) full-frame-image light curves for **asymmetric, single-event and variable-depth dimming**: exocomets, disintegrating planets and dippers. Unclassifiable anomalies come out as a by-product.

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
