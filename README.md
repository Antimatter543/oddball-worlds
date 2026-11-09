# oddball-worlds

**Searching NASA's TESS data for the weirdest exoplanet systems: comets around other stars, planets that are falling apart, and stars that dim for no obvious reason.**

Planet-search pipelines look for clean, repeating, U-shaped dips, where a planet crosses its star and blocks some of its light. Anything messier gets thrown away. That throwaway pile is where the strange things are:

| Target | What it looks like in the data | Example |
|---|---|---|
| **Exocomets** | A lopsided "shark-fin" dip: sharp drop, slow recovery, as a comet's tail sweeps across the star | β Pictoris |
| **Disintegrating planets** | A dusty-tail transit whose depth changes from orbit to orbit | KIC 12557548 b, BD+05 4868 Ab |
| **Dippers** | Irregular, aperiodic dimming from dust around young stars | Capistrant+2022 sample |
| **Unknown unknowns** | Anything that's an outlier against millions of normal light curves | Boyajian's star |

## Why this, why now

The best automated exocomet search in TESS (Norazman & Kennedy 2025) covered the first two years (Sectors 1–26). TESS has since observed 80+ more sectors at better cadence, and **as far as we can tell, nobody has searched those systematically for exocomets**. A 2026 exocomet review calls ML-assisted searches "the logical next step".

## Approach (Track A)

1. **Data:** TESS-SPOC full-frame-image light curves from Sector 27 onwards, binned and stored compactly.
2. **Injection-recovery:** inject synthetic exocomet, dusty-tail and dipper signals into *real* light curves to measure how complete the search actually is.
3. **Detection:** a classical asymmetric-dip matched filter as the baseline, plus anomaly scores from learned light-curve embeddings (e.g. EncoTESS).
4. **Systematics rejection:** spacecraft momentum dumps, scattered light, and signals shared by neighbouring stars on the same CCD.
5. **Output:** a catalogue with completeness numbers, and a gallery of the strangest objects.

**First go/no-go gate:** recover ≥50% of injected 0.3%-deep events, and rank the known oddballs (β Pic, BD+05 4868 Ab, known dippers) in the top 1% without being told about them. If not, we stop and rethink.

See [`docs/PLAN.md`](docs/PLAN.md) for the full plan, validation strategy and failure catalogue.

## Status

🚧 Day 0: scaffolding. Nothing to see yet.

## Parked: Track B (Gaia DR4)

Gaia DR4 lands on 2 Dec 2026. A neural orbit-fitter for stars with hidden companions may join later as a second track. See `docs/PLAN.md`.

## Credits and prior art

This builds on a lot of other people's work: TESS/MAST, lightkurve, the TESS-SPOC and QLP teams, Norazman & Kennedy (exocomets), Capistrant et al. (dippers), Hon, Vanderburg & Rappaport (disintegrating planets), and the EncoTESS authors. Full references are in [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md).

## License

MIT
