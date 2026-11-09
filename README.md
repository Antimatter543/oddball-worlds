# oddball-worlds

**Hunting for the weirdest stars in NASA's TESS data: stars that behave like nothing in any catalogue.**

TESS has recorded the brightness of millions of stars every few minutes for years. Almost all of them do something boring or well understood. We're after the ones that don't.

| Target | What it looks like in the data | Example |
|---|---|---|
| **Unknown unknowns** | Anything that's an outlier against millions of normal light curves | Boyajian's star ("the alien megastructure star") |
| **Stars being born** | Young stars still wrapped in dusty disks; irregular, deep, aperiodic dimming ("dippers") | Capistrant+2022 sample |
| **Exocomets** | A lopsided "shark-fin" dip as a comet's tail sweeps across the star | β Pictoris |
| **Disintegrating planets** | A dusty-tail transit whose depth changes from orbit to orbit | BD+05 4868 Ab |

## Why this, why now

Most anomaly searches of TESS used the first two years, or only the ~20k stars observed at 2-minute cadence. TESS's extended mission (Sector 27 onward, mid-2020 to now) has full-frame images of millions of stars at 10-minute and 200-second cadence, and that data has barely been searched for oddballs. For example, the best automated exocomet search stopped at Sector 26.

## Approach (Track A)

1. **Data:** TESS-SPOC full-frame-image light curves from Sector 27 onwards, binned and stored compactly.
2. **Injection-recovery:** inject synthetic dipper, exocomet and dusty-tail signals into *real* light curves to measure how complete the search actually is.
3. **Detection:** anomaly scores from learned light-curve embeddings (e.g. EncoTESS), with classical baselines (single-event and asymmetric-dip searches) alongside.
4. **Systematics rejection:** spacecraft momentum dumps, scattered light, and signals shared by neighbouring stars on the same CCD.
5. **Output:** a catalogue with completeness numbers, and a gallery of the strangest objects.

**First go/no-go gate:** recover ≥50% of injected 0.3%-deep events, and rank the known oddballs (Boyajian's star, known dippers, β Pic, BD+05 4868 Ab) in the top 1% without being told about them. If not, we stop and rethink.

See [`docs/PLAN.md`](docs/PLAN.md) for the full plan, validation strategy and failure catalogue, and [`docs/ML_DESIGN.md`](docs/ML_DESIGN.md) for how the models are trained and evaluated.

## Next: exploding stars

A planned second track watches the Vera Rubin Observatory's public alert stream for **young stars that suddenly flare up**. These are FU Orionis and EXor outbursts: a forming star swallows a burst of material from its disk and brightens 100-fold for months to years. The professional alert brokers mostly chase supernovae in other galaxies, so these young stars in our own galaxy get little attention. We'd check each candidate against catalogues of young stars and look back at its earlier TESS light curve.

## Status

🚧 Milestone 1a/1b: data pipeline and known-positive list.

```bash
uv sync
uv run python -I scripts/ingest_sector.py 32 --limit 2000 --include-known   # quick subset
uv run python -I scripts/ingest_sector.py 32                                # full sector (resumable)
uv run python -I scripts/plot_sanity.py 32                                  # -> outputs/sanity_*.png
uv run python -I scripts/build_known_positives.py                           # -> docs/known_positives.csv
uv run pytest
```

Each sector becomes `data/tess-spoc/sNNNN/` (or `$ODDBALL_DATA/...`): Parquet shards of
median-normalised PDCSAP flux binned to 30 min on a global BTJD grid (plus per-bin error, point
count, OR of quality flags, SAP flux and background), a metadata table and `provenance.json`.
No FITS is kept; nothing is detrended at ingest. Layout details: `src/oddball/tess/ingest.py`.

## Parked: Gaia DR4

Gaia DR4 lands on 2 Dec 2026. A neural orbit-fitter for stars with hidden companions may join later. See `docs/PLAN.md`.

## Credits and prior art

This builds on a lot of other people's work: TESS/MAST, lightkurve, the TESS-SPOC and QLP teams, Norazman & Kennedy (exocomets), Capistrant et al. (dippers), Hon, Vanderburg & Rappaport (disintegrating planets), and the EncoTESS authors. Full references are in [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md).

## License

MIT
