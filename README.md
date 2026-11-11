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

## Parked: young stars flaring up (Rubin)

**Status (Nov 2026): built, tested, parked.** As of Nov 2026, Rubin isn't sending alerts from the Galactic plane, where the young stars are: 0 of 118k catalogued young stars have plane coverage, and only 25 young stars match any Rubin object at all. The detector works on real Rubin supernovae as a positive control. We'll re-run `scripts/rubin_coverage_summary.py` monthly and resume when the plane gets alerts. Access notes: [`docs/rubin_access.md`](docs/rubin_access.md).

This track watches the Vera Rubin Observatory's public alert stream for **young stars that suddenly flare up**. These are FU Orionis and EXor outbursts: a forming star swallows a burst of material from its disk and brightens 100-fold for months to years. The professional alert brokers mostly chase supernovae in other galaxies, so these young stars in our own galaxy get little attention. We'd check each candidate against catalogues of young stars and look back at its earlier TESS light curve.

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

### Cloud ingest (Kaggle)

A sector pulls ~67 GB of FITS to keep ~2.7 GB of Parquet, so the download can run in a private,
CPU-only Kaggle kernel instead, and only the Parquet comes home. Needs an authenticated Kaggle CLI
(`~/.kaggle`). The kernel pip-installs this repo at a pinned commit, so that commit must be on
GitHub (default: current `main`).

```bash
uv run --group cloud python -I scripts/kaggle_ingest.py run 33 --limit 2000      # quick slice
uv run --group cloud python -I scripts/kaggle_ingest.py run 33 34 35 36 37       # one kernel per sector
uv run --group cloud python -I scripts/kaggle_ingest.py status 33 34 35 36 37
uv run --group cloud python -I scripts/kaggle_ingest.py download 33              # re-pull a finished run
uv run --group cloud python -I scripts/kaggle_ingest.py verify 33                # check a local sector dir
```

`run` keeps at most `--max-concurrent` kernels going (default 4; Kaggle allows 5 batch CPU
sessions per account), downloads each output to `data/tess-spoc/.kaggle-staging/`, verifies it
(shard and row counts against the metadata table and provenance, every shard decodes) and only
then moves it to `data/tess-spoc/sNNNN/`. It never overwrites an existing sector dir.
`provenance.json` gains a `kaggle` block (git SHA, kernel, timings, bytes downloaded) and a
`kaggle_download` block; each run is also logged to `data/kaggle/runs.jsonl`.

### Labelling gallery (human in the loop)

A local gallery for looking at a scorer's top-ranked stars and labelling them
(`docs/ML_DESIGN.md` §7). It works with any score file `outputs/scores/<scorer>/sNNNN/real.parquet`.

```bash
uv run python -I scripts/gallery_build.py --scorer ssl_v2_iforest --sector 32 --n 60   # ~5 min
uv run python -I scripts/gallery_serve.py          # then open http://127.0.0.1:8765/
```

- **Choosing stars:** 70% are the top-ranked stars and 30% are exploration. The exploration half
  is random picks from the rest of the top 5% plus "uncertain" picks from ranks 5–25%, weighted
  toward stars whose sub-scores disagree. Each card records why it was chosen. Known positives
  are never shown. Stars that are already labelled are skipped, so a rebuild shows new ones;
  `--relabel` brings them back. Cards are shuffled by default (`--order rank` keeps rank order).
- **Each card** (pre-rendered PNG in `outputs/gallery/<name>/cards/`) has these panels:
  - The whole sector in PDCSAP and SAP, with momentum dumps, orbit edges, scattered-light and
    other quality-flagged bins marked.
  - The scorer's time-resolved anomaly. This is the ML reconstruction surprise when the
    scorer's model wrote one for this star (`outputs/ml/*/timeresolved_*`, top 500 of the
    combined score only). Otherwise it is the classical 2-h dip SNR.
  - A zoom on the most anomalous window.
  - The 5 nearest Tmag ≤ 14 stars on the same camera/CCD, shown over the same window.
  - The star in the other ingested sectors.

  The page beside each card shows the scores and the percentile of each sub-score, the
  `sys_flag` systematics columns, and the known-class model's guess. It also shows catalogue
  matches: SIMBAD (TAP, CfA mirror first), VSX and Gaia DR3 variability from the known-class
  cache, and the ExoFOP TOI/CTOI lists. A one-line verdict flags TOIs, catalogued young stars
  and dimmers, and known common variables.
- **Catalogue cache:** results are cached under `data/gallery/xmatch/` with their provenance
  (service, query, UTC). Use `--offline` to build from the cache only and `--refresh-xmatch` to
  re-query.
- **Labels:** `interesting`, `systematic`, `known-boring-variable` and `unsure`, plus a note.
  Keys: `i`/`s`/`b`/`u` (or `1`–`4`), `j`/`k` to move, `n` to add a note, `h` to hide labelled
  cards. Labels are appended to [`labels/labels_v1.csv`](labels/labels_v1.csv) with the
  columns `tic, sector, scorer, rank, split, reason, label, note, utc, gallery`. Commit this
  file. It is append-only and the latest row per (tic, sector) wins. Every write is locked and
  fsynced, so a crash or restart loses nothing; a half-written last line is skipped.
- **Safety:** the server binds to 127.0.0.1 only and checks the Host header. Writing a label
  needs a JSON POST that carries a per-run token. Every field is validated: the label must be
  from the fixed set, the TIC must be one of this gallery's cards, and the note is capped at
  500 characters and made safe for CSV and spreadsheets. Catalogue text is HTML-escaped. The
  server uses no external JS or CDN.
- **Pixel check:** `scripts/pixel_check.py` draws a TESScut difference image, signal-locus map and per-pixel light curves per card (`cards/*_pix.png`); `scripts/vet_dossier.py` adds the catalogue research.

## Parked: Gaia DR4

Gaia DR4 lands on 2 Dec 2026. A neural orbit-fitter for stars with hidden companions may join later. See `docs/PLAN.md`.

## Credits and prior art

This builds on a lot of other people's work: TESS/MAST, lightkurve, the TESS-SPOC and QLP teams, Norazman & Kennedy (exocomets), Capistrant et al. (dippers), Hon, Vanderburg & Rappaport (disintegrating planets), and the EncoTESS authors. Full references are in [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md).

## License

MIT
