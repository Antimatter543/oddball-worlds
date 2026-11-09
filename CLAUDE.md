# oddball-worlds: agent notes

Read `docs/PLAN.md` first (milestones, go/no-go gate, failure catalogue). Any training or scoring work must follow `docs/ML_DESIGN.md`.

## Hard rules
- **Never commit data.** All data lives under `data/` (gitignored), or under `$ODDBALL_DATA` if set.
- **Never keep raw FITS.** Download, then convert to compact Parquet, then delete. Disk has about 189 GB free.
- **Memory is tight:** 15 GB RAM with often only 3–5 GB free, and a 4 GB GPU (GTX 1650 Super). Stream data and use small batches.
- **Run Python with `-I`** when it reads downloaded files.
- **No accounts or signups.** If a data service needs registration, stop and report it.
- **Record data provenance** for every run: sector, product, version, download date.

## Stack
- `uv` for environments (`uv sync`, `uv run`). Python 3.11–3.13.
- `lightkurve`/`astroquery` for MAST. Use direct HTTP bulk routes for scale (the HLSP curl scripts).
- `pytest` for tests, `ruff` for lint.

## Conventions
- Library code in `src/oddball/`, runnable entry points in `scripts/`, tests in `tests/`.
- Time system: TESS BTJD (BJD_TDB − 2457000) internally. Convert other surveys explicitly.
- Every claim about results needs the numbers behind it and the command that produced them.
