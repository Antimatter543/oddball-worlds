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

## Gotchas (learned)
- **MAST tops out near ~30 files/s** (~12 MB/s). A 10-min sector (S27–55) is ~160k light curves, ~67 GB downloaded, ~2.6 GB stored, ~1.6 h locally. 200-s sectors (S56+) move ~3× the bytes. For new sectors, prefer cloud ingest on Kaggle.
- **PDC can suppress real signals:** a ~2% dip in PDS 110 (S32, BTJD 2190–2192) is clear in SAP but weak in PDCSAP. Detectors must look at `sap_flux` too.
- **Faint, crowded stars (T≈15) blow up in PDCSAP** (±100% swings from the crowding correction). Don't trust their amplitudes, and stratify scores by Tmag.
- **About 0.02% of light curves have non-positive median PDCSAP.** They're logged in `failures.csv`. A future fix is to fall back to SAP.
- **VizieR:** use the catalogue id (e.g. `J/ApJS/263/14`), not `/table1`. The CDS Strasbourg servers sometimes time out; the CfA mirrors work.
- **Splits:** always use `oddball.common.splits.split_of(tic)`. Never invent a new split.
- **SAP is scattered-light heavy.** 22–26% of every CCD's S32 bins are common-mode in SAP (PDCSAP 2–6%). Any per-star "share of evidence in systematic bins" rule therefore flags most quiet stars (the v0 `sys_flag` flagged 77.5%). Veto per event and on PDCSAP instead (`cmv2`, ML_DESIGN §12).
