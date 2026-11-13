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
- **"Short event" tests need the half-maximum width.** A bright asteroid crossing's run above SNR 3 is many times longer than its core, so a run-length cut drops exactly the brightest movers (the first `cmv3` pass, ML_DESIGN §13).
- **Store eyeballing labels per star** (`labels/`, with tic, label and source). The first S32 triage kept only counts, and the labels had to be rebuilt from the cards.
- **Replicated injection sets (`yso1`) need real-sector masks and per-replica cross labels.** A set with 20 copies of each host makes every event "common" in its own map. Use `veto_inj_set.py --real-masks-only` and `veto3_inj_set.py --real-masks-only`; the v0 `sys_flag`/`classical_cm` are meaningless there (ML_DESIGN §15).
- **`ssl_v2_recon` used to depend on row order** (fixed 2026-11-10, ML_DESIGN §16): scoring masks came from one RNG stream per run. They are now seeded per (seed, host TIC), so a matched null copy scores like its host. The score is still sensitive to *which* masks are drawn (re-drawing them moves a real star's percentile by median 0.02), so treat single-star recon percentiles as noisy. Use matched null copies to check new scorers.
- **Per-sector state in `runs/<run>/score/`:** `ref_state_sNNNN.pkl` (the IsolationForest is fitted on each sector's own train+val); encodings and surprise files made with per-TIC masks end in `_pt`. The old single `ref_state.pkl` is S32 only.
- **Other sectors use S32's frozen rules:** `apply_veto_rule.py` (cmv2/cmv3 flags) and `ensemble_v3.py --fixed ens_v3 --fixed-sector 32 --sets` (no injection sets there). Never re-tune on a sector's real-star results.
- **`evaluate.py --compare` writes `__inj-<set>` dirs for non-v1 sets.** Before 2026-11-10 every set overwrote the v1 report.
- **SkyBoT `-loc=C57` is geocentric, not TESS.** For (535) Montague, C57 and `500` agree to 0.02″ while Horizons from TESS (`500@-95`) is 189″ away. `scripts/skybot_check.py` queries the geocentre with a 12′ cone and applies the TESS parallax from Horizons vectors (`--observer tess`, default).
- **VizieR CfA mirror can return sentinel magnitudes** (Gaia Gmag = 999999999) or empty answers. `oddball.gallery.research` asks the IDIA mirror first, rejects sentinel rows, re-asks an empty answer on the other mirror and matches rows back to targets by coordinates (`_q` is unreliable).
- **External queries use `oddball.common.http.ANON_USER_AGENT`**: no names or project identifiers in headers, URLs or payloads.
- **TESS-SPOC FFI light curves at MAST stop at sector 86** (Nov 2026; the S3 bucket copy stops at 81). Half of the Track C survey's missing star-sectors are S87+. Re-check before any re-run; newer sectors only exist as QLP / TARS / TGLC or as TESScut cutouts, which are 11-23% noisier on the 2-hour scale.
- **TGLC sectors 12-55 are not listed by MAST's search** but are in the public bucket; see `spikes/mdwarf_stack/followup/products.py` for the path layout.
- **Large TIC cone queries through astroquery can stall for minutes.** Use small cones or the catalogue files, and always set a timeout.
- **A detrend can eat most of a long dip:** the frozen biweight removed ~78% of a 4-hour dip in one sector. Injections for any "would we have seen it" test must go in before detrending, at the raw depth.
