# Rubin/LSST alert access via ALeRCE (anonymous)

Verified by hand on **2026-11-09** (probe scripts plus `src/oddball/rubin/alerce_api.py`). No account, key or token is needed for anything below.

## Endpoints

| What | URL |
|---|---|
| **Production LSST host** | `https://api-lsst.alerce.online/` |
| OpenAPI specs (per service) | `https://api-lsst.alerce.online/<service>/openapi.json`, Swagger UI at `/<service>/docs` |
| Services | `object_api`, `lightcurve_api`, `magstat_api`, `classifier_api`, `probability_api`, `features_api`, `stamps_api`, `crossmatch_api` |

- The `alerce` Python client (v2.4.0, `AlerceSearchMultiSurvey`) hard-codes this same production host in `default_config.json`. The staging hostnames I guessed (`api-lsst-staging.alerce.online`, `api-lsst-dev.alerce.online`) do not resolve. `api.staging.alerce.online` returns 404. Nothing staging-only is needed.
- The old ZTF host `api.alerce.online` serves ZTF only.
- At the time of probing, object_api was v0.2.24.

### Routes used

| Route | Key params | Notes |
|---|---|---|
| `GET object_api/list_objects` | `survey_id=lsst`, `ra`, `dec`, `radius` (**arcsec**), `n_det=[min,max]`, `firstmjd=[a,b]`, `lastmjd=[a,b]`, `class_name`, `classifier`, `probability`, `oid` (list of up to 200), `page`, `page_size` (**max 200**, HTTP 422 above that), `count`, `order_by`, `order_mode` | Send `survey_id`. Without it, only ZTF is searched. `survey` is a deprecated alias. Lists are sent as repeated params (`n_det=2&n_det=1000000`). |
| `GET object_api/object` | `survey_id`, `oid`, `return_survey_extra=true` | Full DIAObject record: per-band `*_psfFluxMean`, `*_scienceFluxMean`, `*_fpFluxMean`, and so on |
| `GET lightcurve_api/detections` | `survey_id`, `oid` | Alert DIASources (full LSST schema fields; see below) |
| `GET lightcurve_api/forced-photometry` | `survey_id`, `oid` | DIAForcedSources: `mjd`, `band`, `visit`, `detector`, `psfFlux`(+Err), `scienceFlux`(+Err) |
| `GET lightcurve_api/conesearch/objects_by_coordinates` | `ra`, `dec`, `radius` (**degrees**), `neighbors` | Fast (0.4–4 s) but **mixes ZTF and LSST** and is capped at `neighbors`. In the plane, ZTF dominates. Not used. |
| `GET probability_api/probability` | `survey_id`, `oid` | All classifier rankings |
| `GET classifier_api/classifiers` | `survey_id=lsst` | `stamp_classifier_rubin_beta` (v2.0.1) and `_20260421` (v2.0.2): classes SN, AGN, VS, asteroid, bogus. `nuclear_classifier`: nuclear / off-nuclear. No light-curve (periodic-variable) classifier for LSST yet. |
| `GET crossmatch_api/crossmatch` | `survey_id`, `oid` | Precomputed Gaia/AllWISE/... neighbours |
| `GET object_api/htmx/tns/?ra=&dec=` | | HTML snippet of TNS matches (anonymous). It could be parsed for TNS names, but I did not use it. |

### Query patterns

- **Cone search.** `list_objects?survey_id=lsst&ra=..&dec=..&radius=<arcsec>&page_size=200&page=N`. A 1° cone (3600") works. Without `count`, a cone takes about 0.6–2 s. `count=true` adds a full count and was sometimes very slow (up to 80 s on a plane field).
- **Time ranges.** `firstmjd=[a,b]` and `lastmjd=[a,b]` filter on the object's first and last detection.
- **Galactic latitude.** There is no server-side l/b filter. Convert an l/b grid to RA/Dec cones client-side (`scripts/rubin_coverage.py`, `crossmatch.make_tiles`).
- **Bulk access.** There is no bulk download. A deep page through the full 10.5M-object list without a cone (`order_by=oid`) timed out with HTTP 504 after about 56 s. Tile the sky instead.
- **Duplicates.** `list_objects` can return the **same oid more than once** (one row per classifier row). Deduplicate on `oid`.

### Rate limits

- There are no published limits and no rate-limit or `Retry-After` headers.
- I saw sporadic **HTTP 504 after ~56 s** under load, on deep pages and on `count=true`.
- The client keeps at least 0.25–0.5 s between requests per thread, uses 2 threads at most, and backs off exponentially on 429 and 5xx.
- Every response is cached as gzipped JSON under `data/rubin/cache/`, so re-runs make no requests.

## What comes back

- **Object** (`list_objects` item): `oid` (= diaObjectId), `meanra`, `meandec`, `sigmara/dec`, `firstmjd`, `lastmjd`, `deltamjd`, `n_det`, `n_forced`, `n_non_det`, `class_name`, `classifier_name`, `probability`, `ranking`.
- **Detection** (DIASource): `mjd`, `band` (1=g 2=r 3=i 4=z 5=y 6=u), `visit`, `detector`, `psfFlux` (difference image), `scienceFlux` (direct image), `templateFlux`, `apFlux`, `snr`, `reliability` (real/bogus score), `isNegative`, `isDipole`, dipole fit fields, trail fields, `extendedness`, `ssObjectId`, and the full `pixelFlags_*` set (`saturated`/`saturatedCenter`, `bad`, `cr`, `edge`, `interpolated`, `suspect`, `streak`, `injected`, ...), plus `psfFlux_flag*` and `glint_trail`.
- **Forced photometry** (DIAForcedSource): `mjd`, `band`, `visit`, `detector`, `psfFlux`, `psfFluxErr`, `scienceFlux`, `scienceFluxErr`. **No pixel flags.**
- **Units.** Fluxes are in **nJy**, so mag_AB = 31.4 − 2.5 log10(flux). Times are MJD.

**Total flux.** `scienceFlux` is the PSF flux measured on the unsubtracted image at the object position. It gives the total brightness directly, without needing the template epoch. The detector uses it rather than reconstructing template + difference flux, because a template taken during an outburst would hide the event in the difference flux.

## Coverage and depth (as of 2026-11-09)

- **Volume.** 10,474,330 LSST objects. `firstmjd` values go back to **MJD 60973 (2025-10-26)**, before the public stream opened on 24 Feb 2026. `lastmjd` reaches 61352 (2026-11-08), so the data is current to the previous night.
- **Light-curve depth in practice.** Forced photometry is only served from about **MJD 61203 (2026-07-15)** onward, and only for a subset of visits. For example, objects detected since MJD 60983 with about 190 detections have about 50 forced points, all after 61203. No forced point predates an object's first detection. **The promised "up to 12 months of precovery forced photometry" is not present** in what ALeRCE serves.
- **Consequences for the light curves:**
  - Most epochs are alert detections. These exist only when the star differs from the template.
  - Quiescent epochs of a star in outburst are usually missing, so the detector falls back to `templateFlux` (carried on every detection) as the baseline.
  - The young-star matches have only 2–6 points each.
- **Galactic plane.** **Almost no alerts.** One-degree cones on Orion, Lupus 3, Cha I, CrA, Carina, the Lagoon (M8), Serpens and many |b| < 5° grid points return **zero** LSST objects. The YSO-bearing fields that do have alerts are mainly Upper Sco / ρ Oph and the bulge outskirts. The tile run below gives the numbers. This is presumably a template-availability limitation of year-1 alert production (no difference imaging without a template). It is not a broker filter: ALeRCE's ZTF holdings in the same cones are full.

This table summarises the young-star tile run (`scripts/rubin_crossmatch.py`, then `scripts/rubin_coverage_summary.py`, on 2026-11-09).
- **Tiles:** 2° cells holding at least one SPICY, Gaia DR3 YSO or known-YSO entry, Dec ≤ +35°.
- **Objects counted:** only LSST objects with n_det ≥ 2.
- **Star mode:** in tiles with ≤ 50 catalogue stars and > 200 objects, only objects within 2.5" of a catalogue star were pulled. The last column therefore undercounts the field total.

| \|b\| (deg) | tiles | tiles with LSST objects | YSO catalogue rows | rows in those tiles | LSST objects returned |
|---|---|---|---|---|---|
| 0–2 | 254 | **0 (0.0%)** | 118,822 | 0 | 0 |
| 2–5 | 377 | 1 (0.3%) | 22,065 | 91 | 182 |
| 5–10 | 538 | 28 (5.2%) | 9,880 | 279 | 226 |
| 10–20 | 912 | 98 (10.7%) | 20,474 | 2,406 | 30,386 |
| > 20 | 1,313 | 116 (8.8%) | 5,649 | 1,475 | 23,309 |

The YSO-bearing fields that do have alerts are dominated by **Upper Sco / ρ Oph** (l ≈ 352–356°, b ≈ +14 to +22°).

### Time coverage: an ingestion gap

These are global object counts by first-detection window:

| First detection | Objects |
|---|---|
| 2025-10-26 – 2026-02-24 (MJD 60970–61101) | ~3.25 M |
| 2026-02-24 – 2026-05-24 (61101–61193) | ~18.6 k |
| 2026-05-24 – 2026-08-16 (61193–61236) | ~7.06 M (3.8 M in the last 12 days) |
| 2026-08-16 – 2026-10-31 (61236–61343) | **0** |
| 2026-10-31 onward | resumes (e.g. EDFS RA 60, Dec −48; RA 300–340) |

- From these counts alone I can't tell whether the gap is a Rubin downtime or an ALeRCE ingestion gap.
- In the young-star fields (Upper Sco and the bulge edge), **no object has data after MJD 61235 (2026-08-16)**. Upper Sco is now in evening twilight, so it won't come back this year.
- The effective backtest window for young stars is therefore **2026-06-25 to 2026-08-16, about 7 weeks**.

## Not used: Lasair, Fink

Both need registration for their APIs and Kafka streams, so they are out of scope under the project's no-accounts rule.

## Mirrors

On 2026-11-09, CDS Strasbourg (`vizier.cds.unistra.fr`, `simbad.cds.unistra.fr`, `tapvizier`) timed out from here. The pipeline therefore uses the **CfA mirrors** `vizier.cfa.harvard.edu` and `simbad.harvard.edu`. Override them with `ODDBALL_VIZIER_SERVER` and `ODDBALL_SIMBAD_SERVER`.
