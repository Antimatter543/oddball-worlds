# Vetting TESS anomalies: what professionals check, and what we can automate

Research notes, 2026-11-10. This collects how the SPOC pipeline, the TESS release notes, and the
vetting tools and dipper searches reject false positives, and maps each check onto our data
(30-min binned PDCSAP, SAP, BKG and QUALITY; no pixels). Payoff ratings are **expected, not
measured**. Each check must earn its place on val injections and the null set (PLAN.md
"Validation strategy" #3), never on the known positives.

**Glossary (plain language).**
- **Light curves and photometry.** A *light curve* is a star's brightness over time. TESS
  measures it by adding up the light in a small patch of pixels around the star, the
  *aperture*. *SAP* is that raw sum after the sky *background* is subtracted. The background is
  estimated from nearby faint pixels and stored as its own column (BKG). *PDCSAP* is SAP after
  the pipeline removes trends shared with other stars and corrects for light from neighbours
  that falls in the aperture (*crowding*, measured by CROWDSAP).
- **Pixels and the instrument.** Each TESS pixel is 21″ across, so the aperture often holds
  other stars too. A *blend* is a dip that really belongs to one of those neighbours. The sky
  is watched in ~27-day *sectors*, each split into two ~13.7-day *orbits*. Near the start and
  end of each orbit, sunlight reflected from the Earth or Moon (*scattered light*) raises the
  background. Every few days the spacecraft fires thrusters to unload its reaction wheels (a
  *momentum dump*), which briefly jolts the pointing. *QUALITY flags* are bits the pipeline
  sets on cadences it knows are suspect.
- **Pixel-level checks.** The *centroid* is the light-weighted centre of the star's image. If
  it moves during a dip, the dip probably comes from somewhere else. A *difference image*
  (average image during the dip minus the image outside it) shows directly which pixel lost
  light. *TESScut* is the public service that cuts those pixels out of the full-frame images.
  *SkyBoT* is a public service that lists known asteroids in a patch of sky at a given time.
- **Eclipsing binaries.** An *eclipsing binary* (EB) is two stars orbiting each other and
  taking turns blocking each other's light. That gives strictly periodic, same-shaped
  *primary* and *secondary eclipses*. When the two stars are similar, alternate eclipses
  differ slightly (*odd/even*). The eclipse is *V-shaped* when the stars only graze each
  other, and *flat-bottomed* when one passes fully in front of the other. A *dipper*
  instead dims irregularly, by different amounts each time, because dust is passing in front
  of the star.

## 0. What we already have, and what is one cheap request away

Everything here was checked first-hand on 2026-11-10 (S32 files, anonymous, no account).

| Source | What it gives | Cost | Status |
|---|---|---|---|
| Our shards | PDCSAP, SAP, `bkg` (= SAP_BKG / median SAP), OR of QUALITY, `npts`; meta has camera, CCD, RA/Dec, Tmag, Teff, radius, CROWDSAP, FLFRCSAP; tess-point col/row (cached by `cmv2`) | 0 | have |
| **SAP_BKG semantics** | SDPDD Table: "Estimated background flux contribution to the target aperture. **Already subtracted from SAP_FLUX.**" So `sap_flux + bkg` is the **raw aperture flux** before background subtraction (both are already divided by the SAP median at ingest). | 0 | verified [SDPDD] |
| The same TESS-SPOC LC FITS we already download | `MOM_CENTR1/2` (flux-weighted centroid), `PSF_CENTR1/2`, `POS_CORR1/2` (pointing/DVA motion) and an `APERTURE` image (11×11 px in the S32 file checked). Its bits mark the optimal-aperture pixels and the **pixels used for the background** (bit 3, value 4). **`ingest.py` drops all of these.** | ~400 kB per star from MAST | verified |
| QLP HLSP (T < 13.5) | Independent photometry: `SAP_FLUX`, `KSPSAP_FLUX` in a best aperture plus **`KSPSAP_FLUX_SML` / `KSPSAP_FLUX_LAG` (small and large apertures)**, its own `SAP_BKG`, `SAP_X/SAP_Y` centroids. Header gives the aperture radii (e.g. `SMALLAP 1.75`, `BESTAP 2.5`, `LARGEAP 3.0` px). Per-sector curl scripts. | ~100 kB per star | verified |
| TESScut | Pixel cutout of the FFIs. An 11×11 px S32 cutout took 6 s and 9 MB. Limit: 10 concurrent requests per user. | 6 s, 9 MB per star-sector | verified [TESSCUT] |
| SkyBoT cone search | Known asteroids in a cone at an epoch, **observer code `C57` = TESS**. Returns name, V, offset, motion in RA/Dec ("/h). | ~10 s per query | verified [SKYBOT] |
| VizieR (CfA / IDIA mirrors) | EB catalogues beyond Prša: Kostov+2025 `J/ApJS/279/50` (7,936 new EBs in table3, 2,065 known in table4, **872,720 unvetted NN-flagged targets in table2**); Gaia DR3 EB candidates `I/358/veb`; Gaia DR3 non-single stars `I/357` (incl. `tboeb` eclipsing solutions) | cone queries | ids verified |
| TESS Data Release Notes (DRN) | Per sector: orbit start/end times, momentum-dump times, cameras with Moon/Earth glints, **a table of fireflies/fireworks (FFI start/end, cameras)**, a **list of smear-affected columns (camera, CCD, column, bright star)**, quality-bit usage | 1 PDF per sector | read DRN48 (S32) |

**Most important single fact:** the asteroid-in-the-background-annulus failure needs no new data
to detect. If a dip in SAP is matched by a rise in BKG, then `sap_flux + bkg` stays flat: the
star didn't fade, the background estimate went up. [CASSESE] spell this mechanism out
(object in the background pixels → "the target appears dimmer"; in the aperture → brighter; one
flyby can give **dip-spike-dip**).

## 1. How professionals vet (one line each)

- **SPOC Data Validation** [DV] runs a battery on every detection:
  - odd/even depth and timing χ²;
  - weak-secondary search at the same period;
  - difference-image PRF centroid offsets against both the out-of-transit centroid and the
    catalogue position (3σ "radius of confusion", 0.2″ floor);
  - flux-weighted centroid motion (χ²₂ test, plus a source-offset estimate);
  - an optical-ghost test, which correlates the core-aperture and 1-px-halo flux with the
    transit model (core > halo for 98.1% of golden KOIs);
  - a bootstrap false-alarm probability.
- **TESS DRNs** [DRN48]:
  - quality bits per sector (bit 5/16 "Argabrightening" = sudden background change, "in
    practice ... glints and unstable pointing at times near momentum dumps"; bit 13/4096 SPOC
    scattered light; bit 12/2048 predicted stray light in FFIs; bit 16/32768 = too few stars
    for PDC to correct that cadence);
  - Earth/Moon boresight angles (trouble at Earth < ~25–35°, Moon < ~35°);
  - firefly/fireworks and smear-column tables;
  - TPS's de-emphasis of cadences where many TCEs pile up.
- **Planet Hunters TESS** [PHT]:
  - The science team sorted the top 500 per sector into ~5% keep, ~37% EB and ~58% discard.
    "The most common reasons for discarding ... momentum dumps and ... background events."
  - The LATTE reports [LATTE] then show momentum dumps, background flux ("asteroids or
    fireflies"), x/y centroids, a two-aperture test, in/out-of-transit difference images,
    per-pixel LCs, Gaia stars brighter than V = 15, the 5 nearest TESS targets' LCs, and BLS.
  - Final-stage discards were ~40% (background) EBs, ~25% systematics, ~15% background events
    and ~10% other.
  - A blend is confirmed when the centroids are separated by > 1 px, and neighbours more than
    5 mag fainter are ruled out as the source.
- **LEO-Vetter** [LEO], fully automated, the closest model for us:
  - flux-level tests: asymmetry (ASYM > 10), single-event domination, the "data gap" test
    (≥ 50% of events within 2 durations of a ≥ 0.3-d gap = scattered light), chases /
    uniqueness (Modshift), V-shape (b + Rp/R★ ≥ 1.5), secondary (MS4–6), odd/even (> 3σ),
    unphysical duration;
  - the one pixel test is a difference-image PRF offset > 15″ on FFI cutouts.
- **DAVE** [DAVE], **TESS-ExoClass** [TEC], **exovetter** [EXOV] and **vetting** [VET] are
  Robovetter-style metrics:
  - DAVE: Modshift, photocentre, shape;
  - TEC: a decision tree over SPOC DV metrics;
  - exovetter: LPP, Modshift, odd/even, SWEET, centroid;
  - vetting: a stand-alone TPF centroid-offset test.
- **TRICERATOPS** [TRI] turns TESS pixels plus TIC/Gaia neighbours into scenario probabilities.
  It flags a "likely nearby false positive" at NFPP > 0.1. It is built for periodic transits,
  not dippers.
- **Exocomet search, S1–26** [NK25]:
  - Manual vetting started with the background and centroid columns. Events coincident with
    a background rise were "mainly caused by nearby asteroids passing through background pixels".
  - They then re-checked raw flux, other sectors and QLP, and corrected each depth for
    dilution by the brightest close neighbour.
  - **A time-reversed null search** turned up only systematics and "EBs with unsubtracted
    background flux".
- **Dipper / Boyajian searches:**
  - **Boyajian+16** [B16]: the dips appear in SAP, not just PDC; no gaps or cosmic rays at dip
    times; no centroid shift; neighbours quiet; column anomalies, crosstalk and reflection ruled
    out.
  - **Capistrant+22** [C22]: single-pixel TESScut light curves, each source judged on- or
    off-target by 3 people (477 → 414 kept); M > 0.25 for dipping, Q for periodicity [CODY14].
  - **Little Dippers** [A19]: same-module stars within ~5′ quiet; crosstalk checked;
    re-extraction with a smaller aperture that excludes the contaminants.
  - **ZTF main-sequence dippers** [ZTF25]: drop dips within ±0.5 d of nights with > 10⁴ dips;
    drop sources clustered in combined sky-and-dip-time distance. About 90% contamination
    before visual vetting.

## 2. False-positive types → automatable checks

Data column: **have** = in our shards and meta now; **LC-FITS** = re-fetch the TESS-SPOC file
we already download (or keep the columns at ingest); **pix** = TESScut; **API** = another
anonymous service.

| # | False positive | Signature in our data | Automatable check | Data | Expected payoff | Sources |
|---|---|---|---|---|---|---|
| F1 | **Asteroid in the background pixels** (seen) | Sharp 1–4-bin dip in SAP *and* PDCSAP, with a simultaneous **bump in `bkg`**. `sap_flux + bkg` shows no dip. Often dip-spike-dip if the object also crosses the aperture. Neighbours within a few px show `bkg` bumps one after another. Median interaction ~2 h. | **Background-explains-dip, per event.** Measure the event depth in SAP (δ_SAP) and the background rise in the same bins (Δbkg, both already in SAP-median units). Label the event systematic `bkg` when Δbkg ≥ 0.5 δ_SAP and the raw series `sap_flux + bkg` has no significant dip (box SNR < 3). Add this label to `cmv2`'s per-event decision. | have | **High.** Zero new data. It also catches any other background-estimate glitch (glints, firefly in the annulus). Contamination is common: in ecliptic sectors 42–46 and 70–72, **every** 20-s target had minor-planet passes, and > 1% contamination is ~50% of stars at T 13–15 and > 90% at T 15–17 [CASSESE]. **S43 and S44 (in our training set) are ecliptic sectors.** | [CASSESE] §results; [NK25] §4.3; [PHT] §5; [SDPDD] SAP_BKG |
| F2 | **Asteroid track across neighbours** | A chain of `bkg` bumps (and SAP bumps or dips) in stars along a straight line, at a steady speed: main-belt ~1 px per 30-min cadence [PAL20], i.e. ~40″/h; SkyBoT gave 30–40″/h in a test cone. | **Moving-track test.** For each camera, collect positive `bkg`/raw-flux events (SNR > 5, ≤ 4 bins). Fit lines in (RA, Dec, t) with RANSAC, speed 5–300″/h, ≥ 3 stars per track. Flag any target whose event time matches a track passing within ~6 px (the 11×11 stamp holds the background pixels). | have | **High** on ecliptic and near-ecliptic cameras, low elsewhere (asteroid density falls steeply with ecliptic latitude [PAL20]). It also finds **unknown** asteroids, which SkyBoT can't. | [PAL20]; [ZTF25] space-time clustering; [CASSESE] |
| F3 | **Known asteroid, confirmation** | As F1/F2 | **SkyBoT** cone at the event epoch, radius ~3′, `-loc=C57` (TESS observer), `-objFilter=100` (asteroids only). Report V, offset and motion; call it explained if a body with V ≲ 19 passes within ~6 px during the event. **Do not use a geocentric observer:** TESS's parallax for a main-belt object is several arcmin, i.e. many pixels ([CASSESE] needed a 15′ search radius from the geocentre). | API (SkyBoT, anonymous, ~10 s per query, so ~100 queries is ~20 min serial) | **High** for the final gallery (~100–300 events), as a label source for F1/F2, and as a gallery annotation. Alternatives: JPL `sb_ident` needs an observer state vector for spacecraft [SBID]; `tess-ephem` is per named object, not a field search [TEPH]; `jorbit` N-body postdiction [CASSESE] is heavier. | [SKYBOT]; [CASSESE] §5 |
| F4 | **Scattered light at orbit start/end, Earth/Moon glints** | Smooth ramps in SAP and `bkg` near orbit edges and data gaps, camera/CCD-correlated. Residual PDCSAP dips at the edges that are *star-specific* (a local background mis-estimate) slip past the common-mode map. | (a) Add QUALITY bits **16** (Argabrightening), **2048** (predicted stray light, FFIs), **32768** (PDC could not correct) and **128** (pointing > 7″) to `cmv2`'s known windows. (b) **Gap-edge test** [LEO III.12]: label an event systematic if it lies within 2 durations of an orbit start/end or a ≥ 0.3-d gap *and* \|d bkg/dt\| there is above the CCD's 90th percentile. (c) Optional: per-camera Earth/Moon angle (JPL Horizons TESS = −95, plus tess-point boresights); risk mask at Earth < 35°, Moon < 35°. | (a)(b) have; (c) API (Horizons, anonymous) | **High** for (a)(b): trivial, and orbit edges are a top systematic in [PHT] and [NK25]. Low for (c): `bkg` already measures the effect. | [DRN48] §1.3, §2; [LEO] III.12; [SDPDD] Table 32 |
| F5 | **Momentum-dump steps and jitter** | A step, spike or short dip at a dump (bit 32; S32 dumps at 2179.48 and 2193.48). Flux moves in or out of the aperture as the pointing jumps. The dump cadence is dropped at ingest, so the bin has a low `npts`. | (a) Already: `cmv2`'s dump window. (b) **Step-vs-dip model test:** around a dump, compare the BIC of a step (Heaviside) and of a dip. Systematic if the step wins. (c) **Centroid/pointing correlation:** regress the event residual on `POS_CORR`/`MOM_CENTR` and flag when they explain > 50% of its χ². (d) `npts` guard: flag events made mostly of bins with `npts` < 50% of nominal. | (a)(b)(d) have; (c) LC-FITS | **Medium.** Mostly covered by `cmv2` already. (c) matters for bright and saturated stars: η¹ Pic's 0.14% "exocomet" was centroid jitter common to all bright stars nearby [NK25 App. B]. | [DRN48] §1.2; [PHT] §5 (top discard reason); [LEO] III.8/III.11; [NK25] |
| F6 | **CCD-, camera- or focal-plane-wide events** (seen) | Same event time on many stars | Already `cmv2` (CCD and 512-px cell). **Add a sector-wide event-time histogram** across all 16 CCDs. De-weight bins above the median + 3σ, as SPOC TPS does with TCE pile-ups [DRN48 §4.2], and as the ZTF dipper search drops nights with > 10⁴ dips [ZTF25]. Also mark the DRN **fireworks** windows (all cameras, ~10 min). | have + DRN table | **Medium**: cheap. Catches events spread over several CCDs at a low rate per CCD. | [DRN48] §3.2, §4.2; [ZTF25] |
| F7 | **Fireflies** (sunlit particles crossing a camera) | Short, spatially extended moving brightening; `bkg` bumps on part of a camera within minutes | Parse the DRN "Fireflies and Fireworks" table (FFI start/end `YYYYDDDHHMMSS`, cameras) into a windows file per sector. Label events in those windows on the listed cameras. F2's track test also catches them. | DRN (anonymous PDF) | **Low to medium.** Few per sector (S32: 1 fireworks, 2 fireflies), but they look like real events. | [DRN48] §3.2; [PHT] §5 |
| F8 | **Bright-star column artefacts** (smear, bleed) | Events shared by stars on the **same CCD column** as a saturated star, but not by stars nearby off that column | **Same-column test:** for each event, count other stars on the same CCD within ±2 columns that have an event in the same bins, compared with a random-column control. Also look up the DRN smear-column list. | have (col/row from tess-point) + DRN list | **Low to medium.** It covers the column anomaly behind 74 Kepler false-positive TCEs [K6], and our 512-px cells are too coarse to see it. | [DRN48] §3.1; [C14]; [K6]; [B16] |
| F9 | **Saturated target** (T ≲ 6.8; saturation limit per the TESS Instrument Handbook, not re-checked here) | Aperture includes the bleed columns, so pointing jitter moves flux in and out → dips. β Pic (T = 3.7) is one. | Tag T < 7 as `saturated` and require the F5(c) centroid/pointing check before calling an event astrophysical. | have; LC-FITS for (c) | **Low** in count, but it touches known positives (β Pic) and bright gallery picks. | [NK25] App. B; TESS Instrument Handbook |
| F10 | **Blend: a neighbour's EB or variable** | The same event in a neighbour's light curve; the depth doesn't fit the crowding; the centroid moves toward the neighbour during the dip | (a) **Neighbour event match** ("ephemeris matching" [C14]): every LC within 5 px (~105″) with an event overlapping in time. If neighbour depth × neighbour flux ≥ target depth × target flux, the neighbour is the likely source. (b) **Maximum-dilution test:** a contaminant can fake at most a fractional SAP depth of 1 − CROWDSAP from inside the aperture, so if δ_SAP > 1 − CROWDSAP the dip is (mostly) on target. (CROWDSAP counts TIC stars only.) With TIC/Gaia neighbours: a source Δm fainter can fake at most a depth of 10^(−0.4Δm). [PHT] ruled out neighbours more than 5 mag fainter, i.e. ≤ 1% maximum depth. (c) **Flux-weighted centroid motion** [DV III.6]: χ²₂ on the in-event shift of `MOM_CENTR1/2`; source offset ≈ −Δc·(1/δ − 1) [BRY13]. (d) **QLP aperture test** (T < 13.5): on target, depth is equal in `KSPSAP_FLUX_SML` and `_LAG`; a blend from outside gets deeper in the larger aperture [LATTE two-aperture test]. | (a)(b) have (+TIC/Gaia cone); (c) LC-FITS; (d) API (QLP HLSP) | **High.** Blends are failure #2 in PLAN.md. (a) and (b) are free. (b) alone clears most *deep* dippers: a 20% dip needs a contaminant giving ≥ 20% of the aperture flux. | [PHT] §5–6; [C14]; [DV]; [BRY13]; [LATTE]; [VET] |
| F11 | **Off-target source or optical ghost** (pixel level) | Signal stronger just outside the aperture than inside | (a) **Difference-image centroid** from a TESScut cutout: in-event minus out-of-event image, PRF or flux-weighted centroid. Compare with the out-of-event centroid *and* the catalogue position. Fail at > 3σ and > 15″ [LEO IV.5]. (b) **Core/halo ghost test** [DV III.7]: correlate the core-aperture flux and the 1-px-halo-ring flux with the event template; fail if halo > core. (c) Per-pixel light curves for the gallery card [LATTE], [C22]. | pix (+ APERTURE mask from LC-FITS) | **High for the final list only.** This is the standard every professional applies before calling anything real, but at ~6 s and 9 MB per star-sector it is a gallery-stage check (≤ ~300 stars). Delete the cutout after measuring (CLAUDE.md: never keep FITS); keep the numbers and a PNG. | [DV]; [LEO]; [LATTE]; [C22]; [TRI]; [TESSCUT] |
| F12 | **PDC artefacts and crowding over-correction** | Event in PDCSAP only (PDC injected it), or in SAP only (PDC ate it, as with PDS 110). PDCSAP depth ≈ δ_SAP / CROWDSAP, which blows up for faint crowded stars. | **SAP/PDC consistency per event:** require a co-located SAP event with δ_SAP·(1/CROWDSAP) within a factor of 2 of δ_PDC. Report a PDC-only event as `pdc_artefact`. For CROWDSAP < 0.5, rank by δ_SAP, not δ_PDC. Optionally confirm in QLP (independent background, aperture and detrending). | have (+QLP) | **High, cheap.** It is the first thing [B16] and [NK25] checked, and it addresses the CLAUDE.md "T ≈ 15 blow-up" gotcha directly. | [B16] §3.1; [NK25] §4.3; [SDPDD] CROWDSAP; [DRN48] (background algorithm change at S27) |
| F13 | **Method-level null: time-reversed search** | n/a | Run the asymmetric-dip (exocomet) detector on time-reversed light curves. Any excess of "reverse shark fins" measures the systematic floor of that sub-score. | have | **Medium**: one run. It calibrates the exocomet sub-score and gives a null set that comes from the real data. | [NK25] §4.4 |

**Also add to the null-injection set** (so each check can be measured on val, PLAN.md
validation #3):
- an asteroid-like null: a `bkg` bump plus a matching SAP dip, repeated along a straight track
  across neighbours;
- a step at the dump time;
- a single-column event.

## 3. Eclipsing binary vs astrophysical oddball, without a catalogue

Context: ~70% of our real top 1% has no catalogue entry, and `kc_v1`'s shape features call 15
of 18 known dippers "typical EB or rotator" (ML_DESIGN §11). The checks below are the physical
signatures that define an EB, and the ones a dipper, exocomet or disintegrating planet *fails*.
That makes them a better basis than typicality in generic features (a hypothesis to test on
val; see §4).

| # | Discriminator | EB looks like | Oddball looks like | Automatable check | Data | Expected payoff | Sources |
|---|---|---|---|---|---|---|---|
| E1 | **Strict periodicity, constant depth and duration** | ≥ 2 events at one period, depth and duration constant within errors | Dipper: depths vary event to event, quasi-periodic or aperiodic (Q → 1). Exocomet or Boyajian: unique events. | BLS on PDCSAP (and SAP for long events). Phase-fold at P, 2P and P/2. Per-event depth χ² against a common depth (DV-style). Cody Q (periodic Q ≈ 0) and M. EB-like if per-event depth scatter is consistent with noise and Q < 0.1. | have | **High.** This is what separates the deep EBs that fill our top 1% from dippers. | [DV] III.3; [CODY14]; [C22]; [LEO] III.7 (SNR consistency) |
| E2 | **Odd/even depth** | Equal for a true-period EB; unequal when the true period is 2P | n/a (no period) | Odd vs even depth significance, > 3σ ⇒ EB at 2P (exovetter `OddEven`, LEO OE). Needs ≥ 2 odd + 2 even events, so P ≲ 6 d in one sector. More with multiple sectors. | have | **Medium** (EBs only). Resolves half-period EBs that otherwise look like "changing depth". | [DV] III.3; [LEO] IV.4; [EXOV] |
| E3 | **Secondary eclipse at any phase** | A second, shallower dip at phase 0.5 (circular) or elsewhere (eccentric) | None | **Modshift**: convolve the folded LC with the primary's shape; primary, secondary, tertiary and positive-event significances (MS1–MS6). EB if the secondary is significant and the albedo it implies is unphysical. | have | **High** where periodic. Modshift is light-curve-only and available off the shelf (exovetter, DAVE). | [K7]; [K8]; [LEO] IV.3, III.9; [DAVE]; [EXOV] |
| E4 | **Shape: symmetric V or flat bottom vs asymmetric or irregular** | Symmetric, ingress = egress; U/flat (total) or V (grazing; LEO b + Rp/R★ ≥ 1.5) | Exocomet or dusty tail: fast ingress, slow egress. Dipper: irregular, multi-component, varying width. | Per event, fit (i) a symmetric trapezoid or limb-darkened eclipse and (ii) our injection profiles (asymmetric exponential tail, dusty tail; `inject/profiles.py`). Use ΔBIC plus an ingress/egress ratio. Add LEO's ASYM and the skewness-of-fit cut of [NK25] (skew < 8, error < 50%). | have | **High** for single events, where E1–E3 can't run. The only light-curve-level way to tell a long-period EB's lone eclipse from an exocomet. | [LEO] III.5, IV.2; [NK25] §4.2; [A19] §4.1 (dips not flat-bottomed) |
| E5 | **Out-of-eclipse phase curve** | Ellipsoidal (cos 2φ) ± reflection and beaming at the eclipse period; W UMa continuous variation | Dipper: no coherent harmonic at the dip period | Fit cos φ, cos 2φ and sin φ at the BLS period on out-of-event data. An ellipsoidal amplitude > 3σ with the cos 2φ term dominant ⇒ binary. Cross-match the ellipsoidal catalogue `J/MNRAS/522/29` (15,779 FFI candidates). | have | **Medium.** Explains contact and ellipsoidal binaries the eclipse tests miss. | [ELL]; [K8] (SWEET) |
| E6 | **Physical duration** | Duration ≤ (P/π)·(R₁+R₂)/a for TIC R★, M★ | Dips lasting days at P of days, or a/R★ < 1.5 ⇒ not an eclipse by a star (circumstellar dust) | LEO "unphysical duration": q = T/P > 0.5, or q_circ/q < 0.6, or a/R★ < 1.5. Use meta `radius`; get mass from TIC or a Teff relation. | have | **Medium.** Cheap, and puts dippers on the "not an EB" side. | [LEO] III.4; [A19] §4.1 |
| E7 | **Cross-sector ephemeris** | Eclipses recur at predicted times, same depth, in every sector | Dippers change morphology and depth from sector to sector | Predict eclipse times from the best sector's (P, t₀) in our other ingested sectors (S33/38/43/44/55). Score hit fraction and depth agreement. | have (6 sectors) | **High** where sectors overlap; free. [PHT], [NK25] and [B16] all checked other sectors. | [PHT]; [NK25]; [B16] |
| E8 | **Bigger EB catalogues** (soft labels for `kc`) | Listed | Usually not listed | Add to `build_known_class_labels`, keeping the existing young-star and known-positive hard exclusions: **Kostov+2025** FFI S1–82 (`J/ApJS/279/50`: 7,936 new + 2,065 known EBs; table2's **872,720 NN-flagged unvetted** targets *only as a soft feature, never a veto*, because an "eclipse-like" network flags dippers too). **Gaia DR3 EB candidates** `I/358/veb` (2.18M, G < 20, 25–50% complete) and NSS eclipsing solutions `I/357/tboeb`. **IJspeert+2024** `J/A+A/691/A242` (14,573 EBs, 4 yr of TESS) and **IJspeert+2021** `J/A+A/652/A120` (OBA). Ellipsoidal `J/MNRAS/522/29`. | API (VizieR) | **High, cheap.** Prša+2022 covers only 4,584 2-min EBs. Kostov+2025 is built from FFIs, i.e. our stars. | [KOS25]; [MOW23]; VizieR ids verified 2026-11-10 |
| E9 | **Infrared excess and youth** | No excess; main-sequence CMD position | Disk dippers: WISE excess (most of [C22]'s sample), above the main sequence, young-group members | AllWISE W1−W2, W2−W3, W1−W4 excess against the Teff locus; Gaia CMD height above the MS; Kounkel+2020 group membership. **Use only to up-weight the dipper class, never as a veto**: Boyajian's star has no IR excess, and the main-sequence dippers of [A19] and [ZTF25] are not disk-bearing YSOs. | API (VizieR) | **Medium-high** for the YSO-dipper target class. Zero for unknown unknowns. | [C22]; [A16]; [ZTF25] |
| E10 | **Gaia binarity flags** | RUWE > 1.4, NSS SB1/EB orbit, large `rv_amplitude_robust` | Usually single (not always) | Annotate only. [ZTF25] *flagged* RUWE > 1.4 rather than removing such stars. | API (VizieR `I/355`, `I/357`) | **Low-medium**: context for the gallery. A binary can host a disk. | [ZTF25]; [MOW23] |

## 4. Build order (top 5 by payoff ÷ effort for our setup)

Constraints: 4 GB GPU, tight RAM, anonymous services only. None of these needs the GPU. 1, 2
and 4a are pure NumPy over the shards we already stream.

1. **Background, raw-flux and SAP/PDC consistency labels in `cmv2`** (F1, F12, F4a/b, F6
   histogram; about a day). Each event gets `bkg` (Δbkg explains δ_SAP; `sap_flux + bkg` flat),
   `pdc_artefact`, `gap_edge` and extra QUALITY windows (bits 16, 128, 2048, 32768). These are
   per-event labels, so they keep the property that made `cmv2` work: a star keeps its real
   dip. Tune thresholds on val nulls with the new asteroid-like and step nulls. This directly
   targets the observed asteroid-in-annulus and orbit-edge false positives.
2. **EB light-curve tests feeding the explain-away** (E1–E4, E6, E7, plus E8 catalogues; 2–3
   days). The features are:
   - BLS period, per-event depth constancy, odd/even and Modshift secondary (exovetter's
     implementations are fine on CPU);
   - symmetric-vs-asymmetric ΔBIC;
   - the physical-duration test;
   - the cross-sector hit fraction.

   Add them as a new feature table next to `kcf_v1` (the combiner is feature-agnostic, ML_DESIGN
   §9) and add the Kostov+2025 and Gaia `veb` labels. These are the properties dippers
   physically lack, so they should separate EBs from dippers better than the typicality in the
   current features. Measure that on val injections into catalogued EB and YSO hosts only.
3. **Asteroid track test + SkyBoT confirmation** (F2, F3; 1–2 days). An internal RANSAC over
   `bkg` events finds known and unknown movers on every camera. SkyBoT (`C57`) confirms the top
   ~100–300 at ~10 s each, and the result is written as a gallery annotation. Run it first on
   S43/S44 (ecliptic) to size the problem.

   **Status (2026-11-10):** items 1 and 3 are built as the `cmv3` veto (ML_DESIGN §13), so far
   on S32 only. Item 1 also gained F5 (`dumpedge`) and F8 (`column`) from item 4a.
   - The track test is a pair-plus-support line search per CCD, not RANSAC.
   - A single-star dip-spike-dip label (`flank`) catches crossings that have too few
     neighbours for a track.
   - SkyBoT: `scripts/skybot_check.py`, run on the `ens_v2` top 200.
   - S43/S44 are still to do.
4. **Neighbour and blend pack without pixels, then keep centroids** (F10a/b/c, F8, F5c; ~1–2
   days).
   - (a) Neighbour event matching, the CROWDSAP / Δmag maximum-dilution test and the
     same-column test, all from shards and meta.
   - (b) **Change `ingest.py` to keep binned `MOM_CENTR1/2` and `POS_CORR1/2`**, stored as
     float16 offsets from the median. That's ~1.7 GB uncompressed per sector for all four
     columns, or ~0.8 GB for `MOM_CENTR1/2` alone, less after Parquet compression. A stored
     sector is ~2.7 GB today.
   - (c) Until then, re-fetch the LC FITS (~400 kB) for the top ~2,000 per sector and run the
     DV centroid-motion test.
5. **TESScut pixel vetting for the gallery only** (F11; 1–2 days). For each final candidate:
   - a difference-image centroid offset (vs the out-of-event centroid and the catalogue
     position);
   - the core/halo ghost test, using the SPOC `APERTURE` mask from the LC FITS;
   - per-pixel light curves and the Gaia neighbours overlaid on the gallery card (this closes
     the README "TODO: TESScut pixel thumbnails").

   Keep ≤ 10 concurrent requests and delete each cutout after measuring. Add the QLP aperture
   test (F10d) here for T < 13.5.

Deferred: Earth/Moon angle masks (F4c; `bkg` already shows it), DRN firefly and smear-column
parsing (F7, F8 lists; low counts), TRICERATOPS (built for periodic planets), Gaia binarity (E10;
annotation only).

## References

Every arXiv id below was resolved via the arXiv API or opened during this research.

- [A16] Ansdell et al. 2016, Young "dipper" stars in Upper Sco and ρ Oph observed by K2. arXiv:1510.08853
- [A19] Ansdell et al. 2019, The Little Dippers: transits of star-grazing exocomets? arXiv:1811.12414
- [B16] Boyajian et al. 2016, Planet Hunters X. KIC 8462852 – Where's the flux? arXiv:1509.03622 (§3.1)
- [BRY13] Bryson et al. 2013, Identification of background false positives from Kepler data. arXiv:1303.0052
- [C14] Coughlin et al. 2014, Contamination in the Kepler field: 685 KOIs as false positives via ephemeris matching. arXiv:1401.1240
- [C22] Capistrant et al. 2022, TESS dippers (visual survey; single-pixel TESScut vetting). arXiv:2209.03379
- [CASSESE] Cassese et al. 2026, Quantifying the contamination of TESS ecliptic-plane light curves by minor planets. arXiv:2601.02637
- [CODY14] Cody et al. 2014, CSI 2264 (Q and M metrics). arXiv:1401.6582
- [DAVE] Kostov et al. 2019, Discovery and Vetting of Exoplanets I. arXiv:1901.07459
- [DRN48] TESS Sector 32 Data Release Notes (DRN 48). https://archive.stsci.edu/missions/tess/doc/tess_drn/tess_sector_32_drn48_v02.pdf (others in the same directory)
- [DV] Twicken et al. 2018, Kepler Data Validation I: architecture, diagnostic tests and data products. arXiv:1803.04526 (§III.3 EB tests, §III.4 difference images, §III.6 centroid motion, §III.7 ghost, bootstrap); TESS usage in Guerrero et al. 2021, arXiv:2103.12538
- [EXOV] exovetter (LPP, Modshift, OddEven, SWEET, centroid). https://exovetter.readthedocs.io
- [ELL] Ellipsoidal binary candidates in TESS FFIs, VizieR `J/MNRAS/522/29` (15,779 systems)
- [K6] Mullally et al. 2015, Kepler candidates VI (column anomaly: 74 FP TCEs). arXiv:1502.02038
- [K7] Coughlin et al. 2016, Kepler DR24 (Robovetter, Modshift). arXiv:1512.06149
- [K8] Thompson et al. 2018, Kepler DR25 (Robovetter metrics incl. LPP, SWEET). arXiv:1710.06758
- [KOS25] Kostov et al. 2025, The TESS Ten Thousand Catalog. arXiv:2506.05631; VizieR `J/ApJS/279/50`
- [LATTE] Eisner et al. 2020, LATTE (JOSS). https://github.com/noraeisner/LATTE
- [LEO] Kunimoto et al. 2025, LEO-Vetter. arXiv:2509.10619; https://github.com/mkunimoto/LEO-vetter
- [MOW23] Mowlavi et al. 2023, Gaia DR3 eclipsing-binary candidates. arXiv:2211.00929; VizieR `I/358/veb`
- [NK25] Norazman, Kennedy et al. 2025, A search for transiting exocomets in TESS Sectors 1–26. arXiv:2508.04673 (§4.3 manual vetting, §4.4 reversed search, App. B)
- [PAL20] Pál et al. 2020, Solar System objects observed with TESS – DR1. arXiv:2001.05822
- [PHT] Eisner et al. 2021, Planet Hunters TESS II. arXiv:2011.13944 (§5 vetting, §6 multi-star systems)
- [SBID] JPL Small-Body Identification API. https://ssd-api.jpl.nasa.gov/doc/sb_ident.html
- [SDPDD] TESS Science Data Products Description Document, Rev F. https://archive.stsci.edu/missions/tess/doc/EXP-TESS-ARC-ICD-TM-0014-Rev-F.pdf (LC columns, aperture-mask bits, quality Table 32)
- [SKYBOT] SkyBoT cone search, `https://ssp.imcce.fr/webservices/skybot/api/conesearch.php?-ep=<JD>&-ra=<deg>&-dec=<deg>&-rd=<deg>&-loc=C57&-mime=text&-output=basic&-objFilter=100`; Berthier et al. 2016 (K2 extension). arXiv:1602.07153
- [TEC] TESS-ExoClass. https://github.com/christopherburke/TESS-ExoClass
- [TEPH] tess-ephem. https://github.com/SSDataLab/tess-ephem
- [TESSCUT] TESScut API. `https://mast.stsci.edu/tesscut/api/v0.1/astrocut?ra=&dec=&x=&y=&units=px&sector=`; https://astroquery.readthedocs.io/en/stable/mast/mast_cut.html
- [TRI] Giacalone et al. 2021, TRICERATOPS. arXiv:2002.00691
- [VET] Hedges 2021, `vetting` centroid test (RNAAS 5, 262). https://github.com/SSDataLab/vetting
- [ZTF25] A systematic search for main-sequence dipper stars using ZTF. arXiv:2508.03964
- Schmidt 2022, A search for analogs of KIC 8462852: a second list of candidates. arXiv:2111.01208 (Boyajian-analog search; its vetting details were not read for this note)
