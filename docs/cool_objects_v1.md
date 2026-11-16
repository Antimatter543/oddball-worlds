# Cool objects v1: the real non-planet things in the two red-dwarf transit surveys (2026-11-16)

The two Track C surveys searched 4,350 red dwarfs within 50 parsecs and 15,452 at 50-100 parsecs for transiting
planets. Almost everything they flagged was rejected as "not a planet". This note goes back through that pile and
asks a different question: which of those rejected signals are real, interesting objects in their own right
(eclipsing pairs of stars, young stars with clumps of material around them, one-off deep dimmings), and which of
those appear to be new?

## Bottom line

- **About 140 real objects** came out of the 1,256 stars the surveys flagged (counting everything that is not
  noise, an instrument artefact, plain starspots or light leaking in from a neighbouring star). Most were
  already known: 110 stars catalogued as eclipsing binaries and 13 known young stars with dips.
- **18 appear new or only partly known in the catalogues checked.** 14 are not in any of them: 9 eclipsing pairs
  of red dwarfs, 3 young stars with sharp, changing dips ("complex periodic variables"), 1 star with three deep
  dimmings and 1 star with a single deep dip. 4 more are eclipsing pairs that a catalogue lists as a binary or
  variable but without the eclipses found here. All 18 come from the 50-100 parsec survey; everything real in
  the 50-parsec survey was already known.
  "New" here always means *not found in the catalogues listed below, as of 16 Nov 2026*; it is not a literature
  search.
- **The three most interesting:**
  1. **TIC 238891168** (SCR J0241-7719, 274 light-years): the Gaia satellite measured this red dwarf wobbling
     around an unseen partner every 152 days on an orbit seen almost edge-on. TESS caught two eclipses four
     years apart, and they land on the two moments Gaia's orbit says the stars line up. Together the two
     datasets pin the orbit to 152.74-152.82 days. If that is right, the next eclipses fall around
     20 December 2026 and 17 January 2027 (give or take about half a day).
  2. **TIC 220599904 and TIC 394664599**: two young red dwarfs (174 and 311 light-years) whose brightness
     shows sharp notches at fixed points in each spin that appear in some years and vanish in others. This is the
     signature of a rare class (about 50 known) thought to be clumps of gas or dust held in place by the
     star's magnetic field. Neither star is in the lists of such objects checked here.
  3. **TIC 308186412** (277 light-years, a candidate member of a young star group): something passed in front of
     it three times (April 2019, September 2020, May 2021), each time hiding 10-15% of its light for about five
     hours. The period is 258 days, or a multiple of it. A fainter partner star sits 6.5″ away, too close for
     TESS to separate, so either star could be the one being eclipsed.
- **Nothing unexplained** among the stars looked at by eye. Two odd cases are listed at the end; both have
  ordinary likely explanations.
- **Not submitted anywhere.** Fields for 8 AAVSO VSX new-variable entries (eclipsing binaries VSX does not
  list) are prepared below, plus 5 with caveats; nothing was sent and no account was made.

Images (dark style, for a phone), in `outputs/planets_spike/visuals/`:
`cool_01_wobble_and_eclipses.png` (TIC 238891168), `cool_02_three_dimmings.png` (TIC 308186412),
`cool_03_dips_come_and_go.png` (TIC 220599904).

## What was looked at, and how deep

**Inputs** (the survey output as it stands; the frozen search and vetting were not changed or re-run):

| Pile | Signals | Stars |
|---|---|---|
| Long range (10-130 d), every graded signal, both surveys | 1,319 | 656 (50 pc: 252, 100 pc: 404) |
| Short range (1-10 d), at or above the survey's short-range threshold (~0.7% per-star false-alarm rate on nulls) | 2,073 | 876 (50 pc: 279, 100 pc: 597) |
| Union | | 1,256 (276 in both) |

**Depth of treatment** (per star, in `labels/cool_objects_v1.csv`, column `depth`):

| Depth | Stars | What it means |
|---|---|---|
| `full` | 142 | eyeballed light curve, period work; for every kept object also a pixel check and a per-object catalogue look (SIMBAD references read, Gaia DR3 neighbours) |
| `lc_look` | 140 | light curve rebuilt and folded automatically (`cool/ebscan.py`), seen on contact sheets, label from catalogues + rules |
| `catalogue` | 974 | catalogue cross-match only; label from catalogues + the survey's own first-pass class |

The brief allowed full depth on the best ~40 and catalogue-only for the rest. In practice 282 stars got a light-curve look and 142 were labelled by eye (all kept objects among them). Most of the
catalogue-only stars are short-range signals on stars whose light curves look like ordinary starspots
(see "The short-range pile").

**Evidence used for every kept object:**

- *On target:* the survey's pixel check (`vetting/pixel_check.py`, unchanged), run at the deepest eclipses
  (`cool/pixev.py`, `cool/deep.py`). It asks whether the dimming sits on the target's pixels or on a neighbour's
  (TESS pixels are 21″ wide). "Pass" means the dip is on the target or on something closer to it than about
  10″; every such close star is listed, with its Gaia parallax and motion to tell a physical partner from a
  background star.
- *Period:* folded on the best period from a trial grid (the survey period and its ×1/3, ×1/2, ×2, ×3, plus any
  catalogued period), with odd/even and secondary-eclipse checks (`cool/ebscan.py`) and trapezoid fits to each
  eclipse (`cool/deep.py`). For objects with only 2-3 eclipses, every period the data allow is listed
  (`cool/aliases.py`, `cool/gaiaorbit.py`).
- *Catalogues:* next section.

## Catalogues checked

All 1,256 stars were cross-matched in one pass (`cool/catpass.py`, `cool/catgaia.py`, `cool/catextra.py`),
anonymously and cached under `outputs/planets_spike/cool/cat/`. A source that did not answer is "not checked",
never "not found". The kept objects were also checked one by one (`cool/gaiakept.py`, `cool/vizcheck.py`,
`cool/simrefs.py`), asking both VizieR mirrors before believing an empty answer.

| Catalogue | What it lists | Match | Status |
|---|---|---|---|
| AAVSO VSX (`B/vsx/vsx`) | all known variable stars | 30″ on target, 63″ neighbours | checked |
| Prsa+2022 TESS EBs (`J/ApJS/258/16`) | 4,584 EBs, 2-min data S1-26 | TIC id or 10″ | checked |
| Kostov+2025 TESS EBs (`J/ApJS/279/50`) | ~10,000 EBs from full-frame images | TIC id or 10″ | checked |
| IJspeert+2024 (`J/A+A/691/A242`) | EBs among OBAF stars (no M dwarfs expected) | TIC id or 10″ | checked |
| Gaia DR3 eclipsing binaries (`I/358/veb`) | 2.2 million EBs | Gaia source id | checked |
| Gaia DR3 variability classes (`I/358/vclassre`) | | Gaia source id | checked |
| Gaia DR3 two-body orbits (`I/357/tbooc`) | astrometric / spectroscopic / eclipsing orbits | Gaia source id | checked |
| Gaia DR3 rotation (`I/358/vrot`) | | Gaia source id | not checked (query error) |
| Gaia DR3 main table (`I/355/gaiadr3`) | parallax and motion of neighbours, RUWE | 25″ | checked |
| ASAS-SN variables (`II/366`) | | 30″ | checked |
| ZTF periodic variables, Chen+2020 (`J/ApJS/249/18`) | | 5″ | checked |
| ATLAS variables, Heinze+2018 (`J/AJ/156/241`) | | 5″ | checked |
| Capistrant+2022 TESS dippers (`J/ApJS/263/14`) | young stars with dips | 10″ | checked |
| Bouma+2024 complex periodic variables, Zhan+2019, Guenther+2022, Tajiri+2020 | young-star dips and CPVs | via SIMBAD references only | no VizieR table on either mirror; checked for kept objects through their SIMBAD reference lists |
| Marton+2019 young-star candidates, Kounkel+2020 young groups, AllWISE infrared excess | youth | 3-5″ | checked |
| SIMBAD (Harvard mirror TAP) | object type; for kept objects, every reference title | 5″ on target, 63″ neighbours | checked |
| ExoFOP TOI list (2026-11-16) and CTOI list | planet candidates and their false positives | TIC id | checked |

The ESA Gaia archive and the CDS VizieR TAP service could not be reached from here on 16 Nov (connection
timeouts), so the Gaia DR3 tables were read through the VizieR mirrors at CfA and IDIA instead (same tables).
The bulk cross-match finds few Gaia DR3 eclipsing binaries among these stars (3 of 1,255); one-by-one checks of
known EBs that are missing (for example TIC 461579405, 5704681) also came back empty from both mirrors, so this
looks real (Gaia's EB list is thin for faint red dwarfs) rather than a query failure. 11 of the stars have a
Gaia DR3 orbit (all astrometric); only TIC 238891168 also shows eclipses. Of the two others with a deep survey
signal, TIC 231815218 (orbit 308 d, tilt 112 ± 5°) and 144121387 (715 d, 98 ± 1°) are too far from edge-on to
eclipse (231815218's dip was judged an artefact; 144121387's 1.3% signal was not looked at by eye).

## Everything kept

Distances from Gaia DR3 parallax; "depth" is the fraction of light lost at the deepest point; P is the period in
days. "Pixel" is the pixel check at the eclipses. Status: **new** = not found in any catalogue above;
**partly** = a catalogue lists the star as binary or variable but not these eclipses; **known** = catalogued.

**Apparently new (14) or partly known (4)**

| TIC | Other name | Distance | What | P (d) | Depth | Pixel | Status |
|---|---|---|---|---|---|---|---|
| 238891168 | SCR J0241-7719 | 84 pc | Gaia orbit + 2 eclipses | 152.74-152.82 | 27% / 41% | pass | partly (Gaia DR3 astrometric orbit) |
| 220599904 | 2MASS J05320932-4607036 | 53 pc | complex periodic variable | spin 0.6285 | dips 2-5% | pass | new |
| 394664599 | Gaia DR3 5199332484776169600 | 95 pc | complex periodic variable | spin 1.3632 | dips 4-7% | pass | new |
| 22739419 | Gaia DR3 127661138301191168 | 95 pc | complex periodic variable | spin 0.3513 | sharp dips | pass (weak) | new |
| 308186412 | 2MASS J08420090-7113216 | 85 pc | 3 deep eclipses | 257.96 (or 515.9, 773.9) | 10-15% | pass, co-moving star 6.5″ | new |
| 35665763 | LP 588-14 | 95 pc | one deep dip, 13 Oct 2023 | - | 23% | pass | new |
| 435680606 | LP 13-133 | 55 pc | eclipsing pair, near twins | 3.00782 | 40% / 34% | pass | new |
| 162689162 | | 79 pc | eclipsing pair, eccentric | 21.02706 | 21% / 22% | pass | new |
| 344769778 | | 83 pc | eclipsing pair, eccentric | 18.58115 | 25% / 4.4% | pass | partly (ATLAS 'dubious' variable, P 18.19) |
| 468353260 | | 93 pc | eclipsing pair | 5.87063 | 27% / 15% | pass | new |
| 316822151 | | 73 pc | eclipsing pair, eccentric | 5.02091 | 20% / 4.7% | pass, co-moving star 1.8″ | new |
| 147304495 | | 75 pc | eclipsing pair | 13.14687 | 16% / 9.5% | pass | new |
| 10747609 | | 78 pc | eclipsing pair | 5.98383 | 15% / 13% | pass, brighter co-moving star 2.4″ | new |
| 7322727 | | 69 pc | eclipsing pair (shallow) | 1.90517 | 4.5% / 2.0% | pass | new |
| 233466569 | LSPM J1524+3942N | 71 pc | eclipsing pair | 0.51392 | 5.9% / 2.6% | pass, co-moving star 6.6″ | new |
| 446169146 | LP 7-265 | 80 pc | grazing eclipsing pair | 0.80976 (or 0.40488) | 3% / 3% | pass | new |
| 261590581 | LEHPM 5710 | 87 pc | eclipsing pair | 4.45725 | 20% / 14% | pass | partly (VSX/Gaia variable, wrong period 0.0196 d) |
| 94917984 | 2MASS J07344924+2818158 | 92 pc | eclipsing pair | 1.03034 | 6.2% / 5.3% | pass | partly (known double-lined binary, APOGEE) |

**Known, found again (selection; the full list is in `labels/cool_objects_v1.csv`)**

| TIC | Name | What | Found in |
|---|---|---|---|
| 199574208 | CM Dra | classic red-dwarf eclipsing pair, P 1.268 | VSX, Prsa+2022 |
| 142588954 | LSPM J1112+7626 | eccentric eclipsing pair, P 41.03 | VSX |
| 427240420, 358908226, 142979644, 392651143, 408651394, 471012590, 22131853, 156067195 | CzeV390, MNIC V134, NEV364, NV Lyr, CCDM J04404+3127B, LP 133-373, HAT-196-0006238, HAT-148-0000574 | eclipsing pairs | VSX |
| 172517058, 166090445, 91542943, 160492103, 230235519 | | eclipsing pairs (checked by eye) | Prsa+2022 / Kostov+2025 / Gaia |
| 94 more | | catalogued eclipsing binaries, labelled automatically | VSX / Prsa+2022 / Kostov+2025 / ASAS-SN / ZTF / ATLAS / Gaia / SIMBAD |
| 272086159 | TOI-176.01 | planet candidate already shown to be an eclipsing pair | ExoFOP |
| 57830249 | TWA 33 | young star with deep irregular dips (to 68%) | Capistrant+2022, Bouma+2024 |
| 193136669 | TWA 34 | complex periodic variable, P 1.567 | Prsa+2022, Capistrant+2022, Bouma+2024 |
| 2234692, 425933644, 234295610, 402980664, 280945693 | -, UPM J0014-6003, -, LP 12-502, UCAC4 103-046417 | complex periodic variables | Bouma+2024 (and Zhan+2019) |
| 146725602 (+ partner 146725601), 167303776, 241626984, 266842406, 311410991 | | young stars with dips | Capistrant+2022 |
| 294328887 | UCAC4 154-009819 | complex periodic variable, spin 0.3545, dips to 10% | Bouma+2024 |

## The stories, best first

**1. TIC 238891168 (SCR J0241-7719): a wobble and two eclipses that tell the same story.** This red dwarf, 274
light-years away, is one of the stars whose position Gaia saw swing back and forth: Gaia DR3 fits it with an
orbit around an unseen partner, period 152.4 ± 0.5 days, quite stretched (eccentricity 0.56), and tilted 87 ± 3
degrees, i.e. nearly edge-on. Separately, TESS saw it lose about a quarter of its light for four hours in July
2019, and about two-fifths in August 2023. Both dips are on the star's own pixels. Nothing else in eight
sectors.

On a stretched orbit the two moments when the stars line up (one passing in front, then the other) are not half
an orbit apart. Gaia's orbit predicts they are 127 days apart (68% range 120-133). The two TESS eclipses are
1,499.3 days apart, which is 9 orbits plus 128 ± 4 days. They match: the July 2019 dip is one star passing in
front of the other and the August 2023 dip is the reverse, which also explains why they have different depths.
TESS's other sectors then rule out every period near Gaia's except 152.74-152.82 days (`cool/gaiaorbit.py`).
Each eclipse is also within about 7-8 days of where Gaia's orbit alone puts it, inside that orbit's timing
uncertainty. Gaia DR3 catalogues the orbit; none of the catalogues checked lists the eclipses. The partner is
probably another small red dwarf: it blocks up to 41% of the light, so it must be at least about two-thirds the
size of the star it covers. If the period is right the next eclipses are due around 20 December 2026 and
17-18 January 2027 (each ± about half a day), then 21-22 May and 19 June 2027. The star is faint (G 15), so
this needs a telescope; long-period eclipsing red dwarfs whose orbit is also measured by Gaia are rare and are
exactly what is needed to measure red-dwarf masses and sizes.

**2. TIC 220599904 and TIC 394664599: young stars with clumps that come and go.** Both are young red dwarfs that
spin fast (every 15 hours and every 33 hours). Most of the time their brightness rises and falls smoothly as
starspots rotate in and out of view. But in some years sharp notches appear at fixed points in the spin: in
late 2020 TIC 220599904 (174 light-years) dipped 2.5% and 5% twice every spin; in 2018 and again in 2024 the
notches were gone. TIC 394664599 (311 light-years) dipped by 4-7% (single points to 15%) twice per spin in
April 2021 and once per spin in May 2023. Both pass the pixel check. These "complex periodic variables" are thought to be
clumps of gas or dust caught in the star's magnetic field and carried around with it. About 50 are known, most
from TESS. Neither star appears in Capistrant+2022's dipper table, and neither has a Bouma+2024, Zhan+2019 or
Guenther+2022 paper in its SIMBAD references (the CPV lists themselves are not on VizieR). A third, weaker
case is TIC 22739419 (309 light-years, spin 8.4 hours), whose pixel check passes but with a
fainter background star 5″ away also lighting up a little.

**3. TIC 308186412: something passed in front of a young red dwarf three times.** This faint red dwarf, 277
light-years away, is listed as a candidate member of a young star group (SIMBAD "young star candidate";
BANYAN XIII, SPYGLASS I and III, Laos+2022). TESS watched it in 12 sectors at 2-minute cadence. Three times, in
April 2019, September 2020 and May 2021, it dimmed by 10-15% for about five hours. If all three are the same
kind of eclipse the only period the gaps allow is 257.96 days; if one of them is the other star's eclipse,
515.9 or 773.9 days also fit. Gaia lists no orbit or variability for it. A second star 6.5″ away, one
magnitude fainter in Gaia, has the same distance and moves across the sky the same way: it is a physical
partner, and TESS cannot separate the two, so either could be the one being eclipsed. On the brighter star a
10-15% eclipse needs a companion at least about two-fifths of its width (bigger than Jupiter); on the fainter one
the true eclipse is deeper. Either way this is a long-period eclipsing system in a young wide pair, with no
catalogue entry.

**4. Ten eclipsing pairs of red dwarfs that no catalogue lists as eclipsing.** The cleanest is **TIC 435680606** (LP 13-133,
179 light-years): two nearly identical red dwarfs that eclipse each other every 3.008 days, each hiding 34-40% of
the pair's light. Three have stretched (eccentric) orbits: **TIC 162689162** (21.03 days, second eclipse at
0.40 of the orbit), **TIC 344769778** (18.58 days, a 25% eclipse and a 4% one at 0.22 of the orbit; ATLAS lists it as a "dubious"
variable with an 18.19-day period, so it was partly noticed before) and
**TIC 316822151** (5.02 days). Long-period eclipsing red dwarfs like the first two are scarce and are useful
for testing how red dwarfs are built. The rest (468353260, 147304495, 10747609, 7322727, 233466569, 446169146) are
ordinary short-period pairs. TIC 446169146 shows a 3%, V-shaped dip every 0.405 days, perfectly regular for
five years with nothing in between, so it is probably a grazing pair of equal stars with a 0.81-day orbit. Three of the ten have a close partner star that TESS cannot separate (noted in the table).
TIC 147304495 and 7322727 have eclipses that last longer than expected for stars as small as the TIC says,
so the eclipsing stars may be larger than listed.

**5. TIC 35665763 (LP 588-14): one deep dip.** On 13 October 2023 this red dwarf, 310 light-years away, lost 23%
of its light for about an hour and a half, once, in all the TESS data. The dip is on the star's own pixels and
no other star lies within 12″. A single eclipse by a companion on a long orbit is the simplest explanation.

**6. Partly known.** Besides TIC 344769778 above, **TIC 261590581** (LEHPM 5710) is in VSX as a Gaia variable with a period of 0.0196
days; TESS shows it is an eclipsing pair with a 4.457-day period. **TIC 94917984** is known from APOGEE spectra
to be a double-lined binary (two stars seen in the spectrum); TESS shows the pair also eclipses, every 1.030
days. **TIC 238891168** is story 1.

## Apparently uncatalogued: fields for an AAVSO VSX submission

Prepared only. **Nothing has been submitted anywhere and no account was created.** The discoverer field is left
blank on purpose. Positions: Gaia DR3 moved to epoch J2000 with its proper motion (equinox J2000/ICRS).
Magnitudes in the TESS band (TIC Tmag at maximum; minimum from the measured depth, which neighbours' light can
make shallower than the truth, so the range is a lower limit). Epoch: time of the deeper eclipse, BJD (TDB).
Source: `outputs/planets_spike/cool/vsx_fields.csv` (`cool/vsxfields.py`).

| Name / identifier | RA, Dec (J2000) | Type | Mag range (TESS) | Period (d) | Epoch (BJD) | Discoverer | Evidence |
|---|---|---|---|---|---|---|---|
| TIC 435680606 = LP 13-133 = Gaia DR3 565629811152693376 | 01 12 13.95 +79 55 50.3 | EA | 13.61 - 14.18 (sec 14.07) | 3.00782 | 2459744.8715 | | TESS PDCSAP; eclipses 40%/34%, sec. phase 0.499; pixel check pass |
| TIC 468353260 = Gaia DR3 573216475744350592 | 00 48 34.13 +83 53 18.9 | EA | 14.60 - 14.95 (14.77) | 5.87063 | 2459728.5625 | | 27%/15%, sec. 0.500; pixel pass |
| TIC 344769778 = Gaia DR3 1998994263375897472 | 23 38 47.01 +58 21 32.0 | EA (eccentric) | 12.86 - 13.18 (12.91) | 18.58115 | 2458779.1979 | | 25%/4.4%, sec. phase 0.218; pixel pass; ATLAS calls it a 'dubious' variable (P 18.19 d); VSX has a different Gaia source nearby (S type, no period) |
| TIC 316822151 = Gaia DR3 2108199098270536832 | 18 20 48.66 +37 59 22.7 | EA (eccentric) | 13.85 - 14.09 (13.91) | 5.02091 | 2459392.3229 | | 20%/4.7%, sec. phase 0.921; pixel pass; faint co-moving star 1.8″ |
| TIC 162689162 = Gaia DR3 1424108183514450304 | 16 27 05.86 +51 39 04.7 | EA (eccentric) | 14.26 - 14.52 (14.54) | 21.02706 | 2459675.0347 | | 21%/22%, sec. phase 0.398; pixel pass |
| TIC 147304495 = Gaia DR3 6578080208708899968 | 21 40 55.55 -43 07 00.0 | EA | 14.43 - 14.62 (14.54) | 13.14687 | 2459063.8819 | | 16%/9.5%, sec. 0.502; pixel pass |
| TIC 7322727 = Gaia DR3 4800389362244174592 | 05 21 44.18 -43 54 21.9 | EA | 14.91 - 14.96 (14.93) | 1.90517 | 2459144.8368 | | 4.5%/2.0%, sec. 0.492; pixel pass |
| TIC 94917984 = 2MASS J07344924+2818158 = Gaia DR3 872560023937529984 | 07 34 49.24 +28 18 15.9 | EA/SB | 13.07 - 13.14 (13.13) | 1.03034 | 2459500.6840 | | 6.2%/5.3%, sec. 0.500; pixel pass; SB2 (Kounkel+2021) |

With caveats (solid eclipses, but which star or which period is uncertain):

| Name / identifier | RA, Dec (J2000) | Type | Mag range (TESS) | Period (d) | Epoch (BJD) | Discoverer | Evidence and caveat |
|---|---|---|---|---|---|---|---|
| TIC 238891168 = SCR J0241-7719 = Gaia DR3 4633052007552688000 | 02 41 29.55 -77 19 56.1 | EA | 13.79 - 14.36 (14.13) | 152.78 ± 0.05 | 2460172.444 | | two eclipses (BTJD 1673.125, 3172.444), primary and secondary of the Gaia DR3 astrometric orbit (P 152.36 ± 0.45); period from the two together |
| TIC 308186412 = 2MASS J08420090-7113216 = Gaia DR3 5221537843653900544 | 08 42 00.90 -71 13 21.6 | EA | 13.26 - 13.43 | 257.96 (or ×2, ×3) | 2458579.035 | | three eclipses BTJD 1579.035, 2094.972, 2352.917; co-moving star 6.5″ unresolved |
| TIC 10747609 = Gaia DR3 1511212525458984448 | 14 09 32.97 +50 02 55.4 | EA | 13.46 - 13.63 (13.61) | 5.98383 | 2460385.8993 | | brighter co-moving star 2.4″ away may be the eclipsing one |
| TIC 233466569 = LSPM J1524+3942N = Gaia DR3 1388863407004882432 | 15 24 17.98 +39 42 22.2 | EA | 13.28 - 13.35 (13.31) | 0.51392 | 2458977.6215 | | the S component 6.6″ away may be the eclipsing one |
| TIC 446169146 = LP 7-265 = Gaia DR3 1725295239047702016 | 13 26 22.28 +83 14 17.1 | EA (or EW) | 14.56 - 14.59 | 0.80976 (or 0.40488) | 2459421.2170 | | 3% V-shaped dips, phase-coherent over 13 sectors, odd = even; pixel pass; period ambiguous |

TIC 261590581 (LEHPM 5710) would be a revision of an existing VSX entry (type EA, period 4.45725 d, epoch BJD
2458345.2396, range 14.40 - 14.64 TESS) rather than a new one. The three young stars with dips are left out:
VSX has no type for complex periodic variables.

## The short-range pile (1-10 days)

The short-range survey (1-10 day periods) flagged 876 stars above its threshold (`cool/shortpile.py`, best signal
per star):

| What the catalogues say | Stars | Best signal deeper than 1% |
|---|---|---|
| Planet candidate (TOI) or community candidate (CTOI) | 73 | 20 |
| Catalogued eclipsing binary on the star | 110 | 59 |
| Catalogued variable on the star (mostly spotted, flaring stars) | 176 | 28 |
| Catalogued variable next door (within 63″) | 114 | 21 |
| Nothing in the catalogues checked | 403 | 45 |

Most of the 403 uncatalogued stars are spotted stars: their signal is shallower than 1% (358 of 403) and
usually repeats at 1-1.5 days in all of the star's signals, which is the survey's search catching a rotation
pattern. 28 of the 45 deeper ones were looked at by eye: the real ones are in the kept table (TIC 435680606,
468353260, 316822151, 7322727, 233466569, 446169146, the young stars with dips), the rest are blends, off-target
signals or artefacts; one turned out to be a known complex periodic variable (TIC 294328887, Bouma+2024). The
other 17 are weak (MES 7-11, just above the threshold) and were not looked at by eye.

99 uncatalogued stars have a shallow (under 1%), strong (MES ≥ 10) signal that is not a TOI. These are the
survey's own planet-candidate territory, not this note's: 10 of the cleanest were folded here (sheets n-o) and
gave 2 blends or off-target signals, 2 shallow transit-like signals (TIC 152364094, 2.152 d, 0.35%;
TIC 137609439, 4.294 d, 0.2%), and spots or noise for the rest. None of them is claimed as anything.

## Unexplained

Nothing among the 142 stars looked at by eye is left unexplained (the automatically labelled stars were not
judged one by one; see "What was not checked"). Two cases are odd but have likely ordinary explanations:

- **TIC 146725602 and 146725601**, a wide pair 31″ apart moving together: both show the same 1.768-day pattern of
  sharp dips in some sectors. One of the two is a young star with dips (146725602 is in Capistrant+2022's
  dipper list) and TESS's large pixels share its light between the two apertures.
- **TIC 457801578**: one sector of ~15%, ~12-hour dips. VSX (from Gaia) calls it a δ Scuti pulsator, which a red
  dwarf cannot be, so the Gaia light probably belongs to something else. Not pixel-checked.

## What was not checked

- **Gaia DR3 rotation periods (`I/358/vrot`)**: the bulk query failed (empty answer handled badly), so not checked.
- **CPV and dipper lists without VizieR tables** (Bouma+2024, Zhan+2019, Guenther+2022, Tajiri+2020): checked only
  through SIMBAD's reference lists, and only for the kept objects. A star in one of those papers that SIMBAD does
  not link to the paper would be missed.
- **The literature beyond SIMBAD**: no ADS or arXiv full-text search (one paper, Laos+2022, was read for TIC
  308186412). A recent paper that SIMBAD has not ingested could already describe any of the "new" objects.
- **Spectra and radial velocities**: none used, so no masses; companion sizes are lower limits from the depths.
- **Pixel checks** were run only for the kept objects and for candidates that might have been kept: 285903673
  could not be checked (TESScut answered HTTP 502), and 455055150, 229479346, 434171112, 166760940, 239366978, 185247277, 302607052, 152364094, 137609439,
  348905376, 457801578, 233684011, 132529464 were labelled by eye without one.
- **140 light-curve-folded stars** were labelled automatically (catalogue + rules), not by eye; the 18
  deepest uncatalogued folds (3-100%) were eyeballed late (sheets p-r; one new pair, TIC 446169146, came out of
  it), but 9 more uncatalogued folds of 1.5-3% with SNR > 15 were not looked at by eye.
- **974 stars got the catalogue cross-match only.** Most are short-range signals on spotted stars.
- **Process limit:** for a few minutes up to three worker processes ran at once (two light-curve jobs plus a
  catalogue query) instead of the two allowed.

## How to reproduce

All scripts are in `spikes/mdwarf_stack/cool/` (run with `python -I` from `spikes/mdwarf_stack/`); outputs go
to `outputs/planets_spike/cool/` (not committed). Light curves are rebuilt from the survey's cached MAST products.

```
python -I cool/stars.py                     # the 1,256 stars and their signals
python -I cool/triage.py                    # first-pass class per star
python -I cool/catpass.py                   # VSX, TESS EB lists, ASAS-SN/ZTF/ATLAS, SIMBAD, youth lists
python -I cool/catgaia.py                   # Gaia DR3 tables via the VizieR mirrors
python -I cool/catextra.py                  # Capistrant+2022 dippers
python -I cool/catsum.py                    # -> cat_summary.csv
python -I cool/ebscan.py --file <list>      # fold / period work per star; cool/ebtable.py collects them
python -I cool/sheet.py OUT.png TIC ...     # contact sheets for eyeballing
python -I cool/deep.py TIC                  # trapezoid fits + pixel check
python -I cool/pixev.py TIC --times ... --width-h W   # pixel check at given times (cool/pixjobs.py: job files in cool/jobs/)
python -I cool/gaiakept.py TIC ...          # per-object Gaia DR3 look-up (both mirrors)
python -I cool/gaiaorbit.py                 # TIC 238891168: eclipses vs the Gaia orbit
python -I cool/shortpile.py                 # short-range pile summary
python -I cool/labels.py                    # -> labels/cool_objects_v1.csv
python -I cool/vsxfields.py TIC ...         # VSX fields
python -I cool/phone_eb.py / phone_cpv.py   # images
```
