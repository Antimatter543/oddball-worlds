"""Build the known-positive list (oddballs we must recover blind) with Sector>=27 coverage.

    uv run python -I scripts/build_known_positives.py

Sources:
* a curated list below (each entry carries its reference; TICs were resolved with the MAST TIC
  name resolver at <0.1" and cross-checked against the papers where they quote a TIC);
* every dipper (``VarF == 'D'``) in Capistrant et al. 2022, ApJS 263, 14 (VizieR J/ApJS/263/14).

Coverage comes from MAST: the HLSP light-curve products actually archived for each TIC
(provenance TESS-SPOC and QLP), restricted to Sector >= 27. This is "has a light curve we can
ingest", which is stricter than "fell on silicon" (faint stars often have no FFI light curve).

Writes ``docs/known_positives.csv`` (committed) and ``<data>/known_positives.csv``, and prints
per-sector counts so the first ingest sector can be chosen by known-positive density.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from oddball.common.paths import REPO_ROOT, data_root

warnings.filterwarnings("ignore")
MIN_SECTOR = 27
CLASSES = ("exocomet", "disintegrating", "dipper/young-star", "boyajian-like", "other-anomaly")

NK25 = "Norazman & Kennedy 2025, arXiv:2508.04673"
R18 = "Rappaport et al. 2018, MNRAS 474, 1453"
CAP22 = "Capistrant et al. 2022, ApJS 263, 14 (VizieR J/ApJS/263/14)"

# (name, tic, class, reference, note)
CURATED: list[tuple[str, int, str, str, str]] = [
    # --- exocomets
    ("beta Pic", 270577175, "exocomet", f"Zieba et al. 2019, A&A 625, L13; {NK25}",
     "prototype; TESS dips in S4-S7 (primary mission)"),
    ("TIC 280832588", 280832588, "exocomet", NK25, "new candidate, F2V; tentative (S1)"),
    ("TYC 8705-361-1", 73149665, "exocomet", NK25, "new candidate, G-type MS (S12)"),
    ("TIC 143152957", 143152957, "exocomet", NK25, "new candidate, G-type MS (S7)"),
    ("TIC 110969638", 110969638, "exocomet", NK25, "new candidate, giant/supergiant (S7)"),
    ("TIC 229790952", 229790952, "exocomet", NK25, "new candidate, giant (S15)"),
    ("HD 172555", 464405850, "exocomet", "Kiefer et al. 2014, A&A 561, L10; " + NK25,
     "spectroscopic exocomets; no TESS transit seen in S66 (NK25)"),
    ("KIC 3542116", 122519163, "exocomet", R18, "Kepler exocomet transits"),
    ("KIC 11084727", 26584368, "exocomet", R18, "Kepler exocomet transit"),
    ("KIC 8027456 (HD 182952)", 159719806, "exocomet", "Kennedy et al. 2019, MNRAS 482, 5587",
     "Kepler exocomet transit"),
    # --- disintegrating planets
    ("BD+05 4868 A", 466376085, "disintegrating", "Hon et al. 2025, arXiv:2501.05431",
     "P=1.27 d; TESS S55, S82"),
    ("KIC 12557548", 417679385, "disintegrating", "Rappaport et al. 2012, ApJ 752, 1",
     "Kepler-1520 b; T=15.0, likely below FFI-LC limits"),
    ("K2-22", 363445338, "disintegrating", "Sanchis-Ojeda et al. 2015, ApJ 812, 112", "T=14.0"),
    ("KOI-2700", 272938824, "disintegrating", "Rappaport et al. 2014, ApJ 784, 40", "T=14.6"),
    ("WD 1145+017", 902906874, "disintegrating", "Vanderburg et al. 2015, Nature 526, 546",
     "white dwarf debris; T=17.3, likely no FFI LC"),
    # --- Boyajian-like (aperiodic deep dimming of non-young stars)
    ("KIC 8462852 (Boyajian's star)", 185336364, "boyajian-like",
     "Boyajian et al. 2016, MNRAS 457, 3988", ""),
    ("ASASSN-21qj", 182582608, "boyajian-like", "Kenworthy et al. 2023, Nature 622, 251",
     "dust from a planetary collision"),
    ("TIC 400799224", 400799224, "boyajian-like", "Powell et al. 2021, AJ 162, 299",
     "periodic-ish deep dust occultations"),
    # --- dippers / young stars with disks (beyond the Capistrant table)
    ("EPIC 204278916", 9735122, "dipper/young-star", "Scaringi et al. 2016, MNRAS 463, 2265",
     "Upper Sco; ~25 d of irregular dimming"),
    ("PDS 110", 4206238, "dipper/young-star", "Osborn et al. 2017, MNRAS 471, 740",
     "~25-d-long eclipses"),
    ("RZ Psc", 15693497, "dipper/young-star", "Kennedy et al. 2017, RSOS 4, 160652", "UXor"),
    ("AA Tau", 268510757, "dipper/young-star", "Bouvier et al. 1999; Bouvier et al. 2013",
     "prototype dipper, faded since 2011"),
    # --- other oddballs
    ("EPIC 204376071", 9855014, "other-anomaly", "Rappaport et al. 2019, MNRAS 485, 2681",
     "single 80%-deep 1-day occultation"),
    ("HD 139139", 70652803, "other-anomaly", "Rappaport et al. 2019, MNRAS 488, 2455",
     "'random transiter': 28 aperiodic transit-like dips"),
    ("V1400 Cen (J1407)", 179367009, "other-anomaly", "Mamajek et al. 2012, AJ 143, 72",
     "ring-system eclipse (2007)"),
]  # fmt: skip


def capistrant_dippers() -> pd.DataFrame:
    from astroquery.vizier import Vizier

    last = None
    for server in ("vizier.cfa.harvard.edu", "vizier.cds.unistra.fr"):
        try:
            v = Vizier(columns=["**"], row_limit=-1, timeout=120)
            v.VIZIER_SERVER = server
            t = v.get_catalogs("J/ApJS/263/14")[0]
            break
        except Exception as e:  # noqa: BLE001 - try the next mirror
            last = e
    else:
        raise RuntimeError(f"VizieR J/ApJS/263/14 unavailable: {last}")
    df = t.to_pandas()
    df = df[df["VarF"].astype(str).str.strip() == "D"]
    out = pd.DataFrame({
        # VizieR's SimbadName column is a link field, not a usable name: use the TIC.
        "name": [f"TIC {tic}" for tic in df.TIC],
        "tic": df.TIC.astype("int64"),
        "class": "dipper/young-star",
        "reference": CAP22,
        "note": [
            ("known dipper" if str(k).strip() == "Y" else "new dipper candidate")
            + (f"; age {a:.0f} Myr" if pd.notna(a) else "")
            for k, a in zip(df.Known, df.Age, strict=True)
        ],
    })  # fmt: skip
    return out


def hlsp_sectors(tics: list[int], batch: int = 50) -> dict[tuple[int, str], set[int]]:
    from astroquery.mast import Observations

    cov: dict[tuple[int, str], set[int]] = defaultdict(set)
    for i in range(0, len(tics), batch):
        names = [str(t) for t in tics[i : i + batch]]
        obs = Observations.query_criteria(
            target_name=names, provenance_name=["TESS-SPOC", "QLP"], obs_collection="HLSP"
        )
        for r in obs:
            s = int(r["sequence_number"])
            if MIN_SECTOR <= s < 1000:  # >=1000 are multi-sector products (e.g. s1751)
                cov[(int(r["target_name"]), str(r["provenance_name"]))].add(s)
    return cov


def tic_info(tics: list[int], batch: int = 200) -> pd.DataFrame:
    from astroquery.mast import Catalogs

    parts = []
    for i in range(0, len(tics), batch):
        c = Catalogs.query_criteria(catalog="TIC", ID=tics[i : i + batch])
        parts.append(c["ID", "ra", "dec", "Tmag"].to_pandas())
    df = pd.concat(parts)
    df["tic"] = df.ID.astype("int64")
    return df.drop(columns="ID").drop_duplicates("tic")


def _fmt(s: set[int]) -> str:
    return " ".join(str(x) for x in sorted(s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "known_positives.csv"))
    a = ap.parse_args()

    cur = pd.DataFrame(CURATED, columns=["name", "tic", "class", "reference", "note"])
    cap = capistrant_dippers()
    # Curated entries win on duplicates (they carry the more specific reference).
    kp = pd.concat([cur, cap[~cap.tic.isin(cur.tic)]], ignore_index=True)
    dup = cap[cap.tic.isin(cur.tic)]
    for _, r in dup.iterrows():
        kp.loc[kp.tic == r.tic, "note"] += f"; also in {CAP22.split(',')[0]}"
    assert set(kp["class"]) <= set(CLASSES)

    tics = kp.tic.astype(int).tolist()
    info = tic_info(tics)
    kp = kp.merge(info, on="tic", how="left")
    cov = hlsp_sectors(tics)
    kp["sectors_spoc"] = [_fmt(cov.get((t, "TESS-SPOC"), set())) for t in kp.tic]
    kp["sectors_qlp"] = [_fmt(cov.get((t, "QLP"), set())) for t in kp.tic]
    kp["sectors_any"] = [
        _fmt(cov.get((t, "TESS-SPOC"), set()) | cov.get((t, "QLP"), set())) for t in kp.tic
    ]
    kp["ra"] = kp.ra.round(6)
    kp["dec"] = kp.dec.round(6)
    kp["tmag"] = kp.Tmag.round(3)
    cols = ["name", "tic", "class", "ra", "dec", "tmag", "sectors_spoc", "sectors_qlp",
            "sectors_any", "reference", "note"]  # fmt: skip
    kp = kp[cols]

    out = Path(a.out)
    kp.to_csv(out, index=False)
    data_root().mkdir(parents=True, exist_ok=True)
    shutil.copy(out, data_root() / "known_positives.csv")
    print(f"wrote {out} ({len(kp)} rows) and {data_root() / 'known_positives.csv'}")

    print("\nper class: total / with any S>=27 SPOC-or-QLP LC / with S>=27 TESS-SPOC LC")
    for c in CLASSES:
        sub = kp[kp["class"] == c]
        print(f"  {c:18s} {len(sub):4d} {(sub.sectors_any != '').sum():4d} "
              f"{(sub.sectors_spoc != '').sum():4d}")  # fmt: skip

    per = defaultdict(Counter)
    for _, r in kp.iterrows():
        for s in r.sectors_spoc.split():
            per[int(s)][r["class"]] += 1
    rank = sorted(per.items(), key=lambda kv: (-sum(kv[1].values()), -len(kv[1]), kv[0]))
    print("\nTESS-SPOC sectors by known-positive count (top 15):")
    for s, c in rank[:15]:
        print(f"  S{s:3d}: {sum(c.values()):3d}  " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))


if __name__ == "__main__":
    main()
