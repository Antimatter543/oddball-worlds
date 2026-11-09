"""Cross-check outburst candidates against SIMBAD and VSX (anonymous; CfA/Harvard mirrors).

    uv run python -I scripts/rubin_crosscheck.py [--csv outputs/rubin/candidates.csv]

Adds: simbad_id, simbad_otype, simbad_sep, simbad_n (within --radius), vsx_name, vsx_type,
vsx_range, vsx_sep. Writes <csv stem>_crosscheck.csv next to the input.

Known-variable rejection: a VSX type that is not a young-star class (e.g. Mira, LPV, EA,
RRAB, CV/NOVA) marks the candidate ``known_nonyso_variable`` (artefact of the YSO catalogue
or a chance match).
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord

SIMBAD_SERVER = os.environ.get("ODDBALL_SIMBAD_SERVER", "simbad.harvard.edu")
VIZIER_SERVER = os.environ.get("ODDBALL_VIZIER_SERVER", "vizier.cfa.harvard.edu")
# VSX types for young stars (GCVS/VSX nomenclature) - anything else is a non-YSO variable
YSO_VSX = re.compile(r"^(?:IN|INA|INB|INS|INSA|INSB|INT|INTA|INTB|IS|ISA|ISB|IT|FUOR|EXOR|"
                     r"UXOR|YSO|TTS|CTTS|WTTS|DIP|ROT|BY|FL|UV|UVN|L|S|VAR|MISC)$")


def yso_like_vsx(t: str) -> bool:
    parts = [p for p in re.split(r"[|+/:,() ]+", t.upper()) if p]
    return bool(parts) and all(YSO_VSX.match(p) for p in parts)


def simbad_check(c: SkyCoord, radius: float):
    from astroquery.simbad import Simbad, conf

    conf.server = SIMBAD_SERVER
    s = Simbad()
    s.add_votable_fields("otype")
    try:
        t = s.query_region(c, radius=radius * u.arcsec)
    except Exception as e:  # noqa: BLE001 - log and continue
        return {"simbad_error": f"{type(e).__name__}: {str(e)[:80]}"}
    if t is None or len(t) == 0:
        return {"simbad_id": "", "simbad_otype": "", "simbad_sep": float("nan"), "simbad_n": 0}
    df = t.to_pandas()
    cols = {c.lower(): c for c in df.columns}
    ra, dec = df[cols["ra"]].astype(float), df[cols["dec"]].astype(float)
    sep = c.separation(SkyCoord(ra.to_numpy() * u.deg, dec.to_numpy() * u.deg)).arcsec
    i = int(sep.argmin())
    return {"simbad_id": str(df[cols["main_id"]].iloc[i]),
            "simbad_otype": str(df[cols["otype"]].iloc[i]), "simbad_sep": float(sep[i]),
            "simbad_n": len(df)}


def vsx_check(c: SkyCoord, radius: float):
    from astroquery.vizier import Vizier

    v = Vizier(columns=["Name", "Type", "max", "min", "Period", "RAJ2000", "DEJ2000"],
               vizier_server=VIZIER_SERVER, timeout=120)
    try:
        tl = v.query_region(c, radius=radius * u.arcsec, catalog="B/vsx/vsx")
    except Exception as e:  # noqa: BLE001 - log and continue
        return {"vsx_error": f"{type(e).__name__}: {str(e)[:80]}"}
    if not tl:
        return {"vsx_name": "", "vsx_type": "", "vsx_range": "", "vsx_sep": float("nan")}
    t = tl[0].to_pandas()
    sep = c.separation(SkyCoord(t["RAJ2000"].to_numpy(float) * u.deg,
                                t["DEJ2000"].to_numpy(float) * u.deg)).arcsec
    i = int(sep.argmin())
    r = t.iloc[i]
    return {"vsx_name": str(r["Name"]), "vsx_type": str(r["Type"]),
            "vsx_range": f"{r['max']}-{r['min']}", "vsx_sep": float(sep[i])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/rubin/candidates.csv")
    ap.add_argument("--radius", type=float, default=3.0, help="arcsec")
    args = ap.parse_args()
    path = Path(args.csv)
    cand = pd.read_csv(path)
    rows = []
    for r in cand.itertuples():
        c = SkyCoord(r.meanra * u.deg, r.meandec * u.deg)
        d = {"oid": r.oid}
        d.update(simbad_check(c, args.radius))
        d.update(vsx_check(c, args.radius))
        t = d.get("vsx_type") or ""
        d["known_nonyso_variable"] = bool(t) and not yso_like_vsx(t)
        rows.append(d)
        print(r.name, d, flush=True)
        time.sleep(0.5)
    out = cand.merge(pd.DataFrame(rows), on="oid", how="left")
    out_path = path.with_name(path.stem + "_crosscheck.csv")
    out.to_csv(out_path, index=False)
    print("->", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
