"""Assemble a tidy Rubin light curve from ALeRCE detections + forced photometry.

Brightness is taken from ``scienceFlux`` (PSF flux measured on the direct, unsubtracted
image at the object position), not the difference flux ``psfFlux``. That makes Delta-mag
independent of the template epoch (a template taken in outburst would otherwise hide or
invert the event) and of bad subtractions. The difference flux is kept for a
consistency check only.

Times stay in MJD (TAI-ish, as served); convert explicitly if comparing to TESS BTJD.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from oddball.rubin.alerce_api import BAND_MAP

ZP_NJY = 31.4  # AB zero point for fluxes in nJy

# detection-level flags that make that visit's photometry untrustworthy
BAD_DET_FLAGS = [
    "pixelFlags_saturatedCenter", "pixelFlags_badCenter", "pixelFlags_bad",
    "pixelFlags_edge", "pixelFlags_nodataCenter", "pixelFlags_offimage",
    "pixelFlags_interpolatedCenter", "pixelFlags_suspectCenter", "pixelFlags_crCenter",
    "pixelFlags_streakCenter", "pixelFlags_injectedCenter", "psfFlux_flag",
    "psfFlux_flag_edge", "psfFlux_flag_noGoodPixels", "glint_trail",
]
LC_COLUMNS = ["mjd", "band", "visit", "src", "sci_flux", "sci_flux_err", "diff_flux",
              "diff_flux_err", "mag", "magerr", "is_det", "bad", "saturated", "reliability",
              "is_dipole"]


def flux_to_mag(flux, flux_err):
    flux = np.asarray(flux, float)
    flux_err = np.asarray(flux_err, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where(flux > 0, ZP_NJY - 2.5 * np.log10(flux), np.nan)
        magerr = np.where(flux > 0, 1.0857362 * flux_err / flux, np.nan)
    return mag, magerr


def _band_name(b) -> str:
    try:
        return BAND_MAP[int(b)]
    except (TypeError, ValueError, KeyError):
        return str(b)


def _truthy(v) -> bool:
    return bool(v) and v not in ("false", "False", 0)


def assemble(detections: list[dict], forced: list[dict]) -> pd.DataFrame:
    """One row per visit. Forced photometry supplies the flux where available (measured at a
    fixed position every visit); detection-only visits use the detection's science flux.
    Detection flags/reliability are attached to the matching visit either way."""
    rows: dict[object, dict] = {}
    for d in forced or []:
        key = d.get("visit") or ("fp", d.get("measurement_id"))
        rows[key] = {
            "mjd": d.get("mjd"), "band": _band_name(d.get("band")), "visit": d.get("visit"),
            "src": "fp", "sci_flux": d.get("scienceFlux"), "sci_flux_err": d.get("scienceFluxErr"),
            "diff_flux": d.get("psfFlux"), "diff_flux_err": d.get("psfFluxErr"), "is_det": False,
            "bad": False, "saturated": False, "reliability": np.nan, "is_dipole": False,
        }
    for d in detections or []:
        if d.get("ssObjectId") not in (None, 0, "0"):
            continue  # associated with a solar-system object
        key = d.get("visit") or ("det", d.get("measurement_id"))
        bad = any(_truthy(d.get(f)) for f in BAD_DET_FLAGS)
        sat = _truthy(d.get("pixelFlags_saturated")) or _truthy(d.get("pixelFlags_saturatedCenter"))
        rel = d.get("reliability")
        extra = {"is_det": True, "bad": bad, "saturated": sat,
                 "reliability": np.nan if rel is None else float(rel),
                 "is_dipole": _truthy(d.get("isDipole"))}
        if key in rows:
            rows[key].update(extra)
        else:
            rows[key] = {
                "mjd": d.get("mjd"), "band": _band_name(d.get("band")), "visit": d.get("visit"),
                "src": "det", "sci_flux": d.get("scienceFlux"),
                "sci_flux_err": d.get("scienceFluxErr"), "diff_flux": d.get("psfFlux"),
                "diff_flux_err": d.get("psfFluxErr"), **extra,
            }
    if not rows:
        return pd.DataFrame(columns=LC_COLUMNS)
    df = pd.DataFrame(list(rows.values()))
    for c in ("mjd", "sci_flux", "sci_flux_err", "diff_flux", "diff_flux_err"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["mag"], df["magerr"] = flux_to_mag(df["sci_flux"], df["sci_flux_err"])
    return df[LC_COLUMNS].sort_values("mjd").reset_index(drop=True)
