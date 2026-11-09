"""Sky tiling + KD-tree cross-match between young-star catalogues and Rubin objects.

Rubin objects are pulled per sky tile (~1 deg cones) rather than per star: one ALeRCE query
per tile instead of one per catalogue entry. Tiles with no catalogue entries are skipped.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

MATCH_COLUMNS = ["cat", "cat_id", "cat_ra", "cat_dec", "oid", "sep_arcsec", "n_rubin_near", "rank"]


def tile_ids(ra, dec, cell_deg: float = 1.5) -> np.ndarray:
    """Tile id ("band_col") of each point, consistent with ``make_tiles``."""
    band, col, _, _ = _cells(np.asarray(ra, float) % 360.0, np.asarray(dec, float), cell_deg)
    return np.char.add(np.char.add(band.astype(str), "_"), col.astype(str))


def _cells(ra, dec, cell_deg):
    band = np.floor((dec + 90.0) / cell_deg).astype(int)
    lo = band * cell_deg - 90.0
    hi = lo + cell_deg
    dec_eq = np.where((lo < 0) & (hi > 0), 0.0, np.minimum(np.abs(lo), np.abs(hi)))
    nra = np.maximum(1, np.ceil(360.0 * np.cos(np.radians(dec_eq)) / cell_deg)).astype(int)
    step = 360.0 / nra
    col = np.minimum(np.floor(ra / step).astype(int), nra - 1)
    return band, col, lo, step


def make_tiles(ra, dec, cell_deg: float = 1.5, dec_max: float = 35.0) -> pd.DataFrame:
    """Dec bands of height ``cell_deg``; each band split into ``ceil(360 cos(dec_eq)/cell)``
    RA cells, so no cell edge is wider than ``cell_deg``. Keeps cells holding >= 1 point.

    ``radius_deg`` circumscribes the cell (half-diagonal, +1% slack), so every point in a
    cell lies inside its tile's cone.
    """
    ra = np.asarray(ra, float) % 360.0
    dec = np.asarray(dec, float)
    keep = dec <= dec_max
    ra, dec = ra[keep], dec[keep]
    band, col, lo, step = _cells(ra, dec, cell_deg)
    df = pd.DataFrame({"band": band, "col": col, "dec_c": lo + cell_deg / 2, "step": step})
    g = df.groupby(["band", "col"], as_index=False).agg(
        dec_c=("dec_c", "first"), step=("step", "first"), n_cat=("dec_c", "size"))
    g["ra_c"] = (g["col"] + 0.5) * g["step"]
    g["radius_deg"] = 1.01 * np.hypot(cell_deg / 2, cell_deg / 2)
    g["tile_id"] = g["band"].astype(str) + "_" + g["col"].astype(str)
    return g[["tile_id", "ra_c", "dec_c", "radius_deg", "n_cat"]].reset_index(drop=True)


def match(cat: pd.DataFrame, objs: pd.DataFrame, radius_arcsec: float = 1.0,
          crowd_arcsec: float = 2.5) -> pd.DataFrame:
    """Match catalogue rows (ra, dec, cat, cat_id) to Rubin objects (oid, meanra, meandec).

    One row per (catalogue entry, Rubin object) pair within ``radius_arcsec``, with
    ``sep_arcsec``; ``n_rubin_near`` = Rubin objects within ``crowd_arcsec`` of the catalogue
    star (>1 means a possible blend or split detection); ``rank`` = 0 for the nearest object.
    """
    if len(cat) == 0 or len(objs) == 0:
        return pd.DataFrame(columns=MATCH_COLUMNS)
    c_cat = SkyCoord(ra=cat["ra"].to_numpy() * u.deg, dec=cat["dec"].to_numpy() * u.deg)
    c_obj = SkyCoord(ra=objs["meanra"].to_numpy() * u.deg, dec=objs["meandec"].to_numpy() * u.deg)
    big = max(radius_arcsec, crowd_arcsec) * u.arcsec
    # c_obj.search_around_sky(c_cat, r) -> (indices into c_cat, indices into c_obj, sep, dist)
    i_cat, i_obj, sep, _ = c_obj.search_around_sky(c_cat, big)
    pairs = pd.DataFrame({"i_cat": i_cat, "i_obj": i_obj, "sep_arcsec": sep.to_value(u.arcsec)})
    near = pairs[pairs["sep_arcsec"] <= crowd_arcsec].groupby("i_cat").size()
    pairs = pairs[pairs["sep_arcsec"] <= radius_arcsec].copy()
    pairs["n_rubin_near"] = pairs["i_cat"].map(near).fillna(0).astype(int)
    pairs = pairs.sort_values(["i_cat", "sep_arcsec"])
    pairs["rank"] = pairs.groupby("i_cat").cumcount()
    ic, io = pairs["i_cat"].to_numpy(), pairs["i_obj"].to_numpy()
    return pd.DataFrame({
        "cat": cat["cat"].to_numpy()[ic],
        "cat_id": cat["cat_id"].to_numpy()[ic],
        "cat_ra": cat["ra"].to_numpy()[ic],
        "cat_dec": cat["dec"].to_numpy()[ic],
        "oid": objs["oid"].to_numpy()[io],
        "sep_arcsec": pairs["sep_arcsec"].to_numpy(),
        "n_rubin_near": pairs["n_rubin_near"].to_numpy(),
        "rank": pairs["rank"].to_numpy(),
    })
