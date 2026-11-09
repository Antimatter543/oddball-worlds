"""Tests for sky tiling, cross-match and light-curve assembly (no network)."""

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from oddball.rubin.crossmatch import make_tiles, match
from oddball.rubin.lightcurve import assemble, flux_to_mag


def test_tiles_cover_every_point():
    rng = np.random.default_rng(0)
    ra = rng.uniform(0, 360, 3000)
    dec = np.degrees(np.arcsin(rng.uniform(-1, np.sin(np.radians(30)), 3000)))
    dec[:5] = [-89.5, -88.0, 0.0, 29.9, -45.0]
    t = make_tiles(ra, dec, cell_deg=2.0)
    assert t["n_cat"].sum() == len(ra)
    pts = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    cen = SkyCoord(ra=t["ra_c"].to_numpy() * u.deg, dec=t["dec_c"].to_numpy() * u.deg)
    _, sep, _ = pts.match_to_catalog_sky(cen)
    # every point lies inside the nearest tile's cone (nearest centre is at most as far as
    # its own cell's centre)
    assert np.all(sep.deg <= t["radius_deg"].iloc[0])


def test_tiles_respect_dec_max():
    t = make_tiles([10.0, 20.0], [40.0, -10.0], cell_deg=2.0, dec_max=35.0)
    assert len(t) == 1 and t["n_cat"].iloc[0] == 1


def test_match_radius_and_crowding():
    cat = pd.DataFrame({"cat": ["a", "a", "b"], "cat_id": ["s1", "s2", "s3"],
                        "ra": [100.0, 100.1, 200.0], "dec": [-30.0, -30.0, 10.0]})
    d = 1 / 3600
    objs = pd.DataFrame({
        "oid": [1, 2, 3, 4],
        # obj1 0.5" from s1, obj2 2" from s1 (crowding only), obj3 far, obj4 0.3" from s3
        "meanra": [100.0, 100.0, 150.0, 200.0],
        "meandec": [-30.0 + 0.5 * d, -30.0 - 2.0 * d, 0.0, 10.0 + 0.3 * d],
    })
    m = match(cat, objs, radius_arcsec=1.0, crowd_arcsec=2.5)
    assert set(m["cat_id"]) == {"s1", "s3"}
    s1 = m[m["cat_id"] == "s1"].iloc[0]
    assert s1["oid"] == 1
    assert s1["sep_arcsec"] == pytest.approx(0.5, abs=0.01)
    assert s1["n_rubin_near"] == 2  # obj2 at 2" counts as a possible blend
    assert m[m["cat_id"] == "s3"].iloc[0]["n_rubin_near"] == 1


def test_match_empty_inputs():
    m = match(pd.DataFrame(columns=["cat", "cat_id", "ra", "dec"]),
              pd.DataFrame({"oid": [1], "meanra": [0.0], "meandec": [0.0]}))
    assert len(m) == 0


def test_flux_to_mag_zero_point_and_negative_flux():
    mag, err = flux_to_mag([3631e9, 1000.0, -5.0], [0.0, 100.0, 1.0])
    assert mag[0] == pytest.approx(0.0, abs=1e-3)  # 3631 Jy = AB 0
    assert mag[1] == pytest.approx(31.4 - 7.5)
    assert err[1] == pytest.approx(0.10857, rel=1e-3)
    assert np.isnan(mag[2])


def test_assemble_prefers_forced_flux_and_attaches_flags():
    fp = [{"visit": 1, "mjd": 61000.1, "band": 2, "scienceFlux": 10000.0,
           "scienceFluxErr": 100.0, "psfFlux": 50.0, "psfFluxErr": 100.0},
          {"visit": 2, "mjd": 61003.1, "band": 1, "scienceFlux": 20000.0,
           "scienceFluxErr": 100.0, "psfFlux": 9000.0, "psfFluxErr": 100.0}]
    det = [{"visit": 2, "mjd": 61003.1, "band": 1, "scienceFlux": 19990.0,
            "scienceFluxErr": 100.0, "psfFlux": 8990.0, "psfFluxErr": 100.0,
            "pixelFlags_saturatedCenter": True, "reliability": 0.9, "ssObjectId": 0},
           {"visit": 3, "mjd": 61005.1, "band": 3, "scienceFlux": 5000.0,
            "scienceFluxErr": 100.0, "psfFlux": 10.0, "psfFluxErr": 100.0, "ssObjectId": 0},
           {"visit": 4, "mjd": 61006.1, "band": 3, "scienceFlux": 5000.0,
            "scienceFluxErr": 100.0, "psfFlux": 10.0, "psfFluxErr": 100.0,
            "ssObjectId": 123}]
    lc = assemble(det, fp)
    assert list(lc["visit"]) == [1, 2, 3]  # solar-system-associated visit 4 dropped
    v2 = lc[lc["visit"] == 2].iloc[0]
    assert v2["src"] == "fp" and v2["sci_flux"] == 20000.0  # forced flux kept
    assert v2["is_det"] and v2["bad"] and v2["saturated"] and v2["reliability"] == 0.9
    assert v2["band"] == "g"
    assert lc[lc["visit"] == 3].iloc[0]["src"] == "det"


def test_assemble_empty():
    assert assemble([], []).empty
