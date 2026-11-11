"""Unit tests for the vetting helpers: catalogue summaries and the pixel-level checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from oddball.common.http import ANON_USER_AGENT
from oddball.gallery import pixels as P
from oddball.gallery import research as R


def test_user_agent_is_anonymous():
    ua = ANON_USER_AGENT.lower()
    for bad in ("github", "@", "oddball", "antimatter", "http"):
        assert bad not in ua


def test_ms_height_on_and_above_sequence():
    assert abs(R.ms_height(0.82, 4.65)) < 1e-9  # the Sun sits on the sequence
    assert R.ms_height(0.82, 2.65) > 1.9  # 2 mag brighter = above (young or evolved)
    assert np.isnan(R.ms_height(np.nan, 4.0))


def _gaia_rows():
    return pd.DataFrame({
        "Source": [1, 2, 3, 4], "Gmag": [12.0, 12.5, 17.5, 11.0], "BP-RP": [0.8, 1.0, 1.5, 0.5],
        "Plx": [5.0, 1.0, 0.5, 2.0], "e_Plx": [0.05, 0.1, 0.3, 0.05], "RUWE": [1.0, 1.1, 1.0, 3.0],
        "VarFlag": ["NOT_AVAILABLE", "VARIABLE", "NOT_AVAILABLE", "NOT_AVAILABLE"],
        "sep_arcsec": [0.3, 15.0, 30.0, 80.0]})  # fmt: skip


def test_gaia_summary_picks_target_and_counts_blends():
    s = R.gaia_summary(_gaia_rows(), tmag=11.5)
    assert s["gaia_match"] and s["gaia_source"] == "1"
    assert abs(s["dist_pc"] - 200.0) < 1e-6
    assert s["n_nb_1px"] == 1  # source 2 at 15" (dG 0.5); source 3 is dG 5.5
    assert s["n_nb_2px"] == 1
    assert abs(s["brightest_nb_2px_dG"] - 0.5) < 1e-9
    assert s["brightest_nb_2px_varflag"] == "VARIABLE"


def test_gaia_summary_no_close_source():
    g = _gaia_rows()
    g["sep_arcsec"] = g.sep_arcsec + 10
    assert R.gaia_summary(g, 11.5) == {"gaia_match": False}


def test_wise_excess_flags_and_upper_limits():
    base = {"table": "x", "sep_arcsec": 0.5, "ccf": "0000", "ex": "0", "qph": "AAAA",
            "W1mag": 10.0, "W2mag": 9.5, "W3mag": 8.0, "W4mag": 6.0, "e_W1mag": 0.02,
            "e_W2mag": 0.02, "e_W3mag": 0.05, "e_W4mag": 0.1}  # fmt: skip
    s = R.wise_summary(pd.DataFrame([base]))
    assert s["ir_excess"] == "W1-W2,W2-W3,W1-W4"
    lim = dict(base, e_W3mag=np.nan, e_W4mag=np.nan, W2mag=9.9)  # W3/W4 upper limits only
    s = R.wise_summary(pd.DataFrame([lim]))
    assert s["ir_excess"] == ""


def _cube(src_xy=(5.0, 5.0), dip_xy=None, depth=0.2, n=600, seed=0):
    """Synthetic 11x11 cube: a Gaussian star at src_xy; a dip on a star at dip_xy."""
    rng = np.random.default_rng(seed)
    t = 2000.0 + np.arange(n) * P.BIN_D
    yy, xx = np.mgrid[0:11, 0:11]

    def psf(x0, y0, amp):
        return amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * 0.7**2))

    star = psf(*src_xy, 1000.0)
    f = np.repeat(star[None], n, axis=0).astype(float)
    ev = np.abs(t - t[n // 2]) < 0.1
    if dip_xy is not None:
        nb = psf(*dip_xy, 1000.0)
        f += nb[None]
        f[ev] -= depth * nb[None]
    f += rng.normal(0, 2.0, f.shape)
    hdr = ("SIMPLE  =                    T" + " " * 49)  # placeholder, WCS not used below
    return P.Cube(t, f.astype(np.float32), np.zeros(n, np.float32), hdr, 1, 11), t[n // 2]


def _analyse_px(c, t_ev, tx, ty, monkeypatch):
    class W:
        def world_to_pixel_values(self, ra, dec):
            return tx, ty

    monkeypatch.setattr(P, "wcs_of", lambda _c: W())
    return P.analyse(c, 0.0, 0.0, t_ev, 0.2, "dip")


def test_difference_image_on_target(monkeypatch):
    c, t_ev = _cube(src_xy=(5, 5), dip_xy=(5, 5))
    r = _analyse_px(c, t_ev, 5.0, 5.0, monkeypatch)
    assert r["diff_peak_snr"] > 20
    assert r["diff_offset_px"] < 0.3
    assert "on target" in P.verdict(r)


def test_difference_image_flags_neighbour(monkeypatch):
    c, t_ev = _cube(src_xy=(5, 5), dip_xy=(8, 5))  # the dip is on a star 3 px away
    r = _analyse_px(c, t_ev, 5.0, 5.0, monkeypatch)
    assert r["diff_offset_px"] > 2.5
    assert "OFF target" in P.verdict(r)


def test_signal_locus_follows_variable_star(monkeypatch):
    rng = np.random.default_rng(1)
    c, _ = _cube(src_xy=(5, 5), dip_xy=None)
    yy, xx = np.mgrid[0:11, 0:11]
    var = 1000.0 * np.exp(-((xx - 2) ** 2 + (yy - 7) ** 2) / (2 * 0.7**2))
    sig = 0.05 * np.sin(2 * np.pi * (c.time - c.time[0]) / 1.3)
    c.flux += (var[None] * (1 + sig[:, None, None])).astype(np.float32)
    c.flux += rng.normal(0, 1, c.flux.shape).astype(np.float32)

    class W:
        def world_to_pixel_values(self, ra, dec):
            return 5.0, 5.0

    monkeypatch.setattr(P, "wcs_of", lambda _c: W())
    # the template is the (diluted) variability seen in the target's aperture
    r = P.analyse(c, 0.0, 0.0, c.time[300], 0.2, "dip", c.time, 0.01 * sig)
    assert r["locus_peak_snr"] > 10
    assert abs(r["locus_cx"] - 2) < 0.5 and abs(r["locus_cy"] - 7) < 0.5
    assert "variability OFF target" in P.verdict(r)


def test_triage_rows_need_evidence(tmp_path):
    import importlib.util

    from oddball.common.paths import REPO_ROOT

    spec = importlib.util.spec_from_file_location("ta", REPO_ROOT / "scripts" / "triage_append.py")
    ta = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ta)
    ok = {"tic": 1, "cls": "EB", "confidence": "high", "reason": "flat eclipses P=2 d",
          "catalogue_evidence": "Gaia RUWE 1.0", "already_known": "no", "confirm_or_kill": "RVs",
          "research": "full", "vetter": "t", "card": "x.png"}  # fmt: skip
    assert ta.validate(ok) == []
    assert ta.validate({**ok, "reason": ""})
    assert ta.validate({**ok, "cls": "weird"})
    un = {**ok, "cls": "unexplained", "ruled_out": "EB: no period"}
    assert any("systematic" in e for e in ta.validate(un))
    un["ruled_out"] = "EB: no period; systematic: in SAP, not at an edge; blend: on target"
    assert ta.validate(un) == []
    p = tmp_path / "t.csv"
    ta.append(p, ok)
    ta.append(p, {**ok, "tic": 2, "reason": 'comma, "quote"'})
    df = pd.read_csv(p)
    assert list(df.tic) == [1, 2] and df.reason.iloc[1] == 'comma, "quote"'
