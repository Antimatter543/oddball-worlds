import numpy as np
import pandas as pd
import pytest

from oddball.common.splits import known_positive_tics, split_of
from oddball.inject import KINDS, apply_model, binned_model, draw_null_event, draw_params
from oddball.inject.build import select_hosts, stratified_sample
from oddball.inject.profiles import fine_grid, model_from_params
from oddball.inject.sampling import DEPTH_RANGE, DURATION_RANGE, pick_time
from oddball.score.classical import dip_shape

BIN = 30.0 / 1440.0


def host_grid(gap=(2186.0, 2187.6)):
    t = (104363 + np.arange(1245) + 0.5) * BIN
    valid = ~((t > gap[0]) & (t < gap[1]))
    return t, valid


def _fine(p, t):
    tf = fine_grid(t[0] - 1, t[-1] + 1)
    return tf, model_from_params(p, tf)


@pytest.mark.parametrize("kind", KINDS)
def test_profile_peak_is_depth_and_baseline_is_one(kind):
    t, valid = host_grid()
    for seed in range(20):
        p = draw_params(kind, np.random.default_rng(seed), t, valid)
        assert DEPTH_RANGE[0] <= p["depth"] <= DEPTH_RANGE[1]
        assert DURATION_RANGE[0] <= p["duration_d"] <= DURATION_RANGE[1]
        tf, m = _fine(p, t)
        if kind == "outburst":
            assert m.max() == pytest.approx(1 + p["depth"], rel=1e-6)
            assert m.min() >= 1.0 - 1e-12
        else:
            assert m.min() == pytest.approx(1 - p["depth"], rel=1e-6)
            assert m.max() <= 1.0 + 1e-12
        # t0 is where the deviation peaks
        assert abs(m[np.argmin(np.abs(tf - p["t0_btjd"]))] - 1) == pytest.approx(p["depth"],
                                                                                 rel=1e-3)
        # far from a single-component event the model is flat
        far = {"single_dip": 3.0, "exocomet": 12.0, "outburst": 12.0}.get(kind)
        if far:
            away = np.abs(tf - p["t0_btjd"]) > far * p["duration_d"]
            if away.any():
                assert np.abs(m[away] - 1).max() < 1e-3 * p["depth"]


def _half_depth_sides(m, i):
    d = 1 - m[i]
    below = (1 - m) >= d / 2
    left = i
    while left > 0 and below[left - 1]:
        left -= 1
    right = i
    while right < len(m) - 1 and below[right + 1]:
        right += 1
    return i - left + 0.5, right - i + 0.5


@pytest.mark.parametrize("dur_h", [4.0, 8.0, 24.0])
def test_exocomet_asymmetry_survives_30min_binning(dur_h):
    dur = dur_h / 24
    p = {"kind": "exocomet", "depth": 0.01, "duration_d": dur, "sign": -1, "norm": 1.0,
         "components": [{"shape": "fred", "tc": 2180.0, "rise": 0.05 * dur,
                         "tau": 0.95 * dur / np.log(2), "rise_kind": "linear"}]}
    t = (104363 + np.arange(1245) + 0.5) * BIN
    m = binned_model(p, t, BIN)
    i = int(np.argmin(m))
    ingress, egress = _half_depth_sides(m, i)
    assert egress / ingress >= 3.0, (ingress, egress)
    # binned depth for a sharp event stays close to the true depth
    assert 1 - m[i] > 0.8 * p["depth"]
    # the scorer's asymmetry statistic sees it too (noiseless residual)
    r = (m - 1.0)[None, :]
    w = max(4, int(dur / BIN))
    _, _, asym, _ = dip_shape(r, np.array([i - w // 2]), np.array([w]), BIN)
    assert asym[0] > 0.4


def test_single_dip_binned_is_symmetric():
    p = {"kind": "single_dip", "depth": 0.01, "duration_d": 0.5, "sign": -1, "norm": 1.0,
         "components": [{"shape": "gauss", "tc": 2180.0 + 0.5 * BIN + 104363 * 0, "fwhm": 0.5}]}
    t = (104363 + np.arange(1245) + 0.5) * BIN
    p["components"][0]["tc"] = float(t[300])
    m = binned_model(p, t, BIN)
    ingress, egress = _half_depth_sides(m, int(np.argmin(m)))
    assert ingress == egress


def test_sap_depth_diluted_by_crowdsap():
    t = (104363 + np.arange(400) + 0.5) * BIN
    p = {"kind": "single_dip", "depth": 0.02, "duration_d": 0.5, "sign": -1, "norm": 1.0,
         "components": [{"shape": "gauss", "tc": float(t[200]), "fwhm": 0.5}]}
    flux = np.ones_like(t)
    f, s, m = apply_model(flux, flux, t, p, BIN, crowdsap=0.25)
    assert 1 - f.min() == pytest.approx(1 - m.min(), rel=1e-5)
    assert 1 - s.min() == pytest.approx(0.25 * (1 - m.min()), rel=1e-4)
    f2, s2, _ = apply_model(flux, flux, t, p, BIN, crowdsap=np.nan)  # missing -> undiluted
    assert np.allclose(f2, s2)
    nanflux = flux.copy()
    nanflux[10] = np.nan
    f3, _, _ = apply_model(nanflux, flux, t, p, BIN, crowdsap=1.0)
    assert np.isnan(f3[10])


def test_pick_time_avoids_gaps():
    t, valid = host_grid()
    rng = np.random.default_rng(0)
    for _ in range(200):
        tc = pick_time(rng, t, valid, -0.25, 0.25)
        assert valid[np.argmin(np.abs(t - tc))]
        w = (t >= tc - 0.25) & (t <= tc + 0.25)
        assert valid[w].mean() >= 0.75


def test_null_event_identical_and_aligned_across_hosts():
    t, valid = host_grid()
    cover = valid.astype(float)
    p = draw_null_event("null_ccd", np.random.default_rng(3), t, cover, 0.01)
    assert p["is_null"]
    # three hosts on one CCD whose grids start one bin apart: the same absolute-time model
    ms = []
    for off in (0, 1, 2):
        th = t[off:off + 1200]
        ms.append(binned_model(p, th, BIN))
    for off, m in zip((1, 2), ms[1:], strict=True):
        np.testing.assert_allclose(m[: 1200 - 2], ms[0][off: off + 1200 - 2])
    assert ms[0].min() < 1 - 0.5 * p["depth"]
    pd_ = draw_null_event("null_dump", np.random.default_rng(4), t, cover, 0.005,
                          dump_times=np.array([2179.48, 2193.48]))
    assert pd_["dump_btjd"] in (2179.48, 2193.48)
    assert 0 <= pd_["t0_btjd"] - pd_["dump_btjd"] < 0.05  # dip starts at the dump


def _fake_meta(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    tics = rng.choice(np.arange(10_000_000, 99_000_000), size=n, replace=False)
    tics[:5] = sorted(known_positive_tics())[:5]
    return pd.DataFrame({
        "tic": tics, "tmag": rng.uniform(8, 15, n), "camera": rng.integers(1, 5, n),
        "ccd": rng.integers(1, 5, n), "crowdsap": rng.uniform(0.5, 1, n),
        "shard": rng.integers(0, 3, n),
    })


def test_host_selection_split_pure_no_known_and_null_groups_per_ccd():
    meta = _fake_meta()
    astro, nulls = select_hosts(meta, "val", 200, 5, 10, np.random.default_rng(1))
    known = known_positive_tics()
    all_hosts = pd.concat([astro, *nulls])
    assert all(split_of(t) == "val" for t in all_hosts.tic)
    assert not all_hosts.tic.isin(known).any()
    assert all_hosts.tic.is_unique  # astro and null hosts disjoint
    assert len(astro) == 200
    for g in nulls:
        assert g.camera.nunique() == 1 and g.ccd.nunique() == 1
        assert len(g) == 10
    assert len({(g.camera.iloc[0], g.ccd.iloc[0]) for g in nulls}) == 5


def test_stratified_sample_is_proportional():
    df = pd.DataFrame({"tic": np.arange(1000), "stratum": np.repeat([0, 1, 2], [600, 300, 100])})
    s = stratified_sample(df, 100, np.random.default_rng(0))
    assert s.stratum.value_counts().to_dict() == {0: 60, 1: 30, 2: 10}
