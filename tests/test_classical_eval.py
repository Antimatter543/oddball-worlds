import numpy as np
import pandas as pd

from oddball.common.strata import stratum_id, tmag_bin
from oddball.eval.harness import midrank_pct, rank_within
from oddball.inject.profiles import binned_model
from oddball.score.classical import (
    StratumECDF,
    channel_features,
    common_mode_bins,
    nanquantile_last,
)

BIN = 30.0 / 1440.0


def _batch(n=40, seed=0):
    rng = np.random.default_rng(seed)
    t = (104363 + np.arange(1245) + 0.5) * BIN
    f = 1 + 1e-3 * rng.standard_normal((n, t.size))
    f[:, 560:640] = np.nan  # mid-sector gap
    return t, f, np.full_like(f, 1e-3)


def test_dip_found_with_depth_duration_and_asymmetry():
    t, f, e = _batch()
    p = {"kind": "exocomet", "depth": 0.01, "duration_d": 0.5, "sign": -1, "norm": 1.0,
         "components": [{"shape": "fred", "tc": float(t[300]), "rise": 0.025,
                         "tau": 0.475 / np.log(2), "rise_kind": "linear"}]}
    f[0] *= binned_model(p, t, BIN)
    o = channel_features(f, e, t, BIN)
    snr = o.feats["dip_snr"]
    assert snr[0] > 20 and snr[0] > 3 * np.nanmax(snr[1:])
    assert abs(o.feats["dip_t"][0] - t[300]) < 0.1
    assert 0.007 < o.feats["dip_depth"][0] < 0.012
    assert 0.25 < o.feats["dip_dur_d"][0] < 0.75
    assert o.feats["dip_asym"][0] > 0.3
    assert o.feats["M"][0] < 0 < o.feats["M"][0] + 1  # dipper: negative flux M
    assert o.anom[0, 295:310].any()


def test_outburst_found_by_bump_search():
    t, f, e = _batch(seed=1)
    p = {"kind": "outburst", "depth": 0.01, "duration_d": 0.5, "sign": 1, "norm": 1.0,
         "components": [{"shape": "fred", "tc": float(t[900]), "rise": 0.1,
                         "tau": 0.4 / np.log(2), "rise_kind": "gauss"}]}
    f[3] *= binned_model(p, t, BIN)
    o = channel_features(f, e, t, BIN)
    assert np.argmax(o.feats["bump_snr"]) == 3


def test_periodic_star_has_low_q():
    t, f, e = _batch(seed=2)
    f[0] *= 1 + 0.01 * np.sin(2 * np.pi * t / 2.3)
    rng = np.random.default_rng(5)
    walk = np.cumsum(rng.standard_normal(t.size)) * 1e-3
    f[1] *= 1 + walk - walk.mean()
    o = channel_features(f, e, t, BIN)
    assert o.feats["Q"][0] < 0.1
    assert o.feats["Q"][1] > 0.3
    assert o.feats["var_frac"][0] > 0.9


def test_nanquantile_matches_numpy():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((7, 50))
    a[rng.random(a.shape) < 0.2] = np.nan
    for q in (0.01, 0.5, 0.9):
        np.testing.assert_allclose(nanquantile_last(a, q)[0], np.nanquantile(a, q, axis=1))


def test_stratum_ecdf_and_midrank():
    ref = np.array([[1.0], [2.0], [3.0], [4.0]])
    strata = np.array([0, 0, 1, 1])
    e = StratumECDF().fit(ref, strata)
    out = e.transform(np.array([[2.5], [2.5], [np.nan]]), np.array([0, 1, 0]))[:, 0]
    np.testing.assert_allclose(out, [1.0, 0.0, 0.0])
    assert midrank_pct(np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0]))[0] == 0.375


def test_rank_within_uses_matching_split_and_stratum():
    real = pd.DataFrame({"split": ["val"] * 3 + ["test"] * 3, "stratum": [1, 1, 2, 1, 1, 1],
                         "score": [0.1, 0.2, 0.9, 0.5, 0.6, 0.7]})
    q = pd.DataFrame({"split": ["val", "test", "val"], "stratum": [1, 1, 2],
                      "score": [0.3, 0.3, 0.3]})
    np.testing.assert_allclose(rank_within(real, q, ["split", "stratum"]), [1.0, 0.0, 0.0])


def test_strata_definition():
    assert list(tmag_bin([9.0, 10.0, 10.5, 12.9, 13.5, np.nan])) == [0, 0, 1, 3, 4, 4]
    assert stratum_id(11.5, 4, 4) == 2 * 16 + 15


def test_common_mode_bins_need_fraction_and_count():
    num = np.zeros((2, 16, 10))
    den = np.full((2, 16, 10), 200.0)
    num[0, 3, 5] = 30  # 15% of the CCD dips at once
    num[0, 3, 6] = 3  # too few stars
    cm = common_mode_bins(num, den, np.full((2, 16), 0.01), min_count=5, min_frac=0.05)
    assert cm[0, 3, 5] and not cm[0, 3, 6] and cm.sum() == 1
