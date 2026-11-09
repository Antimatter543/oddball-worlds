import numpy as np
import pytest

from oddball.tess.lightcurve import (
    DEFAULT_QUALITY_BITMASK,
    bin_centres,
    bin_lightcurve,
    normalise,
    quality_mask,
    robust_std,
)

W = 30.0 / 1440.0  # 30 min in days


# ---------------------------------------------------------------- quality mask


def test_quality_mask_drops_default_bits_and_keeps_others():
    q = np.array([0, 32, 128, 512, 2048, 4096, 1, 16384, 256])
    good = quality_mask(q)
    # 32 desat, 128 manual, 512 impulsive, 1 attitude, 16384 bad cal are in the default mask;
    # straylight (2048/4096) and discontinuity (256) are kept on purpose.
    assert good.tolist() == [True, False, False, False, True, True, False, False, True]


def test_quality_mask_requires_finite_arrays():
    q = np.zeros(4, dtype=int)
    f = np.array([1.0, np.nan, 1.0, np.inf])
    e = np.array([1.0, 1.0, np.nan, 1.0])
    assert quality_mask(q, f, e).tolist() == [True, False, False, False]


def test_quality_mask_custom_bitmask_and_combined_bits():
    q = np.array([0, 2048, 32 | 2048])
    assert quality_mask(q, bitmask=2048).tolist() == [True, False, False]
    assert quality_mask(q).tolist() == [True, True, False]
    assert DEFAULT_QUALITY_BITMASK == 17087


# ---------------------------------------------------------------- normalisation


def test_normalise_median_one_and_errors_scaled():
    f = np.array([100.0, 102.0, 98.0, 1e6, np.nan])
    e = np.full(5, 2.0)
    good = np.array([True, True, True, False, False])
    fn, en, med = normalise(f, e, good)
    assert med == 100.0
    assert np.median(fn[good]) == 1.0
    np.testing.assert_allclose(en, 0.02)
    # masked outlier does not move the median but is still scaled
    assert fn[3] == pytest.approx(1e4)


@pytest.mark.parametrize("flux", [[np.nan, np.nan], [-5.0, -3.0], [0.0, 0.0]])
def test_normalise_bad_median_gives_nan(flux):
    f = np.array(flux)
    fn, en, med = normalise(f, np.ones_like(f), np.isfinite(f))
    assert np.isnan(med)
    assert np.all(np.isnan(fn)) and np.all(np.isnan(en))


def test_normalise_no_good_points():
    f = np.array([1.0, 2.0])
    fn, _, med = normalise(f, np.ones(2), np.zeros(2, bool))
    assert np.isnan(med) and np.all(np.isnan(fn))


# ---------------------------------------------------------------- binning


def _cadence(t0, n, dt_min=10.0):
    return t0 + np.arange(n) * dt_min / 1440.0


def test_binning_constant_signal_and_error_propagation():
    # Start exactly on a bin edge so every bin gets 3 x 10-min cadences.
    t = _cadence(100 * W + 1e-9, 30)
    f = np.ones_like(t)
    e = np.full_like(t, 0.003)
    b = bin_lightcurve(t, f, e, np.ones_like(t, bool), W)
    assert b.bin0 == 100 and b.nbins == 10
    assert np.all(b.npts == 3)
    np.testing.assert_allclose(b.flux, 1.0)
    np.testing.assert_allclose(b.flux_err, 0.003 / np.sqrt(3))
    np.testing.assert_allclose(b.time(W), (100 + np.arange(10) + 0.5) * W)


def test_binning_mean_matches_manual():
    t = np.array([0.1, 0.2, 0.9, 1.1]) * W
    f = np.array([1.0, 2.0, 3.0, 10.0])
    b = bin_lightcurve(t, f, np.ones(4), np.ones(4, bool), W)
    assert b.bin0 == 0 and b.nbins == 2
    np.testing.assert_allclose(b.flux, [2.0, 10.0])
    np.testing.assert_allclose(b.flux_err, [np.sqrt(3) / 3, 1.0])
    assert b.npts.tolist() == [3, 1]


def test_binning_gap_gives_empty_interior_bins():
    t = np.concatenate([_cadence(10.0, 30), _cadence(12.0, 30)])  # ~1.8 d gap
    f = np.ones_like(t)
    b = bin_lightcurve(t, f, np.full_like(t, 1e-3), np.ones_like(t, bool), W)
    empty = b.npts == 0
    assert empty.sum() > 50  # the gap is preserved as NaN bins, not squeezed out
    assert np.all(np.isnan(b.flux[empty])) and np.all(np.isnan(b.flux_err[empty]))
    assert np.all(np.isfinite(b.flux[~empty]))
    # first and last bins are non-empty (range is trimmed to the data)
    assert b.npts[0] > 0 and b.npts[-1] > 0
    assert b.nbins == int(np.floor(t.max() / W)) - int(np.floor(t.min() / W)) + 1


def test_binning_ignores_nans_and_masked_points():
    t = _cadence(50 * W + 1e-9, 6)  # two bins of three
    f = np.array([1.0, np.nan, 1.0, 5.0, 5.0, 5.0])
    e = np.array([0.1, 0.1, np.nan, 0.1, 0.1, 0.1])
    good = np.array([True, True, True, True, False, True])
    b = bin_lightcurve(t, f, e, good, W)
    # bin 0: only the first point survives (NaN flux, NaN err); bin 1: 2 of 3 (one masked)
    assert b.npts.tolist() == [1, 2]
    np.testing.assert_allclose(b.flux, [1.0, 5.0])
    np.testing.assert_allclose(b.flux_err, [0.1, np.sqrt(2) * 0.1 / 2])


def test_binning_nan_time_is_dropped():
    t = np.array([0.1 * W, np.nan, 0.5 * W])
    b = bin_lightcurve(t, np.ones(3), np.ones(3), np.ones(3, bool), W, quality=np.array([0, 32, 0]))
    assert b.nbins == 1 and b.npts.tolist() == [2]
    assert b.extras["qflags"].tolist() == [0]  # the NaN-time cadence can't be placed in a bin


def test_binning_all_bad_returns_none():
    t = _cadence(1.0, 5)
    assert bin_lightcurve(t, np.full(5, np.nan), np.ones(5), np.ones(5, bool), W) is None
    assert bin_lightcurve(t, np.ones(5), np.ones(5), np.zeros(5, bool), W) is None


def test_qflags_or_includes_masked_cadences():
    t = _cadence(0.0 + 1e-9, 6)
    q = np.array([0, 32, 0, 0, 4096, 2048])
    good = quality_mask(q)  # drops the desat cadence
    b = bin_lightcurve(t, np.ones(6), np.ones(6), good, W, quality=q)
    assert b.extras["qflags"].tolist() == [32, 4096 | 2048]
    assert b.npts.tolist() == [2, 3]


def test_extras_binned_over_own_mask():
    t = _cadence(1e-9, 6)
    sap = np.array([1.0, 3.0, np.nan, 2.0, 2.0, 2.0])
    good = np.ones(6, bool)
    b = bin_lightcurve(t, np.ones(6), np.ones(6), good, W, extras={"sap": (sap, good)})
    np.testing.assert_allclose(b.extras["sap"], [2.0, 2.0])


def test_global_grid_alignment_across_targets():
    # Two targets with different start times share bin edges on the global grid.
    b1 = bin_lightcurve(_cadence(2000.003, 50), np.ones(50), np.ones(50), np.ones(50, bool), W)
    b2 = bin_lightcurve(_cadence(2000.2, 50), np.ones(50), np.ones(50), np.ones(50, bool), W)
    t1, t2 = b1.time(W), b2.time(W)
    common = np.intersect1d(np.round(t1 / W, 6), np.round(t2 / W, 6))
    assert common.size > 0
    np.testing.assert_allclose(bin_centres(b2.bin0, 1, W)[0] / W % 1, 0.5)


def test_dip_depth_preserved_by_binning():
    # A 0.3% box dip lasting 4 h must survive binning at ~full depth (no smoothing/detrending).
    t = _cadence(1e-9, 1440)  # 10 d
    f = np.ones_like(t)
    dip = (t > 5.0) & (t < 5.0 + 4 / 24)
    f[dip] -= 0.003
    b = bin_lightcurve(t, f, np.full_like(t, 1e-4), np.ones_like(t, bool), W)
    assert np.nanmin(b.flux) == pytest.approx(0.997, abs=1e-9)


def test_robust_std():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 2.0, 20000)
    x[:100] = 1e3  # outliers
    assert robust_std(x) == pytest.approx(2.0, rel=0.05)
    assert np.isnan(robust_std(np.array([1.0, np.nan])))
