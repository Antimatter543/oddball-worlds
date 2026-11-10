import numpy as np
import pandas as pd
import pytest

from oddball.common.splits import split_of
from oddball.inject import binned_model, draw_params
from oddball.inject.build import select_hosts
from oddball.inject.sampling import draw_dusty_tail, draw_eb
from oddball.knownclass import ebtests as E

DT = 30.0 / 1440.0
L = 1240
T = 2174.0 + (np.arange(L) + 0.5) * DT
VALID = np.ones(L, bool)
VALID[600:640] = False  # orbit gap


def _lc(model, rng, noise=1e-3, red=0.002):
    f = 1 + rng.normal(0, noise, L) + red * np.sin(2 * np.pi * T / 3.1 + rng.uniform(0, 6))
    f = f * model
    f[~VALID] = np.nan
    return f


def _eb(P, dur, depth, sec_ratio, t0=2175.3, phase=0.5):
    comps = []
    for k in range(-2, int(30 / P) + 3):
        for tc, amp in ((t0 + k * P, 1.0), (t0 + (k + phase) * P, sec_ratio)):
            if amp > 0:
                comps.append({"shape": "box", "tc": tc, "width": 0.5 * dur, "edge": 0.25 * dur,
                              "amp": amp})  # fmt: skip
    return binned_model({"components": comps, "norm": 1.0, "sign": -1, "depth": depth}, T, DT)


def _run(f, radius=1.0):
    rl, r, y, sig, snr, mask = E.residuals(f[None, :])
    return E.analyse(r[0], y[0], snr[0], mask[0], sig[0], DT, radius, rl[0])


@pytest.mark.parametrize(
    "P,dur,depth,sr",
    [(2.3, 0.15, 0.05, 0.4), (0.8, 0.08, 0.2, 0.9), (5.1, 0.25, 0.02, 0.0), (11.0, 0.3, 0.1, 0.3)],
)
def test_synthetic_eb_is_periodic_eb(P, dur, depth, sr):
    o = _run(_lc(_eb(P, dur, depth, sr), np.random.default_rng(1)))
    assert o["eb_periodic"] > 0.6
    assert o["is_periodic"]
    # the period found is the true one (or, with an equal secondary, half of it)
    assert min(abs(o["P"] / P - 1), abs(o["P"] / (P / 2) - 1)) < 0.01


def test_long_period_single_eclipse_is_single_not_periodic():
    o = _run(_lc(_eb(40.0, 0.4, 0.1, 0.0), np.random.default_rng(2)))
    assert o["eb_periodic"] < 0.1
    assert o["eb_single_raw"] > 0.6


def test_shallow_secondary_found_at_half_phase():
    # secondary below half the primary depth: not a hit of the primary, found as a secondary
    o = _run(_lc(_eb(4.0, 0.2, 0.05, 0.4), np.random.default_rng(3)))
    assert o["eb_periodic"] > 0.6
    assert abs(o["P"] / 4.0 - 1) < 0.01
    assert o["sec_ok"]
    assert abs(o["sec_phase"] - 0.5) < 0.03


@pytest.mark.parametrize(
    "kind", ["aperiodic_dipper", "quasiperiodic_dipper", "boyajian", "exocomet"]
)
def test_dipper_like_signals_are_not_periodic_ebs(kind):
    rng = np.random.default_rng(11)
    vals = []
    for _ in range(6):
        p = draw_params(kind, rng, T, VALID)
        p["depth"] = float(rng.uniform(0.02, 0.2))
        vals.append(_run(_lc(binned_model(p, T, DT), rng))["eb_periodic"])
    assert np.median(vals) < 0.2
    assert max(vals) < 0.6


def test_exocomet_is_not_single_eb():
    rng = np.random.default_rng(12)
    vals = []
    for _ in range(6):
        p = draw_params("exocomet", rng, T, VALID)
        p["depth"] = float(rng.uniform(0.02, 0.1))
        vals.append(_run(_lc(binned_model(p, T, DT), rng))["eb_single_raw"])
    assert np.median(vals) < 0.3  # asymmetric: the shape factor rejects it


def test_dusty_tail_varying_depths_are_not_periodic_ebs():
    rng = np.random.default_rng(5)
    vals = []
    for _ in range(6):
        p = draw_dusty_tail(rng, T, VALID)
        p["depth"] = 0.03
        vals.append(_run(_lc(binned_model(p, T, DT), rng, red=0.0))["eb_periodic"])
    assert np.median(vals) < 0.2


def test_odd_even_depths_read_as_eb_with_secondary_not_as_varying_depth():
    # an EB with primary 5%, secondary 3% at phase 0.5: the folded-at-half-period reading must
    # not call it a dipper; the factor product stays high
    # (a secondary above half the primary depth reads as P/2 with alternating depths)
    o = _run(_lc(_eb(3.0, 0.15, 0.05, 0.6), np.random.default_rng(4)))
    assert abs(o["P"] / 1.5 - 1) < 0.01
    assert o["depth_fex"] > 0.1 > o["depth_fex_oe"]
    assert o["eb_periodic"] > 0.6
    assert o["c_depth"] > 0.8


def test_flat_light_curve_scores_zero():
    o = _run(_lc(np.ones(L), np.random.default_rng(6)))
    assert o["eb_periodic"] == 0.0 and o["eb_single_raw"] == 0.0


def test_informative_monotone():
    assert E.informative(3.0, 1) == 0.0 and E.informative(10.0, 1) == 1.0
    assert E.informative(5.0, 2) < E.informative(5.0, 6) <= E.informative(50.0, 6) == 1.0
    assert E.informative(0.0, 5) == 0.0


def test_unphysical_duration_lowers_factor():
    base = {
        "is_periodic": True,
        "hit_frac": 1.0,
        "depth_fex": 0.0,
        "time_rms": 0.0,
        "time_tol": 1.0,
        "n_hit": 5,
        "snr_ep": 50.0,
        "asym": 0.0,
        "n_extra": 0,
    }
    assert E.factors({**base, "dur_ratio": 1.0})["c_phys"] == 1.0
    assert E.factors({**base, "dur_ratio": 3.5})["c_phys"] == 0.0
    assert E.factors({**base, "n_extra": 2})["c_extra"] == pytest.approx(E.EXTRA_FACTOR**2)


def _row(**k):
    d = {
        "host_tic": 1,
        "eb_anchor_depth": 0.05,
        "eb_fwhm": 0.1,
        "eb_periodic": 0.9,
        "eb_P": 2.0,
        "eb_single_raw": 0.9,
    }
    d.update(k)
    return d


def test_cross_sector_support_rules():
    this = pd.DataFrame([_row(), _row(host_tic=2), _row(host_tic=3, eb_periodic=0.0, eb_P=np.nan),
                         _row(host_tic=4)])  # fmt: skip
    others = pd.DataFrame(
        [
            {**_row(), "tic": 1, "eb_P": 4.0},  # period at 2x: supports
            {**_row(), "tic": 2, "eb_P": 2.5},  # period disagrees: no support
            {
                **_row(),
                "tic": 3,
                "eb_periodic": 0.0,
                "eb_single_raw": 0.8,
                "eb_anchor_depth": 0.065,
            },
            # lone + lone, depth 1.3x: half weight
            {**_row(), "tic": 4, "eb_anchor_depth": 0.2},  # depth 4x: no support
        ]
    ).drop(columns=["host_tic"])
    x = E.cross_sector(this, others)
    np.testing.assert_allclose(x, [0.9, 0.0, 0.4, 0.0])
    tight = others.copy()
    tight.loc[2, "eb_anchor_depth"] = 0.053  # lone + lone, depth 1.06x: full weight
    assert E.cross_sector(this, tight)[2] == pytest.approx(0.8)
    assert (E.cross_sector(this, others.iloc[:0]) == 0).all()


def test_combine_eb_gates_cross_sector_by_own_shape():
    df = pd.DataFrame({"eb_periodic": [0.2, 0.0, 0.0], "eb_single_raw": [0.0, 0.8, 0.8],
                       "eb_xsec": [0.0, 0.0, 0.9], "eb_c_shape": [1.0, 1.0, 0.1],
                       "eb_c_extra": [1.0, 1.0, 1.0]})  # fmt: skip
    np.testing.assert_allclose(E.combine_eb(df, 0.0), [0.2, 0.0, 0.09])
    np.testing.assert_allclose(E.combine_eb(df, 0.5), [0.2, 0.4, 0.4])


def test_draw_eb_physical_and_null():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = draw_eb(rng, T, VALID, radius=1.0)
        assert p["is_null"] and p["kind"] == "null_eb"
        assert 0.4 <= p["period_d"] <= 40
        assert 0.003 <= p["depth"] <= 0.5
        assert p["duration_d"] <= E.max_duration_d(p["period_d"], 1.0) * 1.05 + 1 / 24
        m = binned_model(p, T, DT)
        assert np.all(m <= 1.0 + 1e-9)


def test_draw_dusty_tail_strictly_periodic_varying_depth():
    rng = np.random.default_rng(1)
    p = draw_dusty_tail(rng, T, VALID)
    tc = np.array([c["tc"] for c in p["components"]])
    np.testing.assert_allclose(np.diff(tc), p["period_d"], rtol=1e-9)
    amps = np.array([c["amp"] for c in p["components"]])
    assert amps.max() == pytest.approx(1.0) and amps.min() < 0.9
    assert not p.get("is_null", False)


def test_select_hosts_excludes_other_sets():
    rng = np.random.default_rng(0)
    n = 3000
    tics = rng.choice(np.arange(10_000_000, 99_000_000), size=n, replace=False)
    meta = pd.DataFrame(
        {
            "tic": tics,
            "tmag": rng.uniform(8, 15, n),
            "camera": rng.integers(1, 5, n),
            "ccd": rng.integers(1, 5, n),
        }
    )
    first, _ = select_hosts(meta, "val", 100, 0, 0, np.random.default_rng(1))
    second, _ = select_hosts(
        meta, "val", 100, 0, 0, np.random.default_rng(1), exclude_tics=first.tic
    )
    assert not second.tic.isin(first.tic).any()
    assert all(split_of(t) == "val" for t in second.tic)
