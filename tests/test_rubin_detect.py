"""Synthetic light-curve tests for the Rubin outburst detector."""

import numpy as np
import pandas as pd
import pytest

from oddball.rubin.detect import DetectConfig, bin_nights, detect_band, detect_object

BASE = {"g": 19.5, "r": 18.6, "i": 18.1}
AMP = {"g": 1.2, "r": 1.0, "i": 0.85}  # outbursts are bluer: scale by band


def cadence(rng, t0=61100.0, t1=61360.0, every=3.0, gaps=((61200.0, 61230.0),),
            bands=("g", "r", "i")):
    """Rubin-like: a night every ~`every` days, two visits ~30 min apart, random band."""
    rows = []
    t = t0
    while t < t1:
        if not any(a <= t < b for a, b in gaps):
            night = np.floor(t) + 0.05 + rng.uniform(0, 0.25)  # ~01-07 UT
            b = rng.choice(bands)
            rows += [(night, b), (night + 0.021, b)]
        t += every * rng.uniform(0.6, 1.4)
    return pd.DataFrame(rows, columns=["mjd", "band"])


def make_lc(rng, profile, err=0.03, **kw):
    """profile(t, band) -> magnitude change (negative = brighter)."""
    df = cadence(rng, **kw)
    base = df["band"].map(BASE).to_numpy()
    dm = np.array([profile(t, b) for t, b in zip(df["mjd"], df["band"])])
    df["mag"] = base + dm + rng.normal(0, err, len(df))
    df["magerr"] = err
    df["bad"] = False
    return df


def fuor(t_start=61150.0, rise=30.0, amp=2.5):
    def f(t, b):
        a = amp * AMP[b]
        if t < t_start:
            return 0.0
        return -a * min(1.0, (t - t_start) / rise)
    return f


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def test_flat_star_not_flagged(rng):
    lc = make_lc(rng, lambda t, b: 0.0)
    r = detect_object(lc)
    assert not r.flagged
    assert "no_sustained_brightening" in r.reasons


def test_fuor_like_rise_flagged(rng):
    lc = make_lc(rng, fuor(t_start=61150.0, rise=30.0, amp=2.5))
    r = detect_object(lc)
    assert r.flagged, r.reasons
    p = r.primary
    assert p.peak_dmag == pytest.approx(2.5 * AMP[p.band], abs=0.25)
    assert p.rise_resolved
    # true rise to 80% of peak: 24 d. The interval brackets it up to night sampling.
    assert p.rise_time_lo <= 24.0 + 3.0
    assert p.rise_time >= 24.0 - 3.0
    assert p.rise_time_lo <= p.rise_time
    assert "multiband_confirmed" in r.flags


def test_rise_time_tracks_fast_rise(rng):
    # short rise, then fade back: the peak is at the end of the rise
    def f(t, b):
        a = 2.0 * AMP[b]
        if t < 61150:
            return 0.0
        if t < 61160:
            return -a * (t - 61150) / 10
        return -a * max(0.0, 1 - (t - 61160) / 120)
    r = detect_object(make_lc(rng, f, bands=("r",)))
    assert r.flagged, r.reasons
    assert r.primary.rise_time < 20.0


def test_single_visit_spike_rejected(rng):
    lc = make_lc(rng, lambda t, b: 0.0)
    lc.loc[40, "mag"] -= 2.5
    r = detect_object(lc)
    assert not r.flagged


def test_single_night_spike_rejected(rng):
    lc = make_lc(rng, lambda t, b: 0.0)
    lc.loc[[40, 41], "mag"] -= 2.5  # both visits of one night
    assert lc.loc[40, "mjd"] == pytest.approx(lc.loc[41, "mjd"], abs=0.05)
    r = detect_object(lc)
    assert not r.flagged


def test_two_isolated_spike_nights_rejected(rng):
    lc = make_lc(rng, lambda t, b: 0.0, bands=("r",))
    lc.loc[[20, 21, 60, 61], "mag"] -= 2.0  # two far-apart nights, faint in between
    r = detect_object(lc)
    assert not r.flagged


def test_gap_inside_outburst_does_not_break_run(rng):
    # outburst spans the season gap (61200-61230): still one consecutive run of nights
    lc = make_lc(rng, fuor(t_start=61185.0, rise=5.0, amp=2.0), bands=("r",))
    r = detect_object(lc)
    assert r.flagged, r.reasons
    assert r.primary.t_run_start < 61200 < r.primary.t_run_end


def test_slow_small_drift_not_flagged(rng):
    lc = make_lc(rng, lambda t, b: -0.6 * (t - 61100) / 260)
    r = detect_object(lc)
    assert not r.flagged


def test_slow_large_rise_flagged_with_long_rise(rng):
    lc = make_lc(rng, lambda t, b: -1.8 * AMP[b] * (t - 61100) / 260)
    r = detect_object(lc)
    assert r.flagged, r.reasons
    assert r.primary.rise_time > 80


def test_below_threshold_not_flagged(rng):
    lc = make_lc(rng, fuor(amp=0.8 / AMP["i"]), bands=("i",))
    r = detect_object(lc)
    assert not r.flagged


def test_noisy_faint_star_needs_significance(rng):
    # 1.1 mag "brightening" but 0.5 mag errors -> not significant
    lc = make_lc(rng, fuor(amp=1.1 / AMP["r"]), err=0.5, bands=("r",))
    lc["magerr"] = 0.5
    r = detect_object(lc)
    assert not r.flagged


@pytest.mark.parametrize("dip_frac", [0.1, 0.2, 0.3, 0.45, 0.6])
@pytest.mark.parametrize("seed", range(5))
def test_dipper_not_flagged(dip_frac, seed):
    # Normal state with frequent 1.5 mag, ~1-night dips. When dips fill the faint end, the
    # baseline sits in the dips and the normal state looks "bright"; the interleave and
    # fragmentation guards must reject it.
    rng = np.random.default_rng(seed)
    dips = {}

    def f(t, b):
        n = int(np.floor(t))
        if n not in dips:
            dips[n] = rng.uniform() < dip_frac
        return 1.5 if dips[n] else 0.0
    r = detect_object(make_lc(rng, f))
    assert not r.flagged, (r.primary, r.reasons)


def test_band_inconsistent_rejected(rng):
    lc = make_lc(rng, lambda t, b: (-2.0 if (b == "r" and t > 61150) else 0.0))
    r = detect_object(lc)
    assert not r.flagged
    assert any(x.startswith("band_inconsistent") for x in r.reasons)


def test_single_band_flag(rng):
    lc = make_lc(rng, fuor(), bands=("r",))
    r = detect_object(lc)
    assert r.flagged
    assert "single_band" in r.flags


def test_saturated_baseline_rejected(rng):
    lc = make_lc(rng, fuor())
    lc["mag"] -= 4.0  # baseline ~14.5-15.5: Rubin-saturated
    r = detect_object(lc)
    assert not r.flagged
    assert "baseline_saturated" in r.reasons


def test_bad_flagged_epochs_ignored(rng):
    lc = make_lc(rng, lambda t, b: 0.0, bands=("r",))
    sel = (lc["mjd"] > 61150) & (lc["mjd"] < 61190)
    lc.loc[sel, "mag"] -= 2.0
    lc.loc[sel, "bad"] = True  # e.g. bad-subtraction / cosmic-ray / edge flags
    r = detect_object(lc)
    assert not r.flagged


def test_already_bright_at_start_marks_rise_unresolved(rng):
    # fading from outburst: bright at first epochs, then quiescent
    lc = make_lc(rng, lambda t, b: -2.0 * AMP[b] if t < 61160 else 0.0)
    r = detect_object(lc)
    assert r.flagged, r.reasons
    assert not r.primary.rise_resolved
    assert "rise_unresolved" in r.flags


def test_too_few_nights():
    r = detect_band([61000.1, 61000.12, 61003.1], [18.0, 18.0, 17.0], [0.02] * 3)
    assert r.status == "too_few"


def test_bin_nights_groups_same_night_pairs():
    t = np.array([61000.10, 61000.12, 61001.05, 61001.30])
    tn, mn, en = bin_nights(t, np.array([18.0, 18.2, 17.0, 17.2]), np.full(4, 0.04))
    assert tn.size == 2
    assert mn == pytest.approx([18.1, 17.1])
    assert en[0] == pytest.approx(0.04 / np.sqrt(2))


def test_config_threshold_respected(rng):
    lc = make_lc(rng, fuor(amp=1.5 / AMP["r"]), bands=("r",))
    assert detect_object(lc).flagged
    assert not detect_object(lc, DetectConfig(dmag_min=2.0)).flagged


# --- template (reference image) baseline: alert-only light curves -------------------------

def _alert_only(lc, keep):
    """Mimic alert-only sampling: keep only visits that differ from the template."""
    return lc[keep].reset_index(drop=True)


def _with_template(lc, mags):
    lc = lc.copy()
    f = 10 ** (-0.4 * (lc["band"].map(mags).to_numpy() - 31.4))
    lc["template_flux"] = f
    lc["template_flux_err"] = 0.01 * f
    return lc


def test_outburst_seen_only_in_alerts_uses_template(rng):
    # quiescent epochs never alert; only the outburst epochs are in the light curve
    lc = make_lc(rng, fuor(t_start=61150.0, rise=5.0, amp=2.0))
    lc = _with_template(_alert_only(lc, lc["mjd"] > 61160), BASE)
    assert detect_object(lc, DetectConfig(use_template=False)).flagged is False
    r = detect_object(lc)
    assert r.flagged, r.reasons
    assert "template_baseline" in r.flags
    assert r.primary.baseline_mag == pytest.approx(BASE[r.primary.band], abs=0.01)


def test_two_alert_nights_with_template_flagged_but_one_is_not(rng):
    lc = make_lc(rng, lambda t, b: -1.5, bands=("r",))
    nights = np.floor(lc["mjd"] - 0.667)
    first2 = nights.isin(sorted(nights.unique())[:2])
    first1 = nights.isin(sorted(nights.unique())[:1])
    assert detect_object(_with_template(lc[first2], BASE)).flagged
    assert not detect_object(_with_template(lc[first1], BASE)).flagged


def test_dip_only_alerts_with_bright_template_not_flagged(rng):
    # a dipper: only the dips alert; template holds the normal (bright) state
    lc = make_lc(rng, lambda t, b: 1.5)
    r = detect_object(_with_template(lc, BASE))
    assert not r.flagged


def test_template_brighter_than_lc_faint_end_keeps_lc_baseline(rng):
    # template taken in outburst (brighter): the LC faint end is the quiescent level
    lc = make_lc(rng, fuor(t_start=61150.0, rise=10.0, amp=2.0), bands=("r",))
    lc = _with_template(lc, {b: m - 2.0 for b, m in BASE.items()})
    r = detect_object(lc)
    assert r.flagged, r.reasons
    assert r.primary.baseline_src == "lc"
